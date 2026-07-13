"""
Train een kleine PyTorch MLP op voorberekende BirdNET embeddings.

Gebruik:
    python training/train_mlp.py \
        --embeddings-dir /mnt/usb/embeddings/embeddings_index.csv \
        --output /app/models \
        --epochs 100 \
        --patience 10
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

EMBEDDING_DIM = 1024
MODEL_FILENAME = "mammal_mlp.pt"
TARGET_SPECIES = [
    "vulpes_vulpes",
    "canis_lupus",
    "canis_aureus",
    "martes_martes",
    "martes_foina",
    "meles_meles",
    "lutra_lutra",
    "capreolus_capreolus",
    "cervus_elaphus",
    "sus_scrofa",
    "castor_fiber",
    "sciurus_vulgaris",
    "eliomys_quercinus",
    "lynx_lynx",
    "felis_silvestris",
]

# Soorten in prepared/ die als 'background' worden gelabeld bij het trainen.
# Dit zijn geldige audio-opnames, maar geen NL-doelsoorten voor dit systeem.
BACKGROUND_SPECIES: frozenset[str] = frozenset({
    "gallus_gallus",            # kip — vogel, beste ruis-proxy
    "alces_alces",              # eland — niet in NL
    "bos_taurus",               # koe
    "canis_lupus_familiaris",   # hond
    "capra_hircus",             # geit
    "dama_dama",                # damhert — niet in doellijst
    "equus_caballus",           # paard
    "erinaceus_europaeus",      # egel
    "gulo_gulo",                # veelvraat — niet in NL
    "marmota_marmota",          # marmot — niet in NL
    "myocastor_coypus",         # nutria
    "castor_canadensis",        # Canadese bever — dichte verwant van bever, goede hard negative
    "nyctereutes_procyonoides", # wasbeerhond
    "ondatra_zibethicus",       # muskusrat
    "oryctolagus_cuniculus",    # konijn
    "ovis_aries",               # schaap
    "procyon_lotor",            # wasbeer
    "ursus_arctos",             # beer — niet in NL
})



# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def _species_slug(value: str) -> str:
    """Normaliseer wetenschappelijke naam naar slug-formaat."""
    normalized = value.strip().lower().replace(" ", "_")
    return re.sub(r"_+", "_", normalized)


def _load_species_from_config(species_file: Path) -> list[str]:
    """Laad soorten uit species_config.json in vaste volgorde."""
    if not species_file.exists():
        return TARGET_SPECIES
    with species_file.open(encoding="utf-8") as fh:
        data = json.load(fh)
    species = [_species_slug(item["scientific"]) for item in data.get("species", [])]
    return species if species else TARGET_SPECIES


def load_class_mapping(species_file: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Bouw label-mapping soort -> index en index -> soort."""
    species = _load_species_from_config(species_file)
    class_to_idx = {name: idx for idx, name in enumerate(species)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset die voorberekende .npy embedding bestanden leest."""

    def __init__(
        self,
        index_csv: Path,
        class_to_idx: dict[str, int],
        *,
        max_per_species: int | None = None,
        seed: int = 42,
    ) -> None:
        self.class_to_idx = class_to_idx
        self.samples: list[tuple[Path, int]] = []
        self.samples_per_species: dict[str, int] = {}

        sampled_by_species: dict[str, list[tuple[Path, int]]] = {}
        with index_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                species = _species_slug(row["species_scientific"])
                if species in BACKGROUND_SPECIES:
                    species = "background"
                if species not in self.class_to_idx:
                    continue
                emb_path = Path(row["embedding_file"])
                if not emb_path.exists():
                    continue
                sampled_by_species.setdefault(species, []).append((emb_path, self.class_to_idx[species]))

        rng = random.Random(seed)
        for species in sorted(sampled_by_species):
            candidates = sampled_by_species[species]
            if max_per_species is not None and len(candidates) > max_per_species:
                selected = rng.sample(candidates, max_per_species)
                selected.sort(key=lambda item: str(item[0]))
            else:
                selected = candidates
            self.samples.extend(selected)
            self.samples_per_species[species] = len(selected)

        if not self.samples:
            raise ValueError("Geen geldige embedding-bestanden gevonden in index.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        emb_path, label = self.samples[index]
        embedding = torch.from_numpy(np.load(str(emb_path)).astype(np.float32))
        return embedding, label

    def get_labels(self) -> list[int]:
        return [label for _, label in self.samples]


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------

class MammalMLP(nn.Module):
    """Kleine MLP classifier op BirdNET embeddings."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Klasse-gewichten en sample-gewichten (geen sklearn)
# ---------------------------------------------------------------------------

def _compute_class_weights(
    labels: list[int], num_classes: int, device: torch.device
) -> torch.Tensor:
    """Bereken gebalanceerde klasse-gewichten: weight[c] = total / (n_classes * count[c])."""
    label_array = np.array(labels)
    present_classes = np.unique(label_array)
    n_present = len(present_classes)
    total = len(label_array)

    weights = np.ones(num_classes, dtype=np.float32)
    for cls in present_classes:
        count = int(np.sum(label_array == cls))
        weights[int(cls)] = total / (n_present * count)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def _compute_sample_weights(labels: list[int]) -> torch.DoubleTensor:
    """Bereken sample-gewichten voor WeightedRandomSampler (inverse klassefrequentie)."""
    class_counts = Counter(labels)
    sample_weights = [1.0 / class_counts[label] for label in labels]
    return torch.DoubleTensor(sample_weights)


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------

def _split_indices(
    n: int, train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 42
) -> tuple[list[int], list[int], list[int]]:
    """Splits dataset-indices in train/val/test."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
    desc: str = "train",
) -> tuple[float, float, list[int], list[int]]:
    """Draai één train- of validatie-epoch."""
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    total = 0
    correct = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    iterator = tqdm(dataloader, desc=desc, leave=False)
    for features, labels in iterator:
        features = features.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(features)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
        correct += (preds == labels).sum().item()

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        avg_loss = running_loss / total if total else 0.0
        avg_acc = correct / total if total else 0.0
        iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    epoch_loss = running_loss / total if total else 0.0
    epoch_acc = correct / total if total else 0.0
    return epoch_loss, epoch_acc, all_labels, all_preds


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    patience: int,
    device: torch.device,
    learning_rate: float = 0.001,
    class_weights: torch.Tensor | None = None,
) -> tuple[nn.Module, float]:
    """Train MLP met early stopping en ReduceLROnPlateau."""
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state: dict[str, Any] = copy.deepcopy(model.state_dict())
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc, _, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, desc="Train"
        )

        with torch.no_grad():
            val_loss, val_acc, _, _ = _run_epoch(
                model, val_loader, criterion, device, optimizer=None, desc="Validatie"
            )

        scheduler.step(val_loss)
        print(
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping geactiveerd.")
                break

    model.load_state_dict(best_state)
    return model, best_val_acc


# ---------------------------------------------------------------------------
# Evaluatie (pure numpy, geen sklearn)
# ---------------------------------------------------------------------------

def _numpy_confusion_matrix(
    y_true: list[int], y_pred: list[int], num_classes: int
) -> np.ndarray:
    """Bereken een confusion matrix zonder sklearn."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_mapping: dict[int, str],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float], float]:
    """Evalueer model: confusion matrix + per-soort accuracy."""
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        _, overall_acc, labels, preds = _run_epoch(
            model, dataloader, criterion, device, optimizer=None, desc="Test"
        )

    indices = sorted(class_mapping.keys())
    num_classes = max(indices) + 1
    cm = _numpy_confusion_matrix(labels, preds, num_classes)

    per_species: dict[str, float] = {}
    for idx in indices:
        species = class_mapping[idx]
        row_total = float(cm[idx].sum())
        correct = float(cm[idx, idx])
        per_species[species] = correct / row_total if row_total > 0 else 0.0

    return cm, per_species, overall_acc


# ---------------------------------------------------------------------------
# Model opslaan
# ---------------------------------------------------------------------------

def save_model(
    model: nn.Module,
    output_dir: Path,
    class_mapping: dict[int, str],
    val_accuracy: float,
    training_info: dict[str, Any],
) -> Path:
    """Sla MLP model + metadata op als mammal_mlp.pt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_mapping": class_mapping,
            "model_type": "mlp",
            "input_dim": EMBEDDING_DIM,
            "val_accuracy": val_accuracy,
            "training_info": training_info,
        },
        model_path,
    )
    return model_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        help="Pad naar embeddings_index.csv",
    )
    parser.add_argument("--output", default="models", help="Uitvoermap voor model-checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Aantal epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=64, help="Batchgrootte")
    parser.add_argument("--num-workers", type=int, default=0, help="Aantal DataLoader workers")
    parser.add_argument(
        "--max-per-species",
        type=int,
        default=500,
        help="Maximum aantal clips per soort (voorkomt dominantie)",
    )
    parser.add_argument(
        "--species-file",
        default=str(repo_root / "species_config.json"),
        help="Pad naar species_config.json",
    )
    parser.add_argument(
        "--background-dir",
        default=None,
        help="Map met WAV-bestanden als background/ruis klasse (bijv. feedback/rejected/)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.background_dir:
        print(
            f"ℹ️  Background clips opgegeven: {args.background_dir}\n"
            f"   Voeg deze eerst toe aan embeddings_index.csv via:\n"
            f"   python training/extract_embeddings.py --data {args.background_dir} "
            f"--embeddings-dir /mnt/usb/embeddings --species background\n"
            f"   Daarna opnieuw trainen.",
            file=sys.stderr,
        )

    index_path = Path(args.embeddings_dir)
    output_dir = Path(args.output)
    species_file = Path(args.species_file)
    device = torch.device("cpu")

    if not index_path.exists():
        print(f"Embeddings-index niet gevonden: {index_path}", file=sys.stderr)
        sys.exit(1)
    if args.max_per_species <= 0:
        print("--max-per-species moet groter zijn dan 0", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    class_to_idx, idx_to_class = load_class_mapping(species_file)
    dataset = EmbeddingDataset(
        index_path,
        class_to_idx,
        max_per_species=args.max_per_species,
        seed=args.seed,
    )

    train_idx, val_idx, test_idx = _split_indices(len(dataset), seed=args.seed)

    train_labels = [dataset.samples[i][1] for i in train_idx]
    val_labels = [dataset.samples[i][1] for i in val_idx]

    class_weights = _compute_class_weights(train_labels, len(class_to_idx), device)
    sampler = WeightedRandomSampler(
        weights=_compute_sample_weights(train_labels),
        num_samples=len(train_labels),
        replacement=True,
    )

    from torch.utils.data import Subset

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = MammalMLP(input_dim=EMBEDDING_DIM, num_classes=len(class_to_idx)).to(device)

    model, best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
        class_weights=class_weights.to(device),
    )

    cm, per_species_acc, test_acc = evaluate_model(model, test_loader, idx_to_class, device)

    print("\nConfusion matrix:")
    print(cm)
    print("\nPer-soort accuracy:")
    for species, acc in per_species_acc.items():
        print(f"- {species}: {acc:.4f}")
    print(f"\nOverall accuracy (test): {test_acc:.4f}")

    training_info = {
        "samples_per_species": dataset.samples_per_species,
        "total_samples": len(dataset),
        "clip_distribution": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "max_per_species": args.max_per_species,
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    model_path = save_model(model, output_dir, idx_to_class, best_val_acc, training_info)
    print(f"\nModel opgeslagen: {model_path}")
    print(f"Beste validatie-accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
