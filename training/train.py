"""
Train een compact CNN-model op 5s WAV-chunks uit index.csv.

Gebruik:
python training/train.py --data /mnt/usb/prepared/index.csv --output models/ --epochs 30 --batch-size 32
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

MEL_PARAMS: dict[str, int] = {
    "sample_rate": 16000,
    "n_mels": 64,
    "n_fft": 1024,
    "hop_length": 512,
}

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
    "lynx_lynx",
]

CLIP_DURATION_S = 5
EARLY_STOPPING_PATIENCE = 5
MODEL_FILENAME = "mammal_cnn.pt"


def _species_slug(value: str) -> str:
    """Normaliseer wetenschappelijke naam naar slug-formaat."""
    normalized = value.strip().lower().replace(" ", "_")
    return re.sub(r"_+", "_", normalized)


def create_mel_transform(mel_params: dict[str, int]) -> torchaudio.transforms.MelSpectrogram:
    """Maak de mel-spectrogram transformatie."""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=mel_params["sample_rate"],
        n_mels=mel_params["n_mels"],
        n_fft=mel_params["n_fft"],
        hop_length=mel_params["hop_length"],
    )


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


class AudioChunkDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset die WAV-chunks uit index.csv leest en omzet naar mel-spectrogrammen."""

    def __init__(self, index_csv: Path, class_to_idx: dict[str, int], mel_params: dict[str, int]) -> None:
        self.class_to_idx = class_to_idx
        self.mel_params = mel_params
        self.expected_samples = mel_params["sample_rate"] * CLIP_DURATION_S
        self.mel_transform = create_mel_transform(mel_params)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.samples: list[tuple[Path, int]] = []

        with index_csv.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                species = _species_slug(row["species_scientific"])
                if species not in self.class_to_idx:
                    continue
                file_path = Path(row["file"])
                self.samples.append((file_path, self.class_to_idx[species]))

        if not self.samples:
            raise ValueError("Geen geldige trainingssamples gevonden in index.csv")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        file_path, label = self.samples[index]
        audio, sample_rate = sf.read(str(file_path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        waveform = torch.from_numpy(audio).unsqueeze(0)

        if sample_rate != self.mel_params["sample_rate"]:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.mel_params["sample_rate"],
            )

        length = waveform.shape[1]
        if length < self.expected_samples:
            pad_size = self.expected_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif length > self.expected_samples:
            waveform = waveform[:, : self.expected_samples]

        mel = self.mel_transform(waveform)
        mel_db = self.to_db(mel).to(dtype=torch.float32)
        return mel_db, label


class MammalCNN(nn.Module):
    """Klein CNN-model voor soortclassificatie op mel-spectrogrammen."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def _set_seed(seed: int) -> None:
    """Zet random seeds voor reproduceerbaarheid."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_dataset(dataset: Dataset[tuple[torch.Tensor, int]]) -> tuple[Dataset, Dataset, Dataset]:
    """Splits dataset in train/val/test met verhouding 70/15/15."""
    total = len(dataset)
    train_size = int(total * 0.70)
    val_size = int(total * 0.15)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


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
    device: torch.device,
    learning_rate: float = 0.001,
) -> tuple[nn.Module, float]:
    """Train model met early stopping en ReduceLROnPlateau."""
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state: dict[str, Any] = copy.deepcopy(model.state_dict())
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            desc="Train",
        )

        with torch.no_grad():
            val_loss, val_acc, _, _ = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                desc="Validatie",
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
            if no_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping geactiveerd.")
                break

    model.load_state_dict(best_state)
    return model, best_val_acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_mapping: dict[int, str],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float], float]:
    """Evalueer model en bereken confusion matrix + per-soort accuracy."""
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        _, overall_acc, labels, preds = _run_epoch(
            model,
            dataloader,
            criterion,
            device,
            optimizer=None,
            desc="Test",
        )

    indices = sorted(class_mapping.keys())
    cm = confusion_matrix(labels, preds, labels=indices)

    per_species: dict[str, float] = {}
    for idx in indices:
        species = class_mapping[idx]
        row_total = float(cm[idx].sum())
        correct = float(cm[idx, idx])
        per_species[species] = correct / row_total if row_total > 0 else 0.0

    return cm, per_species, overall_acc


def save_model(
    model: nn.Module,
    output_dir: Path,
    class_mapping: dict[int, str],
    mel_params: dict[str, int],
    val_accuracy: float,
) -> Path:
    """Sla model + metadata op naar models/mammal_cnn.pt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_mapping": class_mapping,
            "mel_params": mel_params,
            "val_accuracy": val_accuracy,
        },
        model_path,
    )
    return model_path


def parse_args() -> argparse.Namespace:
    """Parse command line argumenten."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Pad naar prepared/index.csv")
    parser.add_argument("--output", default="models", help="Uitvoermap voor model-checkpoint")
    parser.add_argument("--epochs", type=int, default=30, help="Aantal epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batchgrootte")
    parser.add_argument(
        "--species-file",
        default=str(repo_root / "species_config.json"),
        help="Pad naar species_config.json",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="Aantal DataLoader workers")
    return parser.parse_args()


def main() -> None:
    """Start training en evaluatie voor het CNN-model."""
    args = parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    species_file = Path(args.species_file)

    if not data_path.exists():
        print(f"Indexbestand niet gevonden: {data_path}", file=sys.stderr)
        sys.exit(1)

    _set_seed(args.seed)

    class_to_idx, idx_to_class = load_class_mapping(species_file)
    dataset = AudioChunkDataset(data_path, class_to_idx, MEL_PARAMS)
    train_set, val_set, test_set = _split_dataset(dataset)

    if len(train_set) == 0 or len(val_set) == 0 or len(test_set) == 0:
        print("Dataset te klein voor train/val/test split.", file=sys.stderr)
        sys.exit(1)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cpu")
    model = MammalCNN(num_classes=len(class_to_idx)).to(device)

    model, best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=device,
    )

    cm, per_species_acc, test_acc = evaluate_model(model, test_loader, idx_to_class, device)

    print("\nConfusion matrix:")
    print(cm)
    print("\nPer-soort accuracy:")
    for species, acc in per_species_acc.items():
        print(f"- {species}: {acc:.4f}")
    print(f"\nOverall accuracy (test): {test_acc:.4f}")

    model_path = save_model(model, output_dir, idx_to_class, MEL_PARAMS, best_val_acc)
    print(f"\nModel opgeslagen: {model_path}")
    print(f"Beste validatie-accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
