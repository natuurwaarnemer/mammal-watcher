"""
Vergelijkt twee matching-strategieen voor NatureLM-audio-training over de eerste N samples:

1. HUIDIGE logica (download_naturelm.py): alleen task~"taxonomic" + laatste-2-woorden van 'output'.
2. METADATA logica: metadata.species (volledige binomiale naam) tegen species_targets.yaml,
   ongeacht task-type.

Doel: is de "dataset uitgeput"-conclusie van 2026-07-13 terecht, of mist de huidige logica
doelsoorten die wel aanwezig zijn maar in andere task-types / velden staan?

Gebruik:
    HF_TOKEN=$(cat /mnt/usb/hf_cache/token) HF_HOME=/mnt/usb/hf_cache \
        venv/bin/python scripts/compare_matching.py --limit 200000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

NATURELM_DATASET = "EarthSpeciesProject/NatureLM-audio-training"


def _slug(scientific: str) -> str:
    return re.sub(r"\s+", "_", scientific.strip().lower())


def _load_targets(yaml_path: Path) -> dict[str, str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return {s["scientific"].strip().lower(): _slug(s["scientific"]) for s in data["species"]}


def _extract_species_from_output(output: str) -> str:
    if not output or not isinstance(output, str):
        return ""
    words = output.strip().split()
    if len(words) >= 2:
        return " ".join(words[-2:])
    return output.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200_000)
    parser.add_argument(
        "--species-file", type=Path, default=Path(__file__).parent.parent / "dataset" / "species_targets.yaml"
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()

    targets = _load_targets(args.species_file)
    print(f"Doelsoorten geladen: {len(targets)}", flush=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("FOUT: 'datasets' package niet gevonden — draai dit binnen de venv.", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset(NATURELM_DATASET, split="train", streaming=True)

    hits_current: dict[str, int] = Counter()
    hits_metadata: dict[str, int] = Counter()
    metadata_hit_tasks: dict[str, Counter] = defaultdict(Counter)  # slug -> task-type counts
    task_type_counts: Counter = Counter()
    scanned = 0
    errors = 0

    for sample in ds:
        if scanned >= args.limit:
            break
        scanned += 1

        task = sample.get("task") or ""
        task_type_counts[task] += 1

        # --- Methode 1: huidige logica ---
        if "taxonomic" in task:
            output = sample.get("output") or ""
            species_name = _extract_species_from_output(output).lower()
            if species_name in targets:
                hits_current[targets[species_name]] += 1

        # --- Methode 2: metadata-logica ---
        meta_raw = sample.get("metadata")
        if meta_raw:
            try:
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                meta_species = (meta.get("species") or "").strip().lower()
            except Exception:
                meta_species = ""
                errors += 1
            if meta_species in targets:
                slug = targets[meta_species]
                hits_metadata[slug] += 1
                metadata_hit_tasks[slug][task] += 1

        if scanned % args.progress_every == 0:
            print(
                f"[{scanned:,}/{args.limit:,}] huidig={sum(hits_current.values())} hits "
                f"({len(hits_current)} soorten) | metadata={sum(hits_metadata.values())} hits "
                f"({len(hits_metadata)} soorten) | fouten={errors}",
                flush=True,
            )

    print("\n=== EINDRESULTAAT ===")
    print(f"Samples gescand: {scanned:,}")
    print(f"Parse-fouten metadata: {errors}")
    print(f"\nTask-type verdeling (top 20):")
    for task, count in task_type_counts.most_common(20):
        print(f"  {task or '(leeg)'}: {count:,}")

    print(f"\n--- Methode 1 (huidig, task~taxonomic + laatste-2-woorden) ---")
    print(f"Totaal hits: {sum(hits_current.values())}, unieke soorten: {len(hits_current)}")
    for slug, n in sorted(hits_current.items(), key=lambda x: -x[1]):
        print(f"  {slug}: {n}")

    print(f"\n--- Methode 2 (metadata.species, alle task-types) ---")
    print(f"Totaal hits: {sum(hits_metadata.values())}, unieke soorten: {len(hits_metadata)}")
    for slug, n in sorted(hits_metadata.items(), key=lambda x: -x[1]):
        task_breakdown = ", ".join(f"{t}={c}" for t, c in metadata_hit_tasks[slug].most_common(5))
        print(f"  {slug}: {n}  [{task_breakdown}]")

    only_metadata = set(hits_metadata) - set(hits_current)
    only_current = set(hits_current) - set(hits_metadata)
    print(f"\n--- Verschil ---")
    print(f"ALLEEN gevonden via metadata (gemist door huidige logica): {sorted(only_metadata)}")
    print(f"ALLEEN gevonden via huidige logica: {sorted(only_current)}")
    print(f"Soorten met 0 hits via BEIDE methodes: {sorted(set(targets.values()) - set(hits_current) - set(hits_metadata))}")


if __name__ == "__main__":
    main()
