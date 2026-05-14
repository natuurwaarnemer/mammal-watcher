#!/usr/bin/env bash
# MammalRadar hertraining script
# Gebruik: bash retrain.sh [--skip-download] [--skip-prepare] [--max-per-species 1000]
#
# Stappen:
#   1. Download nieuwe audio van GBIF (23 soorten, max 1000 per soort)
#   2. Bereid dataset voor (10s chunks, nieuwe index.csv)
#   3. Train CNN model (gebalanceerd, max 1000 per soort)
#   4. Herstart mammal-watcher container

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_DIR="/mnt/usb/audio"
PREPARED_DIR="/mnt/usb/prepared"
MODEL_DIR="${HOME}/mammal-watcher/models"
SPECIES_FILE="${REPO_ROOT}/species_config.json"
MAX_PER_SPECIES=1000
SKIP_DOWNLOAD=0
SKIP_PREPARE=0

usage() {
  echo "Gebruik: bash retrain.sh [--skip-download] [--skip-prepare] [--max-per-species N]"
}

read_val_accuracy() {
  local model_path="$1"
  if [[ ! -f "$model_path" ]]; then
    return 0
  fi
  python - "$model_path" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
value = checkpoint.get("val_accuracy")
print("" if value is None else value)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --skip-prepare)
      SKIP_PREPARE=1
      shift
      ;;
    --max-per-species)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      MAX_PER_SPECIES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Onbekende optie: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$MAX_PER_SPECIES" =~ ^[0-9]+$ ]] || [[ "$MAX_PER_SPECIES" -le 0 ]]; then
  echo "--max-per-species moet een positief geheel getal zijn" >&2
  exit 1
fi

mkdir -p "$AUDIO_DIR" "$PREPARED_DIR" "$MODEL_DIR"
cd "$REPO_ROOT"

OLD_MODEL="${MODEL_DIR}/mammal_cnn.pt"
OLD_VAL_ACCURACY="$(read_val_accuracy "$OLD_MODEL" || true)"

echo "========================================"
echo "MammalRadar hertraining"
echo "Repo: $REPO_ROOT"
echo "Audio: $AUDIO_DIR"
echo "Prepared: $PREPARED_DIR"
echo "Models: $MODEL_DIR"
echo "Max per soort: $MAX_PER_SPECIES"
echo "========================================"

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  echo
  echo "[1/4] Download GBIF-audio"
  python dataset/download_gbif.py \
    --output "$AUDIO_DIR" \
    --max-per-species "$MAX_PER_SPECIES" \
    --species-file "$SPECIES_FILE"
else
  echo
  echo "[1/4] Download overgeslagen (--skip-download)"
fi

if [[ "$SKIP_PREPARE" -eq 0 ]]; then
  echo
  echo "[2/4] Dataset voorbereiden naar 10s chunks"
  python dataset/prepare_dataset.py \
    --input "$AUDIO_DIR" \
    --output "$PREPARED_DIR" \
    --species-file "$SPECIES_FILE"
else
  echo
  echo "[2/4] Prepare overgeslagen (--skip-prepare)"
fi

echo
echo "[3/4] Model trainen"
python training/train.py \
  --data "$PREPARED_DIR/index.csv" \
  --output "$MODEL_DIR" \
  --epochs 30 \
  --max-per-species "$MAX_PER_SPECIES" \
  --species-file "$SPECIES_FILE"

echo
echo "[4/4] mammal-watcher herstarten"
docker compose restart mammal-watcher

NEW_MODEL="${MODEL_DIR}/mammal_cnn.pt"
NEW_VAL_ACCURACY="$(read_val_accuracy "$NEW_MODEL" || true)"

echo
echo "========================================"
echo "Hertraining afgerond"
echo "Oude val_accuracy: ${OLD_VAL_ACCURACY:-onbekend}"
echo "Nieuwe val_accuracy: ${NEW_VAL_ACCURACY:-onbekend}"
echo "Modelbestand: $NEW_MODEL"
echo "========================================"
