#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/LLD-MMRI/images}"
CSV_DIR="${CSV_DIR:-$SCRIPT_DIR/3subfold_lesionclassifier}"
RESULT_DIR="${RESULT_DIR:-$SCRIPT_DIR/results}"
JSON_NAME="${JSON_NAME:-ps32_res10_lr1e-4_depth64_epoch100.json}"

NUM_CLASSES="${NUM_CLASSES:-7}"
LR="${LR:-1e-4}"
NET="${NET:-3dres}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_EPOCH="${NUM_EPOCH:-100}"
RESUME="${RESUME:-False}"
CUDA="${CUDA:-True}"
AMP="${AMP:-True}"
FOLDS="${FOLDS:-1 2 3 4 5}"
WEIGHT_PATH="${WEIGHT_PATH:-}"

mkdir -p "$RESULT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

for fold in $FOLDS; do
  CSV_PATH="$CSV_DIR/fold${fold}.csv"
  if [[ ! -f "$CSV_PATH" ]]; then
    echo "Error: missing csv file: $CSV_PATH" >&2
    exit 1
  fi

  echo "Running fold${fold}..."
  cmd=(
    "$PYTHON_BIN" "$SCRIPT_DIR/do_main.py"
    --data_dir "$DATA_DIR"
    --csv_path "$CSV_PATH"
    --resume "$RESUME"
    --cuda "$CUDA"
    --amp "$AMP"
    --result_dir "$RESULT_DIR"
    --json_dir "$JSON_NAME"
    --num_classes "$NUM_CLASSES"
    --lr "$LR"
    --net "$NET"
    --batch_size "$BATCH_SIZE"
    --num_epoch "$NUM_EPOCH"
    --n_fold "fold${fold}"
  )
  if [[ -n "$WEIGHT_PATH" ]]; then
    cmd+=(--weight_path "$WEIGHT_PATH")
  fi
  "${cmd[@]}"
done
