#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/parallel}"
SCORE_SCRIPT="${SCORE_SCRIPT:-data/parallel/score_parquet.py}"
LARGE_PAIRS=(fr-pt en-pt en-fr)

for pair in "${LARGE_PAIRS[@]}"; do
  python "$SCORE_SCRIPT" \
    --merge "${DATA_DIR}/parquet/${pair}/split.json" \
    --output "${DATA_DIR}/comet-score/${pair}.parquet" \
    --output_dir "${DATA_DIR}/comet-score" \
    --overwrite
done
