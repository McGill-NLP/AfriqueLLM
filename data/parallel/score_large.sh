#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/parallel}"
GPUS="${GPUS:-4}"
SCORE_SCRIPT="${SCORE_SCRIPT:-data/parallel/score_parquet.py}"
SPLIT_SCRIPT="${SPLIT_SCRIPT:-data/parallel/split.py}"
TARGET_GB="${TARGET_GB:-5}"
LARGE_PAIRS=(fr-pt en-pt en-fr)

for pair in "${LARGE_PAIRS[@]}"; do
  parquet_file="${DATA_DIR}/parquet/${pair}.parquet"
  split_dir="${DATA_DIR}/parquet/${pair}"
  split_metadata="${split_dir}/split.json"
  mkdir -p "$split_dir"

  if [ ! -f "$split_metadata" ]; then
    python "$SPLIT_SCRIPT" "$parquet_file" \
      --target-gb "$TARGET_GB" \
      --metadata "$split_metadata" \
      --output-dir "$split_dir"
  fi

  mapfile -t shard_files < <(python - "$split_metadata" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    for part in json.load(f)["parts"]:
        print(part["path"])
PY
)

  for GPU_ID in $(seq 0 "$((GPUS - 1))"); do
    (
      for i in "${!shard_files[@]}"; do
        if [ "$((i % GPUS))" -eq "$GPU_ID" ]; then
          CUDA_VISIBLE_DEVICES="$GPU_ID" python "$SCORE_SCRIPT" \
            --score "${shard_files[$i]}" \
            --gpus 1 \
            --num_workers 8 \
            --output_dir "${DATA_DIR}/comet-score/${pair}/" \
            --batch_size 256 \
            --no_plots
        fi
      done
    ) &
  done

  wait
done
