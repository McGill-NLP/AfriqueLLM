#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/parallel}"
GPUS="${GPUS:-4}"
SCORE_SCRIPT="${SCORE_SCRIPT:-data/parallel/score_parquet.py}"

NORMAL_PAIRS=(
  af-sw af-so pt-sw ar-sw ar-so ar-pt af-fr fr-sw fr-so ar-fr af-am am-sw am-so am-fr
  af-ha ha-sw ha-so fr-ha am-ha af-rw rw-sw rw-so fr-rw am-rw ha-rw af-zu sw-zu so-zu
  ar-zu fr-zu am-zu ha-zu rw-zu af-ig ig-sw ig-so fr-ig am-ig ha-ig ig-rw ig-zu fr-mg
  af-xh sw-xh so-xh ar-xh fr-xh am-xh ha-xh rw-xh xh-zu ig-xh af-sn sn-sw sn-so fr-sn
  am-sn ha-sn rw-sn sn-zu ig-sn sn-xh af-yo sw-yo so-yo fr-yo am-yo ha-yo rw-yo yo-zu
  ig-yo xh-yo sn-yo af-ny ny-sw ny-so fr-ny am-ny ha-ny rw-ny ny-zu ig-ny ny-xh ny-sn
  ny-yo af-st st-sw so-st fr-st am-st ha-st rw-st st-zu ig-st st-xh sn-st st-yo ny-st
  af-ti sw-ti so-ti fr-ti am-ti ha-ti rw-ti ti-zu ig-ti ti-xh sn-ti ti-yo ny-ti st-ti
  af-om om-sw om-so ar-om fr-om am-om ha-om rw-om om-zu ig-om om-xh om-sn om-yo ny-om
  om-st om-ti af-tn sw-tn so-tn fr-tn am-tn ha-tn rw-tn tn-zu ig-tn tn-xh sn-tn tn-yo
  ny-tn st-tn ti-tn om-tn af-en en-sw en-so ar-en am-en en-ha en-rw en-zu en-ig en-mg
  en-xh en-sn en-yo en-ny en-st en-ti en-om en-tn so-sw
)

for GPU_ID in $(seq 0 "$((GPUS - 1))"); do
  (
    for i in "${!NORMAL_PAIRS[@]}"; do
      if [ "$((i % GPUS))" -eq "$GPU_ID" ]; then
        pair="${NORMAL_PAIRS[$i]}"
        CUDA_VISIBLE_DEVICES="$GPU_ID" python "$SCORE_SCRIPT" \
          --score "${DATA_DIR}/parquet/${pair}.parquet" \
          --gpus 1 \
          --num_workers 8 \
          --output_dir "${DATA_DIR}/comet-score/" \
          --batch_size 256 \
          --no_plots
      fi
    done
  ) &
done

wait
