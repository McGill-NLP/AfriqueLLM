#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: train_multinode.sh CONFIG DATASET OUTPUT_DIR}
DATASET=${2:?Usage: train_multinode.sh CONFIG DATASET OUTPUT_DIR}
OUTPUT_DIR=${3:?Usage: train_multinode.sh CONFIG DATASET OUTPUT_DIR}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-${SLURM_NNODES:-1}}
MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
MASTER_PORT=${MASTER_PORT:-60210}
NODE_RANK=${SLURM_PROCID:-0}
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

srun bash -c 'FORCE_TORCHRUN=1 \
  DISABLE_VERSION_CHECK=1 \
  NNODES='"$NNODES"' \
  NODE_RANK=${SLURM_PROCID:-0} \
  MASTER_ADDR='"$MASTER_ADDR"' \
  MASTER_PORT='"$MASTER_PORT"' \
  llamafactory-cli train '"$CONFIG"' \
  dataset='"$DATASET"' \
  output_dir='"$OUTPUT_DIR"' \
  dataset_dir=${DATASET_DIR:-data/mixture} \
  tokenized_path=${TOKENIZED_PATH:-data/tokenized/'"$DATASET"'}'
