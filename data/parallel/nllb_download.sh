#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/parallel/moses}"
PARQUET_DIR="${PARQUET_DIR:-data/parallel/parquet}"
JOBS="${JOBS:-16}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PARQUET_DIR"

echo "PLACEHOLDER"