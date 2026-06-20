# Parallel Data

**Due to copyright restrictions, we cannot redistribute the NLLB Moses files. You must download them yourself and place them in `data/parallel/moses`. Then you can run the following scripts to convert to parquet, score with COMET, and format for LLaMA-Factory.**

Download NLLB Moses files, convert to parquet, score with COMET, then format for LLaMA-Factory.

```bash
OUTPUT_DIR=data/parallel/moses \
PARQUET_DIR=data/parallel/parquet \
bash data/parallel/nllb_download.sh
```

```bash
DATA_DIR=data/parallel GPUS=4 bash data/parallel/score_normal.sh
DATA_DIR=data/parallel GPUS=4 bash data/parallel/score_large.sh
DATA_DIR=data/parallel bash data/parallel/score_merge.sh
```

```bash
python data/parallel/parallel_data.py \
  --config data/parallel/parallel_nllb.yaml \
  --orig_dir data/parallel/parquet \
  --score_dir data/parallel/comet-score \
  --jobs 8
```

Outputs are written under `data/parallel/processed/<config_name>/`.
