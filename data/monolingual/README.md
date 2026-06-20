# Monolingual

Build African monolingual pretraining mixtures from `<YourOrganization>/<YourRepo>`.

```bash
python data/monolingual/mixture.py \
  --config data/monolingual/mixture-config/unimax-full.csv \
  --output data/mixture \
  --model_name google/gemma-3-4b-pt \
  --num_workers 8
```

Useful options:

- `--fraction 1/3`: create a smaller mixture.
- `--subsets yor_Latn hau_Latn`: process selected languages.
- `--dataset_id <HF_DATASET>` and `--token <TOKEN>`: use another dataset repo if needed.

`download.ipynb` keeps the source-download notes for FineWeb2, MADLAD-400, Wura, and extra language sources. It prepares local parquet files under `preprocessed_data/<lang>/`.

To create the Hugging Face dataset layout consumed by `mixture.py` and `remaining_languages_5times.py`, push one config per language and one split per source:

```bash
python data/monolingual/push_dataset_repo.py \
  --input preprocessed_data \
  --repo_id <YourOrganization>/<YourRepo> \
  --revision languageSubset_datasetSplit \
  --private \
  --dry_run
```

Remove `--dry_run` to upload. Expected split names are `fineweb2`, `madlad400`, `wura_documentLevel`, `finewebEdu`, and `extraData`.
