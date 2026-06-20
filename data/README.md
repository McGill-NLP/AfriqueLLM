# Data Recipes

Public data preparation entry points for LLaMA-Factory pretraining/SFT files.

## Folders

- `monolingual/`: African monolingual mixture from public HF datasets.
- `code-math/`: FineMath and Cornstack Python sampling.
- `parallel/`: NLLB/OPUS bitext download, COMET scoring, and SFT/PT formatting.
- `synthetic/`: local-model synthetic translation for organized web/math sources.
- `extended-languages/`: extra African language export with repeated sampling.

## Quick Commands

```bash
python data/monolingual/mixture.py \
  --config data/monolingual/mixture-config/unimax-full.csv \
  --dataset_id <YourOrganization>/<YourRepo> \
  --output data/mixture

python data/monolingual/push_dataset_repo.py \
  --input preprocessed_data \
  --repo_id <YourOrganization>/<YourRepo> \
  --revision languageSubset_datasetSplit \
  --private \
  --dry_run

python data/code-math/source_recipes.py finemath \
  --output data/mixture/finemath-5b \
  --max_tokens 5000000000

python data/code-math/source_recipes.py cornstack \
  --output data/mixture/cornstack-python-5b \
  --max_tokens 5000000000

python data/extended-languages/remaining_languages_5times.py \
  --dataset <YourOrganization>/<YourRepo> \
  --output data/mixture/remaining-30-languages-5times

python data/synthetic/prepare_sources.py \
  --source all \
  --output data/synthetic/source

python data/synthetic/synthetic_translate.py \
  --backend vllm \
  --model <MODEL> \
  --input data/synthetic/source/organized-web/all.jsonl \
  --targets afr_Latn swa_Latn hau_Latn \
  --output data/synthetic/translated
```

Monolingual notebook reference: `XuTiany1/data_pretrain`.
