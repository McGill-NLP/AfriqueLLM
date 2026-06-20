# Synthetic Data

Generate multilingual synthetic data with a local translation model.

## Prepare Sources

```bash
python data/synthetic/prepare_sources.py \
  --source all \
  --samples_per_domain 2000 \
  --math_samples 2000 \
  --output data/synthetic/source
```

This samples:

- `WebOrganizer/TopicAnnotations-Llama-3.1-8B` into `organized-web/*.jsonl`
- `nvidia/OpenMathReasoning` into `math/openmathreasoning.jsonl`

## Translate

```bash
python data/synthetic/synthetic_translate.py \
  --backend vllm \
  --model <MODEL> \
  --tensor_parallel_size <TP> \
  --input data/synthetic/source/organized-web/all.jsonl \
  --targets afr_Latn swa_Latn amh_Ethi hau_Latn yor_Latn \
  --output data/synthetic/translated
```

For CPU/smaller debug runs:

```bash
python data/synthetic/synthetic_translate.py \
  --backend transformers \
  --model <MODEL> \
  --input data/synthetic/source/math/openmathreasoning.jsonl \
  --targets fra_Latn \
  --max_samples 20 \
  --output data/synthetic/debug
```
