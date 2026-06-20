# AfriqueLLM

[![Paper](https://img.shields.io/badge/arXiv-2601.06395-b31b1b.svg)](https://arxiv.org/abs/2601.06395)
[![Models](https://img.shields.io/badge/Hugging%20Face-AfriqueLLM-yellow)](https://huggingface.co/collections/McGill-NLP/afriquellm)

AfriqueLLM is a suite of open language models adapted to African languages through continued pre-training. This repository contains the public data recipes, training configs, and evaluation commands used for the release.

- Paper: [AfriqueLLM: How Data Mixing and Model Architecture Impact Continued Pre-training for African Languages](https://arxiv.org/abs/2601.06395)
- Models: [McGill-NLP AfriqueLLM collection](https://huggingface.co/collections/McGill-NLP/afriquellm)
- Evaluation: lm-eval based benchmark commands in [`eval/INSTRUCTION.md`](eval/INSTRUCTION.md)

## Model Zoo

The Hugging Face collection is the source of truth for released model paths.

| Model path | Base model | Training tokens | Config | Notes |
|---|---|---:|---|---|
| [`McGill-NLP/AfriqueQwen-14B`](https://huggingface.co/McGill-NLP/AfriqueQwen-14B) | `Qwen/Qwen3-14B-Base` | ~26B | `train/configs/qwen3-14b.yaml` | flagship Qwen model |
| [`McGill-NLP/AfriqueQwen-8B`](https://huggingface.co/McGill-NLP/AfriqueQwen-8B) | `Qwen/Qwen3-8B-Base` | ~26B | `train/configs/qwen3-8b.yaml` | Qwen 8B |
| [`McGill-NLP/AfriqueQwen-4B`](https://huggingface.co/McGill-NLP/AfriqueQwen-4B) | `Qwen/Qwen3-4B-Base` | ~26B | `train/configs/qwen3-4b.yaml` | Qwen 4B |
| [`McGill-NLP/AfriqueQwen3.5-4B`](https://huggingface.co/McGill-NLP/AfriqueQwen3.5-4B) | `Qwen/Qwen3.5-4B-Base` | ~26B | `train/configs/qwen3.5-4b.yaml` | Qwen 3.5 4B |
| [`McGill-NLP/AfriqueQwen3.5-4B-ExtendedCM`](https://huggingface.co/McGill-NLP/AfriqueQwen3.5-4B-ExtendedCM) | `Qwen/Qwen3.5-4B-Base` | ~34B | `train/configs/qwen3.5-4b.yaml` | extended code/math |
| [`McGill-NLP/AfriqueGemma-12B`](https://huggingface.co/McGill-NLP/AfriqueGemma-12B) | `google/gemma-3-12b-pt` | ~26B | `train/configs/gemma3-12b-pt.yaml` | Gemma 12B |
| [`McGill-NLP/AfriqueGemma-4B`](https://huggingface.co/McGill-NLP/AfriqueGemma-4B) | `google/gemma-3-4b-pt` | ~26B | `train/configs/gemma3-4b-pt.yaml` | Gemma 4B |
| [`McGill-NLP/AfriqueLlama-8B`](https://huggingface.co/McGill-NLP/AfriqueLlama-8B) | `meta-llama/Llama-3.1-8B` | ~26B | `train/configs/llama3.1-8b.yaml` | Llama 3.1 8B |

See each Hugging Face model card for supported languages, evaluation results, and license details.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "McGill-NLP/AfriqueQwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

prompt = "Bawo ni o se n se?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

For serving:

```bash
vllm serve McGill-NLP/AfriqueQwen-14B --tensor-parallel-size <TP>
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U transformers datasets huggingface_hub tqdm
pip install -U vllm lm_eval
```

Training uses LLaMA-Factory. Install it so `llamafactory-cli` is available before running the training configs.

## Repository Layout

| Path | Purpose |
|---|---|
| `data/monolingual/` | African monolingual download notes and UniMax mixture creation |
| `data/code-math/` | FineMath and Cornstack Python sampling recipes |
| `data/parallel/` | NLLB/OPUS download, COMET scoring, and parallel-data formatting |
| `data/extended-languages/` | additional African language export recipe |
| `data/synthetic/` | synthetic translation source preparation and local translation pipeline |
| `train/configs/` | one LLaMA-Factory continued-pretraining YAML per base model |
| `train/deepspeed/` | ZeRO-1 and ZeRO-2 DeepSpeed configs |
| `eval/` | lm-eval commands and task-alias expansion |

Run commands from the repository root unless noted otherwise.

## Data Sources

The CPT mixture combines:

- African monolingual text from public web corpora, including FineWeb2, WURA, and MADLAD-400.
- Code and math data from Cornstack Python and FineMath.
- Synthetic translated data from organized web and math sources.
- Parallel data from NLLB/OPUS-style bitext sources for formatting and evaluation support.

## Data Preparation

Use `data/monolingual/download.ipynb` to collect and preprocess monolingual sources into `preprocessed_data/<lang>/`. Then create the HF dataset repo layout used by the mixture scripts:

```bash
python data/monolingual/push_dataset_repo.py \
  --input preprocessed_data \
  --repo_id <YourOrganization>/<YourRepo> \
  --revision languageSubset_datasetSplit \
  --private \
  --dry_run
```

Remove `--dry_run` to upload. Then create the monolingual UniMax mixture:

```bash
python data/monolingual/mixture.py \
  --config data/monolingual/mixture-config/unimax-full.csv \
  --dataset_id <YourOrganization>/<YourRepo> \
  --output data/mixture
```

Prepare math and code data:

```bash
python data/code-math/source_recipes.py finemath \
  --output data/mixture/finemath-5b \
  --max_tokens 5000000000

python data/code-math/source_recipes.py cornstack \
  --output data/mixture/cornstack-python-5b \
  --max_tokens 5000000000
```

Prepare extended-language and parallel data:

```bash
python data/extended-languages/remaining_languages_5times.py \
  --dataset <YourOrganization>/<YourRepo> \
  --output data/mixture/remaining-30-languages-5times

python data/parallel/parallel_data.py \
  --config data/parallel/parallel_nllb.yaml \
  --orig_dir data/parallel/parquet \
  --score_dir data/parallel/comet-score
```

See [`data/README.md`](data/README.md) for folder-level details.

## Training

Training configs use LLaMA-Factory with full-parameter continued pre-training, BF16, sequence packing, Flash Attention, Liger kernels, and DeepSpeed. The top comment in each YAML records the actual node/GPU, ZeRO stage, batch size, and gradient accumulation used for that run. The scripts are used for demonstration only. For reproducibility, please adjust paths, cluster settings, and hyperparameters as needed, and run end-to-end.

Single-node or launcher-managed run:

```bash
llamafactory-cli train train/configs/qwen3.5-4b.yaml \
  dataset_dir=data/mixture \
  dataset=mixture-unimax-full \
  output_dir=outputs/qwen3.5-4b
```

Generic multi-node launcher:

```bash
bash train/train_multinode.sh \
  train/configs/qwen3.5-4b.yaml \
  mixture-unimax-full \
  outputs/qwen3.5-4b
```

Training config summary:

| Config | GPUs | ZeRO | Batch/GPU | Grad accum |
|---|---:|---:|---:|---:|
| `train/configs/gemma3-4b-pt.yaml` | 4 nodes x 4 GPUs | 2 | 8 | 2 |
| `train/configs/gemma3-12b-pt.yaml` | 16 nodes x 4 GPUs | 1 | 1 | 4 |
| `train/configs/qwen3-4b.yaml` | 4 nodes x 8 GPUs | 1 | 8 | 1 |
| `train/configs/qwen3-8b.yaml` | 16 nodes x 4 GPUs | 1 | 4 | 1 |
| `train/configs/qwen3-14b.yaml` | 16 nodes x 4 GPUs | 2 | 2 | 2 |
| `train/configs/qwen3.5-4b.yaml` | 8 nodes x 8 GPUs | 2 | 2 | 2 |
| `train/configs/llama3.1-8b.yaml` | 16 nodes x 4 GPUs | 1 | 4 | 1 |

## Evaluation

Use lm-eval with vLLM for standard benchmarks:

```bash
TASKS=$(python eval/expand_tasks.py afrimgsm_light_cot_tasks)
lm_eval \
  --model vllm \
  --model_args pretrained=<MODEL>,dtype=bfloat16,max_model_len=16384,tensor_parallel_size=<TP>,data_parallel_size=<DP>,trust_remote_code=True \
  --tasks "$TASKS" \
  --num_fewshot 8 \
  --batch_size auto \
  --output_path <OUTPUT_DIR> \
  --include_path <TASK_DIR> \
  --write_out --log_samples
```

Long-document and full benchmark commands are listed in [`eval/INSTRUCTION.md`](eval/INSTRUCTION.md).

## Reproducibility Note

This public release is a cleaned version of the original experiment code. Because the pipelines are large-scale, we did not rerun every data preparation, training, and evaluation job end-to-end after sanitization. The workflow steps are preserved, while private paths, cluster-specific settings, credentials, caches, logs, generated datasets, and checkpoints were removed or replaced with placeholders.

## Citation

```bibtex
@misc{yu2026afriquellmdatamixingmodel,
      title={AfriqueLLM: How Data Mixing and Model Architecture Impact Continued Pre-training for African Languages},
      author={Hao Yu and Tianyi Xu and Michael A. Hedderich and Wassim Hamidouche and Syed Waqas Zamir and David Ifeoluwa Adelani},
      year={2026},
      eprint={2601.06395},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.06395},
}
```
