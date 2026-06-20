# lm-eval Commands

Use `<TASK_DIR>` for custom lm-eval task YAMLs. Remove `--include_path <TASK_DIR>` if the tasks are already installed.

Expand project task-group aliases before calling raw `lm_eval`:

```bash
python eval/expand_tasks.py --list
TASKS=$(python eval/expand_tasks.py afrimgsm_extra_cot_tasks)
```

## Single Task

vLLM:

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=<MODEL>,dtype=bfloat16,max_model_len=<MAX_LEN>,tensor_parallel_size=<TP>,data_parallel_size=<DP>,gpu_memory_utilization=0.85,trust_remote_code=True \
  --tasks <TASKS> \
  --batch_size auto \
  --output_path <OUTPUT_DIR> \
  --include_path <TASK_DIR> \
  --write_out --log_samples --cache_requests true
```

Hugging Face:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=<MODEL>,dtype=bfloat16,parallelize=True,trust_remote_code=True \
  --tasks <TASKS> \
  --batch_size auto \
  --output_path <OUTPUT_DIR> \
  --include_path <TASK_DIR> \
  --write_out --log_samples --cache_requests true
```

## AfroLite Core

```bash
for GROUP in \
  afrimgsm_light_cot_tasks \
  afrimmlu_light_tasks \
  afrixnli_light_tasks \
  belebele_light_tasks \
  sib_light_tasks \
  african_flores_light_tasks \
  injongointent_light_tasks_prompt_1 \
  injongointent_light_tasks_prompt_2 \
  injongointent_light_tasks_prompt_3 \
  injongointent_light_tasks_prompt_4 \
  injongointent_light_tasks_prompt_5
do
  TASKS=$(python eval/expand_tasks.py "$GROUP")
  SHOTS=5
  case "$GROUP" in afrimgsm*) SHOTS=8 ;; esac
  lm_eval \
    --model vllm \
    --model_args pretrained=<MODEL>,dtype=bfloat16,max_model_len=16384,tensor_parallel_size=<TP>,data_parallel_size=<DP>,gpu_memory_utilization=0.85,trust_remote_code=True \
    --tasks "$TASKS" \
    --num_fewshot "$SHOTS" \
    --batch_size auto \
    --output_path "<OUTPUT_DIR>/$GROUP" \
    --include_path <TASK_DIR> \
    --write_out --log_samples --cache_requests true
done
```

## AfroLite Extra

```bash
for GROUP in \
  afrimgsm_extra_cot_tasks \
  afrimmlu_extra_tasks \
  afrixnli_extra_tasks \
  injongointent_extra_tasks \
  african_flores_custom_tasks \
  belebele_custom_tasks \
  sib_custom_tasks
do
  TASKS=$(python eval/expand_tasks.py "$GROUP")
  SHOTS=5
  case "$GROUP" in afrimgsm*) SHOTS=8 ;; esac
  lm_eval \
    --model vllm \
    --model_args pretrained=<MODEL>,dtype=bfloat16,max_model_len=16384,tensor_parallel_size=<TP>,data_parallel_size=<DP>,gpu_memory_utilization=0.85,trust_remote_code=True \
    --tasks "$TASKS" \
    --num_fewshot "$SHOTS" \
    --batch_size auto \
    --output_path "<OUTPUT_DIR>/$GROUP" \
    --include_path <TASK_DIR> \
    --write_out --log_samples --cache_requests true
done
```

## Full Benchmark Aliases

```bash
for GROUP in \
  afrimgsm_cot_tasks \
  afrimmlu_tasks \
  afrixnli_tasks \
  belebele_tasks \
  sib_tasks \
  african_flores_tasks \
  injongointent_tasks \
  mmmlu_tasks
do
  TASKS=$(python eval/expand_tasks.py "$GROUP")
  SHOTS=5
  case "$GROUP" in afrimgsm*) SHOTS=8 ;; esac
  lm_eval \
    --model vllm \
    --model_args pretrained=<MODEL>,dtype=bfloat16,max_model_len=16384,tensor_parallel_size=<TP>,data_parallel_size=<DP>,gpu_memory_utilization=0.85,trust_remote_code=True \
    --tasks "$TASKS" \
    --num_fewshot "$SHOTS" \
    --batch_size auto \
    --output_path "<OUTPUT_DIR>/$GROUP" \
    --include_path <TASK_DIR> \
    --write_out --log_samples --cache_requests true
done
```

## Long Document

DAMT doc10 health, all prompts:

```bash
TASKS=$(python eval/expand_tasks.py damt_doc10_health_tasks)
lm_eval \
  --model hf \
  --model_args pretrained=<MODEL>,dtype=bfloat16,parallelize=True,trust_remote_code=True,max_length=32768 \
  --tasks "$TASKS" \
  --num_fewshot 3 \
  --batch_size 1 \
  --gen_kwargs max_gen_toks=4096,do_sample=False \
  --output_path <OUTPUT_DIR>/damt_doc10_health_tasks \
  --include_path <TASK_DIR> \
  --write_out --log_samples --cache_requests true
```

DAMT doc10 health, one prompt per job:

```bash
for GROUP in \
  damt_doc10_health_tasks_prompt_1 \
  damt_doc10_health_tasks_prompt_2 \
  damt_doc10_health_tasks_prompt_3 \
  damt_doc10_health_tasks_prompt_4
do
  TASKS=$(python eval/expand_tasks.py "$GROUP")
  lm_eval \
    --model hf \
    --model_args pretrained=<MODEL>,dtype=bfloat16,parallelize=True,trust_remote_code=True,max_length=32768 \
    --tasks "$TASKS" \
    --num_fewshot 3 \
    --batch_size 1 \
    --gen_kwargs max_gen_toks=4096,do_sample=False \
    --output_path "<OUTPUT_DIR>/$GROUP" \
    --include_path <TASK_DIR> \
    --write_out --log_samples --cache_requests true
done
```
