# Code And Math

Prepare public math/code pretraining data.

```bash
python data/code-math/source_recipes.py finemath \
  --output data/mixture/finemath-5b \
  --max_tokens 5000000000

python data/code-math/source_recipes.py cornstack \
  --output data/mixture/cornstack-python-5b \
  --max_tokens 5000000000
```

Sources:

- `HuggingFaceTB/finemath`, config `finemath-4plus`
- `nomic-ai/cornstack-python-v1`

The notebooks are short references for the same recipes.
