# Extended Languages

Export the additional African language data and repeat sampling five times.

```bash
python data/extended-languages/remaining_languages_5times.py \
  --dataset <YourOrganization>/<YourRepo> \
  --output data/mixture/remaining-30-languages-5times \
  --sample_times 5
```

Use `--languages` to restrict the run and `--no-streaming` when a local datasets cache is preferred.
