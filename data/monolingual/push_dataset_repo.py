"""Push preprocessed monolingual parquet files to the HF dataset layout used by training scripts."""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi


DEFAULT_REVISION = "languageSubset_datasetSplit"
DEFAULT_REPO_ID = "<YourOrganization>/<YourRepo>"


def infer_split_name(lang: str, path: Path) -> str | None:
    stem = path.stem.lower()
    if "documentlevel" in stem:
        return "wura_documentLevel"
    if "finedu" in stem or "fwedu" in stem or "finewebedu" in stem:
        return "finewebEdu"
    if lang == "eng_Latn" and "fw2" in stem:
        return "finewebEdu"
    if "fw2" in stem or "fineweb2" in stem:
        return "fineweb2"
    if "ml400" in stem or "madlad" in stem:
        return "madlad400"
    if "extra" in stem:
        return "extraData"
    if "wura" in stem:
        return "wura"
    return None


def collect_language_files(input_dir: Path, languages: set[str] | None):
    for lang_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        lang = lang_dir.name
        if languages and lang not in languages:
            continue

        split_files = defaultdict(list)
        for parquet_file in sorted(lang_dir.glob("*.parquet")):
            split = infer_split_name(lang, parquet_file)
            if split is None:
                logging.warning("Skipping %s: could not infer split name", parquet_file)
                continue
            split_files[split].append(str(parquet_file))

        if split_files:
            yield lang, dict(split_files)
        else:
            logging.warning("No parquet files found for %s", lang)


def read_expected_splits(csv_paths):
    expected = defaultdict(set)
    for csv_path in csv_paths:
        if not csv_path.exists():
            logging.warning("Mixture config not found: %s", csv_path)
            continue
        with csv_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                subset = row.get("subset", "").strip()
                splits = [item.strip() for item in row.get("split", "").split(",") if item.strip()]
                expected[subset].update(splits)
    return expected


def maybe_create_repo(repo_id: str, revision: str, private: bool, token: str | None):
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    if revision and revision != "main":
        api.create_branch(repo_id, repo_type="dataset", branch=revision, exist_ok=True)


def push_language(repo_id, revision, lang, split_files, token, max_shard_size):
    dataset = load_dataset("parquet", data_files=split_files)
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    dataset.push_to_hub(
        repo_id,
        config_name=lang,
        revision=revision,
        token=token,
        max_shard_size=max_shard_size,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Create the language-config/source-split HF dataset repo.")
    parser.add_argument("--input", default="preprocessed_data", help="Directory with <lang>/*.parquet files.")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--max_shard_size", default="1GB")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--mixture_config",
        nargs="*",
        type=Path,
        default=[
            Path("data/monolingual/mixture-config/unimax-full.csv"),
            Path("data/monolingual/mixture-config/unimax-4b.csv"),
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    languages = set(args.languages) if args.languages else None
    expected = read_expected_splits(args.mixture_config)
    collected = list(collect_language_files(input_dir, languages))

    if args.dry_run:
        logging.info("Dry run. Would create/update %s at revision %s", args.repo_id, args.revision)
    else:
        maybe_create_repo(args.repo_id, args.revision, args.private, args.token)

    for lang, split_files in collected:
        missing = expected.get(lang, set()) - set(split_files)
        if missing:
            logging.warning("%s is missing expected splits: %s", lang, ", ".join(sorted(missing)))

        summary = ", ".join(f"{split}={len(files)} files" for split, files in sorted(split_files.items()))
        if args.dry_run:
            logging.info("Would push config %s with %s", lang, summary)
            continue

        logging.info("Pushing config %s with %s", lang, summary)
        push_language(args.repo_id, args.revision, lang, split_files, args.token, args.max_shard_size)


if __name__ == "__main__":
    main()
