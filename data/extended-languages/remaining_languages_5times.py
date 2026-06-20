"""Build LLaMA-Factory JSONL files for the remaining pretrain languages."""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path


from datasets import get_dataset_split_names, load_dataset
from tqdm import tqdm


DATASET_ID = "<YourOrganization>/<YourRepo>"
REVISION = "languageSubset_datasetSplit"
DEFAULT_OUTPUT = "data/remaining-30-languages-5times"
DEFAULT_CACHE = None

LANGUAGES = [
    "run_Latn", "lug_Latn", "tso_Latn", "lin_Latn", "ewe_Latn", "wol_Latn",
    "sag_Latn", "aka_Latn", "twi_Latn", "kbp_Latn", "bam_Latn", "nso_Latn",
    "fon_Latn", "ssw_Latn", "tzm_Tfng", "kab_Latn", "kea_Latn", "nqo_Nkoo",
    "mos_Latn", "kmb_Latn", "knc_Arab", "knc_Latn", "dyu_Latn", "taq_Latn",
    "dik_Latn", "luo_Latn", "fuv_Latn", "bem_Latn", "kik_Latn", "kam_Latn",
    "kon_Latn", "lua_Latn",
]

FALLBACK_SPLITS = {
    "aka_Latn": ["madlad400"],
    "bam_Latn": ["fineweb2", "madlad400"],
    "bem_Latn": ["fineweb2"],
    "dik_Latn": ["fineweb2", "madlad400"],
    "dyu_Latn": ["fineweb2", "madlad400"],
    "ewe_Latn": ["fineweb2", "madlad400"],
    "fon_Latn": ["fineweb2", "madlad400"],
    "fuv_Latn": ["fineweb2", "madlad400"],
    "kab_Latn": ["fineweb2"],
    "kam_Latn": ["fineweb2"],
    "kbp_Latn": ["fineweb2", "madlad400"],
    "kea_Latn": ["fineweb2"],
    "kik_Latn": ["fineweb2"],
    "kmb_Latn": ["fineweb2", "madlad400"],
    "knc_Arab": ["fineweb2"],
    "knc_Latn": ["fineweb2"],
    "kon_Latn": ["madlad400"],
    "lin_Latn": ["fineweb2", "madlad400"],
    "lua_Latn": ["fineweb2"],
    "lug_Latn": ["fineweb2", "madlad400"],
    "luo_Latn": ["fineweb2"],
    "mos_Latn": ["fineweb2"],
    "nqo_Nkoo": ["fineweb2"],
    "nso_Latn": ["fineweb2"],
    "run_Latn": ["fineweb2", "madlad400"],
    "sag_Latn": ["fineweb2", "madlad400"],
    "ssw_Latn": ["fineweb2", "madlad400"],
    "taq_Latn": ["fineweb2"],
    "tso_Latn": ["fineweb2", "madlad400"],
    "twi_Latn": ["fineweb2"],
    "tzm_Tfng": ["fineweb2", "madlad400"],
    "wol_Latn": ["fineweb2", "madlad400"],
}


def get_splits(dataset_id, language, revision, token):
    try:
        return get_dataset_split_names(dataset_id, language, revision=revision, token=token)
    except Exception as error:
        logging.warning("Could not discover splits for %s: %s", language, error)
        return FALLBACK_SPLITS[language]


def iter_split(args, language, split, epoch):
    dataset = load_dataset(
        args.dataset,
        name=language,
        split=split,
        revision=args.revision,
        token=args.token,
        cache_dir=None if args.streaming else args.cache_dir,
        streaming=args.streaming,
    )
    if args.seed is None:
        return dataset
    if args.streaming:
        return dataset.shuffle(args.seed + epoch, None, args.shuffle_buffer)
    return dataset.shuffle(seed=args.seed + epoch)


def write_language(args, language):
    splits = get_splits(args.dataset, language, args.revision, args.token)
    output_file = Path(args.output) / f"{language}_data.jsonl"
    rows = 0

    with output_file.open("w", encoding="utf-8") as writer:
        for epoch in range(args.sample_times):
            for split in splits:
                dataset = iter_split(args, language, split, epoch)
                desc = f"{language}/{split} epoch {epoch + 1}/{args.sample_times}"
                for item in tqdm(dataset, desc=desc):
                    text = item.get("text", "")
                    if text:
                        writer.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        rows += 1

    logging.info("Wrote %s rows to %s", rows, output_file)
    return output_file.name


def write_dataset_info(output_dir, file_names):
    dataset_info = {
        file_name.removesuffix("_data.jsonl") + "_dataset": {
            "file_name": file_name,
            "columns": {"prompt": "text"},
        }
        for file_name in file_names
    }
    info_path = Path(output_dir) / "dataset_info.json"
    info_path.write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_dataset_cache(cache_dir):
    if not cache_dir:
        logging.info("No cache directory configured.")
        return
    cache_path = Path(cache_dir) / "multilingualPretrain___pretrain_dataset"
    if cache_path.is_symlink():
        cache_path.unlink()
        logging.info("Deleted broken datasets cache symlink: %s", cache_path)
    elif cache_path.exists():
        shutil.rmtree(cache_path)
        logging.info("Deleted broken datasets cache: %s", cache_path)


def main():
    parser = argparse.ArgumentParser(description="Export remaining pretrain languages as LLaMA-Factory JSONL files.")
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE)
    parser.add_argument("--token", default=None, help="Optional Hugging Face read token.")
    parser.add_argument("--sample_times", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025, help="Set to -1 to keep original order.")
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--clean_cache", action="store_true")
    parser.add_argument("--clean_cache_only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args.seed = None if args.seed < 0 else args.seed
    if args.clean_cache or args.clean_cache_only:
        clean_dataset_cache(args.cache_dir)
    if args.clean_cache_only:
        return
    Path(args.output).mkdir(parents=True, exist_ok=True)

    file_names = [write_language(args, language) for language in args.languages]
    write_dataset_info(args.output, file_names)
    logging.info("Done. Output directory: %s", args.output)


if __name__ == "__main__":
    main()
