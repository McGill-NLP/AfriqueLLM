"""Public math/code data recipes for LlamaFactory."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def write_dataset_info(output_dir):
    parent = output_dir.parent
    info_path = parent / "dataset_info.json"
    if info_path.exists():
        dataset_info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        dataset_info = {}

    dataset_info[output_dir.name] = {
        "file_name": f"{output_dir.name}/",
        "columns": {"prompt": "text"},
    }
    info_path.write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_parts(rows, output_dir, part_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    part_id = 1
    buffer = []

    for row in rows:
        buffer.append({"text": row})
        if len(buffer) >= part_size:
            part_path = output_dir / f"data-part-{part_id:03d}.json"
            part_path.write_text(json.dumps(buffer, ensure_ascii=False), encoding="utf-8")
            print(f"Wrote {len(buffer):,} rows to {part_path}")
            part_id += 1
            buffer = []

    if buffer:
        part_path = output_dir / f"data-part-{part_id:03d}.json"
        part_path.write_text(json.dumps(buffer, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(buffer):,} rows to {part_path}")

    write_dataset_info(output_dir)


def token_len(text, tokenizer):
    return len(tokenizer.encode(text, max_length=None, truncation=False))


def iter_finemath(args, tokenizer):
    dataset = load_dataset(
        "HuggingFaceTB/finemath",
        "finemath-4plus",
        split="train",
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
    ).shuffle(seed=args.seed)

    total_tokens = 0
    for i, item in enumerate(tqdm(dataset, desc="finemath")):
        text = item.get("text", "")
        if not text:
            continue
        total_tokens += item.get("token_count") or token_len(text, tokenizer)
        yield text
        if args.max_samples and i + 1 >= args.max_samples:
            break
        if args.max_tokens and total_tokens >= args.max_tokens:
            break


def iter_cornstack(args, tokenizer):
    data_files = {"train": [f"shard-{i:05d}.jsonl.gz" for i in range(1, 240, 2)]}
    dataset = load_dataset(
        "nomic-ai/cornstack-python-v1",
        data_files=data_files,
        split="train",
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
    ).shuffle(seed=args.seed)

    total_tokens = 0
    for i, item in enumerate(tqdm(dataset, desc="cornstack")):
        text = f"{item.get('query', '')}\n\n{item.get('document', '')}".strip()
        if not text:
            continue
        total_tokens += token_len(text, tokenizer)
        yield text
        if args.max_samples and i + 1 >= args.max_samples:
            break
        if args.max_tokens and total_tokens >= args.max_tokens:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare public math/code datasets.")
    parser.add_argument("recipe", choices=["finemath", "cornstack"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--part_size", type=int, default=500_000)
    parser.add_argument("--num_proc", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="google/gemma-3-4b-pt")
    parser.add_argument("--cache_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    output_dir = Path(args.output)

    if args.recipe == "finemath":
        rows = iter_finemath(args, tokenizer)
    else:
        rows = iter_cornstack(args, tokenizer)

    write_parts(rows, output_dir, args.part_size)


if __name__ == "__main__":
    main()
