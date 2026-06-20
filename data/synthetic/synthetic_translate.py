"""Translate JSONL/Parquet sources into LLaMA-Factory training files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline


LANGUAGE_NAMES = {
    "afr_Latn": "Afrikaans",
    "amh_Ethi": "Amharic",
    "arb_Arab": "Arabic",
    "ary_Arab": "Moroccan Arabic",
    "arz_Arab": "Egyptian Arabic",
    "eng_Latn": "English",
    "fra_Latn": "French",
    "hau_Latn": "Hausa",
    "ibo_Latn": "Igbo",
    "kin_Latn": "Kinyarwanda",
    "nya_Latn": "Nyanja",
    "orm_Latn": "Oromo",
    "plt_Latn": "Plateau Malagasy",
    "por_Latn": "Portuguese",
    "sna_Latn": "Shona",
    "som_Latn": "Somali",
    "sot_Latn": "Southern Sotho",
    "swa_Latn": "Swahili",
    "tir_Ethi": "Tigrinya",
    "tsn_Latn": "Tswana",
    "xho_Latn": "Xhosa",
    "yor_Latn": "Yoruba",
    "zul_Latn": "Zulu",
}


def read_rows(path: Path, text_column: str, max_samples: int | None = None) -> list[dict]:
    if path.suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict("records")
    elif path.suffix in {".jsonl", ".json"}:
        rows = []
        with path.open(encoding="utf-8") as f:
            if path.suffix == ".json":
                payload = json.load(f)
                rows = payload if isinstance(payload, list) else [payload]
            else:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
    else:
        rows = [{"text": line.rstrip("\n")} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    rows = [row for row in rows if str(row.get(text_column, "")).strip()]
    return rows[:max_samples] if max_samples else rows


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_prompt(tokenizer: AutoTokenizer, text: str, source_name: str, target_name: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional translator. Preserve meaning, formatting, numbers, "
                "math notation, code blocks, and named entities. Return only the translation."
            ),
        },
        {
            "role": "user",
            "content": f"Translate this {source_name} text into {target_name}:\n\n{text}",
        },
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nTranslation:"


def load_vllm(args: argparse.Namespace):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    return llm, params


def generate_vllm(args: argparse.Namespace, prompts: list[str]) -> list[str]:
    llm, params = load_vllm(args)
    outputs = llm.generate(prompts, params)
    return [item.outputs[0].text.strip() for item in outputs]


def generate_transformers(args: argparse.Namespace, prompts: list[str]) -> list[str]:
    generator = pipeline(
        "text-generation",
        model=args.model,
        tokenizer=args.model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )
    outputs: list[str] = []
    for chunk in tqdm(list(batched(prompts, args.batch_size)), desc="generate"):
        result = generator(
            chunk,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            return_full_text=False,
        )
        for item in result:
            if isinstance(item, list):
                item = item[0]
            outputs.append(item["generated_text"].strip())
    return outputs


def write_dataset_info(output_dir: Path, entries: dict[str, dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dataset_info.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(entries)
    path.write_text(json.dumps(old, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def translate_file(args: argparse.Namespace, path: Path, tokenizer: AutoTokenizer) -> dict[str, dict]:
    rows = read_rows(path, args.text_column, args.max_samples)
    if not rows:
        return {}

    source_name = LANGUAGE_NAMES.get(args.source_language, args.source_language)
    info_entries = {}
    for target in args.targets:
        target_name = LANGUAGE_NAMES.get(target, target)
        dataset_name = f"{path.stem}-{target}"
        out_path = Path(args.output) / f"{dataset_name}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"skip existing {out_path}")
            info_entries[dataset_name] = {"file_name": out_path.name, "columns": {"prompt": "text"}}
            continue

        if target == args.source_language:
            translations = [row[args.text_column] for row in rows]
        else:
            prompts = [build_prompt(tokenizer, row[args.text_column], source_name, target_name) for row in rows]
            translations = (
                generate_vllm(args, prompts)
                if args.backend == "vllm"
                else generate_transformers(args, prompts)
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row, translation in zip(rows, translations):
                output = {
                    "text": translation,
                    "source_language": args.source_language,
                    "target_language": target,
                    "source": row.get("source", path.stem),
                    "domain": row.get("domain"),
                }
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
        info_entries[dataset_name] = {"file_name": out_path.name, "columns": {"prompt": "text"}}
        print(out_path)
    return info_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic multilingual data by local model translation.")
    parser.add_argument("--input", nargs="+", required=True, help="JSONL, JSON, text, or parquet files.")
    parser.add_argument("--output", default="data/synthetic/translated")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--source_language", default="eng_Latn")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    entries = {}
    for input_path in args.input:
        entries.update(translate_file(args, Path(input_path), tokenizer))
    write_dataset_info(Path(args.output), entries)


if __name__ == "__main__":
    main()
