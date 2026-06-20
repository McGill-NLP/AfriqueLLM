"""Sample public English sources for synthetic multilingual data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset


DOMAINS = {
    2: "Software_Dev",
    4: "Education_Jobs",
    6: "Entertainment",
    10: "Food_Dining",
    12: "Health",
    13: "History",
    15: "Industrial",
    17: "Politics",
    19: "Science_Tech",
    23: "Travel",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_weborganizer(args: argparse.Namespace) -> list[Path]:
    dataset = load_dataset("WebOrganizer/TopicAnnotations-Llama-3.1-8B", split="train")
    dataset = dataset.filter(
        lambda x: x["top_choice_prob"] >= args.min_topic_prob
        and x["top_choice_index"] in DOMAINS,
        num_proc=args.num_proc,
    )
    dataset = dataset.map(
        lambda x: {"char_len": len(x["text"])},
        num_proc=args.num_proc,
    )

    paths: list[Path] = []
    samples = []
    for domain_id, domain_name in DOMAINS.items():
        pool = dataset.filter(lambda x: x["top_choice_index"] == domain_id, num_proc=args.num_proc)
        if len(pool) == 0:
            continue
        pool = pool.filter(
            lambda x: args.min_chars <= x["char_len"] <= args.max_chars,
            num_proc=args.num_proc,
        )
        sampled = pool.shuffle(seed=args.seed).select(range(min(args.samples_per_domain, len(pool))))
        rows = [
            {
                "text": item["text"],
                "source": "WebOrganizer/TopicAnnotations-Llama-3.1-8B",
                "domain": domain_name,
            }
            for item in sampled
        ]
        path = Path(args.output) / "organized-web" / f"{domain_name}.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
        samples.append(sampled)

    if samples:
        merged = concatenate_datasets(samples)
        rows = [
            {
                "text": item["text"],
                "source": "WebOrganizer/TopicAnnotations-Llama-3.1-8B",
                "domain": DOMAINS[item["top_choice_index"]],
            }
            for item in merged
        ]
        path = Path(args.output) / "organized-web" / "all.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
    return paths


def sample_openmath(args: argparse.Namespace) -> list[Path]:
    dataset = load_dataset("nvidia/OpenMathReasoning", split=args.math_split, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer).take(args.math_samples)
    dataset = Dataset.from_generator(lambda: dataset)

    rows = []
    for item in dataset:
        problem = item.get("problem", "").strip()
        solution = item.get("generated_solution", "").strip()
        if not problem or not solution:
            continue
        rows.append(
            {
                "text": f"<problem>{problem}</problem>\n{solution}",
                "source": "nvidia/OpenMathReasoning",
                "domain": "Math",
            }
        )

    path = Path(args.output) / "math" / "openmathreasoning.jsonl"
    write_jsonl(path, rows)
    return [path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare public sources for synthetic translation.")
    parser.add_argument("--output", default="data/synthetic/source")
    parser.add_argument("--source", choices=["web", "math", "all"], default="all")
    parser.add_argument("--samples_per_domain", type=int, default=2000)
    parser.add_argument("--math_samples", type=int, default=2000)
    parser.add_argument("--math_split", default="cot")
    parser.add_argument("--min_topic_prob", type=float, default=0.3)
    parser.add_argument("--min_chars", type=int, default=200)
    parser.add_argument("--max_chars", type=int, default=20000)
    parser.add_argument("--shuffle_buffer", type=int, default=100000)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs: list[Path] = []
    if args.source in {"web", "all"}:
        outputs.extend(sample_weborganizer(args))
    if args.source in {"math", "all"}:
        outputs.extend(sample_openmath(args))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
