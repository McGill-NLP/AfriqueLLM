"""Score NLLB parquet pairs with COMET and merge scored shards."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from comet.models import download_model, load_from_checkpoint


def natural_key(path: Path) -> list[object]:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def load_comet_model(model_name: str, model_storage_path: str | None = None):
    model_path = download_model(model_name, saving_directory=model_storage_path)
    model = load_from_checkpoint(model_path)
    model.eval()
    model.half()
    return model


def score_file(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name

    if output_path.exists() and not args.overwrite:
        print(f"skip existing {output_path}")
        return output_path

    df = pd.read_parquet(input_path, columns=[args.src_column, args.mt_column])
    pair = re.sub(r"\d+$", "", input_path.stem)
    if "-" in pair:
        src_lang, tgt_lang = pair.split("-", 1)
        if tgt_lang in {"en", "fr", "pt"} and src_lang not in {"", "en", "fr", "pt"}:
            args.src_column, args.mt_column = args.mt_column, args.src_column

    samples = [
        {"src": src, "mt": mt}
        for src, mt in zip(df[args.src_column].astype(str), df[args.mt_column].astype(str))
    ]

    torch.set_float32_matmul_precision("high")
    model = load_comet_model(args.model_name, args.model_storage_path)
    if not args.disable_cache and hasattr(model, "set_embedding_cache"):
        model.set_embedding_cache()

    outputs = model.predict(
        samples=samples,
        batch_size=args.batch_size,
        gpus=args.gpus,
        accelerator="cuda" if args.gpus else "cpu",
        num_workers=args.num_workers,
        progress_bar=not args.quiet,
        length_batching=not args.disable_length_batching,
    )
    scores = getattr(outputs, "scores", None)
    if scores is None:
        scores = getattr(outputs, "predictions", outputs)
    pd.DataFrame({args.score_column: scores}).to_parquet(output_path, index=False)
    print(output_path)
    return output_path


def load_split_manifest(path: Path) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parts = payload.get("parts", [])
    paths = []
    for part in parts:
        part_path = Path(part["path"])
        if not part_path.is_absolute():
            part_path = path.parent / part_path
        paths.append(part_path.expanduser().resolve())
    return paths


def compression_from_metadata(parquet_file: pq.ParquetFile) -> str | None:
    metadata = parquet_file.metadata
    if metadata is None or metadata.num_row_groups == 0:
        return None
    codecs = set()
    row_group = metadata.row_group(0)
    for idx in range(row_group.num_columns):
        codec = row_group.column(idx).compression
        if codec is not None:
            codecs.add(str(getattr(codec, "name", codec)).lower())
    return codecs.pop() if len(codecs) == 1 else None


def merge_files(args: argparse.Namespace) -> Path:
    manifest = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score_dir = Path(args.output_dir).expanduser()
    pair_dir = score_dir / manifest.parent.name
    scored_shards = [pair_dir / p.name for p in load_split_manifest(manifest)]
    missing = [p for p in scored_shards if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing scored shards: {missing[:3]}")

    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
        else:
            raise FileExistsError(output_path)

    first = pq.ParquetFile(scored_shards[0])
    writer_kwargs = {"schema": first.schema_arrow}
    compression = compression_from_metadata(first)
    if compression:
        writer_kwargs["compression"] = compression

    with pq.ParquetWriter(str(output_path), **writer_kwargs) as writer:
        for shard in sorted(scored_shards, key=natural_key):
            shard_file = pq.ParquetFile(shard)
            for row_group_id in range(shard_file.metadata.num_row_groups):
                writer.write_table(shard_file.read_row_group(row_group_id))
    print(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COMET score parquet files.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--score", metavar="PARQUET", dest="score_input")
    mode.add_argument("--merge", metavar="SPLIT_JSON", dest="merge_input")
    parser.add_argument("--output", help="Merged parquet path for --merge.")
    parser.add_argument("--output_dir", default="data/parallel/comet-score")
    parser.add_argument("--src_column", default="sentence1")
    parser.add_argument("--mt_column", default="sentence2")
    parser.add_argument("--score_column", default="comet_score")
    parser.add_argument("--model_name", default="McGill-NLP/ssa-comet-mtl-final")
    parser.add_argument("--model_storage_path", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--disable_length_batching", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_plots", action="store_true", help="Accepted for compatibility.")
    args = parser.parse_args()
    args.input = args.score_input or args.merge_input
    return args


def main() -> None:
    args = parse_args()
    if args.score_input:
        score_file(args)
    else:
        if not args.output:
            raise SystemExit("--output is required with --merge")
        merge_files(args)


if __name__ == "__main__":
    main()
