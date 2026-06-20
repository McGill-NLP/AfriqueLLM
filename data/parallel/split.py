"""Split a large parquet file into row-group shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def split_parquet(source: Path, output_dir: Path, target_gb: float, overwrite: bool) -> dict:
    source = source.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(source)
    schema = parquet_file.schema_arrow
    target_bytes = int(target_gb * 1024**3)
    parts = []
    writer = None
    current_rows = 0
    current_bytes = 0
    current_groups = 0
    part_id = 1
    current_path = None

    def start_writer():
        nonlocal writer, current_rows, current_bytes, current_groups, current_path
        current_path = output_dir / f"{source.stem}{part_id}.parquet"
        if current_path.exists() and not overwrite:
            raise FileExistsError(current_path)
        writer = pq.ParquetWriter(str(current_path), schema=schema)
        current_rows = 0
        current_bytes = 0
        current_groups = 0

    def finish_writer():
        nonlocal writer, current_path
        if writer is None or current_path is None:
            return
        writer.close()
        parts.append(
            {
                "index": part_id,
                "path": str(current_path),
                "num_rows": current_rows,
                "row_groups": current_groups,
                "estimated_bytes": current_bytes,
                "file_size": current_path.stat().st_size,
            }
        )
        writer = None

    for row_group_id in range(parquet_file.metadata.num_row_groups):
        meta = parquet_file.metadata.row_group(row_group_id)
        if writer is None:
            start_writer()
        if current_rows and current_bytes + meta.total_byte_size > target_bytes:
            finish_writer()
            part_id += 1
            start_writer()
        writer.write_table(parquet_file.read_row_group(row_group_id))
        current_rows += meta.num_rows
        current_bytes += meta.total_byte_size
        current_groups += 1

    finish_writer()
    return {"source": str(source), "target_size_bytes": target_bytes, "parts": parts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split parquet by row group.")
    parser.add_argument("source")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--target-gb", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = split_parquet(Path(args.source), Path(args.output_dir), args.target_gb, args.overwrite)
    Path(args.metadata).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.metadata)


if __name__ == "__main__":
    main()
