#!/usr/bin/env python3
"""Download the one required ST-AudioQA test shard and extract a RIR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staudiolm.data import extract_test_rir_member, lookup_test_rir

DATASET_ID = "HBoh/ST-AudioQA"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rir_id", help="RIR ID from the sources or rir_recipes table")
    parser.add_argument("--output", required=True, type=Path, help="Destination .npz path")
    parser.add_argument("--dataset", default=DATASET_ID, help="Hugging Face dataset ID")
    parser.add_argument("--dataset-root", type=Path, help="Use a local ST-AudioQA directory")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")

    if args.dataset_root:
        index_path = args.dataset_root / "rir/test/index.parquet"
    else:
        from huggingface_hub import hf_hub_download

        index_path = Path(hf_hub_download(
            repo_id=args.dataset,
            repo_type="dataset",
            filename="rir/test/index.parquet",
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        ))
    row = lookup_test_rir(index_path, args.rir_id)

    if args.dataset_root:
        shard_path = args.dataset_root / "rir/test" / row["shard"]
    else:
        shard_path = Path(hf_hub_download(
            repo_id=args.dataset,
            repo_type="dataset",
            filename=f"rir/test/{row['shard']}",
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        ))
    extract_test_rir_member(
        shard_path,
        row["member"],
        args.output,
        expected_size=int(row["size_bytes"]),
    )
    print(f"Wrote {args.output} from {row['shard']}:{row['member']}")


if __name__ == "__main__":
    main()
