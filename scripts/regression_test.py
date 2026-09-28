#!/usr/bin/env python3
"""Compare model answers with expected predictions in a portable JSONL manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staudiolm import ST_AUDIOLM_MODEL_ID, STAudioLM, load_foa


def read_manifest(path: Path, count: int) -> list[dict]:
    if count < 1:
        raise ValueError("num-samples must be positive")
    rows, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "audio", "question", "prediction"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Each row needs a non-empty string: {key}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate sample ID: {row['id']}")
            if Path(row["audio"]).is_absolute():
                raise ValueError("Manifest audio paths must be relative to the manifest")
            seen.add(row["id"])
            rows.append(row)
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f"Requested {count} samples; manifest contains {len(rows)}")
    return rows


def compare_predictions(rows: list[dict], predictions: list[str]) -> dict:
    if len(rows) != len(predictions):
        raise ValueError("The number of predictions does not match the manifest")
    mismatches = [
        {"id": row["id"], "expected": row["prediction"], "actual": prediction}
        for row, prediction in zip(rows, predictions)
        if row["prediction"].strip() != prediction.strip()
    ]
    return {
        "num_samples": len(rows),
        "exact_matches": len(rows) - len(mismatches),
        "mismatches": len(mismatches),
        "comparison": "text equality after stripping leading/trailing whitespace",
        "mismatch_examples": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=ST_AUDIOLM_MODEL_ID)
    parser.add_argument("--encoder-model", help="Optional Encoder repository or directory override")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-model")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    rows = read_manifest(args.manifest, args.num_samples)

    start = time.time()
    model = STAudioLM.from_pretrained(
        args.model, encoder_model=args.encoder_model,
        device=args.device, base_model=args.base_model,
        local_files_only=args.local_files_only,
    )
    predictions = []
    for index in range(0, len(rows), args.batch_size):
        batch = rows[index:index + args.batch_size]
        waveforms = [
            load_foa(
                args.manifest.parent / row["audio"],
                input_format=row.get("input_format", "ambix"),
                npy_index=row.get("npy_index"),
            )
            for row in batch
        ]
        predictions.extend(model.generate(
            torch.stack(waveforms), [row["question"] for row in batch],
            max_new_tokens=24, max_seq_length=256,
        ))
        print(f"Compared {min(index + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    report = compare_predictions(rows, predictions)
    report["batch_size"] = args.batch_size
    report["elapsed_seconds"] = round(time.time() - start, 3)
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if report["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
