#!/usr/bin/env python3
"""Reconstruct one ST-AudioQA recording from dry sources and exact RIRs."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staudiolm.synthesis import load_rir_sequence, mix_sources_ambix, render_dynamic_foa


def one_row(path: Path, column: str, value: str) -> dict:
    table = pq.read_table(path)
    selected = table.filter(pc.equal(table[column], value)).to_pylist()
    if len(selected) != 1:
        raise ValueError(f"Expected one {column}={value!r} row in {path}, found {len(selected)}")
    return selected[0]


def read_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, actual_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32, copy=False)
    if int(actual_rate) != int(sample_rate):
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError("Install st-audiolm[data] to resample source audio") from exc
        divisor = math.gcd(int(actual_rate), int(sample_rate))
        mono = resample_poly(
            mono, int(sample_rate) // divisor, int(actual_rate) // divisor
        ).astype(np.float32, copy=False)
    return mono


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--audio-id", required=True)
    parser.add_argument(
        "--source-audio", action="append", default=[], metavar="SOURCE_ID=PATH",
        help="Repeat once for every source listed in sources/<split>.parquet",
    )
    parser.add_argument(
        "--rir", action="append", default=[], metavar="RIR_ID=PATH",
        help="Repeat once for every RIR listed in sources/<split>.parquet",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_paths = {key: Path(value) for key, value in (item.split("=", 1) for item in args.source_audio)}
    rir_paths = {key: Path(value) for key, value in (item.split("=", 1) for item in args.rir)}
    source_table = pq.read_table(args.dataset_root / "sources" / f"{args.split}.parquet")
    sources = source_table.filter(pc.equal(source_table["audio_id"], args.audio_id)).to_pylist()
    if not sources:
        raise ValueError(f"Unknown audio_id: {args.audio_id}")
    rendered = []
    for source in sorted(sources, key=lambda item: item["source_index"]):
        source_id, rir_id = source["source_id"], source["rir_id"]
        if source_id not in source_paths or rir_id not in rir_paths:
            raise ValueError(f"Missing --source-audio {source_id}=... or --rir {rir_id}=...")
        recipe = one_row(
            args.dataset_root / "rir" / "recipes" / f"{args.split}.parquet",
            "source_id", source_id,
        )
        rirs = load_rir_sequence(rir_paths[rir_id])
        if source["source_motion_mode"] == "static_freeze":
            keyframe = max(0, min(int(source["static_keyframe_index"]), len(rirs) - 1))
            rirs = [rirs[keyframe] for _ in rirs]
        wet = render_dynamic_foa(
            read_mono(source_paths[source_id], int(recipe["sample_rate"])), rirs,
            sample_rate=int(recipe["sample_rate"]), duration_s=float(recipe["duration_s"]),
            keyframe_interval_s=float(recipe["keyframe_interval_s"]),
        )
        rendered.append(wet * float(source["gain_linear"]))
    ambix = mix_sources_ambix(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, ambix, 32_000, subtype="FLOAT")
    print(f"Wrote {args.output} {ambix.shape} AmbiX ACN/SN3D [W,Y,Z,X]")


if __name__ == "__main__":
    main()
