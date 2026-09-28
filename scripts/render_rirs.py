#!/usr/bin/env python3
"""Render RIRs from one ST-AudioQA recipe Parquet file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staudiolm.rir import FOARenderer, pack_rirs, trajectory_positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mp3d-root", required=True, type=Path)
    parser.add_argument("--scene-dataset-config", required=True, type=Path)
    parser.add_argument("--materials-json", required=True, type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--direct-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    rows = pq.read_table(args.recipes).to_pylist()
    if args.max_items is not None:
        rows = rows[: args.max_items]
    args.output.mkdir(parents=True)
    completed = []
    with FOARenderer(
        mp3d_root=args.mp3d_root,
        scene_dataset_config=args.scene_dataset_config,
        materials_json=args.materials_json,
        indirect=not args.direct_only,
    ) as renderer:
        for index, row in enumerate(rows):
            recipe = dict(row)
            recipe["time_profile_params"] = json.loads(recipe.pop("time_profile_params_json"))
            positions = trajectory_positions(recipe)
            rirs = [
                renderer.render_ir(
                    scene_id=recipe["scene_id"],
                    agent_position=recipe["agent_position"],
                    sensor_offset=recipe["sensor_offset"],
                    source_position=position,
                )
                for position in positions
            ]
            packed, lengths = pack_rirs(rirs)
            output = args.output / f"{recipe['rir_id']}.npz"
            np.savez(
                output, rirs=packed, lengths=lengths,
                sample_rate=np.int32(recipe["sample_rate"]),
                direct_only=np.bool_(args.direct_only),
            )
            completed.append({"rir_id": recipe["rir_id"], "file": output.name})
            print(f"[{index + 1}/{len(rows)}] {recipe['rir_id']}", flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps({"rirs": completed}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
