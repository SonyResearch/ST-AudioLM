#!/usr/bin/env python3
"""Create or download and verify the pinned public renderer configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

MATERIALS_URL = (
    "https://raw.githubusercontent.com/facebookresearch/rlr-audio-propagation/"
    "4fd446b4abb5c71fb7a232a083bbddd65f25fc6f/"
    "RLRAudioPropagationPkg/data/mp3d_material_config.json"
)
SCENE_CONFIG = {
    "stages": {
        "paths": {".glb": ["*/*.glb"]},
        "default_attributes": {
            "shader_type": "flat",
            "nav_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.navmesh",
            "semantic_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%_semantic.ply",
            "semantic_descriptor_filename": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.house",
            "up": [0, 0, 1],
            "front": [0, 1, 0],
            "semantic_up": [0, 1, 0],
            "semantic_front": [0, 0, -1],
            "origin": [0, 0, 0],
        },
    },
    "objects": {},
    "light_setups": {},
    "scene_instances": {"default_attributes": {"default_lighting": "no_lights"}},
}
ASSETS = {
    "mp3d.scene_dataset_config.json": {
        "sha256": "b56050135d429839ca7066f335de51897d27e9741810bff8e13972fe539b0f5f",
        "content": json.dumps(SCENE_CONFIG, indent=2).encode(),
    },
    "mp3d_material_config.json": {
        "sha256": "77cc3b16f45843b1138020837d4f78aeaf0abda2427f658291b632d84afbb1f4",
        "url": MATERIALS_URL,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rir_renderer_assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, specification in ASSETS.items():
        expected = specification["sha256"]
        destination = args.output / filename
        temporary = destination.with_suffix(destination.suffix + ".download")
        if "content" in specification:
            temporary.write_bytes(specification["content"])
        else:
            urllib.request.urlretrieve(specification["url"], temporary)
        actual = sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
            )
        temporary.replace(destination)
        print(f"Verified {destination} sha256={actual}")


if __name__ == "__main__":
    main()
