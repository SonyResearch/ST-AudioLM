#!/usr/bin/env python3
"""Run ST-Audio Encoder independently on one 10-second FOA clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staudiolm import ENCODER_MODEL_ID, STAudioEncoder, load_foa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="4-channel, 32-kHz AmbiX WAV/FLAC or .npy")
    parser.add_argument(
        "--model", default=ENCODER_MODEL_ID,
        help="ST-Audio Encoder Hugging Face model ID or local directory",
    )
    parser.add_argument("--input-format", choices=("ambix", "legacy_model"), default="ambix")
    parser.add_argument("--npy-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional .npz output path")
    args = parser.parse_args()

    waveform = load_foa(
        args.audio, input_format=args.input_format, npy_index=args.npy_index
    )
    model = STAudioEncoder.from_pretrained(
        args.model, device=args.device, local_files_only=args.local_files_only
    )
    with torch.inference_mode():
        result = model(waveform)

    arrays = {
        "audio_tokens": result["audio_tokens"][0].float().cpu().numpy(),
        "temporal_tokens": result["spatial_tokens"][0].float().cpu().numpy(),
        "semantic_token": result["semantic_token"][0].float().cpu().numpy(),
        "semantic_logits": result["clip_logits"][0].float().cpu().numpy(),
        "direction": result["direction"][0].float().cpu().numpy(),
        "distance": result["distance_log"][0].float().exp().cpu().numpy(),
        "activity_probability": result["activity_logits"][0].float().sigmoid().cpu().numpy(),
    }
    if args.output:
        if args.output.suffix.lower() != ".npz":
            parser.error("--output must end in .npz")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
    print(json.dumps({
        "audio_tokens": list(arrays["audio_tokens"].shape),
        "direction": list(arrays["direction"].shape),
        "distance": list(arrays["distance"].shape),
        "activity_probability": list(arrays["activity_probability"].shape),
        "output": str(args.output) if args.output else None,
    }, indent=2))


if __name__ == "__main__":
    main()
