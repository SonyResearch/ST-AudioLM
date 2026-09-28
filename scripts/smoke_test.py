#!/usr/bin/env python3
"""Exercise an inference bundle using synthetic audio or a user-provided clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staudiolm import (
    ENCODER_MODEL_ID,
    ST_AUDIOLM_MODEL_ID,
    STAudioEncoder,
    STAudioLM,
    load_foa,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-model")
    parser.add_argument("--model", default=ST_AUDIOLM_MODEL_ID)
    parser.add_argument("--level", choices=("encoder", "full"), default="encoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audio", type=Path, help="Optional 32-kHz, 10-second FOA clip")
    parser.add_argument("--input-format", choices=("ambix", "legacy_model"), default="ambix")
    parser.add_argument("--npy-index", type=int)
    parser.add_argument("--question", default="What sound events are present in the audio clip?")
    parser.add_argument("--base-model")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.audio:
        waveform = load_foa(args.audio, input_format=args.input_format, npy_index=args.npy_index)
    else:
        generator = torch.Generator().manual_seed(0)
        waveform = torch.randn(4, 320000, generator=generator) * 0.01

    with torch.inference_mode():
        if args.level == "encoder":
            encoder = STAudioEncoder.from_pretrained(
                args.encoder_model or ENCODER_MODEL_ID,
                device=args.device,
                local_files_only=args.local_files_only,
            )
            outputs = encoder(waveform)
            temporal, semantic = outputs["spatial_tokens"], outputs["semantic_token"]
            if temporal.shape != (1, 40, 768) or semantic.shape != (1, 768):
                raise RuntimeError("Unexpected encoder token shapes")
            if not torch.isfinite(temporal).all() or not torch.isfinite(semantic).all():
                raise RuntimeError("Encoder returned non-finite tokens")
            report = {"temporal_tokens": list(temporal.shape), "semantic_token": list(semantic.shape)}
        else:
            model = STAudioLM.from_pretrained(
                args.model,
                encoder_model=args.encoder_model,
                device=args.device, base_model=args.base_model,
                local_files_only=args.local_files_only,
            )
            embeddings = model.encode_audio(waveform)
            if embeddings.shape != (1, 41, 4096) or not torch.isfinite(embeddings).all():
                raise RuntimeError("Invalid audio embeddings")
            prediction = model.generate(waveform, args.question)[0]
            if not prediction:
                raise RuntimeError("Generation returned an empty answer")
            report = {"audio_embeddings": list(embeddings.shape), "prediction": prediction}
    report["input"] = "user_audio" if args.audio else "synthetic_noise"
    report["status"] = "passed"
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
