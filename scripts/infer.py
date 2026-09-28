#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staudiolm.audio import load_foa
from staudiolm import ST_AUDIOLM_MODEL_ID, STAudioLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ST-AudioLM on one 10-second FOA clip")
    parser.add_argument("audio", help="4-channel, 32-kHz AmbiX WAV/FLAC or .npy")
    parser.add_argument("question")
    parser.add_argument("--input-format", choices=("ambix", "legacy_model"), default="ambix")
    parser.add_argument("--npy-index", type=int)
    parser.add_argument(
        "--model", default=ST_AUDIOLM_MODEL_ID,
        help="ST-AudioLM Hugging Face model ID or local directory",
    )
    parser.add_argument(
        "--encoder-model",
        help="Optional Encoder repository or directory override",
    )
    parser.add_argument("--base-model", help="Optional local OLMo-2 directory or Hugging Face ID")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached/local weights only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    waveform = load_foa(
        args.audio,
        input_format=args.input_format,
        npy_index=args.npy_index,
    )
    model = STAudioLM.from_pretrained(
        args.model,
        encoder_model=args.encoder_model,
        base_model=args.base_model,
        device=args.device,
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    print(model.generate(waveform, args.question, max_new_tokens=args.max_new_tokens)[0])


if __name__ == "__main__":
    main()
