"""Reconstruct ST-AudioQA FOA clips from dry sources and released RIRs.

The functions in this module mirror the waveform-generation path used for the
released benchmark.  They intentionally operate on NumPy arrays so that data
reconstruction does not require loading either neural model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .audio import model_foa_to_ambix

SAMPLE_RATE = 32_000
DURATION_S = 10.0
TARGET_DBFS = -14.0
CROSSFADE_MS = 50.0
LEGACY_CHANNEL_MAP = (0, 3, 2, 1)
LEGACY_CHANNEL_SIGNS = (1.0, 1.0, -1.0, 1.0)


def normalize_mono(audio: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """RMS-normalize a mono waveform using the benchmark convention."""

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected mono audio [samples], got {x.shape}")
    # Keep NumPy's float32 reduction here: this matches the original dataset
    # generator rather than silently changing its numerical path.
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    if rms == 0.0:
        return x
    current_dbfs = 20.0 * np.log10(rms)
    gain = 10.0 ** ((float(target_dbfs) - current_dbfs) / 20.0)
    return (x * gain).astype(np.float32, copy=False)


def fit_duration(audio: np.ndarray, num_samples: int) -> np.ndarray:
    """Crop or repeat a source exactly as in benchmark generation."""

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected mono audio [samples], got {x.shape}")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if x.size == 0:
        return np.zeros(num_samples, dtype=np.float32)
    if x.shape[0] >= num_samples:
        return x[:num_samples].astype(np.float32, copy=False)
    repetitions = int(math.ceil(num_samples / x.shape[0]))
    return np.tile(x, repetitions)[:num_samples].astype(np.float32, copy=False)


def apply_legacy_rir_conversion(rir: np.ndarray) -> np.ndarray:
    """Convert a raw four-channel SoundSpaces RIR to model tensor order."""

    x = np.asarray(rir, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError(f"Expected an RIR shaped [samples, 4], got {x.shape}")
    ordered = x[:, list(LEGACY_CHANNEL_MAP)]
    signs = np.asarray(LEGACY_CHANNEL_SIGNS, dtype=np.float32)
    return (ordered * signs[None, :]).astype(np.float32, copy=False)


def load_rir_sequence(path: str | Path, *, convert_channels: bool = True) -> list[np.ndarray]:
    """Load a released padded RIR sequence and restore each true length."""

    with np.load(Path(path), allow_pickle=False) as data:
        if "rirs" not in data or "lengths" not in data:
            raise ValueError(f"RIR archive must contain rirs and lengths: {path}")
        packed = np.asarray(data["rirs"], dtype=np.float32)
        lengths = np.asarray(data["lengths"], dtype=np.int64)
    if packed.ndim != 3 or packed.shape[2] != 4:
        raise ValueError(f"Expected packed RIRs [keyframes, samples, 4], got {packed.shape}")
    if lengths.shape != (packed.shape[0],):
        raise ValueError(f"RIR length vector does not match packed data: {lengths.shape}")
    result = []
    for index, length in enumerate(lengths):
        if int(length) < 1 or int(length) > packed.shape[1]:
            raise ValueError(f"Invalid RIR length at keyframe {index}: {int(length)}")
        rir = packed[index, : int(length)]
        result.append(apply_legacy_rir_conversion(rir) if convert_channels else rir)
    return result


def fftconvolve_mono_foa(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """FFT-convolve one mono signal with one four-channel RIR."""

    dry = np.asarray(audio, dtype=np.float32)
    ir = np.asarray(rir, dtype=np.float32)
    if dry.ndim != 1:
        raise ValueError(f"Expected mono audio [samples], got {dry.shape}")
    if ir.ndim != 2 or ir.shape[1] != 4:
        raise ValueError(f"Expected RIR [samples, 4], got {ir.shape}")
    full_length = dry.shape[0] + ir.shape[0] - 1
    fft_length = 1
    while fft_length < full_length:
        fft_length *= 2
    dry_fft = np.fft.rfft(dry, n=fft_length)
    channels = []
    for channel in range(ir.shape[1]):
        rir_fft = np.fft.rfft(ir[:, channel], n=fft_length)
        channels.append(
            np.fft.irfft(dry_fft * rir_fft, n=fft_length)[:full_length]
        )
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


def crossfade_weights(
    num_audio_samples: int,
    rir_length: int,
    num_keyframes: int,
    segment_length: int,
    fade_length: int,
) -> list[np.ndarray]:
    """Build normalized keyframe crossfades used by dynamic rendering."""

    if min(num_audio_samples, rir_length, num_keyframes, segment_length) < 1:
        raise ValueError("Audio, RIR, keyframe, and segment lengths must be positive")
    total_length = num_audio_samples + rir_length - 1
    weights: list[np.ndarray] = []
    for keyframe in range(num_keyframes):
        weight = np.zeros(total_length, dtype=np.float32)
        start = keyframe * segment_length
        end = min((keyframe + 1) * segment_length, total_length)
        if start >= total_length:
            weights.append(weight)
            continue
        weight[start:end] = 1.0
        if keyframe > 0 and fade_length > 0:
            fade_start = max(start - fade_length, 0)
            fade_end = min(start + fade_length, total_length)
            ramp = np.linspace(0.0, 1.0, fade_end - fade_start, dtype=np.float32)
            weight[fade_start:fade_end] = np.maximum(weight[fade_start:fade_end], ramp)
        if keyframe + 1 < num_keyframes and fade_length > 0:
            fade_start = max(end - fade_length, 0)
            fade_end = min(end + fade_length, total_length)
            ramp = np.linspace(1.0, 0.0, fade_end - fade_start, dtype=np.float32)
            weight[fade_start:fade_end] = np.minimum(
                np.maximum(weight[fade_start:fade_end], ramp), 1.0
            )
        if keyframe == num_keyframes - 1 and end < total_length:
            weight[end:] = 1.0
        weights.append(weight)
    denominator = np.maximum(sum(weights), 1e-6)
    return [weight / denominator for weight in weights]


def render_dynamic_foa(
    dry_audio: np.ndarray,
    rirs: Iterable[np.ndarray],
    *,
    sample_rate: int = SAMPLE_RATE,
    duration_s: float = DURATION_S,
    keyframe_interval_s: float = 0.25,
    crossfade_ms: float = CROSSFADE_MS,
    normalize_dry: bool = True,
) -> np.ndarray:
    """Render one source in the checkpoint's internal FOA convention."""

    sequence = [np.asarray(rir, dtype=np.float32) for rir in rirs]
    if not sequence:
        raise ValueError("At least one RIR keyframe is required")
    num_samples = int(round(float(duration_s) * int(sample_rate)))
    dry = fit_duration(dry_audio, num_samples)
    if normalize_dry:
        dry = normalize_mono(dry)
    max_rir_length = max(rir.shape[0] for rir in sequence)
    segment_length = int(round(float(keyframe_interval_s) * int(sample_rate)))
    fade_length = int(round(float(crossfade_ms) * int(sample_rate) / 1000.0))
    weights = crossfade_weights(
        num_samples, max_rir_length, len(sequence), segment_length, fade_length
    )
    wet = np.zeros((num_samples + max_rir_length - 1, 4), dtype=np.float32)
    for rir, weight in zip(sequence, weights):
        convolved = fftconvolve_mono_foa(dry, rir)
        if convolved.shape[0] < wet.shape[0]:
            padded = np.zeros_like(wet)
            padded[: convolved.shape[0]] = convolved
            convolved = padded
        elif convolved.shape[0] > wet.shape[0]:
            convolved = convolved[: wet.shape[0]]
        wet += convolved * weight[:, None]
    return wet[:num_samples].astype(np.float32, copy=False)


def mix_sources_ambix(sources: Iterable[np.ndarray], *, peak_guard: float = 0.0) -> np.ndarray:
    """Mix internal FOA sources and return public sample-first AmbiX audio."""

    arrays = [np.asarray(source, dtype=np.float32) for source in sources]
    if not arrays:
        raise ValueError("At least one rendered source is required")
    shape = arrays[0].shape
    if len(shape) != 2 or shape[1] != 4 or any(item.shape != shape for item in arrays):
        raise ValueError("All rendered sources must share shape [samples, 4]")
    mixed = np.sum(arrays, axis=0, dtype=np.float32)
    if peak_guard > 0:
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > peak_guard:
            mixed *= float(peak_guard) / peak
    ambix = model_foa_to_ambix(mixed.T).T
    return ambix.numpy().astype(np.float32, copy=False)
