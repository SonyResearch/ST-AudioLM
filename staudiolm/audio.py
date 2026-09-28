"""Audio input validation and conversion for the released ST-AudioLM checkpoint."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch

InputFormat = Literal["ambix", "legacy_model"]
SAMPLE_RATE = 32_000
NUM_SAMPLES = 320_000


def _channel_first_foa(waveform: torch.Tensor) -> torch.Tensor:
    """Return FOA as ``[4, samples]`` and reject ambiguous shapes."""

    if waveform.ndim != 2:
        raise ValueError(f"Expected a 2-D FOA waveform, got shape {tuple(waveform.shape)}")
    if waveform.shape[0] == 4 and waveform.shape[1] != 4:
        return waveform
    if waveform.shape[1] == 4 and waveform.shape[0] != 4:
        return waveform.transpose(0, 1)
    raise ValueError(
        "Expected exactly four FOA channels in either [4, samples] or "
        f"[samples, 4] layout, got {tuple(waveform.shape)}"
    )


def ambix_to_model_foa(waveform: torch.Tensor) -> torch.Tensor:
    """Convert conventional AmbiX ACN/SN3D to the checkpoint's legacy tensor.

    Public input is ``[W, Y_left, Z_up, X_front]``.  The training tensors were
    saved as ``[W, Y_right, X_front, Z_up]`` with N3D first-order scaling.
    Therefore the exact conversion is ``[W, -sqrt(3)Y, sqrt(3)X, sqrt(3)Z]``.
    """

    x = _channel_first_foa(waveform)
    scale = math.sqrt(3.0)
    return torch.stack((x[0], -scale * x[1], scale * x[3], scale * x[2]), dim=0)


def model_foa_to_ambix(waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert the checkpoint's legacy FOA tensor to AmbiX ACN/SN3D.

    This is the exact inverse of :func:`ambix_to_model_foa`.  It is useful
    when reconstructing a public waveform from the original SoundSpaces RIRs,
    whose channel conversion was part of the dataset-generation pipeline.
    """

    x = _channel_first_foa(torch.as_tensor(waveform, dtype=torch.float32))
    scale = math.sqrt(3.0)
    return torch.stack((x[0], -x[1] / scale, x[3] / scale, x[2] / scale), dim=0)


def prepare_foa(
    waveform: torch.Tensor | np.ndarray,
    *,
    input_format: InputFormat = "ambix",
    expected_samples: int | None = NUM_SAMPLES,
) -> torch.Tensor:
    """Validate and convert one FOA clip to float32 model input."""

    x = torch.as_tensor(waveform, dtype=torch.float32)
    x = _channel_first_foa(x)
    if input_format == "ambix":
        x = ambix_to_model_foa(x)
    elif input_format != "legacy_model":
        raise ValueError(f"Unsupported input_format={input_format!r}")
    if expected_samples is not None and x.shape[1] != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples (10 s at {SAMPLE_RATE} Hz), got {x.shape[1]}"
        )
    if not torch.isfinite(x).all():
        raise ValueError("FOA waveform contains NaN or infinity")
    return x.contiguous()


def load_foa(
    path: str | Path,
    *,
    input_format: InputFormat = "ambix",
    npy_index: int | None = None,
    expected_samples: int | None = NUM_SAMPLES,
) -> torch.Tensor:
    """Load a WAV/FLAC or NumPy FOA clip.

    ``npy_index`` selects a row from a sharded ``[N, 4, T]`` array.  NumPy
    inputs are not implicitly treated as internal tensors; pass
    ``input_format='legacy_model'`` for the original precomputed dataset.
    """

    path = Path(path).expanduser()
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r")
        if npy_index is not None:
            array = array[npy_index]
        elif array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError("A sharded .npy file requires --npy-index")
            array = array[0]
        waveform = torch.from_numpy(np.asarray(array).copy())
    else:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("soundfile is required for WAV/FLAC input") from exc
        array, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if int(sample_rate) != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
        waveform = torch.from_numpy(array)
    return prepare_foa(waveform, input_format=input_format, expected_samples=expected_samples)
