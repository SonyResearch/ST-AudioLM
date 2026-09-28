"""Checkpoint-compatible log-mel and active-intensity frontend."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def _build_mel_filter(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # HTK-style mel conversion for a self-contained frontend.
    def hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)

    def mel_to_hz(freq_mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (freq_mel / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs, device=device, dtype=dtype)
    mel_min = hz_to_mel(torch.tensor(fmin, device=device, dtype=dtype))
    mel_max = hz_to_mel(torch.tensor(fmax, device=device, dtype=dtype))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device, dtype=dtype)
    hz_points = mel_to_hz(mel_points)

    fb = torch.zeros(n_mels, n_freqs, device=device, dtype=dtype)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        up = (fft_freqs - left) / (center - left + 1e-8)
        down = (right - fft_freqs) / (right - center + 1e-8)
        fb[m - 1] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


@dataclass
class FoaFeatureConfig:
    sample_rate: int = 32000
    n_fft: int = 1024
    hop_length: int = 320
    win_length: int = 1024
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 14000.0
    feature_mode: str = "logmel_iv"
    log_compression: str = "torch_log"
    iv_normalization: str = "total_power"
    foa_channel_map: tuple[int, int, int, int] = (0, 1, 2, 3)
    foa_channel_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    eps: float = 1e-8


class FoaFeatureFrontend(nn.Module):
    """
    Spatial-AST-style FOA frontend.

    `feature_mode="logmel_only"`:
        log-mel for 4 FOA channels.

    `feature_mode="logmel_iv"`:
        Four channel-wise log-mel maps plus three W-directional cross-spectra.

    Notes:
    - The renderer-native channel order is configurable. The final checkpoint
      consumes already basis-corrected legacy tensors with the identity map;
      public AmbiX conversion happens in ``staudiolm.audio``.
    - `log_compression` controls whether mel energies are compressed with natural log
      or librosa-style power-to-dB.
    - `iv_normalization` controls the denominator used for FOA intensity proxies.
    """

    def __init__(self, cfg: FoaFeatureConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "window",
            torch.hann_window(cfg.win_length),
            persistent=False,
        )
        self.register_buffer(
            "mel_filter",
            _build_mel_filter(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft,
                n_mels=cfg.n_mels,
                fmin=cfg.fmin,
                fmax=cfg.fmax,
            ),
            persistent=False,
        )

    @property
    def out_channels(self) -> int:
        if self.cfg.feature_mode == "logmel_only":
            return 4
        if self.cfg.feature_mode == "logmel_iv":
            return 7
        raise ValueError(f"Unsupported feature_mode: {self.cfg.feature_mode}")

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        bsz, channels, time = waveform.shape
        flat = waveform.reshape(bsz * channels, time)
        spec = torch.stft(
            flat,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        return spec.reshape(bsz, channels, spec.shape[-2], spec.shape[-1])

    def _mel_project(self, value: torch.Tensor) -> torch.Tensor:
        # value: [B, C, F, T] -> [B, C, T, M]
        mel = torch.einsum("mf,bcft->bcmt", self.mel_filter.to(value), value)
        return mel.transpose(-1, -2)

    def _compress_mel(self, mel: torch.Tensor) -> torch.Tensor:
        if self.cfg.log_compression == "torch_log":
            return torch.log(mel + self.cfg.eps)
        if self.cfg.log_compression == "power_to_db":
            # Match librosa.power_to_db(mel_spectra, ref=1.0, amin=1e-10, top_db=80)
            # as used in the DCASE SELD FOA baseline, applied per sample/channel map.
            amin = 1e-10
            log_spec = 10.0 * torch.log10(torch.clamp(mel, min=amin))
            db_floor = torch.amax(log_spec, dim=(-2, -1), keepdim=True) - 80.0
            return torch.maximum(log_spec, db_floor)
        raise ValueError(f"Unsupported log_compression: {self.cfg.log_compression}")

    def _iv_denominator(self, power: torch.Tensor) -> torch.Tensor:
        if self.cfg.iv_normalization == "total_power":
            return power.sum(dim=1, keepdim=True) + self.cfg.eps
        if self.cfg.iv_normalization == "dcase_foa":
            w_power = power[:, 0:1]
            xyz_power_mean = power[:, 1:4].sum(dim=1, keepdim=True) / 3.0
            return w_power + xyz_power_mean + self.cfg.eps
        raise ValueError(f"Unsupported iv_normalization: {self.cfg.iv_normalization}")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 4:
            raise ValueError(f"Expected [B, 4, T] waveform, got {tuple(waveform.shape)}")

        spec = self._stft(waveform)
        w_idx, x_idx, y_idx, z_idx = self.cfg.foa_channel_map
        foa = spec[:, [w_idx, x_idx, y_idx, z_idx]]
        signs = spec.new_tensor(self.cfg.foa_channel_signs).view(1, 4, 1, 1)
        foa = foa * signs

        power = foa.abs() ** 2
        logmel = self._compress_mel(self._mel_project(power))
        if self.cfg.feature_mode == "logmel_only":
            return logmel

        w = foa[:, 0:1]
        xyz = foa[:, 1:4]
        denom = self._iv_denominator(power)
        intensity = torch.real(torch.conj(w) * xyz) / denom
        intensity_mel = self._mel_project(intensity)
        return torch.cat([logmel, intensity_mel], dim=1)
