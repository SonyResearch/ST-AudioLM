"""Public standalone interface for ST-Audio Encoder."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .checkpoint import load_encoder_backbone


class STAudioEncoder(nn.Module):
    """Encode a prepared FOA waveform into semantic and temporal features."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        local_files_only: bool = False,
    ) -> "STAudioEncoder":
        backbone = load_encoder_backbone(
            model_id_or_path, local_files_only=local_files_only
        )
        model = cls(backbone)
        model.to_device(device)
        model.eval()
        return model

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def to_device(self, device: str | torch.device) -> "STAudioEncoder":
        self.backbone.to(device=torch.device(device), dtype=torch.float32)
        return self

    @torch.inference_mode()
    def forward(self, foa_waveforms: torch.Tensor) -> dict[str, torch.Tensor]:
        if foa_waveforms.ndim == 2:
            foa_waveforms = foa_waveforms.unsqueeze(0)
        if foa_waveforms.ndim != 3 or foa_waveforms.shape[1] != 4:
            raise ValueError(
                "Expected prepared FOA waveforms with shape [B, 4, samples] or [4, samples]"
            )
        output = self.backbone(foa_waveforms.to(self.device, dtype=torch.float32))
        if output["spatial_tokens"].shape[1:] != (40, 768):
            raise RuntimeError(
                f"Expected temporal tokens [B, 40, 768], got {tuple(output['spatial_tokens'].shape)}"
            )
        if output["semantic_token"].shape[1:] != (768,):
            raise RuntimeError(
                f"Expected semantic token [B, 768], got {tuple(output['semantic_token'].shape)}"
            )
        output["audio_tokens"] = torch.cat(
            (output["spatial_tokens"], output["semantic_token"].unsqueeze(1)), dim=1
        )
        return output

    @torch.inference_mode()
    def encode(self, foa_waveforms: torch.Tensor) -> torch.Tensor:
        """Return 40 temporal tokens followed by one semantic token."""
        return self.forward(foa_waveforms)["audio_tokens"]
