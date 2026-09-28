"""Coordinate helpers retained for checkpoint-compatible spatial outputs."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def normalize_unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def az_el_to_unit_xyz(
    azimuth_deg: torch.Tensor,
    elevation_deg: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    az = azimuth_deg.to(dtype=torch.float32) * (math.pi / 180.0)
    el = elevation_deg.to(dtype=torch.float32) * (math.pi / 180.0)
    out = torch.stack(
        [
            torch.cos(el) * torch.sin(az),
            torch.sin(el),
            -torch.cos(el) * torch.cos(az),
        ],
        dim=-1,
    )
    return normalize_unit_vector(out, eps=eps)


def unit_xyz_to_az_el(direction: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    direction = normalize_unit_vector(direction, eps=eps)
    x = direction[..., 0]
    y = direction[..., 1].clamp(min=-1.0, max=1.0)
    z = direction[..., 2]
    azimuth = torch.atan2(x, -z) * (180.0 / math.pi)
    azimuth = torch.remainder(azimuth + 180.0, 360.0) - 180.0
    elevation = torch.asin(y) * (180.0 / math.pi)
    return azimuth, elevation


def angular_error_deg(pred_direction: torch.Tensor, target_direction: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = normalize_unit_vector(pred_direction, eps=eps)
    target = normalize_unit_vector(target_direction, eps=eps)
    cosine = torch.sum(pred * target, dim=-1).clamp(min=-1.0, max=1.0)
    return torch.acos(cosine) * (180.0 / math.pi)


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=value.device, dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(eps)
