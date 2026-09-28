"""Dynamic 40-bin Spatial-AST encoder used by the final checkpoint."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_base import Block, SpatialASTFOA, trunc_normal_
from .geometry import normalize_unit_vector


class DynamicSpatialASTFOA(SpatialASTFOA):
    def __init__(
        self,
        *args,
        num_time_bins: int = 40,
        temporal_depth: int = 2,
        temporal_num_heads: int = 8,
        dynamic_decoder_type: str = "temporal_pool",
        source_decoder_depth: int = 2,
        source_decoder_num_heads: int = 8,
        source_decoder_dropout: float = 0.0,
        direction_head_type: str = "vector",
        num_dynamic_sources: int = 1,
        **kwargs,
    ):
        super().__init__(*args, direction_head_type="none", **kwargs)
        if direction_head_type not in ("vector", "vmf"):
            raise ValueError(f"Unsupported dynamic direction_head_type: {direction_head_type}")
        if dynamic_decoder_type not in ("temporal_pool", "source_query"):
            raise ValueError(f"Unsupported dynamic_decoder_type: {dynamic_decoder_type}")
        self.dynamic_direction_head_type = direction_head_type
        self.dynamic_decoder_type = dynamic_decoder_type
        self.num_time_bins = int(num_time_bins)
        self.num_dynamic_sources = int(num_dynamic_sources)
        if self.num_dynamic_sources < 1:
            raise ValueError(f"num_dynamic_sources must be >= 1, got {self.num_dynamic_sources}")
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_time_bins, self.embed_dim))
        self.temporal_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=temporal_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                for _ in range(int(temporal_depth))
            ]
        )
        self.temporal_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        if self.dynamic_decoder_type == "source_query":
            self.source_decoder_queries = nn.Parameter(torch.zeros(1, self.num_dynamic_sources, self.embed_dim))
            self.source_decoder_time_embed = nn.Parameter(torch.zeros(1, self.num_time_bins, self.embed_dim))
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=int(source_decoder_num_heads),
                dim_feedforward=int(self.embed_dim * 4),
                dropout=float(source_decoder_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.source_decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(source_decoder_depth))
            self.source_memory_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.source_decoder_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        else:
            self.source_decoder_queries = None
            self.source_decoder_time_embed = None
            self.source_decoder = None
            self.source_memory_norm = None
            self.source_decoder_norm = None
        direction_dim = 3 if direction_head_type == "vector" else 4
        if self.num_dynamic_sources > 1 and self.dynamic_decoder_type == "temporal_pool":
            self.dynamic_source_queries = nn.Parameter(torch.zeros(1, 1, self.num_dynamic_sources, self.embed_dim))
            self.dynamic_slot_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.dynamic_direction_head = nn.Linear(self.embed_dim, direction_dim)
            self.dynamic_distance_log_head = nn.Linear(self.embed_dim, 1)
            self.dynamic_activity_head = nn.Linear(self.embed_dim, 1)
            self.dynamic_slot_semantic_head = nn.Linear(self.embed_dim, self.num_classes)
        else:
            self.dynamic_source_queries = None
            self.dynamic_slot_norm = None
            self.dynamic_slot_semantic_head = (
                nn.Linear(self.embed_dim, self.num_classes)
                if self.dynamic_decoder_type == "source_query"
                else None
            )
            self.dynamic_direction_head = nn.Linear(self.embed_dim, direction_dim)
            self.dynamic_distance_log_head = nn.Linear(self.embed_dim, 1)
            self.dynamic_activity_head = nn.Linear(self.embed_dim, 1)
        trunc_normal_(self.temporal_pos_embed, std=0.02)
        if self.dynamic_source_queries is not None:
            trunc_normal_(self.dynamic_source_queries, std=0.02)
        if self.source_decoder_queries is not None:
            trunc_normal_(self.source_decoder_queries, std=0.02)
        if self.source_decoder_time_embed is not None:
            trunc_normal_(self.source_decoder_time_embed, std=0.02)
        heads = [self.dynamic_direction_head, self.dynamic_distance_log_head, self.dynamic_activity_head]
        if self.dynamic_slot_semantic_head is not None:
            heads.append(self.dynamic_slot_semantic_head)
        for head in heads:
            trunc_normal_(head.weight, std=2e-5)
            nn.init.zeros_(head.bias)

    def no_weight_decay(self):
        out = set(super().no_weight_decay())
        out.add("temporal_pos_embed")
        out.add("dynamic_source_queries")
        out.add("source_decoder_queries")
        out.add("source_decoder_time_embed")
        return out

    def _backbone_tokens(self, waveforms, mask_t_prob: float = 0.0, mask_f_prob: float = 0.0):
        x = self.frontend(waveforms)
        x = self.frontend_bn(x)
        if x.shape[2] < self.target_frame:
            x = F.interpolate(x, (self.target_frame, x.shape[3]), mode="bicubic", align_corners=True)

        x = self.conv_downsample(x)
        if self.training:
            x = x.transpose(-2, -1)
            x = self.freqm(x)
            x = self.timem(x)
            x = x.transpose(-2, -1)

        patch_tokens = self.patch_embed(x)
        patch_tokens = patch_tokens + self.pos_embed[:, 1:, :]
        cls_tokens = self.cls_tokens.expand(patch_tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = self.pos_drop(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        return tokens

    def _temporal_readout(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        bsz, num_patches, dim = patch_tokens.shape
        t_bins, f_bins = self.patch_embed.patch_hw
        if num_patches != t_bins * f_bins:
            raise ValueError(f"Expected {t_bins * f_bins} patch tokens, got {num_patches}")
        x = patch_tokens.reshape(bsz, t_bins, f_bins, dim).mean(dim=2)
        if t_bins != self.num_time_bins:
            x = F.interpolate(
                x.transpose(1, 2),
                size=self.num_time_bins,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
        x = x + self.temporal_pos_embed
        for blk in self.temporal_blocks:
            x = blk(x)
        return self.temporal_norm(x)

    def _source_query_decode(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = patch_tokens.shape[0]
        memory = self.source_memory_norm(patch_tokens)
        queries = self.source_decoder_queries[:, :, None, :] + self.source_decoder_time_embed[:, None, :, :]
        queries = queries.reshape(1, self.num_dynamic_sources * self.num_time_bins, self.embed_dim)
        queries = queries.expand(bsz, -1, -1)
        decoded = self.source_decoder(tgt=queries, memory=memory)
        decoded = self.source_decoder_norm(decoded)
        slot_time_tokens = decoded.reshape(bsz, self.num_dynamic_sources, self.num_time_bins, self.embed_dim)
        slot_time_tokens = slot_time_tokens.permute(0, 2, 1, 3).contiguous()
        source_tokens = slot_time_tokens.mean(dim=1)
        return slot_time_tokens, source_tokens

    def forward(
        self,
        waveforms,
        mask_t_prob: float = 0.0,
        mask_f_prob: float = 0.0,
        static_only: bool = False,
    ):
        tokens = self._backbone_tokens(waveforms, mask_t_prob=0.0, mask_f_prob=0.0)
        dis_token = self.dis_norm(tokens[:, 0])
        doa_token = self.doa_norm(tokens[:, 1])
        cls_token = self.fc_norm(tokens[:, 2])
        static_out = {
            "clip_logits": self.head(cls_token),
            "static_distance_logits": self.distance_head(dis_token),
            "static_azimuth_logits": self.azimuth_head(doa_token),
            "static_elevation_logits": self.elevation_head(doa_token),
        }
        if static_only:
            return static_out

        patch_tokens = tokens[:, self.num_cls_tokens :]
        spatial_tokens = self._temporal_readout(patch_tokens)

        if self.dynamic_decoder_type == "source_query":
            slot_tokens, slot_clip_tokens = self._source_query_decode(patch_tokens)
            raw_direction = self.dynamic_direction_head(slot_tokens)
            direction = normalize_unit_vector(raw_direction[..., :3])
            distance_log = self.dynamic_distance_log_head(slot_tokens).squeeze(-1)
            activity_logits = self.dynamic_activity_head(slot_tokens).squeeze(-1)
            slot_logits = self.dynamic_slot_semantic_head(slot_clip_tokens)
            slot_union_logits = torch.logsumexp(slot_logits, dim=1)
            clip_logits = static_out["clip_logits"]
            if self.num_dynamic_sources == 1:
                raw_direction = raw_direction.squeeze(2)
                direction = direction.squeeze(2)
                distance_log = distance_log.squeeze(2)
                activity_logits = activity_logits.squeeze(2)
        elif self.num_dynamic_sources > 1:
            slot_tokens = spatial_tokens[:, :, None, :] + self.dynamic_source_queries
            slot_tokens = self.dynamic_slot_norm(slot_tokens)
            raw_direction = self.dynamic_direction_head(slot_tokens)
            direction = normalize_unit_vector(raw_direction[..., :3])
            distance_log = self.dynamic_distance_log_head(slot_tokens).squeeze(-1)
            activity_logits = self.dynamic_activity_head(slot_tokens).squeeze(-1)
            slot_clip_tokens = slot_tokens.mean(dim=1)
            slot_logits = self.dynamic_slot_semantic_head(slot_clip_tokens)
            slot_union_logits = torch.logsumexp(slot_logits, dim=1)
            clip_logits = static_out["clip_logits"]
        else:
            raw_direction = self.dynamic_direction_head(spatial_tokens)
            direction = normalize_unit_vector(raw_direction[..., :3])
            distance_log = self.dynamic_distance_log_head(spatial_tokens).squeeze(-1)
            activity_logits = self.dynamic_activity_head(spatial_tokens).squeeze(-1)
            slot_tokens = None
            slot_logits = None
            slot_union_logits = None
            clip_logits = static_out["clip_logits"]
        out = {
            **static_out,
            "global_clip_logits": static_out["clip_logits"],
            "static_clip_logits": static_out["clip_logits"],
            "clip_logits": clip_logits,
            "global_tokens": torch.stack([dis_token, doa_token, cls_token], dim=1),
            "semantic_token": cls_token,
            "spatial_tokens": spatial_tokens,
            "direction": direction,
            "direction_raw": raw_direction,
            "distance_log": distance_log,
            "activity_logits": activity_logits,
        }
        if slot_tokens is not None:
            out["slot_tokens"] = slot_tokens
        if slot_logits is not None:
            out["slot_logits"] = slot_logits
        if slot_union_logits is not None:
            out["slot_union_logits"] = slot_union_logits
        if self.dynamic_direction_head_type == "vmf":
            out["direction_kappa"] = F.softplus(raw_direction[..., 3]).clamp_min(1e-8)
        return out


def build_dynamic_AST_FOA(**kwargs):
    return DynamicSpatialASTFOA(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
