"""Spatial-AST backbone definition required by the final encoder checkpoint."""

# Portions are adapted from BAT / Spatial-AST, licensed under CC BY-NC 4.0.
# See third_party_licenses/SPATIAL_AST_LICENSE.

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from .frontend import FoaFeatureConfig, FoaFeatureFrontend


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        return drop_path(x, self.drop_prob, self.training)


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02):
    return nn.init.trunc_normal_(tensor, std=std)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedNew(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        _, _, h, w = self.get_output_shape(img_size)
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, self.in_chans, img_size[0], img_size[1])).shape

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class VisionTransformerBase(nn.Module):
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}


class SimpleTimeMasking(nn.Module):
    def __init__(self, mask_param: int):
        super().__init__()
        self.mask_param = mask_param

    def forward(self, x: torch.Tensor):
        if not self.training or self.mask_param <= 0:
            return x
        _, _, freq_bins, time_bins = x.shape
        width = int(torch.randint(low=0, high=min(self.mask_param, time_bins) + 1, size=(1,)).item())
        if width == 0:
            return x
        start = int(torch.randint(low=0, high=max(1, time_bins - width + 1), size=(1,)).item())
        x = x.clone()
        x[:, :, :, start : start + width] = 0.0
        return x


class SimpleFrequencyMasking(nn.Module):
    def __init__(self, mask_param: int):
        super().__init__()
        self.mask_param = mask_param

    def forward(self, x: torch.Tensor):
        if not self.training or self.mask_param <= 0:
            return x
        _, _, freq_bins, _ = x.shape
        width = int(torch.randint(low=0, high=min(self.mask_param, freq_bins) + 1, size=(1,)).item())
        if width == 0:
            return x
        start = int(torch.randint(low=0, high=max(1, freq_bins - width + 1), size=(1,)).item())
        x = x.clone()
        x[:, :, start : start + width, :] = 0.0
        return x


class SpatialASTFOA(VisionTransformerBase):
    def __init__(
        self,
        num_classes=355,
        num_cls_tokens=3,
        img_size=(1024, 128),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        feature_mode="logmel_iv",
        log_compression="torch_log",
        iv_normalization="total_power",
        foa_channel_map=(0, 1, 2, 3),
        foa_channel_signs=(1.0, 1.0, 1.0, 1.0),
        frontend_bn=False,
        direction_head_type="none",
        freq_mask_param=48,
        time_mask_param=192,
        target_frame=1024,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_cls_tokens = num_cls_tokens
        self.target_frame = target_frame
        if direction_head_type not in ("none", "vector", "vmf"):
            raise ValueError(f"Unsupported direction_head_type: {direction_head_type}")
        self.direction_head_type = direction_head_type

        self.frontend = FoaFeatureFrontend(
            FoaFeatureConfig(
                sample_rate=32000,
                n_fft=1024,
                hop_length=320,
                win_length=1024,
                n_mels=128,
                fmin=50.0,
                fmax=14000.0,
                feature_mode=feature_mode,
                log_compression=log_compression,
                iv_normalization=iv_normalization,
                foa_channel_map=foa_channel_map,
                foa_channel_signs=foa_channel_signs,
            )
        )
        self.frontend_bn = (
            nn.BatchNorm2d(self.frontend.out_channels, affine=False)
            if frontend_bn
            else nn.Identity()
        )

        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, embed_dim))
        torch.nn.init.normal_(self.cls_tokens, std=0.02)

        self.patch_embed = PatchEmbedNew(
            img_size=img_size,
            patch_size=(16, 16),
            in_chans=1,
            embed_dim=embed_dim,
            stride=16,
        )
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.conv_downsample = nn.Sequential(
            conv3x3(self.frontend.out_channels, 1),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )

        self.timem = SimpleTimeMasking(time_mask_param)
        self.freqm = SimpleFrequencyMasking(freq_mask_param)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.dis_norm = norm_layer(embed_dim)
        self.doa_norm = norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)
        self.distance_head = nn.Linear(embed_dim, 21)
        self.azimuth_head = nn.Linear(embed_dim, 360)
        self.elevation_head = nn.Linear(embed_dim, 180)
        self.direction_head = None
        if direction_head_type != "none":
            direction_dim = 3 if direction_head_type == "vector" else 4
            self.direction_head = nn.Linear(embed_dim, direction_dim)

        trunc_normal_(self.head.weight, std=2e-5)
        trunc_normal_(self.distance_head.weight, std=2e-5)
        trunc_normal_(self.azimuth_head.weight, std=2e-5)
        trunc_normal_(self.elevation_head.weight, std=2e-5)
        if self.direction_head is not None:
            trunc_normal_(self.direction_head.weight, std=2e-5)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
            nn.init.zeros_(self.distance_head.bias)
            nn.init.zeros_(self.azimuth_head.bias)
            nn.init.zeros_(self.elevation_head.bias)
            if self.direction_head is not None:
                nn.init.zeros_(self.direction_head.bias)

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        n, _, dim = x.shape
        t_bins, f_bins = 64, 8
        x = x.reshape(n, t_bins, f_bins, dim)

        len_keep_t = int(t_bins * (1 - mask_t_prob))
        noise_t = torch.rand(n, t_bins, device=x.device)
        ids_shuffle_t = torch.argsort(noise_t, dim=1)
        ids_keep_t = ids_shuffle_t[:, :len_keep_t]
        index_t = ids_keep_t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, f_bins, dim)
        x = torch.gather(x, dim=1, index=index_t)

        x = x.permute(0, 2, 1, 3)
        len_keep_f = int(f_bins * (1 - mask_f_prob))
        noise_f = torch.rand(n, f_bins, device=x.device)
        ids_shuffle_f = torch.argsort(noise_f, dim=1)
        ids_keep_f = ids_shuffle_f[:, :len_keep_f]
        index_f = ids_keep_f.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_t, dim)
        x_masked = torch.gather(x, dim=1, index=index_f)
        x_masked = x_masked.permute(0, 2, 1, 3)
        x_masked = x_masked.reshape(n, len_keep_f * len_keep_t, dim)
        return x_masked, None, None

    def forward_features_mask(self, x, mask_t_prob, mask_f_prob):
        bsz = x.shape[0]
        x = x + self.pos_embed[:, 1:, :]
        if mask_t_prob > 0.0 or mask_f_prob > 0.0:
            x, _, _ = self.random_masking_2d(x, mask_t_prob, mask_f_prob)

        cls_tokens = self.cls_tokens.expand(bsz, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, waveforms, mask_t_prob=0.0, mask_f_prob=0.0):
        x = self.frontend(waveforms)
        x = self.frontend_bn(x)
        if x.shape[2] < self.target_frame:
            x = nn.functional.interpolate(
                x,
                (self.target_frame, x.shape[3]),
                mode="bicubic",
                align_corners=True,
            )

        x = self.conv_downsample(x)
        if self.training:
            x = x.transpose(-2, -1)
            x = self.freqm(x)
            x = self.timem(x)
            x = x.transpose(-2, -1)

        x = self.patch_embed(x)
        x = self.forward_features_mask(x, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob)

        dis_token = self.dis_norm(x[:, 0])
        doa_token = self.doa_norm(x[:, 1])
        cls_token = self.fc_norm(x[:, 2])

        classifier = self.head(cls_token)
        distance = self.distance_head(dis_token)
        azimuth = self.azimuth_head(doa_token)
        elevation = self.elevation_head(doa_token)
        if self.direction_head is not None:
            direction = self.direction_head(doa_token)
            return classifier, distance, azimuth, elevation, direction
        return classifier, distance, azimuth, elevation


def build_AST_FOA(**kwargs):
    model = SpatialASTFOA(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
