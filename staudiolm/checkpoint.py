"""Portable ST-Audio Encoder and ST-AudioLM release bundles."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from safetensors.torch import load_file

from .encoder import build_dynamic_AST_FOA

BASE_MODEL_ID = "allenai/OLMo-2-1124-7B-Instruct"
ST_AUDIOLM_MODEL_ID = "HBoh/ST-AudioLM"
ENCODER_MODEL_ID = ST_AUDIOLM_MODEL_ID
ENCODER_SUBDIR = "encoder"
LM_SUBDIR = "model"

_ENCODER_ARCHITECTURE = {
    "num_classes": 355,
    "feature_mode": "logmel_iv",
    "log_compression": "power_to_db",
    "iv_normalization": "dcase_foa",
    "frontend_bn": True,
    "foa_channel_map": [0, 1, 2, 3],
    "foa_channel_signs": [1.0, 1.0, 1.0, 1.0],
    "num_time_bins": 40,
    "temporal_depth": 2,
    "temporal_num_heads": 8,
    "dynamic_decoder_type": "temporal_pool",
    "direction_head_type": "vector",
    "num_dynamic_sources": 1,
}

_ENCODER_RELEASE_CONFIG = {
    "format_version": 1,
    "model_type": "st_audio_encoder",
    "sample_rate": 32000,
    "num_samples": 320000,
    "input_format": "AmbiX ACN/SN3D",
    "input_channel_order": ["W", "Y", "Z", "X"],
    "num_temporal_tokens": 40,
    "embedding_dim": 768,
    "architecture": _ENCODER_ARCHITECTURE,
}

_LLM_RELEASE_CONFIG = {
    "format_version": 1,
    "model_type": "st_audiolm",
    "base_model": BASE_MODEL_ID,
    "encoder_model": ENCODER_MODEL_ID,
    "audio_token": "<audio>",
    "audio_token_order": "temporal_then_semantic",
    "num_audio_tokens": 41,
    "connector": {"type": "direct", "input_dim": 768, "output_dim": 4096},
}


def encoder_release_config() -> dict:
    """Return the supported standalone-encoder configuration."""
    return deepcopy(_ENCODER_RELEASE_CONFIG)


def lm_release_config() -> dict:
    """Return the supported ST-AudioLM configuration."""
    return deepcopy(_LLM_RELEASE_CONFIG)


def _read_and_validate_config(root: Path, expected: dict) -> dict:
    path = root / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing inference configuration: {path}")
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    if config != expected:
        differences = sorted(
            key for key in set(config) | set(expected)
            if config.get(key) != expected.get(key)
        )
        raise ValueError(f"Unsupported {expected['model_type']} configuration fields: {differences}")
    return config


def resolve_model_dir(
    model_id_or_path: str | Path,
    *,
    allow_patterns: list[str],
    subdir: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a component directory from a local or Hugging Face bundle."""
    value = str(model_id_or_path)
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        nested = candidate / subdir if subdir else candidate
        if subdir and nested.is_dir():
            return nested
        return candidate
    if candidate.exists() or value.startswith((".", "/", "~")):
        raise FileNotFoundError(f"Model directory does not exist: {candidate}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to load a Hugging Face model ID") from exc
    patterns = (
        [f"{subdir}/{pattern}" for pattern in allow_patterns]
        if subdir else allow_patterns
    )
    snapshot = Path(snapshot_download(
        repo_id=value,
        allow_patterns=patterns,
        local_files_only=local_files_only,
    ))
    root = snapshot / subdir if subdir else snapshot
    if not root.is_dir():
        raise FileNotFoundError(f"Missing {subdir!r} component in {value}")
    return root


def validate_encoder_bundle(
    model_id_or_path: str | Path,
    *,
    local_files_only: bool = False,
) -> tuple[Path, dict]:
    root = resolve_model_dir(
        model_id_or_path,
        allow_patterns=["config.json", "encoder.safetensors"],
        subdir=ENCODER_SUBDIR,
        local_files_only=local_files_only,
    )
    config = _read_and_validate_config(root, _ENCODER_RELEASE_CONFIG)
    if not (root / "encoder.safetensors").is_file():
        raise FileNotFoundError(f"Missing encoder weights: {root / 'encoder.safetensors'}")
    return root, config


def validate_lm_bundle(
    model_id_or_path: str | Path,
    *,
    local_files_only: bool = False,
) -> tuple[Path, dict]:
    root = resolve_model_dir(
        model_id_or_path,
        allow_patterns=["config.json", "connector.safetensors", "adapter/*", "tokenizer/*"],
        subdir=LM_SUBDIR,
        local_files_only=local_files_only,
    )
    config = _read_and_validate_config(root, _LLM_RELEASE_CONFIG)
    required = (
        "connector.safetensors",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete ST-AudioLM bundle; missing: {missing}")
    with (root / "adapter" / "adapter_config.json").open(encoding="utf-8") as handle:
        adapter = json.load(handle)
    if adapter.get("base_model_name_or_path") != config["base_model"]:
        raise ValueError("Adapter must reference the public base-model ID in config.json")
    return root, config


def load_encoder_backbone(
    model_id_or_path: str | Path,
    *,
    local_files_only: bool = False,
):
    """Load the raw encoder module with strict state-dictionary checking."""
    root, config = validate_encoder_bundle(
        model_id_or_path, local_files_only=local_files_only
    )
    encoder = build_dynamic_AST_FOA(**config["architecture"])
    state = load_file(str(root / "encoder.safetensors"), device="cpu")
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False).eval()
    return encoder
