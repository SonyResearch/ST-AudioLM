import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from staudiolm.checkpoint import (
    BASE_MODEL_ID,
    ENCODER_MODEL_ID,
    encoder_release_config,
    lm_release_config,
    load_encoder_backbone,
    validate_encoder_bundle,
    validate_lm_bundle,
)
from staudiolm.encoder_model import STAudioEncoder


class CheckpointMetadataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def make_encoder_bundle(self) -> Path:
        root = self.root / "encoder"
        root.mkdir()
        self.write_json(root / "config.json", encoder_release_config())
        save_file({}, str(root / "encoder.safetensors"))
        return root

    def test_encoder_config_survives_directory_move(self):
        root = self.make_encoder_bundle()
        moved = self.root / "different-location"
        root.rename(moved)
        _, config = validate_encoder_bundle(moved)
        self.assertEqual(config, encoder_release_config())

    def test_encoder_and_lm_are_separate_release_units(self):
        encoder = encoder_release_config()
        lm = lm_release_config()
        self.assertEqual(lm["encoder_model"], ENCODER_MODEL_ID)
        self.assertNotIn("architecture", lm)
        self.assertNotIn("base_model", encoder)

    def test_combined_local_repository_resolves_both_components(self):
        root = self.root / "combined"
        encoder = root / "encoder"
        model = root / "model"
        for directory in (encoder, model, model / "adapter", model / "tokenizer"):
            directory.mkdir(parents=True, exist_ok=True)
        self.write_json(encoder / "config.json", encoder_release_config())
        (encoder / "encoder.safetensors").touch()
        self.write_json(model / "config.json", lm_release_config())
        self.write_json(model / "adapter/adapter_config.json", {
            "base_model_name_or_path": BASE_MODEL_ID,
        })
        for name in (
            "connector.safetensors", "adapter/adapter_model.safetensors",
            "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
        ):
            (model / name).touch()
        encoder_root, _ = validate_encoder_bundle(root)
        model_root, _ = validate_lm_bundle(root)
        self.assertEqual(encoder_root, encoder)
        self.assertEqual(model_root, model)

    def test_rejects_changed_encoder_frontend(self):
        root = self.make_encoder_bundle()
        config = encoder_release_config()
        config["architecture"]["foa_channel_map"] = [0, 3, 2, 1]
        self.write_json(root / "config.json", config)
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_encoder_bundle(root)

    def test_missing_encoder_weights(self):
        root = self.root / "encoder"
        root.mkdir()
        self.write_json(root / "config.json", encoder_release_config())
        with self.assertRaisesRegex(FileNotFoundError, "encoder weights"):
            validate_encoder_bundle(root)

    def test_encoder_uses_strict_state_loading(self):
        root = self.root / "encoder"
        root.mkdir()
        self.write_json(root / "config.json", encoder_release_config())
        model = torch.nn.Linear(2, 2)
        state = model.state_dict()
        save_file(state, str(root / "encoder.safetensors"))
        with patch("staudiolm.checkpoint.build_dynamic_AST_FOA", return_value=model):
            self.assertIs(load_encoder_backbone(root), model)
            state.pop("bias")
            save_file(state, str(root / "encoder.safetensors"))
            with self.assertRaises(RuntimeError):
                load_encoder_backbone(root)

    def test_lm_validation_does_not_require_encoder_weights(self):
        root = self.root / "lm"
        for directory in (root, root / "adapter", root / "tokenizer"):
            directory.mkdir(parents=True, exist_ok=True)
        self.write_json(root / "config.json", lm_release_config())
        self.write_json(root / "adapter/adapter_config.json", {
            "base_model_name_or_path": BASE_MODEL_ID,
        })
        for name in (
            "connector.safetensors", "adapter/adapter_model.safetensors",
            "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
        ):
            (root / name).touch()
        _, config = validate_lm_bundle(root)
        self.assertEqual(config, lm_release_config())

    def test_public_encoder_exposes_combined_audio_tokens(self):
        class FakeBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, waveforms):
                batch = waveforms.shape[0]
                return {
                    "spatial_tokens": torch.zeros(batch, 40, 768),
                    "semantic_token": torch.ones(batch, 768),
                }

        model = STAudioEncoder(FakeBackbone())
        output = model(torch.zeros(4, 320000))
        self.assertEqual(output["audio_tokens"].shape, (1, 41, 768))
        self.assertTrue(torch.equal(output["audio_tokens"][:, -1], output["semantic_token"]))

if __name__ == "__main__":
    unittest.main()
