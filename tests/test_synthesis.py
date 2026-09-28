import tempfile
import unittest
from pathlib import Path

import numpy as np

from staudiolm.rir import pack_rirs, trajectory_positions
from staudiolm.synthesis import (
    load_rir_sequence,
    mix_sources_ambix,
    render_dynamic_foa,
)


class RIRSynthesisTest(unittest.TestCase):
    def test_linear_recipe_uses_bin_centers(self):
        positions = trajectory_positions({
            "duration_s": 1.0,
            "keyframe_interval_s": 0.25,
            "start_source_position": [0.0, 0.0, 0.0],
            "end_source_position": [8.0, 0.0, 0.0],
        })
        np.testing.assert_allclose(positions[:, 0], [1.0, 3.0, 5.0, 7.0])

    def test_rir_npz_round_trip_and_channel_conversion(self):
        raw = np.arange(24, dtype=np.float32).reshape(6, 4)
        packed, lengths = pack_rirs([raw[:3], raw])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rirs.npz"
            np.savez(path, rirs=packed, lengths=lengths)
            actual = load_rir_sequence(path)
        np.testing.assert_array_equal(actual[0][:, 0], raw[:3, 0])
        np.testing.assert_array_equal(actual[0][:, 1], raw[:3, 3])
        np.testing.assert_array_equal(actual[0][:, 2], -raw[:3, 2])
        np.testing.assert_array_equal(actual[0][:, 3], raw[:3, 1])

    def test_static_impulse_reconstruction(self):
        dry = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rir = np.zeros((1, 4), dtype=np.float32)
        rir[0] = [1.0, 2.0, 3.0, 4.0]
        rendered = render_dynamic_foa(
            dry, [rir], sample_rate=8, duration_s=1.0,
            keyframe_interval_s=1.0, crossfade_ms=0.0, normalize_dry=False,
        )
        self.assertEqual(rendered.shape, (8, 4))
        np.testing.assert_array_equal(rendered[0], rir[0])
        ambix = mix_sources_ambix([rendered])
        self.assertEqual(ambix.shape, (8, 4))
        self.assertTrue(np.isfinite(ambix).all())


if __name__ == "__main__":
    unittest.main()
