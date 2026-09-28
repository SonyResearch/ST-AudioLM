import math
import unittest

import torch

from staudiolm.audio import ambix_to_model_foa, model_foa_to_ambix, prepare_foa


class AudioAdapterTest(unittest.TestCase):
    def test_ambix_acn_sn3d_conversion(self):
        ambix = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # W, Y, Z, X
        actual = ambix_to_model_foa(ambix).squeeze(1)
        scale = math.sqrt(3.0)
        expected = torch.tensor([1.0, -2.0 * scale, 4.0 * scale, 3.0 * scale])
        torch.testing.assert_close(actual, expected)

    def test_accepts_sample_first(self):
        x = torch.zeros(32, 4)
        self.assertEqual(prepare_foa(x, expected_samples=32).shape, (4, 32))

    def test_public_conversion_is_invertible(self):
        ambix = torch.randn(4, 32)
        reconstructed = model_foa_to_ambix(ambix_to_model_foa(ambix))
        torch.testing.assert_close(reconstructed, ambix)

    def test_rejects_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "Expected 320000 samples"):
            prepare_foa(torch.zeros(4, 100))

    def test_legacy_is_not_reordered(self):
        x = torch.randn(4, 32)
        torch.testing.assert_close(prepare_foa(x, input_format="legacy_model", expected_samples=32), x)


if __name__ == "__main__":
    unittest.main()
