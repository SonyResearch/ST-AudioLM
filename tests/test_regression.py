import json
import tempfile
import unittest
from pathlib import Path

from scripts.regression_test import compare_predictions, read_manifest


class RegressionTest(unittest.TestCase):
    def test_short_dataset_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            row = {"id": "one", "audio": "clip.wav", "question": "Where?", "prediction": "left"}
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(read_manifest(path, 1), [row])
            with self.assertRaisesRegex(ValueError, "contains 1"):
                read_manifest(path, 2)

    def test_comparison_reports_every_mismatch(self):
        rows = [{"id": "one", "prediction": "left"}, {"id": "two", "prediction": "right"}]
        report = compare_predictions(rows, [" left\n", "front"])
        self.assertEqual(report["exact_matches"], 1)
        self.assertEqual(report["mismatches"], 1)
        self.assertEqual(report["mismatch_examples"][0]["id"], "two")
        with self.assertRaises(ValueError):
            compare_predictions(rows, ["left"])


if __name__ == "__main__":
    unittest.main()
