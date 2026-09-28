import io
import tarfile
import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # Optional data dependency.
    pa = None
    pq = None

from staudiolm.data import extract_test_rir_member, lookup_test_rir


@unittest.skipUnless(pa is not None and pq is not None, "requires st-audiolm[data]")
class TestRIRExtractionTest(unittest.TestCase):
    def test_lookup_and_safe_single_member_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.parquet"
            pq.write_table(pa.table({
                "rir_id": ["rir-1"],
                "shard": ["test-rirs-00000.tar"],
                "member": ["rirs/rir-1.npz"],
                "size_bytes": [4],
            }), index)
            row = lookup_test_rir(index, "rir-1")
            self.assertEqual(row["shard"], "test-rirs-00000.tar")

            shard = root / row["shard"]
            with tarfile.open(shard, "w") as archive:
                info = tarfile.TarInfo(row["member"])
                info.size = 4
                archive.addfile(info, io.BytesIO(b"test"))
            output = root / "rir-1.npz"
            extract_test_rir_member(shard, row["member"], output, expected_size=4)
            self.assertEqual(output.read_bytes(), b"test")

    def test_rejects_index_size_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "one.tar"
            with tarfile.open(shard, "w") as archive:
                info = tarfile.TarInfo("rirs/one.npz")
                info.size = 3
                archive.addfile(info, io.BytesIO(b"abc"))
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                extract_test_rir_member(
                    shard, "rirs/one.npz", Path(directory) / "out.npz", expected_size=4
                )


if __name__ == "__main__":
    unittest.main()
