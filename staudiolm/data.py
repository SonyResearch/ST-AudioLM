"""Helpers for locating and extracting released ST-AudioQA test RIRs."""

from __future__ import annotations

import tarfile
from pathlib import Path


def lookup_test_rir(index_path: str | Path, rir_id: str) -> dict:
    """Return the unique test-RIR index row for ``rir_id``."""

    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install st-audiolm[data] to read ST-AudioQA tables") from exc

    table = pq.read_table(index_path)
    rows = table.filter(pc.equal(table["rir_id"], rir_id)).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"Expected one test RIR with rir_id={rir_id!r}, found {len(rows)}")
    return rows[0]


def extract_test_rir_member(
    shard_path: str | Path,
    member_name: str,
    output_path: str | Path,
    *,
    expected_size: int | None = None,
) -> Path:
    """Extract one indexed NPZ member without trusting tar paths."""

    shard_path = Path(shard_path)
    output_path = Path(output_path)
    with tarfile.open(shard_path, mode="r:") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise FileNotFoundError(f"{member_name!r} is not present in {shard_path}") from exc
        if not member.isfile():
            raise ValueError(f"Indexed RIR member is not a regular file: {member_name}")
        if expected_size is not None and member.size != int(expected_size):
            raise ValueError(
                f"Indexed size mismatch for {member_name}: expected {expected_size}, got {member.size}"
            )
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(f"Could not read {member_name} from {shard_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
    return output_path
