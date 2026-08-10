from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from change_perception import TileGroupedBatchSampler  # noqa: E402


def test_tile_grouped_batches_are_homogeneous_and_complete() -> None:
    records = [
        {"tile_id": "a"},
        {"tile_id": "a"},
        {"tile_id": "a"},
        {"tile_id": "b"},
        {"tile_id": "b"},
        {"tile_id": "c"},
    ]
    sampler = TileGroupedBatchSampler(records, batch_size=2, seed=7)
    batches = list(iter(sampler))
    flattened = sorted(index for batch in batches for index in batch)
    assert flattened == list(range(len(records)))
    for batch in batches:
        assert len({records[index]["tile_id"] for index in batch}) == 1


def test_tile_grouped_batches_shuffle_deterministically_per_epoch() -> None:
    records = [{"tile_id": str(index)} for index in range(8)]
    left = TileGroupedBatchSampler(records, batch_size=1, seed=3)
    right = TileGroupedBatchSampler(records, batch_size=1, seed=3)
    assert list(left) == list(right)
    assert list(left) == list(right)
