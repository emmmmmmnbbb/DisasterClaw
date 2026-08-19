"""Frozen event-disjoint partitions shared by training, calibration, and benchmarks.

Must match `paper/generated/strict_loc_manifest.json` split_audit.events.
"""

from __future__ import annotations

TRAIN_EVENTS = (
    "guatemala-volcano",
    "hurricane-florence",
    "hurricane-matthew",
    "joplin-tornado",
    "lower-puna-volcano",
    "midwest-flooding",
    "portugal-wildfire",
    "santa-rosa-wildfire",
    "socal-fire",
    "sunda-tsunami",
    "tuscaloosa-tornado",
    "woolsey-fire",
)
VAL_EVENTS = ("hurricane-harvey", "mexico-earthquake")
TEST_EVENTS = ("hurricane-michael", "palu-tsunami")
HOLDOUT_EVENTS = ("nepal-flooding", "moore-tornado", "pinery-bushfire")

EVAL_EVENTS = TEST_EVENTS + HOLDOUT_EVENTS
LEAK_EVENTS = TRAIN_EVENTS + VAL_EVENTS


def event_partition(disaster: str) -> str:
    name = str(disaster or "").strip().lower()
    if name in TRAIN_EVENTS:
        return "train"
    if name in VAL_EVENTS:
        return "val"
    if name in TEST_EVENTS:
        return "test"
    if name in HOLDOUT_EVENTS:
        return "holdout"
    return "unknown"


def assert_eval_only(disaster: str) -> None:
    part = event_partition(disaster)
    if part in {"train", "val"}:
        raise ValueError(
            f"event leakage: {disaster!r} belongs to the {part} partition; "
            "evidence-rich / end-to-end items may only come from test+holdout"
        )
