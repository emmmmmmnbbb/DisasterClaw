#!/usr/bin/env python3
"""Construct one frozen xBD scene, replay its recorded actions, and score it.

This intentionally replays recorded observations and answers. It is a small
deterministic reproduction of scene linkage and scoring, not a new model run.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def verified_file(relative: str, expected: str) -> Path:
    path = HERE / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    return path


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"invalid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def window(position: dict, fov_deg: float) -> dict:
    lat, lon = position["lat"], position["lon"]
    altitude = min(1330.215, max(443.405, position["alt"]))
    half = altitude * math.tan(math.radians(fov_deg) / 2.)
    dlat = half / 110540.
    dlon = half / (111320. * math.cos(math.radians(lat)))
    return {"west": lon - dlon, "south": lat - dlat,
            "east": lon + dlon, "north": lat + dlat}


def main() -> None:
    scene = load("scene.json")
    episode = load("episode.json")
    if episode["qid"].split("__", 1)[0] != scene["tile_id"]:
        raise ValueError("episode and scene tile IDs differ")
    dimensions = {}
    for stage, asset in scene["images"].items():
        dimensions[stage] = png_size(verified_file(asset["file"], asset["sha256"]))
    if set(dimensions.values()) != {(1024, 1024)}:
        raise ValueError(f"unexpected image dimensions: {dimensions}")
    labels_path = verified_file(scene["post_label_file"], scene["post_label_sha256"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    uid = episode["target"]["ref_id"]
    target = [f for f in labels["features"]["xy"]
              if f.get("properties", {}).get("uid") == uid]
    if len(target) != 1:
        raise ValueError(f"expected one target UID, got {len(target)}")
    subtype = target[0]["properties"]["subtype"]
    reference = "无损伤" if subtype == "no-damage" else "损伤"
    if reference != episode["source_gt_answer"]:
        raise ValueError("annotation-derived reference differs from frozen score")
    trace = []
    for step in episode["steps"]:
        footprint = window(step["position"], scene["observation_model"]["fov_deg"])
        trace.append({"step": step["step"], "observation_id": step["observation_id"],
                      "alt_m": round(step["position"]["alt"], 2),
                      "window": {k: round(v, 7) for k, v in footprint.items()},
                      "candidate_answer": step["candidate_answer"],
                      "decision": step["decision"], "action": step["action"]})
    if [t["step"] for t in trace] != list(range(len(trace))):
        raise ValueError("nonconsecutive trace steps")
    if trace[-1]["candidate_answer"] != episode["recorded_answer"]:
        raise ValueError("final trace answer differs from recorded answer")
    result = {"qid": episode["qid"], "mode": "frozen_trace_replay",
              "scene": {"tile_id": scene["tile_id"], "pre_png": list(dimensions["pre"]),
                        "post_png": list(dimensions["post"]), "target_uid": uid,
                        "target_subtype": subtype},
              "question": episode["question"], "trace": trace,
              "score": {"reference": reference, "answer": episode["recorded_answer"],
                        "correct": episode["recorded_answer"] == reference}}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
