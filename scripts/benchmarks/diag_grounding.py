#!/usr/bin/env python3
"""一次性诊断：目标处 vs 起点处，YOLO/VLM grounding 各返回什么。

用于判断"早停"根因：起点（目标不在视场）时 VLM 是否幻觉出坐标。
    cd backend && set -a && source ../.env && set +a && \
      python ../scripts/benchmarks/diag_grounding.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO / "backend"
sys.path.insert(0, str(BACKEND))

TESTSET = BACKEND / "data" / "benchmarks" / "vln_testset.json"
N_ITEMS = 4


def probe(app, instruction, lat, lon, alt, tag):
    import xbd_store
    from vln_navigator import Observation, parse_instruction, ground_with_yolo
    entry = xbd_store.find_tile_containing(lat, lon, stage_priority=("post_disaster",))
    if entry is None:
        print(f"    [{tag}] 该点无 POST 覆盖")
        return
    app.state.activate_xbd_tile(entry)
    app.state.adapter.reset_origin(lat, lon, alt=alt)
    app.state._sync_world_from_adapter()
    result, snap, _ = app._vln_perceive("diag")
    if result is None:
        print(f"    [{tag}] 感知不可用")
        return
    parsed = parse_instruction(instruction)
    obs = Observation.from_perception(result)
    cc = result.detection.get("class_counts") or {}
    yolo = ground_with_yolo(obs, parsed.get("target_classes") or [])
    vlm = app._vln_vlm_ground(parsed, obs)
    print(f"    [{tag}] alt={alt} radius={result.patch_radius_m:.0f}m det类目={cc or '∅'} "
          f"degraded={result.degraded}")
    print(f"        YOLO: {('命中 ' + yolo.label + f' xy={tuple(round(v,2) for v in yolo.norm_xy)}') if (yolo and yolo.present) else '未命中'}")
    print(f"        VLM : {('命中 xy=' + str(tuple(round(v,2) for v in vlm.norm_xy))) if (vlm and vlm.present) else '未命中'}  :: {vlm.reason if vlm else ''}")


def main():
    items = json.loads(TESTSET.read_text(encoding="utf-8"))["items"][:N_ITEMS]
    print("[diag] 导入 app（加载模型，数分钟）...")
    import app
    print("[diag] ready\n")
    for it in items:
        g = it["goals"][-1]
        s = it["start"]
        print(f"题: {it['instruction']}  (难度 {it['difficulty']}, sp={it['shortest_path_m']}m)")
        probe(app, it["instruction"], g["lat"], g["lon"], 30.0, "目标处")
        probe(app, it["instruction"], s["lat"], s["lon"], s.get("alt", 30.0), "起点处")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
