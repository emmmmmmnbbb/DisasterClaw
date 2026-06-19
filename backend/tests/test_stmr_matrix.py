"""
backend/tests/test_stmr_matrix.py — P1 STMR 文字矩阵单测

验收点：
    1. 文本含图例、UAV 标记 'U'、网格维度正确。
    2. 放入的目标语义码出现在矩阵里。
    3. 方位摘要把目标的方向/距离算对。

运行：`python backend/tests/test_stmr_matrix.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geo import meters_to_latlon  # noqa: E402
from semantic_map import CLASS_CODE, SemanticMap  # noqa: E402
from stmr_matrix import build_stmr, object_bearing_summary  # noqa: E402

ORIGIN_LAT = 31.2304
ORIGIN_LON = 121.4737


def _map_with_destroyed_north() -> SemanticMap:
    """造一张图：UAV 在原点，正北 ~30m 处有一栋完全损毁建筑。"""
    smap = SemanticMap(ORIGIN_LAT, ORIGIN_LON, cell_size_m=5.0)
    # 让目标落在 patch 上方中央（ny 小 = 北），radius 60m → 北 +30m
    det = {"class_name": "完全损毁建筑", "conf": 0.85, "bbox": [48, 18, 52, 22]}
    smap.mark_observation(
        ORIGIN_LAT, ORIGIN_LON, radius_m=60.0,
        detections=[det], patch_width=100, patch_height=100,
    )
    return smap


def test_text_structure() -> None:
    smap = _map_with_destroyed_north()
    out = build_stmr(smap, ORIGIN_LAT, ORIGIN_LON, window_m=100.0, grid_n=20)
    text = out["text"]
    assert "图例" in text and "网格" in text, "文本应含图例与网格"
    assert "U" in text, "应标出无人机位置 U"
    assert len(out["matrix"]) == 20 and len(out["matrix"][0]) == 20, "网格应为 20x20"
    # UAV 在中心格
    lines = [l for l in text.splitlines() if set(l.split()) and "U" in l.split()]
    assert lines, "网格行里应出现 U"
    print("[OK] STMR 文本结构：图例/网格/U 标记/维度")


def test_target_code_in_matrix() -> None:
    smap = _map_with_destroyed_north()
    out = build_stmr(smap, ORIGIN_LAT, ORIGIN_LON)
    flat = [c for row in out["matrix"] for c in row]
    assert CLASS_CODE["完全损毁建筑"] in flat, "矩阵应含完全损毁建筑码"
    print("[OK] 目标语义码进入矩阵")


def test_bearing_summary() -> None:
    smap = _map_with_destroyed_north()
    objs = object_bearing_summary(smap, ORIGIN_LAT, ORIGIN_LON)
    assert len(objs) == 1, f"应有 1 个目标，实际 {len(objs)}"
    o = objs[0]
    assert o["class_name"] == "完全损毁建筑"
    assert o["bearing"] == "北", f"目标应在正北，实际 {o['bearing']}"
    assert 20 <= o["dist_m"] <= 40, f"距离应约 30m，实际 {o['dist_m']}"
    print(f"[OK] 方位摘要：{o['class_name']} 在 {o['bearing']} 约 {o['dist_m']:.0f}m")


def _run_all() -> int:
    tests = [test_text_structure, test_target_code_in_matrix, test_bearing_summary]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {t.__name__}: {exc}")
    print(f"\n{'='*48}\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run_all())
