"""
backend/tests/test_semantic_map.py — P0 语义地图单元测试

验收点（对应 docs/vln_rescue_agent_实施计划.md 的 P0）：
    1. pixel→geo 投影：检测框中心按几何约定落到正确栅格，往返误差在 1 格内。
    2. explored 随观测增长，移动后新增格子。
    3. degraded 视场只记 explored、不投影检测框。
    4. 序列化（to_dict / snapshot）结构完整、目标计数一致。
    5. to_local_matrix（STMR 前身）能把目标语义填进正确网格。

可直接运行：`python backend/tests/test_semantic_map.py`（无需 pytest），
也兼容 `pytest backend/tests/test_semantic_map.py`。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# 让测试能 import backend 包内模块（semantic_map / geo）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geo import latlon_to_meters, meters_to_latlon  # noqa: E402
from semantic_map import (  # noqa: E402
    CLASS_CODE,
    LAYER_LANDMARKS,
    LAYER_OBJECTS,
    SemanticMap,
)

ORIGIN_LAT = 31.2304
ORIGIN_LON = 121.4737


def _approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def test_projection_accuracy() -> None:
    """检测框中心 (nx=0.75, ny=0.25) → +30m 北 / +30m 东，落到正确栅格。"""
    smap = SemanticMap(ORIGIN_LAT, ORIGIN_LON, cell_size_m=5.0)
    # UAV 在原点正上方，patch 半径 60m，patch 100x100 px。
    det = {"class_name": "完全损毁建筑", "conf": 0.9, "bbox": [70.0, 20.0, 80.0, 30.0]}
    smap.mark_observation(
        uav_lat=ORIGIN_LAT,
        uav_lon=ORIGIN_LON,
        radius_m=60.0,
        detections=[det],
        degraded=False,
        risk_level="high",
        patch_width=100,
        patch_height=100,
    )
    objs = smap.objects(LAYER_LANDMARKS)
    assert len(objs) == 1, f"应投影出 1 个建筑目标，实际 {len(objs)}"
    obj = objs[0]

    # 期望落点：北 +30m、东 +30m
    exp_lat, exp_lon = meters_to_latlon(ORIGIN_LAT, ORIGIN_LON, 30.0, 30.0)
    north_err, east_err = latlon_to_meters(exp_lat, exp_lon, obj["lat"], obj["lon"])
    err_m = math.hypot(north_err, east_err)
    assert err_m <= smap.cell_size_m, f"投影往返误差 {err_m:.2f}m 超过 1 格"
    assert obj["class_name"] == "完全损毁建筑"
    assert _approx(obj["conf"], 0.9, 1e-6)
    print(f"[OK] 投影精度：误差 {err_m:.3f}m，落点 cell=({obj['gi']},{obj['gj']})")


def test_explored_growth() -> None:
    """两次不同位置观测后，explored 栅格增多。"""
    smap = SemanticMap(ORIGIN_LAT, ORIGIN_LON, cell_size_m=5.0)
    smap.mark_observation(ORIGIN_LAT, ORIGIN_LON, radius_m=40.0)
    first = smap.stats()["explored_cells"]
    assert first > 0, "首次观测应产生 explored 栅格"

    # 向北移动约 100m 再观测
    far_lat, far_lon = meters_to_latlon(ORIGIN_LAT, ORIGIN_LON, 100.0, 0.0)
    smap.mark_observation(far_lat, far_lon, radius_m=40.0)
    second = smap.stats()["explored_cells"]
    assert second > first, f"移动后 explored 应增长：{first} -> {second}"
    print(f"[OK] explored 增长：{first} -> {second} 格")


def test_degraded_skips_objects() -> None:
    """退化视场只记 explored、不投影检测框。"""
    smap = SemanticMap(ORIGIN_LAT, ORIGIN_LON, cell_size_m=5.0)
    det = {"class_name": "车辆", "conf": 0.8, "bbox": [10.0, 10.0, 20.0, 20.0]}
    smap.mark_observation(
        ORIGIN_LAT, ORIGIN_LON, radius_m=60.0,
        detections=[det], degraded=True,
        patch_width=100, patch_height=100,
    )
    stats = smap.stats()
    assert stats["surrounding_objects"] == 0, "degraded 不应写入目标"
    assert stats["explored_cells"] > 0, "degraded 仍应记 explored"
    print("[OK] degraded：只记 explored，跳过检测框投影")


def test_serialization() -> None:
    """to_dict / snapshot 结构完整、目标计数一致。"""
    smap = SemanticMap(ORIGIN_LAT, ORIGIN_LON, cell_size_m=5.0, instruction="找完全损毁的建筑")
    dets = [
        {"class_name": "完全损毁建筑", "conf": 0.7, "bbox": [60, 60, 70, 70]},
        {"class_name": "车辆", "conf": 0.6, "bbox": [30, 30, 36, 36]},
    ]
    smap.mark_observation(
        ORIGIN_LAT, ORIGIN_LON, radius_m=60.0,
        detections=dets, patch_width=100, patch_height=100,
    )
    snap = smap.snapshot()
    full = smap.to_dict()
    for key in ("origin", "cell_size_m", "stats", "objects"):
        assert key in snap and key in full, f"缺少字段 {key}"
    assert snap["stats"]["landmarks"] == 1
    assert snap["stats"]["surrounding_objects"] == 1
    assert len(snap["objects"]) == 2
    assert snap["instruction"] == "找完全损毁的建筑"
    print("[OK] 序列化：snapshot / to_dict 字段完整，目标计数一致")


def test_local_matrix() -> None:
    """STMR 前身：目标语义填进正确网格码。"""
    smap = SemanticMap(ORIGIN_LAT, ORIGIN_LON, cell_size_m=5.0)
    det = {"class_name": "完全损毁建筑", "conf": 0.9, "bbox": [50, 30, 60, 40]}
    smap.mark_observation(
        ORIGIN_LAT, ORIGIN_LON, radius_m=60.0,
        detections=[det], patch_width=100, patch_height=100,
    )
    out = smap.to_local_matrix(ORIGIN_LAT, ORIGIN_LON, window_m=100.0, grid_n=20)
    flat = [c for row in out["matrix"] for c in row]
    assert CLASS_CODE["完全损毁建筑"] in flat, "矩阵应包含完全损毁建筑的类别码"
    assert CLASS_CODE["已探索空地"] in flat, "矩阵应有已探索空地铺底"
    print(f"[OK] 局部矩阵：{out['grid_n']}x{out['grid_n']}，含损毁建筑码 {CLASS_CODE['完全损毁建筑']}")


def _run_all() -> int:
    tests = [
        test_projection_accuracy,
        test_explored_growth,
        test_degraded_skips_objects,
        test_serialization,
        test_local_matrix,
    ]
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
