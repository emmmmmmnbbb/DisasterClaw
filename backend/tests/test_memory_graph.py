"""
backend/tests/test_memory_graph.py — P3 记忆拓扑图 + 图搜索 单测

验收点（对应 docs/vln_rescue_agent_实施计划.md 的 P3）：
    1. 阈值合并：邻近轨迹点并入同一节点，远点新建。
    2. 连边 + Dijkstra 最短路：沿轨迹可达；返回米权最短。
    3. landmark 匹配：文本 scorer 把"完全损毁的建筑"对到含该标签的节点。
    4. plan：相似指令命中记忆图，给出 walk 航点；分数低于阈值返回 None（交回探索）。
    5. 序列化往返：save/load 后节点/边/标签一致。

运行：`python backend/tests/test_memory_graph.py`
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geo import meters_to_latlon  # noqa: E402
from memory_graph import MemoryGraph, text_match_scorer  # noqa: E402

LAT, LON = 31.2304, 121.4737


def _ll(n_m: float, e_m: float) -> tuple[float, float]:
    return meters_to_latlon(LAT, LON, n_m, e_m)


def test_merge_and_new() -> None:
    g = MemoryGraph(merge_radius_m=15.0)
    a = g.add_waypoint(LAT, LON, labels={"车辆": 1})
    # 10m 内 → 合并
    la, lo = _ll(8.0, 0.0)
    b = g.add_waypoint(la, lo, labels={"完全损毁建筑": 1})
    assert a == b, "8m 内应合并到同一节点"
    # 100m 外 → 新建
    la2, lo2 = _ll(100.0, 0.0)
    c = g.add_waypoint(la2, lo2)
    assert c != a, "100m 外应新建节点"
    assert g.stats()["nodes"] == 2
    print(f"[OK] 合并/新建：{g.stats()}")


def test_path_dijkstra() -> None:
    g = MemoryGraph(merge_radius_m=10.0)
    pts = [
        {"lat": LAT, "lon": LON, "labels": {}},
        {"lat": _ll(50, 0)[0], "lon": _ll(50, 0)[1], "labels": {}},
        {"lat": _ll(100, 0)[0], "lon": _ll(100, 0)[1], "labels": {"完全损毁建筑": 2}},
    ]
    ids = g.add_trajectory(pts, instruction="找完全损毁的建筑", landmarks=["完全损毁建筑"])
    assert len(ids) == 3
    path = g.shortest_path(ids[0], ids[-1])
    assert path == ids, f"应沿轨迹可达 {ids}，实际 {path}"
    print(f"[OK] Dijkstra 最短路：{path}")


def test_landmark_match() -> None:
    g = MemoryGraph()
    g.add_trajectory(
        [{"lat": LAT, "lon": LON, "labels": {"完全损毁建筑": 3}}],
        instruction="找完全损毁的建筑", landmarks=["完全损毁建筑"],
    )
    matches = g.match_nodes("完全损毁的建筑", text_match_scorer, top_k=3)
    assert matches and matches[0][1] > 0.34, matches
    # 不相关短语低分
    low = g.match_nodes("蓝色游泳池", text_match_scorer)
    assert (not low) or low[0][1] < 0.34
    print(f"[OK] 地标匹配：完全损毁的建筑→{matches[0][1]:.2f}")


def test_plan_hit_and_miss() -> None:
    g = MemoryGraph(merge_radius_m=10.0)
    pts = [
        {"lat": LAT, "lon": LON, "labels": {}},
        {"lat": _ll(60, 0)[0], "lon": _ll(60, 0)[1], "labels": {}},
        {"lat": _ll(120, 0)[0], "lon": _ll(120, 0)[1], "labels": {"完全损毁建筑": 2}},
    ]
    g.add_trajectory(pts, instruction="找完全损毁的建筑", landmarks=["完全损毁建筑"])

    # 相似指令、起点靠近首节点 → 命中，给出 walk
    plan = g.plan(["完全损毁的建筑"], LAT, LON)
    assert plan is not None, "相似指令应命中记忆图"
    assert plan["mode"] == "graph_walk", plan["mode"]
    assert len(plan["waypoints"]) >= 2
    # 终点航点应接近 120m 处目标节点
    last = plan["waypoints"][-1]
    from geo import latlon_to_meters
    n, e = latlon_to_meters(LAT, LON, last["lat"], last["lon"])
    assert abs(n - 120.0) < 12.0, (n, e)
    print(f"[OK] plan 命中：mode={plan['mode']} score={plan['target_score']} hops={len(plan['waypoints'])}")

    # 不相关指令 → None
    miss = g.plan(["蓝色的游泳池"], LAT, LON)
    assert miss is None, "无关指令不应命中"
    print("[OK] plan 未命中无关指令 → None（交回探索）")


def test_plan_direct_when_far_start() -> None:
    g = MemoryGraph(merge_radius_m=10.0)
    g.add_trajectory(
        [{"lat": _ll(500, 500)[0], "lon": _ll(500, 500)[1], "labels": {"完全损毁建筑": 2}}],
        instruction="x", landmarks=["完全损毁建筑"],
    )
    # 起点远离任何节点 → direct 模式，单航点直飞
    plan = g.plan(["完全损毁建筑"], LAT, LON)
    assert plan is not None and plan["mode"] == "direct", plan
    assert len(plan["waypoints"]) == 1
    print("[OK] 起点远 → direct 直飞目标节点")


def test_serialization_roundtrip() -> None:
    g = MemoryGraph(merge_radius_m=12.0)
    pts = [
        {"lat": LAT, "lon": LON, "labels": {"车辆": 1}, "risk": "low"},
        {"lat": _ll(40, 0)[0], "lon": _ll(40, 0)[1], "labels": {"完全损毁建筑": 1}, "risk": "high"},
    ]
    g.add_trajectory(pts, instruction="找完全损毁的建筑", landmarks=["完全损毁建筑"])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mem.json"
        g.save(p)
        g2 = MemoryGraph.load(p)
    assert g2.stats() == g.stats(), (g2.stats(), g.stats())
    # 加载后仍能规划
    plan = g2.plan(["完全损毁的建筑"], LAT, LON)
    assert plan is not None
    print(f"[OK] 序列化往返：{g2.stats()}，加载后可规划")


def _run_all() -> int:
    tests = [
        test_merge_and_new,
        test_path_dijkstra,
        test_landmark_match,
        test_plan_hit_and_miss,
        test_plan_direct_when_far_start,
        test_serialization_roundtrip,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            failed += 1
            print(f"[ERROR] {t.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{'='*48}\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run_all())
