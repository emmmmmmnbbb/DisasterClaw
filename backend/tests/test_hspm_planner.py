"""
backend/tests/test_hspm_planner.py — P1 HSPM 分层规划单测（mock LLM/grounder/STMR）

验收点：
    1. landmark-level：LLM 拆出多地标序列；LLM 失败回退到开放词汇短语。
    2. OROI：看不到子目标时，按 LLM 给的方位输出探索动作。
    3. motion：看得到子目标但未到 → 朝质心 fly_relative；到达 → 推进/完成。
    4. 多地标推进：到第一个 → hover 推进；到最后一个 → stop arrived。
    C3（OROI 打分融合，`VLN_OROI_SCORE`/`HspmConfig.use_oroi_score`）：
    5. score_bearings_llm + _prior_score + score_oroi：LLM 打分 + 方向先验 + frontier
       融合后选中信号一致指向的方位。
    6. LLM 打分全失败/为 None 时，score_oroi 仍能靠先验 + frontier 选出有信息量的
       方位，不会死板回退到"北"。
    7. HspmNavigator 在 use_oroi_score=True 且注入 semantic_map_provider 时，
       step() 走打分融合分支且不崩。

运行：`python backend/tests/test_hspm_planner.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hspm_planner import (  # noqa: E402
    HspmConfig,
    HspmNavigator,
    OroiScoreWeights,
    plan_landmarks,
    reason_oroi,
    score_bearings_llm,
    score_oroi,
)
from semantic_map import SemanticMap  # noqa: E402
from vln_navigator import GroundHit, Observation  # noqa: E402

# alt 低于 arrival_confirm_alt_m(22)，使"到达"用例直接确认（不触发降高核验分支）。
SNAP = {"lat": 31.2304, "lon": 121.4737, "alt": 18.0}


def _obs(patch_radius_m: float = 60.0, degraded: bool = False) -> Observation:
    return Observation(
        detections=[],
        patch_width=256,
        patch_height=256,
        patch_radius_m=patch_radius_m,
        risk_level="none",
        scene_text="",
        degraded=degraded,
        patch_path="",
    )


def _llm_landmarks(_msgs, _t, _mt):
    return json.dumps({"landmarks": ["红色屋顶的仓库", "完全损毁的建筑"], "thought": "顺序经过"})


def _llm_oroi_east(_msgs, _t, _mt):
    return json.dumps({"bearing": "东", "reason": "东侧建筑密集"})


def test_plan_landmarks_multi() -> None:
    lms = plan_landmarks("先到红色屋顶的仓库再找完全损毁的建筑", _llm_landmarks)
    assert lms == ["红色屋顶的仓库", "完全损毁的建筑"], lms
    print(f"[OK] landmark 序列: {lms}")


def test_plan_landmarks_fallback() -> None:
    # LLM=None → 回退到开放词汇短语
    lms = plan_landmarks("飞到北侧寻找蓝色的建筑", None)
    assert len(lms) == 1 and "蓝色" in lms[0], lms
    print(f"[OK] landmark 回退: {lms}")


def test_oroi_bearing() -> None:
    b, _ = reason_oroi("找建筑", "蓝色的建筑", "（地图文本）", "", _llm_oroi_east)
    assert b == "东", b
    # 非法方位回退方向先验
    b2, _ = reason_oroi("找建筑", "x", "map", "西南", lambda *a: '{"bearing":"啥"}')
    assert b2 == "西南", b2
    print("[OK] OROI 方位（含非法回退方向先验）")


def test_step_unseen_explores() -> None:
    """看不到子目标 → 按 OROI 方位探索。"""
    nav = HspmNavigator(
        config=HspmConfig(),
        grounder=lambda p, o: GroundHit(present=False, reason="未见", source="vlm"),
        llm_chat=_llm_oroi_east,
        stmr_provider=lambda snap: {"text": "（地图）"},
    )
    nav.reset("找蓝色的建筑")
    dec = nav.step(_obs(), SNAP)
    assert dec.action == "fly_relative", dec.action
    assert dec.params["east_m"] > 0 and abs(dec.params["north_m"]) < 1e-6, dec.params
    assert not dec.matched
    print(f"[OK] 未见→朝东探索: N{dec.params['north_m']} E{dec.params['east_m']}")


def test_step_seen_moves_toward() -> None:
    """看得到但远 → 朝质心步进。norm_xy=(1.0,0.5) → 正东最大。"""
    nav = HspmNavigator(
        grounder=lambda p, o: GroundHit(present=True, norm_xy=(1.0, 0.5), source="vlm", reason="命中"),
        llm_chat=None,
    )
    nav.reset("找蓝色的建筑")
    dec = nav.step(_obs(patch_radius_m=60.0), SNAP)
    assert dec.action == "fly_relative" and dec.matched, dec
    assert dec.params["east_m"] > 0, dec.params
    print(f"[OK] 命中未到→朝目标步进: E{dec.params['east_m']}")


def test_multi_landmark_progress() -> None:
    """两地标：先到第一个→hover 推进；再到第二个→stop arrived。"""
    # 目标始终在质心(已到达)：norm_xy≈中心
    nav = HspmNavigator(
        grounder=lambda p, o: GroundHit(present=True, norm_xy=(0.5, 0.5), source="vlm", reason="正下方"),
        llm_chat=_llm_landmarks,
    )
    nav.reset("先到红色屋顶的仓库再找完全损毁的建筑")
    assert nav.landmarks == ["红色屋顶的仓库", "完全损毁的建筑"]

    d1 = nav.step(_obs(), SNAP)
    assert d1.action == "hover" and nav.lm_idx == 1, (d1.action, nav.lm_idx)

    d2 = nav.step(_obs(), SNAP)
    assert d2.action == "stop" and d2.arrived, (d2.action, d2.arrived)
    print("[OK] 多地标推进：地标1 hover 推进 → 地标2 stop arrived")


def test_score_bearings_llm_fallback_on_invalid() -> None:
    b, reason = score_bearings_llm("t", "s", "map", lambda *a: "not json")
    assert b is None, b
    print(f"[OK] score_bearings_llm 非法输出 → None，reason={reason}")


def test_score_oroi_llm_prior_frontier_agree() -> None:
    """LLM 打分 + 方向先验 + frontier 三路信号一致指向"东" → 应选东。"""
    def _llm_east(_msgs, _t, _mt):
        scores = {b: 0.1 for b in ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]}
        scores["东"] = 0.9
        return json.dumps({"scores": scores, "reason": "东侧建筑密集"})

    def _frontier_east(bearing: str) -> float:
        return 0.9 if bearing == "东" else 0.2

    bearing, reason = score_oroi(
        "找建筑", "蓝色的建筑", "（地图）", "东", _llm_east, frontier_fn=_frontier_east,
    )
    assert bearing == "东", (bearing, reason)
    print(f"[OK] OROI 打分融合三路一致选东: {reason}")


def test_score_oroi_no_llm_uses_frontier() -> None:
    """LLM 不可用（None）、也没有方向先验时，纯靠 frontier 信号选方位，
    不会像 reason_oroi 那样死板回退到"北"。"""
    def _frontier_southeast(bearing: str) -> float:
        return 0.95 if bearing == "东南" else 0.1

    bearing, reason = score_oroi(
        "找建筑", "蓝色的建筑", "", "", None, frontier_fn=_frontier_southeast,
    )
    assert bearing == "东南", (bearing, reason)
    print(f"[OK] LLM 全失败但 frontier 有信号 → 选东南（不是死板回退北）: {reason}")


def test_score_oroi_weights_shape() -> None:
    w = OroiScoreWeights(llm=0.6, prior=0.1, frontier=0.3)
    bearing, _ = score_oroi("t", "s", "map", "", None, frontier_fn=None, weights=w)
    assert bearing in {"北", "东北", "东", "东南", "南", "西南", "西", "西北"}
    print(f"[OK] 自定义权重下 score_oroi 仍返回合法方位: {bearing}")


def test_step_with_oroi_score() -> None:
    """HspmConfig.use_oroi_score=True + 注入 semantic_map_provider → step() 走
    打分融合分支，不崩、动作合法。"""
    smap = SemanticMap(origin_lat=SNAP["lat"], origin_lon=SNAP["lon"], cell_size_m=5.0)
    smap.mark_observation(SNAP["lat"], SNAP["lon"], radius_m=40.0)

    def _llm_neutral(_msgs, _t, _mt):
        scores = {b: 0.5 for b in ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]}
        return json.dumps({"scores": scores, "reason": "均衡"})

    nav = HspmNavigator(
        config=HspmConfig(use_oroi_score=True),
        grounder=lambda p, o: GroundHit(present=False, reason="未见", source="vlm"),
        llm_chat=_llm_neutral,
        stmr_provider=lambda snap: {"text": "（地图）"},
        semantic_map_provider=lambda: smap,
    )
    nav.reset("找蓝色的建筑")
    dec = nav.step(_obs(), SNAP)
    assert dec.action == "fly_relative", dec.action
    assert not dec.matched
    print(f"[OK] use_oroi_score=True 下 step() 走打分融合分支不崩: {dec.reason}")


def test_degraded_no_crash() -> None:
    nav = HspmNavigator(
        grounder=lambda p, o: GroundHit(present=True, norm_xy=(0.5, 0.5), source="vlm"),
        llm_chat=None,
        stmr_provider=lambda snap: None,
    )
    nav.reset("找蓝色的建筑")
    dec = nav.step(_obs(degraded=True), SNAP)
    # degraded → usable=False → 走 OROI 探索分支，不崩
    assert dec.action == "fly_relative"
    print("[OK] degraded 视场不崩，走探索分支")


def _run_all() -> int:
    tests = [
        test_plan_landmarks_multi,
        test_plan_landmarks_fallback,
        test_oroi_bearing,
        test_score_bearings_llm_fallback_on_invalid,
        test_score_oroi_llm_prior_frontier_agree,
        test_score_oroi_no_llm_uses_frontier,
        test_score_oroi_weights_shape,
        test_step_with_oroi_score,
        test_step_unseen_explores,
        test_step_seen_moves_toward,
        test_multi_landmark_progress,
        test_degraded_no_crash,
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
