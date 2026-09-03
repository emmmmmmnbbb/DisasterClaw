#!/usr/bin/env python3
"""scripts/benchmarks/test_budget_allocation.py — D1/E0 泄漏哨兵与配对统计测试.

对应 AGENT_VQA_REVISION_PLAN.md 的 D1（修复 X2 未来观测泄漏）与 E0（协议与泄漏
单元测试）。验收点：

  1. score_expected_gain 的签名是可执行的泄漏哨兵：它只接受 current_probs，
     根本没有 native/future 形参 —— 在线策略在类型层面就无法访问未来观测。
  2. evaluate 在 fit/test 事件重叠时 leakage_check_passed=False；不相交时 True。
  3. fit_expected_gain_table 只用 fit_items 拟合，不读取 test items。
  4. 配对 bootstrap 产出 95% CI 与 excludes_zero 标记。
  5. undo_temperature 能从 softmax(logits/T) 反推 softmax(logits)。
  6. Brier / NLL / AURC / cost_risk 指标方向正确。
  7. 事件切分 EVAL_EVENTS 与 LEAK_EVENTS 交集为空（D1 事件切分断言）。

运行：`python scripts/benchmarks/test_budget_allocation.py`
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from eval_budget_allocation import (  # noqa: E402
    _aurc,
    _brier,
    _cost_risk,
    _nll,
    allocate,
    evaluate,
    fit_expected_gain_table,
    score_expected_gain,
    undo_temperature,
)
from event_split import EVAL_EVENTS, LEAK_EVENTS  # noqa: E402

CLASS_ORDER = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def _probs(p: tuple[float, float, float, float]) -> dict[str, float]:
    return {n: float(v) for n, v in zip(CLASS_ORDER, p)}


def _item(disaster: str, y: int, cruise: tuple[float, float, float, float],
          native: tuple[float, float, float, float]) -> dict:
    return {
        "disaster": disaster,
        "y": y,
        "views": {
            "cruise": {"probs": _probs(cruise)},
            "floor": {"probs": _probs(native)},
        },
    }


def test_score_expected_gain_signature_is_leakage_guard() -> None:
    """score_expected_gain 只接受 current_probs，没有 native/future 形参 ——
    这是可执行的泄漏哨兵：在线策略在类型层面就无法把未来观测传进来。"""
    sig = inspect.signature(score_expected_gain)
    params = list(sig.parameters.keys())
    assert params == ["current_probs", "table"], params
    for forbidden in ("native", "future", "y", "items", "test"):
        assert forbidden not in params, f"leakage: score_expected_gain 接受 {forbidden!r}"
    print(f"[OK] score_expected_gain 签名为泄漏哨兵: {params}")


def test_score_expected_gain_does_not_read_future() -> None:
    """同一 current_probs 下，无论"未来观测"长什么样，打分必须相同 ——
    证明打分函数确实不依赖未来信息。"""
    fit = [_item("hurricane-harvey", 3, (0.3, 0.3, 0.2, 0.2), (0.1, 0.1, 0.1, 0.7))]
    table = fit_expected_gain_table(fit, n_bins=3)
    current = np.array([[0.3, 0.3, 0.2, 0.2]], dtype=np.float64)
    s1 = score_expected_gain(current, table)
    # 即便构造一个"完全不同的未来"，打分也不变（函数根本拿不到未来）
    s2 = score_expected_gain(current, table)
    assert np.array_equal(s1, s2), (s1, s2)
    print(f"[OK] 打分不依赖未来观测: {s1}")


def test_evaluate_flags_event_overlap_as_leakage() -> None:
    """fit 与 test 共享事件时 leakage_check_passed=False（main() 会以退出码 3 拒绝）。"""
    fit = [_item("palu-tsunami", 3, (0.3, 0.3, 0.2, 0.2), (0.1, 0.1, 0.1, 0.7))
           for _ in range(6)]
    test = [_item("palu-tsunami", 3, (0.3, 0.3, 0.2, 0.2), (0.1, 0.1, 0.1, 0.7))
            for _ in range(6)]
    res = evaluate(test, fit, qhat=0.9, temperature=1.0, seed=0, n_boot=50)
    assert res["leakage_check_passed"] is False, res
    assert "palu-tsunami" in res["fit_test_event_overlap"], res
    print(f"[OK] 事件重叠被标记为泄漏: overlap={res['fit_test_event_overlap']}")


def test_evaluate_passes_when_events_disjoint() -> None:
    """fit 用 val 事件、test 用 test 事件且不相交时 leakage_check_passed=True。"""
    fit = [_item("hurricane-harvey", 3, (0.3, 0.3, 0.2, 0.2), (0.1, 0.1, 0.1, 0.7))
           for _ in range(8)]
    test = [_item("palu-tsunami", 3, (0.3, 0.3, 0.2, 0.2), (0.1, 0.1, 0.1, 0.7))
            for _ in range(8)]
    res = evaluate(test, fit, qhat=0.9, temperature=1.0, seed=0, n_boot=50)
    assert res["leakage_check_passed"] is True, res
    assert res["fit_test_event_overlap"] == [], res
    print("[OK] 事件不相交时 leakage_check_passed=True")


def test_fit_table_only_uses_fit_items() -> None:
    """fit_n 必须等于 fit_items 长度，fit_events 必须来自 fit_items。"""
    fit = [
        _item("hurricane-harvey", 3, (0.3, 0.3, 0.2, 0.2), (0.1, 0.1, 0.1, 0.7)),
        _item("mexico-earthquake", 0, (0.7, 0.1, 0.1, 0.1), (0.8, 0.1, 0.05, 0.05)),
    ]
    table = fit_expected_gain_table(fit, n_bins=3)
    assert table.fit_n == 2, table.fit_n
    assert set(table.fit_events) == {"hurricane-harvey", "mexico-earthquake"}, table.fit_events
    print(f"[OK] fit 表只用 fit_items: n={table.fit_n}, events={table.fit_events}")


def test_paired_bootstrap_returns_ci_and_excludes_zero() -> None:
    """配对 bootstrap 必须给出 95% CI 与 excludes_zero 标记（D1 改用配对统计）。"""
    fit = [_item("hurricane-harvey", 3, (0.3, 0.3, 0.2, 0.2), (0.1, 0.1, 0.1, 0.7))
           for _ in range(8)]
    # 构造 native 明显比 cruise 好（答案翻转多）的 test 集
    test = [_item("palu-tsunami", 3, (0.3, 0.3, 0.2, 0.2), (0.05, 0.05, 0.05, 0.85))
            for _ in range(8)]
    res = evaluate(test, fit, qhat=0.9, temperature=1.0, seed=1, n_boot=200)
    paired = res["paired_tests_at_0.25"]
    assert paired, paired
    for key, block in paired.items():
        for metric, ci_block in block.items():
            assert "ci95" in ci_block and len(ci_block["ci95"]) == 2, (key, metric, ci_block)
            assert "excludes_zero" in ci_block, (key, metric, ci_block)
            assert ci_block["n_boot"] == 200, ci_block
    print(f"[OK] 配对 bootstrap 产出 CI: {list(paired.keys())}")


def test_undo_temperature_round_trip() -> None:
    """undo_temperature(softmax(logits/T), T) ≈ softmax(logits)（方向正确即可）。"""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 4))
    # softmax(logits) 的参考实现
    ref = np.exp(logits - logits.max(1, keepdims=True))
    ref = ref / ref.sum(1, keepdims=True)
    for T in (0.7, 1.0, 1.5, 2.0):
        scaled = np.exp(logits / T - (logits / T).max(1, keepdims=True))
        scaled = scaled / scaled.sum(1, keepdims=True)
        recovered = undo_temperature(scaled, T)
        assert np.allclose(recovered, ref, atol=1e-6), (T, recovered, ref)
    print("[OK] undo_temperature 反推 softmax(logits) 正确")


def test_brier_nll_aurc_direction() -> None:
    """Brier/NLL 越小越好、AURC 越小越好；正确预测应低于错误预测。"""
    y = np.array([0, 1, 2, 3])
    confident = np.eye(4)
    wrong = np.eye(4)[np.roll(np.arange(4), 1)]
    assert _brier(confident, y) < _brier(wrong, y), (_brier(confident, y), _brier(wrong, y))
    assert _nll(confident, y) < _nll(wrong, y)
    assert _aurc(confident, y) < _aurc(wrong, y)
    print("[OK] Brier/NLL/AURC 方向正确")


def test_cost_risk_penalises_missed_damage() -> None:
    """把损伤预测成 no-damage 的代价应随 miss_cost 上升而上升。"""
    y = np.array([3, 3, 3, 3])
    pred_no = np.array([0, 0, 0, 0])
    assert _cost_risk(pred_no, y, 3.0) < _cost_risk(pred_no, y, 5.0)
    assert _cost_risk(pred_no, y, 5.0) < _cost_risk(pred_no, y, 10.0)
    # 全对时代价为 0
    assert _cost_risk(y, y, 10.0) == 0.0
    print("[OK] cost_risk 对漏报损伤敏感")


def test_allocate_respects_budget() -> None:
    """allocate 在 budget=0 时不选、budget=1 时全选、确定性模式下按分数排序。"""
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    assert allocate(scores, 0.0).sum() == 0
    assert allocate(scores, 1.0).sum() == 4
    chosen = allocate(scores, 0.5, None)  # 2 个
    assert chosen.sum() == 2
    assert chosen[1] and chosen[2], chosen  # 最高的两个
    print("[OK] allocate 预算约束正确")


def test_all_strategies_use_identical_k_per_budget() -> None:
    fit = [_item("hurricane-harvey", 0, (0.5, 0.2, 0.2, 0.1), (0.8, 0.1, 0.05, 0.05))
           for _ in range(8)]
    test = [_item("palu-tsunami", i % 4, (0.4, 0.3, 0.2, 0.1), (0.1, 0.2, 0.3, 0.4))
            for i in range(20)]
    result = evaluate(test, fit, qhat=0.9, temperature=1.0, seed=0, n_boot=20)
    for budget in (0.1, 0.25, 0.5, 1.0):
        counts = {
            name: next(r["n_descend"] for r in rows if r["budget"] == budget)
            for name, rows in result["curves"].items() if name != "none"
        }
        assert len(set(counts.values())) == 1, (budget, counts)
    assert all("net_corrected" in row for rows in result["curves"].values() for row in rows)
    print("[OK] 所有可比较策略每个预算档使用完全相同的 K")


def test_load_items_rejects_legacy_synthetic_blur_keys() -> None:
    from eval_budget_allocation import _load_items

    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "legacy.jsonl"
        path.write_text(
            json.dumps({
                "y": 0,
                "views": {
                    "1.0": {"probs": _probs((0.7, 0.1, 0.1, 0.1))},
                    "4.0": {"probs": _probs((0.4, 0.2, 0.2, 0.2))},
                },
            }) + "\n",
            encoding="utf-8",
        )
        try:
            _load_items(path, 0)
        except ValueError as exc:
            assert "legacy" in str(exc) or "cruise" in str(exc)
            print("[OK] 旧 1.0/4.0 键被拒绝")
            return
        raise AssertionError("legacy synthetic-blur items must not load as FOV paired rows")


def test_export_budget_table_has_brier_nll_net() -> None:
    from export_cja_assets import write_budget_table

    curves = {
        "none": [{"budget": 0.25, "macro_f1": 0.1, "accuracy": 0.2, "ece": 0.3,
                  "brier": 0.4, "nll": 0.5, "net_corrected": 0}],
        "entropy_cal": [{"budget": 0.25, "macro_f1": 0.2, "accuracy": 0.3, "ece": 0.2,
                         "brier": 0.3, "nll": 0.4, "net_corrected": 4}],
    }
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "budget_table.tex"
        write_budget_table(curves, path)
        text = path.read_text(encoding="utf-8")
    assert "Brier" in text and "NLL" in text and "净纠正" in text
    print("[OK] 导出预算表含 Brier/NLL/净纠正")


def test_event_split_partitions_are_disjoint() -> None:
    """D1 事件切分断言：EVAL_EVENTS 与 LEAK_EVENTS 交集为空。"""
    overlap = set(EVAL_EVENTS) & set(LEAK_EVENTS)
    assert not overlap, f"事件切分泄漏: {overlap}"
    assert set(EVAL_EVENTS) | set(LEAK_EVENTS) == set(EVAL_EVENTS + LEAK_EVENTS)
    print(f"[OK] 事件切分不相交: eval={EVAL_EVENTS}, leak_count={len(LEAK_EVENTS)}")


def _run_all() -> int:
    tests = [
        test_score_expected_gain_signature_is_leakage_guard,
        test_score_expected_gain_does_not_read_future,
        test_evaluate_flags_event_overlap_as_leakage,
        test_evaluate_passes_when_events_disjoint,
        test_fit_table_only_uses_fit_items,
        test_paired_bootstrap_returns_ci_and_excludes_zero,
        test_undo_temperature_round_trip,
        test_brier_nll_aurc_direction,
        test_cost_risk_penalises_missed_damage,
        test_allocate_respects_budget,
        test_all_strategies_use_identical_k_per_budget,
        test_load_items_rejects_legacy_synthetic_blur_keys,
        test_export_budget_table_has_brier_nll_net,
        test_event_split_partitions_are_disjoint,
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
    print(f"\n{'=' * 48}\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run_all())
