"""
backend/tests/test_grounding_parse.py — P1 grounding 修复的纯函数单测

覆盖两处修复：
    1. extract_target_phrase：开放词汇目标短语保留修饰（"蓝色"），不被 YOLO 词典改写。
    2. parse_ground_xy：坐标-or-没有 范式的鲁棒解析（避开 present 布尔陷阱）。

运行：`python backend/tests/test_grounding_parse.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vln_navigator import extract_target_phrase, parse_ground_xy, parse_instruction  # noqa: E402


def test_extract_phrase_keeps_color() -> None:
    p = extract_target_phrase("飞到北侧寻找蓝色的建筑。")
    assert "蓝色" in p, f"应保留‘蓝色’修饰，实际: {p!r}"
    assert "建筑" in p, f"应保留‘建筑’，实际: {p!r}"
    # 方向/动作应被剥掉
    assert "飞到" not in p and "北" not in p, f"方向/动作未剥净: {p!r}"
    print(f"[OK] 开放词汇短语保留修饰: {p!r}")


def test_parse_instruction_has_phrase() -> None:
    parsed = parse_instruction("飞到北侧寻找蓝色的建筑。")
    assert "蓝色" in parsed["target_phrase"], parsed["target_phrase"]
    # target_label 仍是 YOLO 类别（旧行为，给 yolo 模式用），二者解耦
    assert "建筑" in parsed["target_label"]
    print(f"[OK] parse_instruction.target_phrase={parsed['target_phrase']!r} / target_label={parsed['target_label']!r}")


def test_parse_xy_coords() -> None:
    assert parse_ground_xy("0.56,0.49") == (0.56, 0.49)
    assert parse_ground_xy("0.32, 0.78") == (0.32, 0.78)
    # 0~100 / 0~1000 容错
    assert parse_ground_xy("56,49") == (0.56, 0.49)
    print("[OK] 坐标解析（含 0~100/0~1000 容错）")


def test_parse_xy_none() -> None:
    assert parse_ground_xy("没有") is None
    assert parse_ground_xy("画面中没有该目标") is None
    assert parse_ground_xy("") is None
    assert parse_ground_xy("我没看到") is None
    print("[OK] 否定回答解析为 None（未命中）")


def test_parse_xy_verbose_with_coords() -> None:
    # 模型偶尔啰嗦但给了坐标 → 以坐标为准
    assert parse_ground_xy("目标在下方偏左 0.30,0.80") == (0.30, 0.80)
    print("[OK] 啰嗦但含坐标 → 以坐标为准")


def _run_all() -> int:
    tests = [
        test_extract_phrase_keeps_color,
        test_parse_instruction_has_phrase,
        test_parse_xy_coords,
        test_parse_xy_none,
        test_parse_xy_verbose_with_coords,
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
