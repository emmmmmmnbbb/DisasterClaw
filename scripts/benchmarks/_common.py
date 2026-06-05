"""
Shared utilities for the simulation platform benchmark suite.

Provides:
- standard 5 task templates (T1..T5) compiled to fixed plan steps that can be
  injected directly via socketio "execute_action" (planner-bypass mode) or
  recreated as natural-language `ai_task` (LLM mode);
- helpers to load the rescuenet damage_ranking.json so tasks have realistic
  POST-disaster targets;
- run-id / artifact path helpers (mkdir + JSON dump) reused across all bench
  scripts.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "runs" / "benchmarks"
RESCUENET_RANKING = REPO_ROOT / "backend" / "data" / "rescuenet" / "damage_ranking.json"
RESCUENET_MANIFEST = REPO_ROOT / "backend" / "data" / "rescuenet" / "manifest.json"

# The runtime real-time budget the platform is benchmarked against.
SLA = {
    "total_ms": 30_000,        # full chain
    "plan_ms": 3_000,          # LLM planning
    "first_perception_ms": 10_000,  # submit → first perception_result
    "report_ms": 25_000,       # submit → ai_execution_report
}

DEFAULT_HOVER_ALT = 30.0
DEFAULT_SPEED = 14.0


def now_run_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_run_dir(run_id: str) -> Path:
    out = RUNS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_top_damage_tiles(top_n: int = 10) -> list[dict]:
    """Return the top-N rescuenet POST tiles sorted by destroyed-building severity."""
    if not RESCUENET_RANKING.is_file():
        raise FileNotFoundError(
            f"{RESCUENET_RANKING} not found. Run "
            "`python scripts/build_rescuenet_dataset.py --force` first."
        )
    data = json.load(open(RESCUENET_RANKING, "r", encoding="utf-8"))
    items = data.get("items", [])[:top_n]
    return items


def latlon_offset(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    dlat = north_m / 110540.0
    dlon = east_m / (111320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


# --------------------------- Task templates --------------------------- #
#
# Each task definition emits a *fixed plan*: a list of {action, params, reason}
# step dicts. We pick concrete (lat, lon) from rescuenet damage_ranking.json
# so all 5 tasks are guaranteed to land on a real POST-disaster tile.

def _build_tasks(top_tiles: list[dict]) -> dict[str, dict]:
    if len(top_tiles) < 5:
        raise ValueError("need at least 5 rescuenet POST tiles for the bench task templates")

    # Anchor tiles: pick the strongest 5 distinct ones.
    primary = [t for t in top_tiles if t.get("center")][:5]
    if len(primary) < 5:
        raise ValueError("damage_ranking.json missing center for top tiles")

    p1, p2, p3, p4, p5 = primary[:5]

    def fly(t):
        c = t["center"]
        return {
            "action": "fly_to_geo",
            "params": {
                "lat": float(c["lat"]),
                "lon": float(c["lon"]),
                "alt": DEFAULT_HOVER_ALT,
                "speed": DEFAULT_SPEED,
            },
            "reason": f"飞往灾后瓦片 {t['tile_id']} 中心",
        }

    def detect(reason: str = "对当前视场执行 YOLO+SegFormer 灾情检测"):
        return {
            "action": "detect_disaster",
            "params": {"radius_m": 120.0, "use_vlm_summary": True},
            "reason": reason,
        }

    def report(content: str):
        return {
            "action": "report_observation",
            "params": {"content": content, "level": "info"},
            "reason": "汇总并广播本次任务的关键观察",
        }

    # T2 : 5-point snake scan around p2 center ± 60 m
    t2_center = p2["center"]
    t2_offsets_m = [(0, 0), (60, 0), (60, 60), (-60, 60), (-60, -60)]
    t2_steps = []
    for dn, de in t2_offsets_m:
        lat, lon = latlon_offset(t2_center["lat"], t2_center["lon"], dn, de)
        t2_steps.append({
            "action": "fly_to_geo",
            "params": {"lat": lat, "lon": lon, "alt": DEFAULT_HOVER_ALT, "speed": DEFAULT_SPEED},
            "reason": f"洪水范围测绘扫描点 ({dn:+}m, {de:+}m)",
        })
    t2_steps.append(detect("在测绘中心点执行 SegFormer 水域分割与 YOLO 物体识别"))
    t2_steps.append(report("洪水扫描完成，已采集中心视场水域占比"))

    # T5 : multi-point inspection across p1, p3, p5
    t5_steps = []
    for tile, idx in [(p1, 1), (p3, 2), (p5, 3)]:
        t5_steps.append(fly(tile))
        t5_steps.append(detect(f"多点综合巡查第 {idx} 点感知"))
    t5_steps.append(report("多点综合巡查完成，已串联感知三点"))

    return {
        "T1": {
            "name": "建筑损伤评估",
            "label": "T1 建筑损伤评估",
            "task_text": (
                f"请对灾后区域 ({p1['center']['lat']:.6f}, {p1['center']['lon']:.6f}) "
                "进行建筑损伤评估：飞到目标点上空悬停，调用 detect_disaster 执行 "
                "YOLO + SegFormer 联合感知，最后汇报受灾检测结果。"
            ),
            "steps": [fly(p1), detect("评估建筑损伤等级"), report("已完成建筑损伤评估")],
            "anchor_tile": p1,
        },
        "T2": {
            "name": "洪水范围测绘",
            "label": "T2 洪水范围测绘",
            "task_text": (
                f"对 ({t2_center['lat']:.6f}, {t2_center['lon']:.6f}) 周围 60 m 范围执行 "
                "5 点蛇形飞行，扫描洪水范围；在中心点执行 detect_disaster 重点采集水域占比。"
            ),
            "steps": t2_steps,
            "anchor_tile": p2,
        },
        "T3": {
            "name": "道路阻塞侦察",
            "label": "T3 道路阻塞侦察",
            "task_text": (
                f"飞到 ({p3['center']['lat']:.6f}, {p3['center']['lon']:.6f}) 上空，"
                "执行 detect_disaster 重点检测道路是否被碎片阻塞，并汇报。"
            ),
            "steps": [fly(p3), detect("检测道路阻塞情况"), report("已完成道路阻塞侦察")],
            "anchor_tile": p3,
        },
        "T4": {
            "name": "车辆资产清点",
            "label": "T4 车辆资产清点",
            "task_text": (
                f"飞到 ({p4['center']['lat']:.6f}, {p4['center']['lon']:.6f}) 上空，"
                "执行 detect_disaster 重点统计 vehicle 类目标数量，并汇报。"
            ),
            "steps": [fly(p4), detect("YOLO 重点统计车辆数量"), report("已完成车辆资产清点")],
            "anchor_tile": p4,
        },
        "T5": {
            "name": "多点综合巡查",
            "label": "T5 多点综合巡查",
            "task_text": (
                "对 3 个高风险瓦片中心依次执行飞行 + detect_disaster，覆盖完整感知工具链，"
                "最终汇总观察。"
            ),
            "steps": t5_steps,
            "anchor_tile": p1,
        },
    }


_TASKS_CACHE: dict[str, dict] | None = None


def get_tasks(force_reload: bool = False) -> dict[str, dict]:
    """Lazy build the 5 task templates from the latest rescuenet ranking."""
    global _TASKS_CACHE
    if _TASKS_CACHE is None or force_reload:
        tiles = load_top_damage_tiles(top_n=10)
        _TASKS_CACHE = _build_tasks(tiles)
    return _TASKS_CACHE


# --------------------------- helpers ---------------------------------- #


def short_action_chain(steps: list[dict]) -> str:
    return " → ".join(step.get("action", "?") for step in steps)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def stat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(values)
    return {
        "n": len(s),
        "mean": sum(s) / len(s),
        "p50": percentile(s, 0.5),
        "p95": percentile(s, 0.95),
        "min": s[0],
        "max": s[-1],
    }


def env_snapshot() -> dict[str, Any]:
    """Capture the runtime environment so JSON results are self-describing."""
    info: dict[str, Any] = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_mode": os.environ.get("DATASET_MODE", "xbd"),
        "perception_device": os.environ.get("PERCEPTION_DEVICE", "cuda"),
        "python": _python_version(),
    }
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
    except Exception:
        pass
    return info


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def seeded_task_sequence(total: int, seed: int = 42, ids: list[str] | None = None) -> list[str]:
    ids = ids or ["T1", "T2", "T3", "T4", "T5"]
    rng = random.Random(seed)
    return [rng.choice(ids) for _ in range(total)]
