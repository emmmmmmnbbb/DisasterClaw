from __future__ import annotations

import json
import logging
import re

from llm_client import get_client

logger = logging.getLogger(__name__)


class TaskPlanner:
    def __init__(self, hover_altitude_m: float = 30.0):
        self._hover_altitude_m = hover_altitude_m

    def plan(self, task: str, world_state: dict) -> dict:
        try:
            return self._plan_with_llm(task, world_state)
        except Exception as exc:
            logger.exception("LLM planner failed, falling back to rule-based plan")
            return self._fallback_plan(task, world_state, llm_error=str(exc))

    def _plan_with_llm(self, task: str, world_state: dict) -> dict:
        robot = world_state["robots"]["UAV_1"]
        world_map = world_state.get("map", {}) or {}
        active_tile = world_map.get("active_tile") or {}
        client = get_client(module="planner")
        content = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are planning tasks for a single UAV in a disaster response console. "
                        "Output JSON only (no prose, no markdown) with keys summary (string) and "
                        "steps (list). Allowed actions and EXACT param schemas:\n"
                        '  - fly_to_geo:        {"lat": <float>, "lon": <float>, "alt": <float m>, "speed": <float m/s>}\n'
                        '  - fly_relative:      {"north_m": <float>, "east_m": <float>, "up_m": <float>, "speed": <float m/s>}\n'
                        '  - hover:             {"duration": <float sec>}\n'
                        '  - detect_disaster:   {"radius_m": <float, default 120>, "use_vlm_summary": <bool, default true>}\n'
                        '  - mark_target:       {"label": <string>, "lat": <float>, "lon": <float>, "kind": <string>}\n'
                        '  - report_observation:{"content": <string>, "level": "info"|"warn"|"error"}\n'
                        "Each step must be an object with keys action, params, reason. "
                        "Do NOT use synonyms such as latitude/longitude/altitude — use exactly lat/lon/alt. "
                        "The UAV operates over a real xBD satellite tile. All fly_to_geo coordinates "
                        "MUST stay inside latlon_bounds unless the task explicitly says otherwise. "
                        "Prefer fly_to_geo with lat/lon from the active tile over fly_relative. "
                        "NOTE: fly_to_geo automatically aligns the active xBD tile to whichever POST-disaster "
                        "tile covers the target (falls back to PRE if no POST covers it), so you do NOT need "
                        "to emit any tile-switch step before fly_to_geo. "
                        "detect_disaster runs YOLO (building damage, vehicles, water) + SegFormer over the "
                        "UAV's current view; ALWAYS insert one detect_disaster step AFTER the UAV arrives at "
                        "the observation point and BEFORE report_observation when the task mentions "
                        "inspect / survey / damage / disaster / 受灾 / 检测 / 监测 / 侦察 / 查看 / 评估."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task,
                            "hover_altitude_m": self._hover_altitude_m,
                            "robot": robot,
                            "active_tile_id": world_map.get("active_tile_id"),
                            "active_tile": {
                                "disaster": active_tile.get("disaster"),
                                "disaster_type": active_tile.get("disaster_type"),
                                "stage": active_tile.get("stage"),
                                "gsd": active_tile.get("gsd"),
                            } if active_tile else None,
                            "latlon_bounds": world_map.get("latlon_bounds"),
                            "anchor": world_map.get("anchor"),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.2,
        )
        parsed = _extract_json(content)
        if not isinstance(parsed, dict) or "steps" not in parsed:
            raise RuntimeError("LLM returned invalid plan")
        return parsed

    def _fallback_plan(self, task: str, world_state: dict, llm_error: str) -> dict:
        robot = world_state["robots"]["UAV_1"]
        current = robot["position"]
        coords = _extract_latlon(task)
        steps = []
        summary_bits = []

        if coords:
            lat, lon = coords
            steps.append(
                {
                    "action": "fly_to_geo",
                    "params": {"lat": lat, "lon": lon, "alt": self._hover_altitude_m, "speed": 14.0},
                    "reason": "任务中包含明确经纬度，先飞到目标点上空。",
                }
            )
            summary_bits.append(f"飞往目标经纬度 {lat:.6f}, {lon:.6f}")
        elif any(word in task for word in ("向北", "north", "前往北侧", "北边")):
            steps.append(
                {
                    "action": "fly_relative",
                    "params": {"north_m": 180.0, "east_m": 0.0, "up_m": 0.0, "speed": 12.0},
                    "reason": "任务没有经纬度但存在方向词，执行相对移动。",
                }
            )
            summary_bits.append("沿北向做相对机动")

        if any(word in task for word in ("观察", "inspect", "巡查", "查看", "侦察", "observe")):
            steps.append(
                {
                    "action": "hover",
                    "params": {"duration": 4.0},
                    "reason": "到位后悬停，模拟观察窗口。",
                }
            )
            summary_bits.append("悬停观察")

        if any(
            word in task.lower()
            for word in (
                "受灾", "灾情", "灾害", "损伤", "损毁", "倒塌", "监测", "检测",
                "评估", "damage", "disaster", "detect", "survey", "assess",
            )
        ):
            steps.append(
                {
                    "action": "detect_disaster",
                    "params": {"radius_m": 120.0, "use_vlm_summary": True},
                    "reason": "运行 YOLO + SegFormer + VLM，对当前视场做灾情判定。",
                }
            )
            summary_bits.append("视觉灾情检测")

        if any(word in task for word in ("标记", "mark", "记录")):
            label = _extract_label(task) or "AI Marker"
            steps.append(
                {
                    "action": "mark_target",
                    "params": {
                        "label": label,
                        "lat": coords[0] if coords else current["lat"],
                        "lon": coords[1] if coords else current["lon"],
                        "kind": "inspection-point",
                    },
                    "reason": "任务要求在地图上留下标记。",
                }
            )
            summary_bits.append("记录目标点")

        steps.append(
            {
                "action": "report_observation",
                "params": {
                    "content": _build_report(task, coords or (current["lat"], current["lon"])),
                    "level": "info",
                },
                "reason": "任务结束后输出简要态势报告。",
            }
        )

        summary = "，".join(summary_bits) if summary_bits else "保持悬停并输出当前位置报告"
        if llm_error:
            summary = f"{summary}（LLM 不可用，已切换规则规划）"
        return {"summary": summary, "steps": steps}


def _extract_latlon(task: str) -> tuple[float, float] | None:
    patterns = [
        r"(-?\d+\.\d+)\s*[,，]\s*(-?\d+\.\d+)",
        r"lat[:= ]\s*(-?\d+\.\d+).{0,12}?lon[:= ]\s*(-?\d+\.\d+)",
        r"纬度[:： ]\s*(-?\d+\.\d+).{0,12}?经度[:： ]\s*(-?\d+\.\d+)",
    ]
    for pattern in patterns:
        hit = re.search(pattern, task, re.IGNORECASE)
        if hit:
            first = float(hit.group(1))
            second = float(hit.group(2))
            if abs(first) <= 90 and abs(second) <= 180:
                return first, second
    return None


def _extract_label(task: str) -> str | None:
    quoted = re.search(r"[\"“](.+?)[\"”]", task)
    if quoted:
        return quoted.group(1)[:40]
    return None


def _build_report(task: str, coords: tuple[float, float]) -> str:
    lat, lon = coords
    return (
        f"已完成任务“{task[:48]}”，当前 UAV 位于 {lat:.6f}, {lon:.6f} 上空，"
        "处于 30m 悬停高度。本次结果来自本地 mock 规则/AI 规划链路。"
    )


def _extract_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise RuntimeError("No JSON object found")
    return json.loads(match.group(0))
