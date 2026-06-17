"""
backend/vln_navigator.py — BEV 语言目标导航闭环控制器（PoC）

定位（已与用户确认）：
    DisasterClaw 是 2D 俯视（nadir）地理瓦片仿真，不是第一视角 3D AVLN。
    因此这里实现的是 CityNav 式的"地理 BEV 上的语言指代导航"：
        给一句自然语言指令 → 每步用俯视观测对目标做 grounding →
        朝目标质心步进飞行 → 直到目标进入视场中心（到达）或步数预算耗尽。

本模块只负责"决策逻辑"，不做任何 IO / socket / 模型加载：
    - parse_instruction()        : 指令 → 目标 YOLO 类别集合 + 方向先验
    - VlnNavigator.step()        : 观测 + 当前位姿 → 一个低层动作决策
观测与低层飞行由 app.py 的 run_vln_episode 注入（复用 perceive_at /
execute_action），保证与现有感知 / 飞行 / socket 链路解耦。

grounding（PoC 版）：
    复用 perception.py 的 YOLO 检测（已在 RescueNet 域内微调，比 LocateAnything
    零样本更稳）。把指令目标词映射到 YOLO 类别，取匹配框质心相对无人机的方位。
    LocateAnything 开放词汇 grounding 在独立 conda 环境（cu128），本模块只预留
    `locate_ground_fn` 接口，二期再以子进程 / HTTP worker 接入。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── 目标词 → YOLO 类别词典 ──────────────────────────────────────────────
# 值必须与 perception.YOLO_LABEL_MAP 的中文标签一致。
_BUILDING_ALL = ["无损伤建筑", "轻微损伤建筑", "严重损伤建筑", "完全损毁建筑"]
_BUILDING_DAMAGED = ["轻微损伤建筑", "严重损伤建筑", "完全损毁建筑"]

# 建筑类目标按"具体损伤等级 → 泛化受损 → 任意建筑"分层解析，命中越具体优先级越高，
# 避免 "完全损毁的建筑" 被泛化词 "损毁"/"建筑" 污染成全部受损等级。
_SEVERITY_LEXICON: list[tuple[list[str], str]] = [
    (["完全损毁", "完全倒塌", "夷平", "destroyed", "collapsed", "flattened", "leveled"], "完全损毁建筑"),
    (["严重损伤", "严重受损", "重度", "major", "severe"], "严重损伤建筑"),
    (["轻微损伤", "轻微受损", "轻度", "minor", "slight"], "轻微损伤建筑"),
    (["完好", "无损", "未受损", "intact", "undamaged", "no damage"], "无损伤建筑"),
]
_DAMAGED_GENERIC_KW = ["受损", "损毁", "损坏", "损伤", "倒塌", "废墟", "damaged", "damage", "ruined", "rubble"]
_BUILDING_GENERIC_KW = ["建筑", "房屋", "楼房", "楼", "房子", "building", "buildings", "house", "houses", "structure"]
# 与建筑无关、可独立叠加的目标类目。
_INDEPENDENT_LEXICON: list[tuple[list[str], str]] = [
    (["车辆", "汽车", "卡车", "车", "vehicle", "vehicles", "car", "cars", "truck"], "车辆"),
    (["积水", "洪水", "淹没", "内涝", "水池", "水域", "水体", "flood", "flooded", "water", "inundation"], "水池/积水区域"),
]

# 方向词 → (north 单位, east 单位)
_DIRECTION_LEXICON: list[tuple[list[str], tuple[float, float]]] = [
    (["东北", "northeast", "north-east", "ne "], (1.0, 1.0)),
    (["西北", "northwest", "north-west", "nw "], (1.0, -1.0)),
    (["东南", "southeast", "south-east", "se "], (-1.0, 1.0)),
    (["西南", "southwest", "south-west", "sw "], (-1.0, -1.0)),
    (["北", "north", "northern", "northward"], (1.0, 0.0)),
    (["南", "south", "southern", "southward"], (-1.0, 0.0)),
    (["东", "east", "eastern", "eastward"], (0.0, 1.0)),
    (["西", "west", "western", "westward"], (0.0, -1.0)),
]

_DIRECTION_NAME = {
    (1.0, 0.0): "北", (-1.0, 0.0): "南", (0.0, 1.0): "东", (0.0, -1.0): "西",
    (1.0, 1.0): "东北", (1.0, -1.0): "西北", (-1.0, 1.0): "东南", (-1.0, -1.0): "西南",
}


@dataclass
class VlnConfig:
    step_budget: int = 12          # 最多决策步数
    arrival_radius_m: float = 35.0  # 目标质心进入此半径内视为到达
    max_step_m: float = 80.0       # 单步朝目标移动的最大距离（防越界/跳瓦片）
    explore_step_m: float = 90.0   # 未发现目标时的搜索步长
    min_box_area_frac: float = 0.0008  # 过滤过小的噪声框（占 patch 面积比）
    use_llm_stop: bool = False     # 到达候选时是否让 LLM 复核（控时延，默认关）


@dataclass
class Observation:
    """从 perception.PerceptionResult 抽取的、导航需要的最小子集。"""
    detections: list[dict]      # 每项含 class_name / bbox(x1,y1,x2,y2 patch像素) / conf
    patch_width: int
    patch_height: int
    patch_radius_m: float
    risk_level: str = "none"
    scene_text: str = ""
    degraded: bool = False
    patch_path: str = ""        # 当前俯视 patch 的本地路径（供 VLM grounder 读取）

    @classmethod
    def from_perception(cls, result) -> "Observation":
        det = (getattr(result, "detection", None) or {}).get("detections", []) or []
        return cls(
            detections=det,
            patch_width=int(getattr(result, "patch_width", 0) or 0),
            patch_height=int(getattr(result, "patch_height", 0) or 0),
            patch_radius_m=float(getattr(result, "patch_radius_m", 0.0) or 0.0),
            risk_level=str(getattr(result, "risk_level", "none") or "none"),
            scene_text=str(getattr(result, "scene_text", "") or ""),
            degraded=bool(getattr(result, "degraded", False)),
            patch_path=str(getattr(result, "patch_path", "") or ""),
        )


@dataclass
class GroundHit:
    """一次 grounding 的结果：目标是否在视场内、归一化中心、是否到达。

    norm_xy 为目标中心的归一化坐标，左上 (0,0)、右下 (1,1)；YOLO 与 VLM 两种
    grounder 都统一输出该格式，几何换算（→ 米偏移）在导航器内共享。
    """
    present: bool
    norm_xy: Optional[tuple[float, float]] = None
    arrived: bool = False
    label: str = ""
    conf: float = 0.0
    reason: str = ""
    source: str = ""            # "yolo" / "vlm" / "hybrid:*"


@dataclass
class Decision:
    action: str                 # fly_relative / hover / stop
    params: dict
    reason: str
    thought: str
    arrived: bool = False
    matched: bool = False       # 本步是否命中目标
    target_offset_m: Optional[tuple[float, float]] = None  # (north, east)
    target_dist_m: Optional[float] = None


def parse_instruction(instruction: str) -> dict:
    """指令 → {target_classes, target_label, direction, direction_name}。"""
    text = (instruction or "").strip()
    low = text.lower()

    target_classes: list[str] = []
    matched_labels: list[str] = []

    def _add(cls: str) -> None:
        if cls not in target_classes:
            target_classes.append(cls)

    # 1) 建筑类：具体损伤等级优先，其次泛化受损，再次任意建筑（三者择一层）。
    severity_hits: list[str] = []
    for keywords, cls in _SEVERITY_LEXICON:
        for kw in keywords:
            if kw.lower() in low:
                severity_hits.append(cls)
                matched_labels.append(kw)
                break
    if severity_hits:
        for cls in severity_hits:
            _add(cls)
    elif any(kw.lower() in low for kw in _DAMAGED_GENERIC_KW):
        for cls in _BUILDING_DAMAGED:
            _add(cls)
        matched_labels.append("受损建筑")
    elif any(kw.lower() in low for kw in _BUILDING_GENERIC_KW):
        for cls in _BUILDING_ALL:
            _add(cls)
        matched_labels.append("建筑")

    # 2) 独立类目（车辆 / 积水），与建筑可叠加。
    for keywords, cls in _INDEPENDENT_LEXICON:
        for kw in keywords:
            if kw.lower() in low:
                _add(cls)
                matched_labels.append(kw)
                break

    direction: Optional[tuple[float, float]] = None
    for keywords, vec in _DIRECTION_LEXICON:
        if any(kw.strip().lower() in low for kw in keywords):
            direction = vec
            break

    target_label = "、".join(target_classes) if target_classes else (matched_labels[0] if matched_labels else "")
    return {
        "raw": text,
        "target_classes": target_classes,
        "target_label": target_label or "(未识别明确目标)",
        "direction": direction,
        "direction_name": _DIRECTION_NAME.get(direction or (0.0, 0.0), ""),
    }


def ground_with_yolo(
    observation: "Observation",
    target_classes: list[str],
    min_box_area_frac: float = 0.0008,
) -> Optional[GroundHit]:
    """基于 YOLO 检测框做 grounding：选出匹配目标类别中最显著的框，输出归一化中心。

    退化视场（裁到整图 / 贴边 clamp）时几何不可信，返回 None（交由调用方探索）。
    """
    obs = observation
    if obs.degraded or obs.patch_width <= 0 or obs.patch_height <= 0:
        return None
    if not target_classes:
        return None

    pw, ph = obs.patch_width, obs.patch_height
    patch_area = float(pw * ph) or 1.0

    best = None
    best_score = -1.0
    for det in obs.detections:
        cls = det.get("class_name")
        if cls not in target_classes:
            continue
        bbox = det.get("bbox") or det.get("bbox_xyxy")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area / patch_area < min_box_area_frac:
            continue
        cx_px = (x1 + x2) * 0.5
        cy_px = (y1 + y2) * 0.5
        conf = float(det.get("conf", 0.0))
        dist_center = math.hypot(cx_px / pw - 0.5, cy_px / ph - 0.5)
        score = area / patch_area + 0.3 * conf - 0.2 * dist_center
        if score > best_score:
            best_score = score
            best = GroundHit(
                present=True,
                norm_xy=(cx_px / pw, cy_px / ph),
                label=cls,
                conf=conf,
                reason=f"YOLO 命中 {cls} (conf {conf:.2f}, 面积 {area / patch_area:.1%})",
                source="yolo",
            )
    return best


class VlnNavigator:
    """单条指令的导航闭环状态机（每条 episode 用一个实例）。"""

    def __init__(
        self,
        config: Optional[VlnConfig] = None,
        grounder: Optional[Callable[[dict, "Observation"], Optional[GroundHit]]] = None,
        llm_stop_fn: Optional[Callable[[str, str, dict], Optional[bool]]] = None,
        locate_ground_fn: Optional[Callable[[str, str], Optional[list]]] = None,
    ):
        self.config = config or VlnConfig()
        # grounder(parsed, observation) -> GroundHit|None
        #   注入式 grounding 后端：None 时默认用进程内 YOLO（ground_with_yolo）。
        #   app 层据此切换 yolo / vlm / hybrid（见 run_vln_episode）。
        self._grounder = grounder
        # llm_stop_fn(instruction, scene_text, candidate) -> True/False/None
        #   返回 None 表示"无意见"，回退到启发式。
        self._llm_stop_fn = llm_stop_fn
        # locate_ground_fn(image_path, phrase) -> [boxes] —— 二期 LocateAnything 接入点。
        self._locate_ground_fn = locate_ground_fn

        self.parsed: dict = {}
        self.step_index = 0
        self._explore_moves: list[tuple[float, float]] = []
        self._explore_cursor = 0
        self.history: list[dict] = []

    # ── episode 生命周期 ────────────────────────────────────────────
    def reset(self, instruction: str) -> dict:
        self.parsed = parse_instruction(instruction)
        self.step_index = 0
        self._explore_cursor = 0
        self.history = []
        self._explore_moves = self._build_explore_moves(
            self.parsed.get("direction"), self.config.explore_step_m
        )
        return self.parsed

    @staticmethod
    def _build_explore_moves(direction, step_m: float) -> list[tuple[float, float]]:
        """构造搜索移动序列：有方向先验则先沿该方向走几步，再转方形螺旋。"""
        moves: list[tuple[float, float]] = []
        if direction is not None:
            dn, de = direction
            norm = math.hypot(dn, de) or 1.0
            for _ in range(3):
                moves.append((dn / norm * step_m, de / norm * step_m))
        # 方形螺旋：N, E, S, W，腿长 1,1,2,2,3,3...
        dirs = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
        leg, turn = 1, 0
        while len(moves) < 48:
            dn, de = dirs[turn % 4]
            for _ in range(leg):
                moves.append((dn * step_m, de * step_m))
            turn += 1
            if turn % 2 == 0:
                leg += 1
        return moves

    # ── 单步决策 ────────────────────────────────────────────────────
    def step(self, observation: Observation, snapshot: dict) -> Decision:
        self.step_index += 1
        target_classes = self.parsed.get("target_classes") or []

        # 1) 目标 grounding：注入式后端（VLM/hybrid）优先，否则进程内 YOLO。
        hit: Optional[GroundHit] = None
        if self._grounder is not None:
            try:
                hit = self._grounder(self.parsed, observation)
            except Exception:
                hit = None
        else:
            hit = ground_with_yolo(observation, target_classes, self.config.min_box_area_frac)

        usable = (
            hit is not None
            and hit.present
            and hit.norm_xy is not None
            and not observation.degraded
            and observation.patch_radius_m > 0
        )
        if usable:
            north_m, east_m, dist_m = self._offset_from_norm(hit.norm_xy, observation.patch_radius_m)
            label = hit.label or self.parsed.get("target_label")
            src = f"[{hit.source}]" if hit.source else ""
            # 1a) 到达判定：grounder 直接判到达，或目标已进入视场中心半径。
            if hit.arrived or dist_m <= self.config.arrival_radius_m:
                cand = {"class_name": label, "conf": hit.conf, "offset": (north_m, east_m, dist_m)}
                if self._confirm_stop(observation, cand):
                    dec = Decision(
                        action="stop",
                        params={},
                        reason=f"{src}目标「{label}」已位于视场中心（偏移 {dist_m:.0f}m），判定到达。{hit.reason}",
                        thought=f"step{self.step_index}: {src}命中 {label}，偏移 N{north_m:+.0f}/E{east_m:+.0f}m，到达。",
                        arrived=True,
                        matched=True,
                        target_offset_m=(north_m, east_m),
                        target_dist_m=dist_m,
                    )
                    self._log(dec)
                    return dec
            # 1b) 未到达：朝目标步进（限幅，避免越界/跳瓦片）。
            mn, me = self._clamp_step(north_m, east_m, self.config.max_step_m)
            dec = Decision(
                action="fly_relative",
                params={"north_m": round(mn, 1), "east_m": round(me, 1), "up_m": 0.0, "speed": 12.0},
                reason=f"{src}已 grounding 到「{label}」（约 {dist_m:.0f}m，方位 N{north_m:+.0f}/E{east_m:+.0f}m），朝其前进。{hit.reason}",
                thought=f"step{self.step_index}: {src}命中 {label}，朝目标步进 N{mn:+.0f}/E{me:+.0f}m。",
                matched=True,
                target_offset_m=(north_m, east_m),
                target_dist_m=dist_m,
            )
            self._log(dec)
            return dec

        # 2) 未命中目标 → 按搜索序列探索
        mn, me = self._next_explore_move()
        bits = []
        if self.parsed.get("direction_name"):
            bits.append(f"方向先验 {self.parsed['direction_name']}")
        if observation.risk_level not in ("none", ""):
            bits.append(f"当前视场风险 {observation.risk_level}")
        if hit is not None and hit.reason:
            bits.append(hit.reason)
        extra = ("；" + "，".join(bits)) if bits else ""
        dec = Decision(
            action="fly_relative",
            params={"north_m": round(mn, 1), "east_m": round(me, 1), "up_m": 0.0, "speed": 12.0},
            reason=f"当前视场未发现「{self.parsed.get('target_label')}」，按搜索模式移动 N{mn:+.0f}/E{me:+.0f}m{extra}。",
            thought=f"step{self.step_index}: 未命中目标，探索步 N{mn:+.0f}/E{me:+.0f}m。",
            matched=False,
        )
        self._log(dec)
        return dec

    def budget_exhausted(self) -> bool:
        return self.step_index >= self.config.step_budget

    # ── 内部工具 ────────────────────────────────────────────────────
    @staticmethod
    def _offset_from_norm(norm_xy: tuple[float, float], radius_m: float) -> tuple[float, float, float]:
        """归一化中心 (x,y) → 相对无人机的 (north_m, east_m, dist_m)。

        俯视 patch 半边 ≈ radius_m，故全宽对应 2*radius_m 米；图像 y 向下为南。
        """
        nx, ny = norm_xy
        east_m = (nx - 0.5) * 2.0 * radius_m
        north_m = -(ny - 0.5) * 2.0 * radius_m
        return north_m, east_m, math.hypot(north_m, east_m)

    @staticmethod
    def _clamp_step(north_m: float, east_m: float, max_step_m: float) -> tuple[float, float]:
        dist = math.hypot(north_m, east_m)
        if dist <= max_step_m or dist == 0.0:
            return north_m, east_m
        f = max_step_m / dist
        return north_m * f, east_m * f

    def _next_explore_move(self) -> tuple[float, float]:
        if not self._explore_moves:
            return (self.config.explore_step_m, 0.0)
        move = self._explore_moves[self._explore_cursor % len(self._explore_moves)]
        self._explore_cursor += 1
        return move

    def _confirm_stop(self, obs: Observation, candidate: dict) -> bool:
        """到达候选时的停止确认。默认启发式直接停；可选 LLM 复核。"""
        if not self.config.use_llm_stop or self._llm_stop_fn is None:
            return True
        try:
            verdict = self._llm_stop_fn(self.parsed.get("raw", ""), obs.scene_text, candidate)
        except Exception:
            verdict = None
        return True if verdict is None else bool(verdict)

    def _log(self, dec: Decision) -> None:
        self.history.append({
            "step": self.step_index,
            "action": dec.action,
            "matched": dec.matched,
            "arrived": dec.arrived,
            "dist_m": dec.target_dist_m,
            "thought": dec.thought,
        })

    # ── 收尾摘要 ────────────────────────────────────────────────────
    def summarize(self, arrived: bool, snapshot: dict) -> str:
        target = self.parsed.get("target_label", "目标")
        steps = self.step_index
        pos = f"({snapshot.get('lat', 0):.6f}, {snapshot.get('lon', 0):.6f}) @ {snapshot.get('alt', 0):.1f}m"
        if arrived:
            return f"VLN 完成：已导航至「{target}」，共 {steps} 步，UAV 现位于 {pos}。"
        return f"VLN 结束：在 {steps} 步预算内未稳定到达「{target}」，UAV 现位于 {pos}。"
