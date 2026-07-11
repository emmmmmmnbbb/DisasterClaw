"""
backend/hspm_planner.py — HSPM 分层语义规划（P1）

借鉴 CityNavAgent (ACL 2025) 的 Hierarchical Semantic Planning Module，落到
DisasterClaw 的 2D 地理 BEV 设置：

    landmark-level : LLM 把自由指令拆成有序"地标/子目标"序列。
    object-level   : 看不到当前子目标时，喂 STMR 文字矩阵让 LLM 做常识推理，
                     选"下一步朝哪个方位走最可能接近子目标"（OROI 思想）。
    motion-level   : 把"看得到就朝目标质心步进 / 看不到就朝 OROI 方位探索"
                     转成一个 fly_relative 低层动作。

与 vln_navigator.VlnNavigator 接口对齐（reset/step/budget_exhausted/summarize/
history），便于在 run_vln_episode 里用 VLN_PLANNER 开关替换；grounding / 几何
换算 / 防越界全部复用既有实现，避免重复。

所有 LLM 调用通过注入的 `llm_chat` 与 `stmr_provider`，不直接依赖 app/state，
便于单测用 mock 注入。
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from vln_navigator import (
    Decision,
    GroundHit,
    Observation,
    VlnConfig,
    VlnNavigator,
    ground_with_yolo,
    parse_instruction,
)

logger = logging.getLogger(__name__)

# 八方位 → (north 单位, east 单位)
_BEARING_VEC: dict[str, tuple[float, float]] = {
    "北": (1.0, 0.0), "东北": (1.0, 1.0), "东": (0.0, 1.0), "东南": (-1.0, 1.0),
    "南": (-1.0, 0.0), "西南": (-1.0, -1.0), "西": (0.0, -1.0), "西北": (1.0, -1.0),
}

# 类型别名
LlmChat = Callable[[list[dict], float, Optional[int]], str]
StmrProvider = Callable[[dict], Optional[dict]]  # (snapshot) -> {text,...} | None
Grounder = Callable[[dict, Observation], Optional[GroundHit]]


def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def plan_landmarks(instruction: str, llm_chat: Optional[LlmChat]) -> list[str]:
    """landmark-level：把指令拆成有序地标/子目标短语序列。

    LLM 不可用 / 解析失败时，回退到 parse_instruction 的开放词汇短语（单地标）。
    """
    fallback = parse_instruction(instruction).get("target_phrase") or instruction
    if llm_chat is None:
        return [fallback]
    try:
        content = llm_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你在为无人机做视觉语言导航的高层规划。从给定的中文导航指令里，"
                        "按出现顺序抽取一个【地标/目标短语序列】，每个元素是无人机要依次"
                        "经过或最终到达的可见目标（保留颜色/材质/受损程度等修饰词）。"
                        "只输出严格 JSON：{\"landmarks\": [\"...\"], \"thought\": \"...\"}。"
                        "若指令只有一个目标，landmarks 就只含一个元素。"
                    ),
                },
                {"role": "user", "content": f"导航指令：{instruction}"},
            ],
            0.2,
            256,
        )
        data = _extract_json(content) or {}
        lms = [str(x).strip() for x in (data.get("landmarks") or []) if str(x).strip()]
        return lms or [fallback]
    except Exception as exc:
        logger.warning("HSPM landmark planning failed: %s", exc)
        return [fallback]


@dataclass
class OroiScoreWeights:
    """C3（HSPM 运动层工程改进，非 headline）：OROI 打分融合的三路信号权重。"""
    llm: float = 0.5
    prior: float = 0.2
    frontier: float = 0.3


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _prior_score(bearing: str, direction_hint: str) -> float:
    """方向先验一致度 ∈[0,1]：与 direction_hint 的夹角余弦映射到 [0,1]。

    无先验 / 非法方位时返回 0.5（中性，不加分也不减分）。
    """
    if not direction_hint or direction_hint not in _BEARING_VEC or bearing not in _BEARING_VEC:
        return 0.5
    bn, be = _BEARING_VEC[bearing]
    hn, he = _BEARING_VEC[direction_hint]
    b_norm = math.hypot(bn, be) or 1.0
    h_norm = math.hypot(hn, he) or 1.0
    cos = (bn * hn + be * he) / (b_norm * h_norm)
    return _clamp01((cos + 1.0) / 2.0)


def score_bearings_llm(
    instruction: str,
    subgoal: str,
    stmr_text: str,
    llm_chat: Optional[LlmChat],
) -> tuple[Optional[dict[str, float]], str]:
    """让 LLM 给 8 个方位分别打 affordance 分（Say-REAPEx / Say-Score 式打分），

    而不是像 reason_oroi 那样"自由选一个"——即使输出退化，也能和方向先验/frontier
    信号融合，不会整段推理都押在一次离散选择上。

    返回 (scores|None, reason)。scores 缺失/非法时返回 None，由调用方回退权重分配。
    """
    if llm_chat is None or not stmr_text:
        return None, "（无 LLM/地图，跳过 LLM 打分）"
    try:
        content = llm_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是一架无人机，正按子目标在一张俯视语义地图上导航。"
                        "地图是以你为中心的文字网格（行上=北 下=南，列左=西 右=东）。"
                        "当前视场里看不到子目标。请给下面 8 个方位各打一个 0~1 的 "
                        "affordance 分（越可能接近子目标分越高，例如要找受损建筑，"
                        "就给已发现的受损/建筑密集方位打更高分；无线索的方位给中性分）。"
                        "只输出严格 JSON：{\"scores\": {\"北\":0~1,\"东北\":0~1,\"东\":0~1,"
                        "\"东南\":0~1,\"南\":0~1,\"西南\":0~1,\"西\":0~1,\"西北\":0~1}, "
                        "\"reason\": \"...\"}。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"总指令：{instruction}\n当前子目标：{subgoal}\n\n{stmr_text}",
                },
            ],
            0.2,
            260,
        )
        data = _extract_json(content) or {}
        raw_scores = data.get("scores") or {}
        scores: dict[str, float] = {}
        for b in _BEARING_VEC:
            try:
                scores[b] = _clamp01(float(raw_scores.get(b, 0.0)))
            except (TypeError, ValueError):
                scores[b] = 0.0
        if not any(scores.values()):
            return None, "LLM 打分全 0/非法，回退"
        reason = str(data.get("reason", ""))[:80]
        return scores, (reason or "LLM OROI 打分")
    except Exception as exc:
        logger.warning("HSPM OROI 打分失败: %s", exc)
        return None, "OROI 打分异常，回退"


def score_oroi(
    instruction: str,
    subgoal: str,
    stmr_text: str,
    direction_hint: str,
    llm_chat: Optional[LlmChat],
    frontier_fn: Optional[Callable[[str], float]] = None,
    weights: Optional[OroiScoreWeights] = None,
) -> tuple[str, str]:
    """object-level（C3 工程改进，`VLN_OROI_SCORE=1` 开）：LLM affordance + 方向
    先验一致度 + 未探索区域增益 三路信号加权打分，取 argmax 方位。

    比 reason_oroi 的"LLM 自由选一个"更稳：即使 LLM 输出退化/失败（None），
    先验和 frontier 仍能给出有信息量的打分，不会死板回退到"北"。
    """
    weights = weights or OroiScoreWeights()
    llm_scores, llm_reason = score_bearings_llm(instruction, subgoal, stmr_text, llm_chat)

    best_bearing: Optional[str] = None
    best_score = -1.0
    for bearing in _BEARING_VEC:
        s_llm = llm_scores.get(bearing, 0.0) if llm_scores else 0.5  # 无 LLM 信号 → 中性
        s_prior = _prior_score(bearing, direction_hint)
        s_frontier = frontier_fn(bearing) if frontier_fn is not None else 0.5
        total = weights.llm * s_llm + weights.prior * s_prior + weights.frontier * s_frontier
        if total > best_score:
            best_score = total
            best_bearing = bearing

    best_bearing = best_bearing or (direction_hint if direction_hint in _BEARING_VEC else "北")
    reason = (
        f"打分融合选 {best_bearing}(score={best_score:.2f})："
        f"LLM{'✓' if llm_scores else '✗'}={llm_reason}；先验={direction_hint or '无'}"
    )
    return best_bearing, reason[:120]


def reason_oroi(
    instruction: str,
    subgoal: str,
    stmr_text: str,
    direction_hint: str,
    llm_chat: Optional[LlmChat],
) -> tuple[str, str]:
    """object-level：看不到子目标时，让 LLM 在 STMR 文字地图上推理下一步方位。

    返回 (bearing, reason)。LLM 不可用 / 非法输出时回退到方向先验或"北"。
    """
    default_bearing = direction_hint if direction_hint in _BEARING_VEC else "北"
    if llm_chat is None or not stmr_text:
        return default_bearing, "（无 LLM/地图，按方向先验探索）"
    try:
        content = llm_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是一架无人机，正按子目标在一张俯视语义地图上导航。"
                        "地图是以你为中心的文字网格（行上=北 下=南，列左=西 右=东）。"
                        "当前视场里看不到子目标。请用常识推理：朝哪个方位飞，最可能"
                        "接近子目标（例如要找受损建筑，就朝已发现的受损/建筑密集方位走；"
                        "无线索则朝未探索方向扩展）。"
                        "只输出严格 JSON：{\"bearing\": \"北/东北/东/东南/南/西南/西/西北\", \"reason\": \"...\"}。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"总指令：{instruction}\n当前子目标：{subgoal}\n"
                        f"方向先验：{direction_hint or '无'}\n\n{stmr_text}"
                    ),
                },
            ],
            0.2,
            200,
        )
        data = _extract_json(content) or {}
        bearing = str(data.get("bearing", "")).strip()
        reason = str(data.get("reason", ""))[:80]
        if bearing in _BEARING_VEC:
            return bearing, reason or "LLM OROI 推理"
        return default_bearing, f"LLM 方位非法({bearing})，回退 {default_bearing}"
    except Exception as exc:
        logger.warning("HSPM OROI reasoning failed: %s", exc)
        return default_bearing, f"OROI 失败回退 {default_bearing}"


@dataclass
class HspmConfig(VlnConfig):
    """复用 VlnConfig 的预算/半径/步长等字段。"""
    # C3（非 headline）：OROI 打分融合开关，对应文档 P4.5「B1 + OROI-Score」消融行。
    use_oroi_score: bool = False
    oroi_weights: OroiScoreWeights = field(default_factory=OroiScoreWeights)


class HspmNavigator:
    """CityNavAgent 式分层规划的导航状态机（每条 episode 一个实例）。"""

    def __init__(
        self,
        config: Optional[HspmConfig] = None,
        grounder: Optional[Grounder] = None,
        llm_chat: Optional[LlmChat] = None,
        stmr_provider: Optional[StmrProvider] = None,
        semantic_map_provider: Optional[Callable[[], Any]] = None,
    ):
        self.config = config or HspmConfig()
        self._grounder = grounder
        self._llm_chat = llm_chat
        self._stmr_provider = stmr_provider
        # C3（非 headline）：OROI 打分融合里 frontier 项要用；不给就退化成中性分 0.5。
        self._semantic_map_provider = semantic_map_provider

        self.instruction = ""
        self.parsed: dict = {}
        self.landmarks: list[str] = []
        self.lm_idx = 0
        self.step_index = 0
        self.history: list[dict] = []

    # ── 生命周期 ────────────────────────────────────────────────────
    def reset(self, instruction: str) -> dict:
        self.instruction = instruction
        self.parsed = parse_instruction(instruction)
        self.landmarks = plan_landmarks(instruction, self._llm_chat)
        self.lm_idx = 0
        self.step_index = 0
        self.history = []
        # 暴露给上层日志：把 landmark 序列塞进 parsed
        self.parsed["landmarks"] = self.landmarks
        return self.parsed

    @property
    def current_subgoal(self) -> str:
        if 0 <= self.lm_idx < len(self.landmarks):
            return self.landmarks[self.lm_idx]
        return self.parsed.get("target_phrase") or self.instruction

    def budget_exhausted(self) -> bool:
        return self.step_index >= self.config.step_budget

    # ── 单步决策 ────────────────────────────────────────────────────
    def step(self, observation: Observation, snapshot: dict) -> Decision:
        self.step_index += 1
        subgoal = self.current_subgoal

        # 1) grounding 当前子目标
        plike = {
            "raw": self.instruction,
            "target_phrase": subgoal,
            "target_label": subgoal,
            "target_classes": self.parsed.get("target_classes") or [],
        }
        hit: Optional[GroundHit] = None
        if self._grounder is not None:
            try:
                hit = self._grounder(plike, observation)
            except Exception:
                hit = None
        else:
            hit = ground_with_yolo(
                observation, self.parsed.get("target_classes") or [],
                self.config.min_box_area_frac,
            )

        usable = (
            hit is not None and hit.present and hit.norm_xy is not None
            and not observation.degraded and observation.patch_radius_m > 0
        )

        if usable:
            north_m, east_m, dist_m = self._offset(hit.norm_xy, observation.patch_radius_m)
            # 1a) 到达当前子目标 → 推进到下一个；全部完成 → stop
            if dist_m <= self.config.arrival_radius_m:
                self.lm_idx += 1
                done = self.lm_idx >= len(self.landmarks)
                if done:
                    dec = Decision(
                        action="stop", params={},
                        reason=f"[HSPM] 子目标「{subgoal}」已到达（{dist_m:.0f}m），全部地标完成。",
                        thought=f"step{self.step_index}: 到达「{subgoal}」，地标序列完成。",
                        arrived=True, matched=True,
                        target_offset_m=(north_m, east_m), target_dist_m=dist_m,
                    )
                    self._log(dec, subgoal)
                    return dec
                nxt = self.current_subgoal
                dec = Decision(
                    action="hover", params={"duration": 0.5},
                    reason=f"[HSPM] 子目标「{subgoal}」到达，推进到下一个「{nxt}」。",
                    thought=f"step{self.step_index}: 「{subgoal}」达成 → 下一个「{nxt}」。",
                    matched=True, target_offset_m=(north_m, east_m), target_dist_m=dist_m,
                )
                self._log(dec, subgoal)
                return dec
            # 1b) 看得到但未到 → 朝质心步进（motion-level）
            mn, me = VlnNavigator._clamp_step(north_m, east_m, self.config.max_step_m)
            dec = Decision(
                action="fly_relative",
                params={"north_m": round(mn, 1), "east_m": round(me, 1), "up_m": 0.0, "speed": 12.0},
                reason=f"[HSPM] 已 grounding「{subgoal}」(~{dist_m:.0f}m)，朝其前进。{hit.reason}",
                thought=f"step{self.step_index}: 命中「{subgoal}」，朝目标步进 N{mn:+.0f}/E{me:+.0f}m。",
                matched=True, target_offset_m=(north_m, east_m), target_dist_m=dist_m,
            )
            self._log(dec, subgoal)
            return dec

        # 2) 看不到子目标 → object-level OROI 推理方位 → motion 探索一步
        stmr = None
        if self._stmr_provider is not None:
            try:
                stmr = self._stmr_provider(snapshot)
            except Exception:
                stmr = None
        stmr_text = (stmr or {}).get("text", "") if stmr else ""
        direction_hint = self.parsed.get("direction_name", "")
        if self.config.use_oroi_score:
            frontier_fn = self._make_frontier_fn(snapshot)
            bearing, reason = score_oroi(
                self.instruction, subgoal, stmr_text, direction_hint, self._llm_chat,
                frontier_fn=frontier_fn, weights=self.config.oroi_weights,
            )
        else:
            bearing, reason = reason_oroi(
                self.instruction, subgoal, stmr_text, direction_hint, self._llm_chat,
            )
        dn, de = _BEARING_VEC.get(bearing, (1.0, 0.0))
        norm = math.hypot(dn, de) or 1.0
        step_m = self.config.explore_step_m
        mn, me = dn / norm * step_m, de / norm * step_m
        dec = Decision(
            action="fly_relative",
            params={"north_m": round(mn, 1), "east_m": round(me, 1), "up_m": 0.0, "speed": 12.0},
            reason=f"[HSPM] 未见「{subgoal}」，OROI 推理朝 {bearing} 探索。{reason}",
            thought=f"step{self.step_index}: 未见「{subgoal}」，朝 {bearing} 探索 N{mn:+.0f}/E{me:+.0f}m。",
            matched=False,
        )
        self._log(dec, subgoal)
        return dec

    # ── 工具 ────────────────────────────────────────────────────────
    @staticmethod
    def _offset(norm_xy: tuple[float, float], radius_m: float) -> tuple[float, float, float]:
        # VlnNavigator._offset_from_norm 返回 (north_m, east_m, dist_m)
        return VlnNavigator._offset_from_norm(norm_xy, radius_m)

    def _make_frontier_fn(self, snapshot: dict) -> Optional[Callable[[str], float]]:
        """C3：把 SemanticMap.frontier_score 包成 score_oroi 要的 (bearing)->float。

        拿不到语义地图（未开 P0 建图 / provider 为空）时返回 None，
        score_oroi 会自动把 frontier 项当中性分 0.5 处理。
        """
        if self._semantic_map_provider is None:
            return None
        smap = self._semantic_map_provider()
        if smap is None:
            return None
        try:
            lat, lon = float(snapshot["lat"]), float(snapshot["lon"])
        except (KeyError, TypeError, ValueError):
            return None

        def _fn(bearing: str) -> float:
            vec = _BEARING_VEC.get(bearing, (1.0, 0.0))
            try:
                return smap.frontier_score(lat, lon, vec)
            except Exception:
                return 0.5

        return _fn

    def _log(self, dec: Decision, subgoal: str) -> None:
        self.history.append({
            "step": self.step_index,
            "subgoal": subgoal,
            "lm_idx": self.lm_idx,
            "action": dec.action,
            "matched": dec.matched,
            "arrived": dec.arrived,
            "dist_m": dec.target_dist_m,
            "thought": dec.thought,
        })

    def summarize(self, arrived: bool, snapshot: dict) -> str:
        lms = " → ".join(self.landmarks) if self.landmarks else "目标"
        pos = f"({snapshot.get('lat', 0):.6f}, {snapshot.get('lon', 0):.6f}) @ {snapshot.get('alt', 0):.1f}m"
        if arrived:
            return f"HSPM 完成：地标序列「{lms}」已达成，共 {self.step_index} 步，UAV 现位于 {pos}。"
        prog = f"{self.lm_idx}/{len(self.landmarks)}"
        return f"HSPM 结束：{self.step_index} 步内完成地标 {prog}「{lms}」，UAV 现位于 {pos}。"
