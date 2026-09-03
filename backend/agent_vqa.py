r"""backend/agent_vqa.py — Agent-VQA 结构化问答控制器 (D3).

对应 AGENT_VQA_REVISION_PLAN.md 第 5、7 节。智能体接收操作员自然语言问题，在
地理配准俯视环境中搜索目标、收集证据、回答灾情问题，并在证据不足时以有限
预算执行继续搜索 / 居中下降重观测，最终输出结构化决策轨迹。

设计要点 (计划 5.2 / 7.1):
  - 统一结构化输出 schema: answer / confidence / abstain / evidence / decision /
    reason_code。decision 来自封闭集合 {answer, continue_search, reobserve, abstain}。
  - reason_code 是结构化动作理由, 不是思维链; 对外字段不得命名为 chain_of_thought。
  - VLM 输出解析失败时返回显式 invalid_output, 不得静默猜测。
  - evidence.norm_xy 只有在确实定位到目标时才填写。
  - 控制器依赖可注入 (vlm_answer_fn / perceive_fn / search_fn / reobserve_fn),
    使无 torch / 无模型环境也能用桩函数跑单元测试 (计划 7.8)。

信息边界 (计划 7.3 必须避免):
  - 不得从测试条目的 answer 或未来图像读取在线决策信息;
  - 不得把前端可见的 GT 建筑足迹传入智能体观测;
  - 不得通过 item 参数向非 oracle 配置泄漏目标坐标。

本模块不直接做 IO / 模型调用; 几何换算复用 semantic_map.offset_from_norm。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ── schema 常量 (封闭集合) ───────────────────────────────────────────────────

QUESTION_TYPES = ("presence", "damage", "count", "spatial")
DECISIONS = ("answer", "continue_search", "reobserve", "abstain")
REASON_CODES = (
    "sufficient_evidence", "target_missing", "low_confidence",
    "budget_exhausted", "out_of_coverage", "invalid_output",
    "planner_unavailable", "vlm_unavailable", "execution_error",
    "cancelled", "invalid_question",
)
EVIDENCE_SOURCES = ("image", "detector", "change_classifier", "semantic_map", "history")

DAMAGE_CHOICES = ["无损伤", "轻微损伤", "严重损伤", "完全损毁"]
COUNT_CHOICES = ["0", "1", "2", "3+"]
PRESENCE_CHOICES = ["否", "是"]
BEARING_CHOICES = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]

SUBTYPE_TO_LEVEL = {
    "no-damage": "无损伤", "minor-damage": "轻微损伤",
    "major-damage": "严重损伤", "destroyed": "完全损毁",
}
CLASS_TO_SUBTYPE = {
    "无损伤建筑": "no-damage", "轻微损伤建筑": "minor-damage",
    "严重损伤建筑": "major-damage", "完全损毁建筑": "destroyed",
}
DAMAGED_SUBTYPES = ("minor-damage", "major-damage", "destroyed")
SEVERE_SUBTYPES = ("major-damage", "destroyed")


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


# ── 问题解析 (规则式, 无需 LLM) ──────────────────────────────────────────────

@dataclass
class QuestionSpec:
    question_type: str  # presence | damage | count | spatial | invalid_question
    raw: str
    target_phrase: str = ""
    target_subtype: str = ""
    target_subtypes: tuple[str, ...] = ()
    ref_id: str = ""
    needs_target_location: bool = False

    def to_dict(self) -> dict:
        return {
            "question_type": self.question_type, "raw": self.raw,
            "target_phrase": self.target_phrase, "target_subtype": self.target_subtype,
            "target_subtypes": list(self.target_subtypes), "ref_id": self.ref_id,
            "needs_target_location": self.needs_target_location,
        }


_PRESENCE_RE = re.compile(r"是否存在\s*(.+?)\s*[？?]")
_DAMAGE_REF_RE = re.compile(r"标记建筑\s*([A-Za-z0-9_\-:]+)")
_DAMAGE_RE = re.compile(r"(?:标记建筑|建筑)\s*\S*?\s*[的之]?\s*损伤等级")
_COUNT_RE = re.compile(r"有多少栋\s*(.+?)\s*[？?]")
_SPATIAL_RE = re.compile(r"最近\s*的\s*(.+?)\s*位于")

_CLASS_PATTERNS = [
    ("完全损毁建筑", "destroyed"),
    ("严重损伤建筑", "major-damage"),
    ("轻微损伤建筑", "minor-damage"),
    ("无损伤建筑", "no-damage"),
]
_LEVEL_PATTERNS = [
    ("完全损毁", "destroyed"),
    ("严重损伤", "major-damage"),
    ("轻微损伤", "minor-damage"),
    ("无损伤", "no-damage"),
]


def _match_subtype(text: str) -> tuple[str, str]:
    for cn, st in _CLASS_PATTERNS:
        if cn in text:
            return cn, st
    for cn, st in _LEVEL_PATTERNS:
        if cn in text:
            return SUBTYPE_TO_LEVEL[st] + "建筑", st
    return ("", "")


def parse_question(question: str) -> QuestionSpec:
    """规则式问题类型识别 (计划 7.1)。无法识别时返回 invalid_question。"""
    q = (question or "").strip()
    if not q:
        return QuestionSpec("invalid_question", q)

    if _DAMAGE_RE.search(q):
        ref = ""
        rm = _DAMAGE_REF_RE.search(q)
        if rm:
            ref = rm.group(1)
        return QuestionSpec("damage", q, ref_id=ref, needs_target_location=True)

    m = _PRESENCE_RE.search(q)
    if m:
        phrase = m.group(1)
        cn, st = _match_subtype(phrase)
        subs = (st,) if st else ()
        if "严重" in phrase and ("完全" in phrase or "损毁" in phrase):
            subs = SEVERE_SUBTYPES
        return QuestionSpec("presence", q, target_phrase=cn or phrase,
                             target_subtype=st, target_subtypes=subs)

    m = _COUNT_RE.search(q)
    if m:
        phrase = m.group(1)
        cn, st = _match_subtype(phrase)
        subs = (st,) if st else ()
        if "严重" in phrase and ("完全" in phrase or "损毁" in phrase):
            subs = SEVERE_SUBTYPES
        return QuestionSpec("count", q, target_phrase=cn or phrase, target_subtypes=subs)

    m = _SPATIAL_RE.search(q)
    if m:
        phrase = m.group(1)
        cn, st = _match_subtype(phrase)
        return QuestionSpec("spatial", q, target_phrase=cn or phrase,
                            target_subtype=st, target_subtypes=(st,) if st else (),
                            needs_target_location=True)

    return QuestionSpec("invalid_question", q)


# ── 证据束 ──────────────────────────────────────────────────────────────────

@dataclass
class EvidenceBundle:
    """结构化感知证据 (计划 7.5)。精简、固定字段, 供问答控制器使用。"""
    observation_id: str
    source: str = "image"
    target_label: str = ""
    target_subtype: str = ""
    target_conf: float = 0.0
    norm_xy: Optional[list[float]] = None
    matching_count: int = 0
    class_probs: Optional[dict] = None
    detection_source: str = ""
    risk_level: str = ""
    scene_text: str = ""
    degraded: bool = False
    degraded_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "observation_id": self.observation_id, "source": self.source,
            "target_label": self.target_label, "target_subtype": self.target_subtype,
            "target_conf": round(self.target_conf, 4), "norm_xy": self.norm_xy,
            "matching_count": self.matching_count,
            "class_probs": self.class_probs, "detection_source": self.detection_source,
            "risk_level": self.risk_level, "scene_text": self.scene_text,
            "degraded": self.degraded, "degraded_reason": self.degraded_reason,
        }


def build_evidence_from_perception(perception_result: Any, spec: QuestionSpec,
                                    observation_id: str) -> EvidenceBundle:
    """从 PerceptionResult 提取与当前问题相关的证据 (计划 7.5)。

    不把 scene_text 中未经验证的自由文本当作事实标签; 只用检测器/分类器的
    结构化输出与目标框/图像位置。
    """
    dets = (perception_result.detection or {}).get("detections", []) if perception_result else []
    target_subtypes = spec.target_subtypes or (spec.target_subtype,)
    matching = []
    for d in dets:
        cls = d.get("class_name", "")
        sub = CLASS_TO_SUBTYPE.get(cls, "")
        if spec.question_type == "damage" and not sub:
            continue
        if target_subtypes and sub not in target_subtypes:
            continue
        matching.append(d)

    def _center_distance(det: dict) -> float:
        bbox = det.get("bbox") or det.get("bbox_xyxy")
        pw = float(getattr(perception_result, "patch_width", 0) or 0)
        ph = float(getattr(perception_result, "patch_height", 0) or 0)
        if not bbox or pw <= 0 or ph <= 0:
            return float("inf")
        cx = (float(bbox[0]) + float(bbox[2])) * 0.5 / pw
        cy = (float(bbox[1]) + float(bbox[3])) * 0.5 / ph
        return math.hypot(cx - 0.5, cy - 0.5)

    best = None
    if matching:
        # damage 的题面明确指向视场中心标记建筑；spatial 问“最近”目标。两者都应
        # 选择最接近图像中心的匹配检测，而不是选择置信度最高但可能属于另一栋的框。
        if spec.question_type in {"damage", "spatial"}:
            best = min(matching, key=lambda d: (_center_distance(d), -float(d.get("conf", 0.0))))
        else:
            best = max(matching, key=lambda d: float(d.get("conf", 0.0)))
    ev = EvidenceBundle(observation_id=observation_id)
    ev.matching_count = len(matching)
    if perception_result is not None:
        ev.risk_level = getattr(perception_result, "risk_level", "") or ""
        ev.scene_text = getattr(perception_result, "scene_text", "") or ""
        ev.degraded = bool(getattr(perception_result, "degraded", False))
        ev.degraded_reason = getattr(perception_result, "degraded_reason", "") or ""
    if best is not None:
        best_conf = float(best.get("conf", 0.0))
        ev.source = "detector"
        ev.target_label = best.get("class_name", "")
        ev.target_subtype = CLASS_TO_SUBTYPE.get(ev.target_label, "")
        ev.target_conf = best_conf
        ev.class_probs = best.get("class_probs")
        ev.detection_source = "detector"
        bbox = best.get("bbox") or best.get("bbox_xyxy")
        pw = getattr(perception_result, "patch_width", 0) if perception_result else 0
        ph = getattr(perception_result, "patch_height", 0) if perception_result else 0
        if bbox and pw > 0 and ph > 0:
            cx = (float(bbox[0]) + float(bbox[2])) * 0.5 / pw
            cy = (float(bbox[1]) + float(bbox[3])) * 0.5 / ph
            # 全系统统一为图像坐标：[0,1]，左上 (0,0)，右下 (1,1)。
            ev.norm_xy = [round(_clamp01(cx), 4), round(_clamp01(cy), 4)]
    return ev


# ── 结构化回答 + schema 校验 ──────────────────────────────────────────────────

@dataclass
class VqaAnswer:
    question_id: str
    question_type: str
    answer: str = ""
    confidence: float = 0.0
    abstain: bool = False
    evidence: dict = field(default_factory=dict)
    decision: str = "abstain"
    reason_code: str = "invalid_output"
    raw_model_output: str = ""
    schema_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {
            "question_id": self.question_id, "question_type": self.question_type,
            "answer": self.answer, "confidence": round(_clamp01(self.confidence), 4),
            "abstain": bool(self.abstain), "evidence": self.evidence,
            "decision": self.decision, "reason_code": self.reason_code,
        }
        # Keep protocol diagnostics in benchmark artifacts without changing the
        # online decision fields consumed by the controller and frontend.
        if self.schema_errors:
            out["schema_errors"] = list(self.schema_errors)
        if self.raw_model_output:
            out["raw_model_output"] = self.raw_model_output
        return out


def validate_answer_dict(d: dict, spec: QuestionSpec) -> list[str]:
    """校验 VLM 结构化输出是否符合 schema (计划 5.2 / 7.8)。返回错误列表 (空=合法)。"""
    errs = []
    if not isinstance(d, dict):
        return ["not_a_dict"]
    ans = d.get("answer")
    if not isinstance(ans, str) or not ans.strip():
        errs.append("missing_answer")
    elif ans.strip() not in choices_for_question_type(spec.question_type):
        errs.append(f"answer_not_in_choices:{ans.strip()}")
    conf = d.get("confidence")
    if conf is None:
        errs.append("missing_confidence")
    elif isinstance(conf, bool):
        errs.append("confidence_not_number")
    else:
        try:
            cf = float(conf)
            if not math.isfinite(cf) or not (0.0 <= cf <= 1.0):
                errs.append(f"confidence_out_of_range:{cf}")
        except (TypeError, ValueError):
            errs.append("confidence_not_number")
    decision = d.get("decision")
    if decision not in DECISIONS:
        errs.append(f"invalid_decision:{decision}")
    reason = d.get("reason_code")
    if reason is None:
        errs.append("missing_reason_code")
    elif reason not in REASON_CODES:
        errs.append(f"invalid_reason_code:{reason}")
    abstain = d.get("abstain")
    if not isinstance(abstain, bool):
        errs.append("invalid_abstain")
    elif decision == "abstain" and not abstain:
        errs.append("abstain_decision_mismatch")
    elif decision != "abstain" and abstain:
        errs.append("abstain_decision_mismatch")
    reason_by_decision = {
        "answer": {"sufficient_evidence"},
        "continue_search": {"target_missing"},
        "reobserve": {"low_confidence"},
        "abstain": set(REASON_CODES) - {"sufficient_evidence"},
    }
    if decision in reason_by_decision and reason in REASON_CODES:
        if reason not in reason_by_decision[decision]:
            errs.append("decision_reason_mismatch")
    ev = d.get("evidence")
    if not isinstance(ev, dict):
        errs.append("missing_evidence")
    else:
        nxy = ev.get("norm_xy")
        if nxy is not None:
            if not (isinstance(nxy, list) and len(nxy) == 2
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                            and math.isfinite(float(x)) and 0.0 <= float(x) <= 1.0
                            for x in nxy)):
                errs.append("invalid_norm_xy")
        src = ev.get("source")
        if src is None:
            errs.append("missing_evidence_source")
        elif src not in EVIDENCE_SOURCES:
            errs.append(f"invalid_evidence_source:{src}")
    return errs


def choices_for_question_type(question_type: str) -> tuple[str, ...]:
    if question_type == "presence":
        return tuple(PRESENCE_CHOICES)
    if question_type == "damage":
        return tuple(DAMAGE_CHOICES)
    if question_type == "count":
        return tuple(COUNT_CHOICES)
    if question_type == "spatial":
        return tuple(BEARING_CHOICES)
    return ()


def parse_vlm_json_output(text: str, spec: QuestionSpec, question_id: str) -> VqaAnswer:
    """解析 VLM 的 JSON 结构化回答。解析失败返回显式 invalid_output (计划 5.2)。"""
    raw = (text or "").strip()
    # 抽取第一个 JSON 对象 (VLM 常包裹在 ```json ... ``` 或散文里)
    candidate = raw
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            candidate = brace.group(0)
    try:
        d = json.loads(candidate)
    except Exception as exc:
        return VqaAnswer(question_id=question_id, question_type=spec.question_type,
                         decision="abstain", reason_code="invalid_output",
                         raw_model_output=raw,
                         schema_errors=[f"invalid_json:{type(exc).__name__}"])
    errs = validate_answer_dict(d, spec)
    if errs:
        return VqaAnswer(question_id=question_id, question_type=spec.question_type,
                          decision="abstain", reason_code="invalid_output",
                          raw_model_output=raw, schema_errors=errs)
    return VqaAnswer(
        question_id=question_id, question_type=spec.question_type,
        answer=str(d.get("answer", "")).strip(),
        confidence=_clamp01(float(d.get("confidence", 0.0))),
        abstain=bool(d.get("abstain", False)),
        evidence=d.get("evidence") or {},
        decision=d.get("decision", "answer"),
        reason_code=d.get("reason_code", "sufficient_evidence"),
        raw_model_output=raw,
    )


# ── 控制器 ──────────────────────────────────────────────────────────────────

@dataclass
class AgentVqaConfig:
    confidence_threshold: float = 0.5      # 证据充分性阈值
    max_search_steps: int = 6             # continue_search 步数预算
    max_reobservations: int = 2           # reobserve 预算
    oracle: bool = False                  # 仅诊断; 不得部署
    allow_target_leak: bool = False        # oracle 时才允许从 item 读目标坐标
    evidence_level: str = "struct"         # raw | struct | state


# 依赖注入类型 (均为可调用, 便于测试用桩替换)
VlmAnswerFn = Callable[[str, Any, QuestionSpec, str], str]      # (image_bytes, perception, spec, qid) -> json text
PerceiveFn = Callable[[], Any]                                  # () -> PerceptionResult
SearchFn = Callable[[QuestionSpec, int, Any], Optional[dict]]    # (spec, step, perception) -> params | None
ReobserveFn = Callable[[Any, QuestionSpec], Optional[dict]]     # -> {kind, params, reason} | None


@dataclass
class StepRecord:
    """每步结构化决策轨迹 (计划 7.7)。对外不命名为 chain_of_thought。"""
    question_id: str
    observation_id: str
    position: dict
    question_type: str
    candidate_answer: str
    confidence: float
    evidence_ids: list[str]
    decision: str
    reason_code: str
    action: str
    budget_before: int
    budget_after: int
    fallback_used: bool = False
    degraded_reason: str = ""
    evidence: dict = field(default_factory=dict)
    reobserve_kind: str = ""
    reobserve_reason: str = ""
    reobserve_params: dict = field(default_factory=dict)
    entropy_table_loaded: bool = False
    entropy_fallback_used: bool = False
    motion_mode: str = ""
    uncertainty: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id, "observation_id": self.observation_id,
            "position": self.position, "question_type": self.question_type,
            "candidate_answer": self.candidate_answer, "confidence": round(self.confidence, 4),
            "evidence_ids": self.evidence_ids, "decision": self.decision,
            "reason_code": self.reason_code, "action": self.action,
            "budget_before": self.budget_before, "budget_after": self.budget_after,
            "fallback_used": self.fallback_used, "degraded_reason": self.degraded_reason,
            "evidence": self.evidence,
            "reobserve_kind": self.reobserve_kind,
            "reobserve_reason": self.reobserve_reason,
            "reobserve_params": self.reobserve_params,
            "entropy_table_loaded": self.entropy_table_loaded,
            "entropy_fallback_used": self.entropy_fallback_used,
            "motion_mode": self.motion_mode,
            "uncertainty": self.uncertainty,
        }


class AgentVqaController:
    """Agent-VQA 单题闭环控制器 (计划 7.1 / 7.3)。

    依赖通过构造函数注入, 使无 torch / 无模型环境也能用桩函数跑单元测试。
    在线决策只读当前观测; 不读测试条目的 answer 或未来图像; 非 oracle 配置
    不接收目标坐标。
    """

    def __init__(
        self,
        config: Optional[AgentVqaConfig] = None,
        vlm_answer_fn: Optional[VlmAnswerFn] = None,
        perceive_fn: Optional[PerceiveFn] = None,
        search_fn: Optional[SearchFn] = None,
        reobserve_fn: Optional[ReobserveFn] = None,
        get_position_fn: Optional[Callable[[], dict]] = None,
        get_image_bytes_fn: Optional[Callable[[Any], bytes]] = None,
        is_cancelled_fn: Optional[Callable[[], bool]] = None,
    ):
        self.config = config or AgentVqaConfig()
        self._vlm = vlm_answer_fn
        self._perceive = perceive_fn
        self._search = search_fn
        self._reobserve = reobserve_fn
        self._pos = get_position_fn or (lambda: {"lat": 0.0, "lon": 0.0, "alt": 30.0})
        self._img = get_image_bytes_fn or (lambda r: getattr(r, "patch_bytes", b"") or b"")
        self._cancelled = is_cancelled_fn or (lambda: False)
        self.trajectory: list[StepRecord] = []
        self.answer_history: list[VqaAnswer] = []
        self.fallback_used = False
        self.degraded_reason = ""

    def run(self, question: str, question_id: str = "",
             item: Optional[dict] = None,
             on_step: Optional[Callable[[dict], None]] = None) -> VqaAnswer:
        """运行单回合 Agent-VQA (计划 5.4 终止条件)。

        item 仅在 oracle 配置下用于读取目标坐标做诊断; 非 oracle 时忽略 item 的
        answer / target 字段, 不泄漏在线信息。

        on_step: 每完成一次"感知→候选答案→决策"后以该步 trajectory dict 调用一次，
        供前端 socket 实时广播（不阻塞闭环）。默认 None。
        """
        spec = parse_question(question)
        qid = question_id or f"q_{len(self.answer_history)}"
        if spec.question_type == "invalid_question":
            return self._final(qid, spec, "", 0.0, decision="abstain",
                               reason="invalid_question", action="stop")
        if self._perceive is None:
            return self._final(qid, spec, "", 0.0, decision="abstain",
                               reason="planner_unavailable", action="stop")

        search_budget = self.config.max_search_steps
        reobs_budget = self.config.max_reobservations
        last_answer: Optional[VqaAnswer] = None

        for step in range(self.config.max_search_steps + self.config.max_reobservations + 1):
            if self._cancelled():
                return self._final(qid, spec, "", 0.0, decision="abstain",
                                   reason="cancelled", action="stop")
            # 1) 获取当前观测
            try:
                result = self._perceive()
            except Exception as exc:
                self.degraded_reason = f"perception_error:{exc}"
                return self._final(qid, spec, "", 0.0, decision="abstain",
                                   reason="execution_error", action="stop")
            if result is None:
                return self._final(qid, spec, "", 0.0, decision="abstain",
                                   reason="out_of_coverage", action="stop")
            obs_id = getattr(result, "patch_id", f"obs{step}")
            # Capture the pose that produced this observation. reobserve_fn
            # executes motion synchronously, so reading position later would
            # incorrectly attach the post-action altitude to the pre-action image.
            observation_position = self._pos()
            ev = build_evidence_from_perception(result, spec, obs_id)

            # 2) 生成候选答案 (VLM 不可用时规则回退)
            ans = self._candidate_answer(qid, spec, ev, result)
            self.answer_history.append(ans)
            last_answer = ans

            # 3) 判断证据是否充分 -> 决策。重观测触发本身由注入的策略控制器决定，
            # 这样 random/fixed/entropy/conformal/info_gain 才是不同策略，而不是共享
            # 同一个 confidence threshold 后只在执行阶段换名字。
            decision, reason, action = self._decide(spec, ev, ans, search_budget, reobs_budget)

            # 重观测是否发生由注入的策略决定 (random/fixed/entropy/conformal/info_gain)。
            # 不得要求当前帧已经匹配到题面 subtype：A3 的科学点正是
            # “当前 argmax 不是目标类、但 class_probs 熵高 → 下降再看”。
            reobserve_outcome = None
            if decision == "answer" and reobs_budget > 0 and self._reobserve is not None:
                reobserve_outcome = self._safe_reobserve(result, spec)
                if reobserve_outcome is None:
                    decision, reason, action = "abstain", "execution_error", "stop"
                elif reobserve_outcome.get("kind") == "recheck":
                    decision, reason, action = "reobserve", "low_confidence", "fly_relative"

            self._record(qid, obs_id, spec, ans, ev, decision, reason, action,
                          search_budget, reobs_budget, reobserve_outcome,
                          observation_position=observation_position)
            if on_step is not None and self.trajectory:
                try:
                    on_step(self.trajectory[-1].to_dict())
                except Exception:
                    pass

            if decision == "answer":
                return self._finalize_answer(qid, spec, ans, ev, action, reason)
            if decision == "abstain":
                return self._finalize_abstain(qid, spec, ans, ev, reason, action)

            # 4) continue_search: 目标缺失 -> HSPM 搜索
            if decision == "continue_search":
                if search_budget <= 0 or self._search is None:
                    return self._finalize_abstain(qid, spec, ans, ev, "budget_exhausted", "stop")
                params = self._safe_search(spec, step, result)
                if params is None:
                    fail_reason = "execution_error" if self.degraded_reason.startswith("search_error:") else "target_missing"
                    return self._finalize_abstain(qid, spec, ans, ev, fail_reason, "stop")
                search_budget -= 1
                if self._cancelled():
                    return self._finalize_abstain(qid, spec, ans, ev, "cancelled", "stop")
                continue

            # 5) reobserve: 目标在但证据不足 -> 居中下降重观测
            if decision == "reobserve":
                if reobs_budget <= 0 or self._reobserve is None:
                    return self._finalize_abstain(qid, spec, ans, ev, "budget_exhausted", "stop")
                params = (reobserve_outcome or {}).get("params")
                if not params:
                    return self._finalize_abstain(qid, spec, ans, ev, "execution_error", "stop")
                reobs_budget -= 1
                if self._cancelled():
                    return self._finalize_abstain(qid, spec, ans, ev, "cancelled", "stop")
                continue

            break

        return self._finalize_abstain(qid, spec, last_answer or VqaAnswer(qid, spec.question_type),
                                       None, "budget_exhausted", "stop")

    # ── 内部: 候选答案 ────────────────────────────────────────────────────────
    def _candidate_answer(self, qid, spec, ev, result) -> VqaAnswer:
        if self._vlm is None:
            if self.config.evidence_level == "raw":
                return VqaAnswer(qid, spec.question_type, decision="abstain", abstain=True,
                                 reason_code="vlm_unavailable", evidence={"source": "image"})
            return self._rule_fallback(qid, spec, ev)
        try:
            img = self._img(result)
            text = self._vlm(img, result, spec, qid)
        except Exception as exc:
            self.fallback_used = True
            self.degraded_reason = f"vlm_error:{exc}"
            if self.config.evidence_level == "raw":
                return VqaAnswer(qid, spec.question_type, decision="abstain", abstain=True,
                                 reason_code="vlm_unavailable", evidence={"source": "image"})
            return self._rule_fallback(qid, spec, ev)
        ans = parse_vlm_json_output(text, spec, qid)
        if ans.reason_code == "invalid_output":
            self.degraded_reason = "invalid_model_output"
            return ans
        return ans

    def _rule_fallback(self, qid, spec, ev) -> VqaAnswer:
        """VLM 不可用时的规则回退 (计划 3.1 / RQ5)。只用结构化检测证据。"""
        self.fallback_used = True
        if spec.question_type == "presence":
            present = bool(ev.target_subtype and
                            (not spec.target_subtypes or ev.target_subtype in spec.target_subtypes))
            ans = "是" if present else "否"
            conf = ev.target_conf if present else 0.5
            return VqaAnswer(qid, spec.question_type, answer=ans, confidence=conf,
                             decision="answer" if present else "continue_search",
                             reason_code="sufficient_evidence" if present else "target_missing",
                             evidence=ev.to_dict())
        if spec.question_type == "damage":
            level = SUBTYPE_TO_LEVEL.get(ev.target_subtype, "")
            if level:
                return VqaAnswer(qid, spec.question_type, answer=level,
                                  confidence=ev.target_conf, decision="answer",
                                  reason_code="sufficient_evidence", evidence=ev.to_dict())
            return VqaAnswer(qid, spec.question_type, decision="continue_search",
                              reason_code="target_missing", evidence=ev.to_dict())
        if spec.question_type == "count":
            n = ev.matching_count
            bucket = "3+" if n >= 3 else str(n)
            return VqaAnswer(qid, spec.question_type, answer=bucket,
                              confidence=ev.target_conf, decision="answer",
                              reason_code="sufficient_evidence", evidence=ev.to_dict())
        if spec.question_type == "spatial":
            if ev.norm_xy and ev.target_subtype:
                nx, ny = ev.norm_xy
                north = 0.5 - float(ny)
                east = float(nx) - 0.5
                angle = math.degrees(math.atan2(east, north)) % 360.0
                direction = BEARING_CHOICES[int((angle + 22.5) // 45) % 8]
                return VqaAnswer(qid, spec.question_type, answer=direction,
                                  confidence=ev.target_conf, decision="answer",
                                  reason_code="sufficient_evidence", evidence=ev.to_dict())
            return VqaAnswer(qid, spec.question_type, decision="continue_search",
                              reason_code="target_missing", evidence=ev.to_dict())
        return VqaAnswer(qid, spec.question_type, decision="abstain",
                          reason_code="invalid_question", evidence=ev.to_dict())

    # ── 内部: 决策 ────────────────────────────────────────────────────────────
    def _decide(self, spec, ev, ans, search_budget, reobs_budget) -> tuple[str, str, str]:
        if ans.abstain or ans.reason_code in {"invalid_output", "vlm_unavailable"}:
            if ans.reason_code == "invalid_output":
                return "abstain", "invalid_output", "stop"
            if spec.needs_target_location and search_budget > 0:
                return "continue_search", "target_missing", "fly_relative"
            return "abstain", ans.reason_code, "stop"
        # 目标缺失 -> 继续搜索; 预算耗尽 -> 弃答 (计划 5.4: 步数预算耗尽)
        if spec.needs_target_location and not ev.target_subtype and not ans.answer:
            if search_budget > 0:
                return "continue_search", "target_missing", "fly_relative"
            return "abstain", "budget_exhausted", "stop"
        if spec.question_type == "presence" and ans.answer == "否":
            # 题库明确询问“当前视场”，经静态负例缓冲验证后“否”是可评分答案，
            # 不能因 max_search=0 把所有负例强制改成弃答。
            return "answer", "sufficient_evidence", "report_observation"
        # 充分 -> 回答
        return "answer", "sufficient_evidence", "report_observation"

    def _safe_search(self, spec, step, result) -> Optional[dict]:
        try:
            return self._search(spec, step, result)
        except Exception as exc:
            self.degraded_reason = f"search_error:{exc}"
            return None

    def _safe_reobserve(self, result, spec) -> Optional[dict]:
        try:
            return self._reobserve(result, spec)
        except Exception as exc:
            self.degraded_reason = f"reobserve_error:{exc}"
            return None

    # ── 内部: 记录与收尾 ──────────────────────────────────────────────────────
    def _record(self, qid, obs_id, spec, ans, ev, decision, reason, action,
                 search_budget, reobs_budget, reobserve_outcome=None,
                 observation_position=None):
        outcome = reobserve_outcome or {}
        unc = outcome.get("uncertainty")
        self.trajectory.append(StepRecord(
            question_id=qid, observation_id=obs_id,
            position=observation_position or self._pos(),
            question_type=spec.question_type, candidate_answer=ans.answer,
            confidence=ans.confidence, evidence_ids=[obs_id],
            decision=decision, reason_code=reason, action=action,
            budget_before=search_budget + reobs_budget,
            budget_after=search_budget + reobs_budget - (1 if decision in ("continue_search", "reobserve") else 0),
            fallback_used=self.fallback_used, degraded_reason=self.degraded_reason,
            evidence=ev.to_dict() if ev else {},
            reobserve_kind=str(outcome.get("kind") or ""),
            reobserve_reason=str(outcome.get("reason") or ""),
            reobserve_params=dict(outcome.get("params") or {}),
            entropy_table_loaded=bool(outcome.get("entropy_table_loaded")),
            entropy_fallback_used=bool(outcome.get("entropy_fallback_used")),
            motion_mode=str(outcome.get("motion_mode") or ""),
            uncertainty=None if unc is None else float(unc),
        ))

    def _final(self, qid, spec, answer, conf, decision, reason, action) -> VqaAnswer:
        ans = VqaAnswer(qid, spec.question_type, answer=answer, confidence=conf,
                         abstain=decision == "abstain", decision=decision, reason_code=reason)
        self.answer_history.append(ans)
        return ans

    def _finalize_answer(self, qid, spec, ans, ev, action, reason="sufficient_evidence") -> VqaAnswer:
        out = VqaAnswer(qid, spec.question_type, answer=ans.answer,
                         confidence=ans.confidence, abstain=False, evidence=ev.to_dict() if ev else {},
                         decision="answer", reason_code=reason,
                         raw_model_output=ans.raw_model_output,
                         schema_errors=list(ans.schema_errors))
        self.answer_history.append(out)
        return out

    def _finalize_abstain(self, qid, spec, ans, ev, reason, action) -> VqaAnswer:
        out = VqaAnswer(qid, spec.question_type, answer=ans.answer or "",
                         confidence=ans.confidence, abstain=True,
                         evidence=ev.to_dict() if ev else {},
                         decision="abstain", reason_code=reason,
                         raw_model_output=ans.raw_model_output,
                         schema_errors=list(ans.schema_errors))
        self.answer_history.append(out)
        return out

    def trajectory_dicts(self) -> list[dict]:
        return [r.to_dict() for r in self.trajectory]
