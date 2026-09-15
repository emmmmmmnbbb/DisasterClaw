r"""
backend/recheck.py — 灾情不确定性驱动的主动复核（P2，创新主线 C2）

核心思想：让"对灾情判断的把握程度"指挥飞行。看到疑似受灾目标但没把握时，
不急着下结论，而是主动**降高度 + 飞到目标正上方**再感知一次——

    perception 里 patch 半径 = clamp(MIN, alt × factor, MAX)，
    所以降高度 → 视场变小 → 地面分辨率(GSD)变细 → 复核能看得更清。

不确定性来源（两种模式，`RecheckConfig.uncertainty_mode` 切换）：
    - heuristic（默认，向后兼容）：risk_level 暧昧度 + 证据 conf 查表组合。
    - entropy（P5，对应文档第六节"升级接口"）：分布熵 U_t = -Σ p_i log p_i /
      log(K)，p 取自 perception.py 在 `VLN_CHANGE_PERCEPTION=1` 时暴露的
      `class_probs`（4 类损伤的校准 softmax）；没有 class_probs 时自动退化
      为 heuristic，不会因为某一帧缺概率分布而崩。

触发/收尾逻辑（`RecheckConfig.trigger_mode` 四选一，供 E11 六选一对照做基线）：
    - threshold（默认，向后兼容）：`unc >= trigger` 就触发复核。
    - info_gain：用 GSD 条件期望熵表（优先）或线性高度比启发式估计降高后熵下降。
    - conformal：预测集合 |C(x)|>1 则复核（覆盖保证由 val 分区 APS 校准）。
    - fixed / random：对照基线。

闭环：复核到"把握足够 / 预算耗尽 / 到高度下限"后定论——
    confirmed（确认受灾）/ dismissed（证据消退，排除）/ inconclusive（仍存疑），
    连同"不确定性下降量"一并返回，由上层写回语义地图 candidate_goals 层与报告。

已知问题修复（E11 实测发现，2026-07 修复）：episode 常常在 assess() 还没来得及
再次访问某个"复核中"位置之前就已经结束（到达终点 / 步数耗尽），导致该位置的
不确定性变化从未走到上面说的"定论"分支、从未被计入 `resolved_log`，使得
`stats()["avg_uncertainty_reduction"]` 系统性偏向 0——不是复核没起作用，是统计
口径只认"干净收尾"。修复方式：`RecheckController` 现在会持续记录每个位置"最新
一次观测到的不确定性"，并在 `finalize()`（episode 结束时调用一次）里把所有仍
挂在 `_state` 里、没等到定论的位置也按"最新已知的不确定性下降"补记一笔账
（status="episode_end"，不计入 confirmed/dismissed/inconclusive，但计入
avg_uncertainty_reduction 的分子分母），让统计反映复核期间真实发生的不确定性
变化，而不是"有没有幸运地等到一个正式收尾"。

本模块不做任何 IO / 模型调用 / geo 之外的依赖；几何换算复用 semantic_map.offset_from_norm。
便于单测：assess() 是确定性纯函数式接口（仅内部维护按位置去重的复核状态）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from semantic_map import offset_from_norm

try:
    # 改动一：视场收缩阶梯取代合成模糊阶梯。GSD ∝ alt 的关系在两者下都成立，
    # 所以 expected_gsd_gain_ratio() 的公式不变，只有默认高度需要重标定。
    from fov_ladder import ExpectedEntropyTable, eff_gsd_for_alt as effective_gsd_m
except Exception:  # pragma: no cover
    ExpectedEntropyTable = None  # type: ignore
    effective_gsd_m = None  # type: ignore

try:
    import fov_ladder as _FL
    _DEFAULT_ALT_MIN_M = _FL.alt_min_m()
    _DEFAULT_DESCEND_M = _FL.descend_step_m(2)
except Exception:  # pragma: no cover
    _DEFAULT_ALT_MIN_M = 10.0
    _DEFAULT_DESCEND_M = 10.0

# 受灾相关（值得复核）的检测类别——救援关注受损建筑与积水，完好建筑/车辆不算证据。
EVIDENCE_CLASSES = {
    "轻微损伤建筑",
    "严重损伤建筑",
    "完全损毁建筑",
    "受损建筑",
    "水池/积水区域",
}

# 各 risk_level 的"判断暧昧程度"（越大越没把握）。
_RISK_UNCERTAINTY = {"none": 0.2, "low": 0.9, "moderate": 0.6, "high": 0.15}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def best_evidence(
    detections: Optional[list[dict]],
) -> tuple[float, str, Optional[list], Optional[dict]]:
    """Return the detection used by the reobservation controller.

    Four-class backends retain the historical rule: use the highest-confidence
    damaged object and ignore intact buildings.  A binary detector must also be
    allowed to reconsider an uncertain ``no-damage`` prediction, so binary
    detections are ranked by entropy instead.  Otherwise a 0.51/0.49 negative
    prediction could never trigger the mechanism intended to resolve it.
    """
    best_conf, best_cls, best_bbox, best_probs = 0.0, "", None, None
    best_rank = -1.0
    for det in detections or []:
        cls = det.get("class_name", "")
        probs = det.get("class_probs")
        is_binary = isinstance(probs, dict) and set(probs) == {"no-damage", "damaged"}
        if cls not in EVIDENCE_CLASSES and not (is_binary and cls == "无损伤建筑"):
            continue
        conf = float(det.get("conf", 0.0))
        rank = entropy_uncertainty(probs) if is_binary else conf
        if rank > best_rank or (rank == best_rank and conf >= best_conf):
            best_rank = rank
            best_conf = conf
            best_cls = cls
            best_bbox = det.get("bbox") or det.get("bbox_xyxy")
            best_probs = probs
    return best_conf, best_cls, best_bbox, best_probs


def entropy_uncertainty(class_probs: dict[str, float]) -> float:
    """归一化 Shannon 熵 ∈ [0,1]：U_t = -Σ p_i log p_i / log(K)。

    K = len(class_probs)；K<=1 或概率全 0 时返回 0（没有分布可言，视为无不确定性，
    由调用方结合 has_evidence 决定是否真的当作"无需复核"）。
    """
    probs = [max(0.0, float(p)) for p in class_probs.values()]
    total = sum(probs)
    k = len(probs)
    if k <= 1 or total <= 0.0:
        return 0.0
    h = 0.0
    for p in probs:
        p_norm = p / total
        if p_norm > 0.0:
            h -= p_norm * math.log(p_norm)
    return round(_clamp(h / math.log(k), 0.0, 1.0), 3)


def count_bucket_uncertainty(detections: Optional[list[dict]]) -> float:
    """Entropy of the task answer distribution over ``0/1/2/3+``.

    Each binary ChangeOS building contributes a Bernoulli damaged probability.
    Dynamic programming caps the count state at three, matching the benchmark's
    answer vocabulary. This avoids zooming because one of many buildings has
    high entropy when the aggregate answer is already certainly ``3+``.
    """
    damaged_probs: list[float] = []
    for det in detections or []:
        probs = det.get("class_probs")
        if not isinstance(probs, dict) or set(probs) != {"no-damage", "damaged"}:
            continue
        total = max(0.0, float(probs.get("no-damage", 0.0))) + max(
            0.0, float(probs.get("damaged", 0.0))
        )
        if total <= 0.0:
            continue
        damaged_probs.append(_clamp(float(probs.get("damaged", 0.0)) / total, 0.0, 1.0))
    if not damaged_probs:
        return 0.0
    buckets = [1.0, 0.0, 0.0, 0.0]
    for p in damaged_probs:
        nxt = [0.0, 0.0, 0.0, 0.0]
        for count, mass in enumerate(buckets):
            nxt[count] += mass * (1.0 - p)
            nxt[min(3, count + 1)] += mass * p
        buckets = nxt
    return entropy_uncertainty({str(i): p for i, p in enumerate(buckets)})


def spatial_bearing_uncertainty(
    target_geo: Optional[list[float]],
    roi_bounds: Optional[dict[str, float]],
    observation_gsd_m: Optional[float],
    *,
    localization_sigma_factor: float = 3.0,
    min_sigma_m: float = 2.0,
) -> float:
    """Uncertainty that localization noise crosses an eight-bearing boundary.

    The target range and bearing are measured from the registered ROI centre.
    A conservative position sigma is derived from current GSD. The returned
    ratio exceeds 0.5 when angular localization error is larger than the margin
    to the nearest 45-degree class boundary.
    """
    if not isinstance(target_geo, (list, tuple)) or len(target_geo) != 2:
        return 1.0
    if not isinstance(roi_bounds, dict):
        return 1.0
    try:
        lat, lon = float(target_geo[0]), float(target_geo[1])
        center_lat = (float(roi_bounds["south"]) + float(roi_bounds["north"])) * 0.5
        center_lon = (float(roi_bounds["west"]) + float(roi_bounds["east"])) * 0.5
        gsd = float(observation_gsd_m or 0.0)
    except (KeyError, TypeError, ValueError):
        return 1.0
    north = (lat - center_lat) * 111_000.0
    east = (lon - center_lon) * 111_000.0 * math.cos(math.radians(center_lat))
    radius = math.hypot(north, east)
    sigma_m = max(float(min_sigma_m), float(localization_sigma_factor) * max(gsd, 0.0))
    if radius <= 1e-6:
        return 1.0
    bearing = math.degrees(math.atan2(east, north)) % 360.0
    boundary_margin = min(
        abs((bearing - boundary + 180.0) % 360.0 - 180.0)
        for boundary in (22.5 + 45.0 * i for i in range(8))
    )
    angular_sigma = math.degrees(math.atan2(sigma_m, radius))
    return round(_clamp(angular_sigma / max(angular_sigma + boundary_margin, 1e-9), 0.0, 1.0), 3)


def uncertainty_score(
    risk_level: str,
    evidence_conf: float,
    has_evidence: bool,
    class_probs: Optional[dict[str, float]] = None,
    mode: str = "heuristic",
    temperature: float = 1.0,
) -> float:
    """不确定性评分 ∈ [0,1]。

    mode="heuristic"（默认，向后兼容）：risk 暧昧度与证据低置信度各占一半。
    mode="entropy"（P5）：class_probs 给定时用校准熵 entropy_uncertainty()；
        class_probs 缺失时自动退化为 heuristic 公式（不因某一帧没有概率分布而崩）。
    """
    if not has_evidence:
        return 0.0
    if mode in {"entropy", "entropy_raw"} and class_probs:
        probs = class_probs
        if mode == "entropy_raw" and temperature != 1.0:
            raw = {
                name: max(float(value), 1e-12) ** max(float(temperature), 1e-6)
                for name, value in class_probs.items()
            }
            total = sum(raw.values()) or 1.0
            probs = {name: value / total for name, value in raw.items()}
        return entropy_uncertainty(probs)
    risk_unc = _RISK_UNCERTAINTY.get(risk_level, 0.5)
    return round(0.5 * risk_unc + 0.5 * (1.0 - _clamp(evidence_conf, 0.0, 1.0)), 3)


def expected_gsd_gain_ratio(alt: float, descend_step_m: float, alt_min_m: float) -> float:
    """线性高度比启发式（表缺失时的回退）。已到高度下限时比例=1。"""
    if alt <= alt_min_m:
        return 1.0
    alt_after = max(alt - descend_step_m, alt_min_m)
    return _clamp(alt_after / max(alt, 1e-6), 0.0, 1.0)


HORIZONTAL_SPEED_MPS = 12.0
VERTICAL_SPEED_MPS = 10.0


def reobserve_flight_time_s(horizontal_m: float, vertical_m: float) -> float:
    """Method-defined extra flight time: distance / speed, never wall-clock."""
    return float(horizontal_m) / HORIZONTAL_SPEED_MPS + float(vertical_m) / VERTICAL_SPEED_MPS


def predicted_roi_coverage(
    roi_norm_bbox: Optional[list[float]],
    *,
    next_center_norm: tuple[float, float] = (0.5, 0.5),
    next_span_ratio: float = 1.0,
) -> float:
    """Fraction of the task ROI retained by a proposed next observation.

    Coordinates are expressed in the current observation.  A descent scales the
    next square footprint by ``next_span_ratio``; recentering moves its centre.
    The denominator is the complete requested ROI, including any portion already
    outside the current image, so edge truncation is visible in the score.
    """
    if not isinstance(roi_norm_bbox, (list, tuple)) or len(roi_norm_bbox) != 4:
        return 0.0
    try:
        left, top, right, bottom = map(float, roi_norm_bbox)
    except (TypeError, ValueError):
        return 0.0
    area = max(0.0, right - left) * max(0.0, bottom - top)
    if area <= 0.0:
        return 0.0
    ratio = _clamp(float(next_span_ratio), 0.0, 1.0)
    cx, cy = map(float, next_center_norm)
    half = ratio * 0.5
    win_left, win_top, win_right, win_bottom = cx - half, cy - half, cx + half, cy + half
    inter_w = max(0.0, min(right, win_right) - max(left, win_left))
    inter_h = max(0.0, min(bottom, win_bottom) - max(top, win_top))
    return round(_clamp((inter_w * inter_h) / area, 0.0, 1.0), 6)


@dataclass(frozen=True)
class TaskRecheckDecision:
    allow: bool
    motion_mode: str
    utility: float
    uncertainty: float
    detail_gain: float
    predicted_roi_coverage: float
    coverage_loss: float
    estimated_flight_time_s: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "allow": self.allow,
            "motion_mode": self.motion_mode,
            "utility": round(self.utility, 6),
            "uncertainty": round(self.uncertainty, 6),
            "detail_gain": round(self.detail_gain, 6),
            "predicted_roi_coverage": round(self.predicted_roi_coverage, 6),
            "coverage_loss": round(self.coverage_loss, 6),
            "estimated_flight_time_s": round(self.estimated_flight_time_s, 6),
            "reason": self.reason,
        }


def task_conditioned_recheck_decision(
    *,
    question_type: str,
    uncertainty: float,
    alt: float,
    descend_step_m: float,
    alt_min_m: float,
    roi_norm_bbox: Optional[list[float]],
    target_visible: bool,
    target_matched: bool,
    recenter_horizontal_m: float = 0.0,
    uncertainty_trigger: float = 0.5,
    min_roi_coverage: float = 0.98,
    cost_weight: float = 0.05,
    coverage_weight: float = 1.0,
    cost_scale_s: float = 60.0,
    min_utility: float = 0.05,
) -> TaskRecheckDecision:
    """Question-aware, label-free gate for a candidate reobservation.

    Damage questions may centre on their marked building. Count and spatial
    questions keep the current geographic centre and descend only when the full
    task ROI remains visible. Presence positives are already answerable; target
    absence is handled by the search branch instead of a local damage-entropy
    zoom. Parameters are selected on development data and frozen before final.
    """
    qtype = str(question_type or "")
    unc = _clamp(float(uncertainty), 0.0, 1.0)
    alt_after = max(float(alt_min_m), float(alt) - float(descend_step_m))
    ratio = _clamp(alt_after / max(float(alt), 1e-6), 0.0, 1.0)
    detail_gain = max(0.0, 1.0 - ratio)

    if qtype == "presence":
        return TaskRecheckDecision(
            False, "hold", 0.0, unc, detail_gain, 1.0, 0.0, 0.0,
            "presence evidence is answered at the wide view; missing targets use search",
        )
    if qtype == "damage" and not target_visible:
        return TaskRecheckDecision(
            False, "hold", 0.0, unc, detail_gain, 1.0, 0.0, 0.0,
            "marked damage target is outside the current view",
        )
    if qtype not in {"damage", "count", "spatial"}:
        return TaskRecheckDecision(
            False, "hold", 0.0, unc, detail_gain, 0.0, 1.0, 0.0,
            "unsupported question type",
        )

    contextual = qtype in {"count", "spatial"}
    motion_mode = "descend_only" if contextual else "descend_center"
    coverage = (
        predicted_roi_coverage(
            roi_norm_bbox, next_center_norm=(0.5, 0.5), next_span_ratio=ratio,
        )
        if contextual else 1.0
    )
    coverage_loss = 1.0 - coverage
    vertical_m = max(0.0, float(alt) - alt_after)
    horizontal_m = 0.0 if contextual else max(0.0, float(recenter_horizontal_m))
    flight_time = reobserve_flight_time_s(horizontal_m, vertical_m)
    utility = (
        unc * detail_gain
        - float(cost_weight) * (flight_time / max(float(cost_scale_s), 1e-6))
        - float(coverage_weight) * coverage_loss
    )

    reasons = []
    if detail_gain <= 0.0:
        reasons.append("altitude floor reached")
    if unc < float(uncertainty_trigger):
        reasons.append("task uncertainty below trigger")
    if contextual and coverage < float(min_roi_coverage):
        reasons.append("predicted ROI coverage below minimum")
    if utility < float(min_utility):
        reasons.append("expected utility below minimum")
    allow = not reasons
    return TaskRecheckDecision(
        allow, motion_mode, utility, unc, detail_gain, coverage, coverage_loss,
        flight_time, "; ".join(reasons) if reasons else "positive task-conditioned utility",
    )


def info_gain_descend(
    entropy_now: float,
    alt: float,
    descend_step_m: float,
    alt_min_m: float,
    pred_class: str = "",
    entropy_table: Optional["ExpectedEntropyTable"] = None,
) -> float:
    """降高复核的期望熵下降。正式 info_gain 策略必须用拟合表，禁止静默回退高度比。"""
    if entropy_table is not None:
        fitted = entropy_table.info_gain(
            entropy_now, alt, descend_step_m, alt_min_m, pred_class or "no-damage",
        )
        if fitted is None:
            raise RuntimeError(
                "FOV entropy table has no usable bin for the requested class/GSD"
            )
        return max(0.0, round(fitted, 3))
    ratio = expected_gsd_gain_ratio(alt, descend_step_m, alt_min_m)
    expected_after = entropy_now * ratio
    return max(0.0, round(entropy_now - expected_after, 3))


def conformal_aps_scores(probs: list[float], label: int, rng: Optional[random.Random] = None) -> float:
    """Adaptive Prediction Sets nonconformity score for one labelled example."""
    k = len(probs)
    if k <= 0 or label < 0 or label >= k:
        return 1.0
    order = sorted(range(k), key=lambda i: -probs[i])
    cum = 0.0
    u = (rng.random() if rng is not None else 0.5)
    for rank, idx in enumerate(order):
        if idx == label:
            return _clamp(cum + u * max(probs[idx], 0.0), 0.0, 1.0)
        cum += max(probs[idx], 0.0)
    return 1.0


def fit_conformal_qhat(
    rows: list[tuple[dict[str, float], str]],
    alpha: float = 0.1,
    class_order: Optional[list[str]] = None,
) -> float:
    """Fit APS qhat on (class_probs, true_label) rows from the val partition."""
    if class_order:
        names = list(class_order)
    else:
        observed = {str(name) for probs, label in rows for name in probs} | {
            str(label) for _, label in rows
        }
        names = (
            ["no-damage", "damaged"]
            if observed and observed <= {"no-damage", "damaged"}
            else ["no-damage", "minor-damage", "major-damage", "destroyed"]
        )
    name_to_i = {n: i for i, n in enumerate(names)}
    scores: list[float] = []
    rng = random.Random(0)
    for probs_map, y in rows:
        vec = [max(0.0, float(probs_map.get(n, 0.0))) for n in names]
        total = sum(vec) or 1.0
        vec = [v / total for v in vec]
        scores.append(conformal_aps_scores(vec, name_to_i.get(str(y), 0), rng))
    if not scores:
        return 1.0
    scores.sort()
    n = len(scores)
    q_level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    idx = min(n - 1, max(0, int(math.ceil(q_level * n) - 1)))
    return float(scores[idx])


def conformal_predict_set(
    class_probs: dict[str, float],
    qhat: float,
    class_order: Optional[list[str]] = None,
) -> list[str]:
    """Smallest APS set whose cumulative prob reaches qhat."""
    if class_order:
        names = list(class_order)
    elif set(class_probs) == {"no-damage", "damaged"}:
        names = ["no-damage", "damaged"]
    else:
        names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    items = [(n, max(0.0, float(class_probs.get(n, 0.0)))) for n in names]
    total = sum(p for _, p in items) or 1.0
    items = [(n, p / total) for n, p in items]
    items.sort(key=lambda kv: -kv[1])
    chosen: list[str] = []
    cum = 0.0
    tau = _clamp(float(qhat), 0.0, 1.0)
    for name, p in items:
        chosen.append(name)
        cum += p
        if cum >= tau:
            break
    return chosen or [items[0][0]]


@dataclass
class RecheckConfig:
    conf_threshold: float = 0.5      # 证据置信度"够格"阈值
    trigger: float = 0.5             # 触发复核的不确定性阈值（trigger_mode="threshold" 时用）
    # descend_step_m / alt_min_m 必须严格小于调用方的巡航高度，否则"复核"从第一次
    # assess() 起就恒等于"已到高度下限"，永远拿不到 kind="recheck" 的真实降高机动
    # （E11 实测发现的根因，而不是 stats() 的记账问题）。
    #
    # 改动一重标定后：巡航 1330.2 m（3×3 瓦片 / 1.5 m/px）、下限 443.4 m
    # （1 瓦片 / 原生 0.5 m/px），单步 443.4 m 刚好两步到底。默认值直接从
    # fov_ladder 取，避免两处常量漂移。
    descend_step_m: float = _DEFAULT_DESCEND_M   # 每次复核下降的高度
    alt_min_m: float = _DEFAULT_ALT_MIN_M        # 高度下限（= 视场恰好一整瓦片）
    max_rechecks: int = 2            # 同一位置最多复核次数
    recenter_max_m: float = 40.0     # 复核单步水平居中的最大位移
    cell_m: float = 20.0             # 复核去重的位置量化格
    # P5：升级接口开关，默认值向后兼容（等价于升级前的行为）。
    uncertainty_mode: str = "heuristic"   # "heuristic" | "entropy"
    entropy_temperature: float = 1.0       # entropy_raw: 撤销温度标定
    trigger_mode: str = "threshold"       # "threshold" | "info_gain" | "fixed" | "random" | "conformal"
    min_info_gain: float = 0.05           # trigger_mode="info_gain" 时的最小期望熵下降
    random_prob: float = 0.5              # trigger_mode="random" 时的复核概率
    random_seed: int = 0                  # trigger_mode="random" 时的可复现随机种子
    entropy_table_path: str = ""          # info_gain 正式策略必须提供新 FOV 熵表
    conformal_qhat: float = 0.9           # APS 分位数（由 val 拟合）
    conformal_alpha: float = 0.1          # 目标误覆盖率
    motion_mode: str = "descend_center"   # hold | no_op | center_only | descend_only | descend_center | wide_roi


@dataclass
class RecheckOutcome:
    kind: str                              # "skip" | "recheck" | "resolve"
    uncertainty: float                     # 本次观测的不确定性
    label: str = ""                        # 可疑目标类别
    params: Optional[dict] = None          # kind="recheck" 时的 fly_relative 参数
    target_offset_m: Optional[tuple[float, float]] = None  # (north,east) 可疑目标相对 UAV
    status: Optional[str] = None           # kind="resolve" 时：confirmed/dismissed/inconclusive
    uncertainty_before: Optional[float] = None
    reduction: Optional[float] = None      # 不确定性下降量（>0 表示更有把握了）
    count: int = 0                         # 该位置已复核次数
    reason: str = ""
    # Agent-VQA 答案追踪字段 (计划 7.4)。answer_corrected / answer_harmed 仅离线
    # 评测阶段计算，在线控制器不可读取 (通过 record_answer_pair 单独写入)。
    answer_before: Optional[str] = None
    confidence_before: Optional[float] = None
    answer_after: Optional[str] = None
    confidence_after: Optional[float] = None
    answer_changed: Optional[bool] = None
    answer_corrected: Optional[bool] = None   # 离线评测填充
    answer_harmed: Optional[bool] = None      # 离线评测填充


class RecheckController:
    """按位置去重、有预算上限的复核状态机（每条 episode 一个实例）。"""

    def __init__(self, config: Optional[RecheckConfig] = None):
        self.config = config or RecheckConfig()
        valid_motion_modes = {
            "hold", "no_op", "center_only", "descend_only", "descend_center", "wide_roi",
        }
        if self.config.motion_mode not in valid_motion_modes:
            raise ValueError(
                f"invalid motion_mode {self.config.motion_mode!r}; "
                f"expected one of {sorted(valid_motion_modes)}"
            )
        # key=量化位置 → {count, unc0, label}
        self._state: dict[tuple, dict] = {}
        self.resolved_log: list[dict] = []  # 供报告/评测：每次定论的不确定性下降
        self.trigger_count = 0  # 本 episode 新触发复核的位置数（按量化位置/闭环计）
        self._rng = random.Random(self.config.random_seed)  # trigger_mode="random" 专用
        self._entropy_table = None
        path = (self.config.entropy_table_path or "").strip()
        if self.config.trigger_mode == "info_gain":
            if ExpectedEntropyTable is None:
                raise RuntimeError("fov_ladder.ExpectedEntropyTable is unavailable")
            if not path:
                raise ValueError("trigger_mode='info_gain' requires entropy_table_path")
            # Formal experiments are fail-closed: missing/stale/empty tables must
            # stop the run rather than silently changing A5 into a height heuristic.
            self._entropy_table = ExpectedEntropyTable.load(path)

    def _key(self, lat: float, lon: float, track_id: str = "") -> tuple:
        if track_id:
            return ("track", str(track_id))
        # 用经纬度的粗量化做去重（episode 内百米级，误差无所谓）。
        scale = self.config.cell_m / 111_000.0  # 约略：1 度纬度 ≈ 111km
        return (int(round(lat / scale)), int(round(lon / scale)))

    @property
    def entropy_table_loaded(self) -> bool:
        return self._entropy_table is not None

    def _should_recheck(
        self,
        unc: float,
        alt: float,
        class_probs: Optional[dict[str, float]] = None,
    ) -> bool:
        cfg = self.config
        if cfg.trigger_mode == "info_gain":
            pred = ""
            if class_probs:
                pred = max(class_probs.items(), key=lambda kv: float(kv[1]))[0]
            gain = info_gain_descend(
                unc, alt, cfg.descend_step_m, cfg.alt_min_m,
                pred_class=pred, entropy_table=self._entropy_table,
            )
            if self._entropy_table is not None:
                fitted = self._entropy_table.expected_entropy(
                    effective_gsd_m(max(alt - cfg.descend_step_m, cfg.alt_min_m)),
                    pred or "no-damage",
                )
                if fitted is None:
                    raise RuntimeError(
                        "FOV entropy table has no usable bin for the requested class/GSD"
                    )
            return gain > cfg.min_info_gain
        if cfg.trigger_mode == "conformal":
            if not class_probs:
                return unc >= cfg.trigger
            pred_set = conformal_predict_set(class_probs, cfg.conformal_qhat)
            return len(pred_set) > 1
        if cfg.trigger_mode == "fixed":
            return True
        if cfg.trigger_mode == "random":
            return self._rng.random() < cfg.random_prob
        return unc >= cfg.trigger

    def assess(
        self,
        *,
        lat: float,
        lon: float,
        alt: float,
        risk_level: str,
        detections: Optional[list[dict]],
        patch_radius_m: float,
        patch_width: int,
        patch_height: int,
        degraded: bool = False,
        allow_recheck: bool = True,
        track_id: str = "",
    ) -> RecheckOutcome:
        """评估当前观测：跳过 / 触发复核机动 / 定论。

        allow_recheck=False 用于 episode 的全局复核动作预算已经耗尽时：仍接收并
        记账最后一次机动后的新观测，但不再发出新机动。已有 pending 闭环会以
        inconclusive/confirmed 正式收尾；尚未触发的位置直接跳过。
        """
        cfg = self.config
        if cfg.motion_mode == "hold":
            return RecheckOutcome(
                kind="skip", uncertainty=0.0, reason="motion_mode=hold：不执行复观测机动。",
            )
        conf, label, bbox, class_probs = best_evidence(detections)
        has_detection_evidence = bool(label)
        has_evidence = has_detection_evidence or (risk_level not in ("none", ""))
        # 仅分割出水体等、无检测框时，conf 视为低（更不确定）
        eff_conf = conf if has_detection_evidence else 0.3
        unc = uncertainty_score(
            risk_level, eff_conf, has_evidence,
            class_probs=class_probs, mode=cfg.uncertainty_mode,
            temperature=cfg.entropy_temperature,
        )

        key = self._key(lat, lon, track_id=track_id)
        rec = self._state.get(key)

        # ── 1) 把握足够 / 无可疑目标 ────────────────────────────────
        if not has_evidence or not self._should_recheck(unc, alt, class_probs=class_probs):
            if rec is not None:
                # 之前在此处复核过，现在已经有把握 → 定论
                before = rec["unc0"]
                status = "confirmed" if risk_level in ("high", "moderate") else "dismissed"
                reduction = round(before - unc, 3)
                del self._state[key]
                out = RecheckOutcome(
                    kind="resolve", uncertainty=unc, label=rec["label"] or label,
                    status=status, uncertainty_before=before, reduction=reduction,
                    count=rec["count"],
                    reason=f"复核后把握提升（不确定性 {before:.2f}→{unc:.2f}），判定 {status}。",
                )
                self._record(lat, lon, out)
                return out
            return RecheckOutcome(
                kind="skip", uncertainty=unc, label=label,
                reason="把握足够或无可疑灾情目标，无需复核。",
            )

        # ── 2) 可疑且没把握 ─────────────────────────────────────────
        if rec is None:
            if not allow_recheck:
                return RecheckOutcome(
                    kind="skip", uncertainty=unc, label=label,
                    reason="episode 复核动作总预算已耗尽，不再触发新复核。",
                )
            rec = {"count": 0, "unc0": unc, "unc_latest": unc, "label": label, "lat": lat, "lon": lon}
            self._state[key] = rec
            self.trigger_count += 1
        else:
            # 持续刷新"最新观测"，供 finalize() 在 episode 提前结束时补记账用。
            rec["unc_latest"] = unc
            rec["lat"], rec["lon"] = lat, lon

        at_alt_floor = alt <= cfg.alt_min_m + 1e-6
        if rec["count"] >= cfg.max_rechecks or at_alt_floor or not allow_recheck:
            # 预算耗尽 / 到高度下限 → 收尾定论
            before = rec["unc0"]
            reduction = round(before - unc, 3)
            status = "confirmed" if (risk_level == "high" or conf >= cfg.conf_threshold) else "inconclusive"
            del self._state[key]
            if at_alt_floor:
                why = "到达高度下限"
            elif not allow_recheck:
                why = "episode 复核动作总预算耗尽"
            else:
                why = "复核预算耗尽"
            out = RecheckOutcome(
                kind="resolve", uncertainty=unc, label=rec["label"] or label,
                status=status, uncertainty_before=before, reduction=reduction,
                count=rec["count"],
                reason=f"{why}，仍为 risk={risk_level}（不确定性 {before:.2f}→{unc:.2f}），判定 {status}。",
            )
            self._record(lat, lon, out)
            return out

        # 还能复核 → 产出"降高 + 居中"机动
        rec["count"] += 1
        up_m = -min(cfg.descend_step_m, max(0.0, alt - cfg.alt_min_m))
        north_m, east_m = 0.0, 0.0
        offset = None
        allow_center = cfg.motion_mode in {"center_only", "descend_center"}
        allow_descend = cfg.motion_mode in {"descend_only", "descend_center", "wide_roi"}
        if not allow_descend:
            up_m = 0.0
        if allow_center and not degraded and bbox and patch_width > 0 and patch_height > 0 and patch_radius_m > 0:
            cx = (float(bbox[0]) + float(bbox[2])) * 0.5 / patch_width
            cy = (float(bbox[1]) + float(bbox[3])) * 0.5 / patch_height
            north_m, east_m = offset_from_norm((cx, cy), patch_radius_m)
            offset = (north_m, east_m)
            # 居中位移限幅（一步别飞太远，靠多步逼近）
            dist = math.hypot(north_m, east_m)
            if dist > cfg.recenter_max_m and dist > 0:
                f = cfg.recenter_max_m / dist
                north_m, east_m = north_m * f, east_m * f
        params = {
            "north_m": round(north_m, 1),
            "east_m": round(east_m, 1),
            "up_m": round(up_m, 1),
            "speed": 10.0,
        }
        return RecheckOutcome(
            kind="recheck", uncertainty=unc, label=rec["label"] or label,
            params=params, target_offset_m=offset, count=rec["count"],
            uncertainty_before=rec["unc0"],
            reason=(
                f"疑似「{rec['label'] or label or '受灾目标'}」(risk={risk_level}, "
                f"证据conf {conf:.2f}, 不确定性 {unc:.2f})，第 {rec['count']} 次复核："
                f"降高 {abs(up_m):.0f}m + 飞近居中。"
            ),
        )

    def _record(self, lat: float, lon: float, out: RecheckOutcome) -> None:
        self.resolved_log.append({
            "lat": lat,
            "lon": lon,
            "label": out.label,
            "status": out.status,
            "uncertainty_before": out.uncertainty_before,
            "uncertainty_after": out.uncertainty,
            "reduction": out.reduction,
            "rechecks": out.count,
            "answer_before": out.answer_before,
            "confidence_before": out.confidence_before,
            "answer_after": out.answer_after,
            "confidence_after": out.confidence_after,
            "answer_changed": out.answer_changed,
        })

    def finalize(self) -> None:
        """episode 结束时调用一次：把仍处于"复核中"但没等到正式定论的位置，
        按各自最新一次观测到的不确定性补记一笔账（见类顶部文档"已知问题修复"）。

        status="episode_end"，语义上不是"确认/排除/存疑"里的任何一种（没有真正
        走完复核闭环），因此不计入 confirmed/dismissed/inconclusive 计数，但计入
        avg_uncertainty_reduction，避免这部分被复核循环截断的样本从统计里消失。
        """
        for rec in self._state.values():
            before = rec["unc0"]
            after = rec.get("unc_latest", before)
            self.resolved_log.append({
                "lat": rec.get("lat"),
                "lon": rec.get("lon"),
                "label": rec.get("label"),
                "status": "episode_end",
                "uncertainty_before": before,
                "uncertainty_after": after,
                "reduction": round(before - after, 3),
                "rechecks": rec.get("count", 0),
            })
        self._state.clear()

    def stats(self) -> dict:
        """供报告：复核次数与平均不确定性下降。建议在 episode 结束、调用本方法前
        先调用一次 `finalize()`，否则"未收尾"的复核会被排除在统计之外（见类顶部
        文档"已知问题修复"）。"""
        n = len(self.resolved_log)
        red = [r["reduction"] for r in self.resolved_log if r.get("reduction") is not None]
        completed = sum(
            1 for r in self.resolved_log
            if r.get("status") in {"confirmed", "dismissed", "inconclusive"}
        )
        finalized_pending = sum(
            1 for r in self.resolved_log if r.get("status") == "episode_end"
        )
        # Agent-VQA 答案翻转统计 (计划 7.4 / E4)。answer_corrected / answer_harmed
        # 由 score_answer_pairs() 离线填充，在线控制器不读这些字段。
        flips = [r for r in self.resolved_log if r.get("answer_changed")]
        corrected = [r for r in self.resolved_log if r.get("answer_corrected")]
        harmed = [r for r in self.resolved_log if r.get("answer_harmed")]
        return {
            "triggered": self.trigger_count,
            "resolved": n,
            "completed": completed,
            "confirmed": sum(1 for r in self.resolved_log if r.get("status") == "confirmed"),
            "dismissed": sum(1 for r in self.resolved_log if r.get("status") == "dismissed"),
            "inconclusive": sum(1 for r in self.resolved_log if r.get("status") == "inconclusive"),
            "episode_end_pending": finalized_pending,
            "finalized_pending": finalized_pending,
            "uncertainty_reduction_sum": round(sum(red), 3),
            "avg_uncertainty_reduction": round(sum(red) / len(red), 3) if red else 0.0,
            "pending": len(self._state),
            "motion_mode": self.config.motion_mode,
            "entropy_table_loaded": self._entropy_table is not None,
            "entropy_fallback_used": False,
            "n_answer_flip": len(flips),
            "n_corrected": len(corrected),
            "n_harmed": len(harmed),
        }

    def score_answer_pairs(self, gt_answer: dict) -> dict:
        """离线评测：根据 GT 答案填充 answer_corrected / answer_harmed (计划 7.4 / E4)。

        gt_answer: {lat_lon_key: true_answer_str} 或 {recheck_key: true_answer_str}。
        仅在评测阶段调用；在线控制器不可读取。返回 {n_flip, n_corrected, n_harmed}。
        """
        n_flip = n_corrected = n_harmed = 0
        for r in self.resolved_log:
            before = r.get("answer_before")
            after = r.get("answer_after")
            if before is None or after is None:
                continue
            changed = before != after
            r["answer_changed"] = changed
            if changed:
                n_flip += 1
            key = (round(r.get("lat") or 0.0, 5), round(r.get("lon") or 0.0, 5))
            gt = gt_answer.get(key) or gt_answer.get(r.get("label"))
            if gt is not None and changed:
                if before != gt and after == gt:
                    r["answer_corrected"] = True
                    n_corrected += 1
                elif before == gt and after != gt:
                    r["answer_harmed"] = True
                    n_harmed += 1
        return {"n_flip": n_flip, "n_corrected": n_corrected, "n_harmed": n_harmed}
