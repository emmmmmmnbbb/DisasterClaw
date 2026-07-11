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
    - info_gain（P5）：把"复核（降高+居中）"和"维持（不复核）"当成两个候选动作，
      用 GSD-置信度校准曲线的简化确定性代理估计复核动作的期望后验熵下降
      \(H(P_t) - \mathbb E H(P_{t+1}^{descend})\)（"维持"动作的期望熵下降恒为 0），
      仅当该增益超过 `min_info_gain` 才复核——等价于 \(a_t^\star=\arg\max_a
      [H(P_t)-\mathbb E H(P_{t+1}^a)]\) 在两候选动作间取值。
    - fixed（E11 对照基线"固定降高复核"）：只要有可疑证据就必复核，不看不确定性。
    - random（E11 对照基线"随机复核"）：以固定概率 `random_prob` 决定是否复核，
      用同一 seed 复现；作为"复核策略本身有没有信息量"的下界对照。

闭环：复核到"把握足够 / 预算耗尽 / 到高度下限"后定论——
    confirmed（确认受灾）/ dismissed（证据消退，排除）/ inconclusive（仍存疑），
    连同"不确定性下降量"一并返回，由上层写回语义地图 candidate_goals 层与报告。

本模块不做任何 IO / 模型调用 / geo 之外的依赖；几何换算复用 semantic_map.offset_from_norm。
便于单测：assess() 是确定性纯函数式接口（仅内部维护按位置去重的复核状态）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from semantic_map import offset_from_norm

# 受灾相关（值得复核）的检测类别——救援关注受损建筑与积水，完好建筑/车辆不算证据。
EVIDENCE_CLASSES = {
    "轻微损伤建筑",
    "严重损伤建筑",
    "完全损毁建筑",
    "水池/积水区域",
}

# 各 risk_level 的"判断暧昧程度"（越大越没把握）。
_RISK_UNCERTAINTY = {"none": 0.2, "low": 0.9, "moderate": 0.6, "high": 0.15}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def best_evidence(
    detections: Optional[list[dict]],
) -> tuple[float, str, Optional[list], Optional[dict]]:
    """受灾相关检测里 conf 最高的 (conf, class_name, bbox, class_probs)；无则 (0.0, '', None, None)。

    class_probs 是 perception.py 在 `VLN_CHANGE_PERCEPTION=1` 时才会附加的字段
    （4 类损伤的校准 softmax），没开该开关时恒为 None。
    """
    best_conf, best_cls, best_bbox, best_probs = 0.0, "", None, None
    for det in detections or []:
        cls = det.get("class_name", "")
        if cls not in EVIDENCE_CLASSES:
            continue
        conf = float(det.get("conf", 0.0))
        if conf >= best_conf:
            best_conf = conf
            best_cls = cls
            best_bbox = det.get("bbox") or det.get("bbox_xyxy")
            best_probs = det.get("class_probs")
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


def uncertainty_score(
    risk_level: str,
    evidence_conf: float,
    has_evidence: bool,
    class_probs: Optional[dict[str, float]] = None,
    mode: str = "heuristic",
) -> float:
    """不确定性评分 ∈ [0,1]。

    mode="heuristic"（默认，向后兼容）：risk 暧昧度与证据低置信度各占一半。
    mode="entropy"（P5）：class_probs 给定时用校准熵 entropy_uncertainty()；
        class_probs 缺失时自动退化为 heuristic 公式（不因某一帧没有概率分布而崩）。
    """
    if not has_evidence:
        return 0.0
    if mode == "entropy" and class_probs:
        return entropy_uncertainty(class_probs)
    risk_unc = _RISK_UNCERTAINTY.get(risk_level, 0.5)
    return round(0.5 * risk_unc + 0.5 * (1.0 - _clamp(evidence_conf, 0.0, 1.0)), 3)


def expected_gsd_gain_ratio(alt: float, descend_step_m: float, alt_min_m: float) -> float:
    """GSD-置信度校准曲线的简化确定性代理（P5 待确认 5：未来可换真正的贝叶斯/蒙特卡洛版）。

    降高复核后视场半径变小 → 地面分辨率(GSD)变细；用"降高后/降高前的剩余高度比"近似
    "降高后预期熵 / 当前熵"的比例——已到高度下限时比例=1（降不动，预期没有增益）。
    """
    if alt <= alt_min_m:
        return 1.0
    alt_after = max(alt - descend_step_m, alt_min_m)
    return _clamp(alt_after / max(alt, 1e-6), 0.0, 1.0)


def info_gain_descend(entropy_now: float, alt: float, descend_step_m: float, alt_min_m: float) -> float:
    """"降高居中复核"动作的期望信息增益 H(P_t) - E[H(P_{t+1}^{descend})]。

    对照动作"维持（不复核）"的期望信息增益恒为 0（观测不变，熵不变）；
    因此 arg max_a[...] 退化为"该增益是否超过 min_info_gain"的判断。
    """
    ratio = expected_gsd_gain_ratio(alt, descend_step_m, alt_min_m)
    expected_after = entropy_now * ratio
    return max(0.0, round(entropy_now - expected_after, 3))


@dataclass
class RecheckConfig:
    conf_threshold: float = 0.5      # 证据置信度"够格"阈值
    trigger: float = 0.5             # 触发复核的不确定性阈值（trigger_mode="threshold" 时用）
    descend_step_m: float = 20.0     # 每次复核下降的高度
    alt_min_m: float = 30.0          # 高度下限（防止贴地）
    max_rechecks: int = 2            # 同一位置最多复核次数
    recenter_max_m: float = 40.0     # 复核单步水平居中的最大位移
    cell_m: float = 20.0             # 复核去重的位置量化格
    # P5：升级接口开关，默认值向后兼容（等价于升级前的行为）。
    uncertainty_mode: str = "heuristic"   # "heuristic" | "entropy"
    trigger_mode: str = "threshold"       # "threshold" | "info_gain" | "fixed" | "random"
    min_info_gain: float = 0.05           # trigger_mode="info_gain" 时的最小期望熵下降
    random_prob: float = 0.5              # trigger_mode="random" 时的复核概率
    random_seed: int = 0                  # trigger_mode="random" 时的可复现随机种子


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


class RecheckController:
    """按位置去重、有预算上限的复核状态机（每条 episode 一个实例）。"""

    def __init__(self, config: Optional[RecheckConfig] = None):
        self.config = config or RecheckConfig()
        # key=量化位置 → {count, unc0, label}
        self._state: dict[tuple[int, int], dict] = {}
        self.resolved_log: list[dict] = []  # 供报告/评测：每次定论的不确定性下降
        self._rng = random.Random(self.config.random_seed)  # trigger_mode="random" 专用

    def _key(self, lat: float, lon: float) -> tuple[int, int]:
        # 用经纬度的粗量化做去重（episode 内百米级，误差无所谓）。
        scale = self.config.cell_m / 111_000.0  # 约略：1 度纬度 ≈ 111km
        return (int(round(lat / scale)), int(round(lon / scale)))

    def _should_recheck(self, unc: float, alt: float) -> bool:
        """P5：触发判断，`trigger_mode` 二选一。

        threshold：原有阈值判断（向后兼容）。
        info_gain：只有当"降高居中"动作的期望信息增益超过 min_info_gain 才复核，
            等价于在 {descend_center, hold} 两候选动作里 arg max 期望熵下降。
        fixed（E11 基线）：有可疑证据就必复核，不管不确定性数值。
        random（E11 基线）：以固定概率决定，不看任何信号——用来验证"复核策略
            是不是真的有信息量"，而不是随便动就有效果。
        """
        cfg = self.config
        if cfg.trigger_mode == "info_gain":
            gain = info_gain_descend(unc, alt, cfg.descend_step_m, cfg.alt_min_m)
            return gain > cfg.min_info_gain
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
    ) -> RecheckOutcome:
        """评估当前观测：跳过 / 触发复核机动 / 定论。"""
        cfg = self.config
        conf, label, bbox, class_probs = best_evidence(detections)
        has_detection_evidence = bool(label)
        has_evidence = has_detection_evidence or (risk_level not in ("none", ""))
        # 仅分割出水体等、无检测框时，conf 视为低（更不确定）
        eff_conf = conf if has_detection_evidence else 0.3
        unc = uncertainty_score(
            risk_level, eff_conf, has_evidence,
            class_probs=class_probs, mode=cfg.uncertainty_mode,
        )

        key = self._key(lat, lon)
        rec = self._state.get(key)

        # ── 1) 把握足够 / 无可疑目标 ────────────────────────────────
        if not has_evidence or not self._should_recheck(unc, alt):
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
            rec = {"count": 0, "unc0": unc, "label": label}
            self._state[key] = rec

        at_alt_floor = alt <= cfg.alt_min_m + 1e-6
        if rec["count"] >= cfg.max_rechecks or at_alt_floor:
            # 预算耗尽 / 到高度下限 → 收尾定论
            before = rec["unc0"]
            reduction = round(before - unc, 3)
            status = "confirmed" if (risk_level == "high" or conf >= cfg.conf_threshold) else "inconclusive"
            del self._state[key]
            why = "到达高度下限" if at_alt_floor else "复核预算耗尽"
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
        if not degraded and bbox and patch_width > 0 and patch_height > 0 and patch_radius_m > 0:
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
        })

    def stats(self) -> dict:
        """供报告：复核次数与平均不确定性下降。"""
        n = len(self.resolved_log)
        red = [r["reduction"] for r in self.resolved_log if r.get("reduction") is not None]
        return {
            "resolved": n,
            "confirmed": sum(1 for r in self.resolved_log if r.get("status") == "confirmed"),
            "dismissed": sum(1 for r in self.resolved_log if r.get("status") == "dismissed"),
            "inconclusive": sum(1 for r in self.resolved_log if r.get("status") == "inconclusive"),
            "avg_uncertainty_reduction": round(sum(red) / len(red), 3) if red else 0.0,
            "pending": len(self._state),
        }
