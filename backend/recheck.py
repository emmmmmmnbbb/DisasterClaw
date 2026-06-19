"""
backend/recheck.py — 灾情不确定性驱动的主动复核（P2，创新主线 C2）

核心思想：让"对灾情判断的把握程度"指挥飞行。看到疑似受灾目标但没把握时，
不急着下结论，而是主动**降高度 + 飞到目标正上方**再感知一次——

    perception 里 patch 半径 = clamp(MIN, alt × factor, MAX)，
    所以降高度 → 视场变小 → 地面分辨率(GSD)变细 → 复核能看得更清。

不确定性来源：
    - risk_level：'low'（"轻度受灾或可疑"）最暧昧 → 不确定性最高；'high' 最笃定 → 最低。
    - 证据置信度：受灾相关检测框（受损建筑 / 积水）里最高的 conf，越低越不确定。

闭环：复核到"把握足够 / 预算耗尽 / 到高度下限"后定论——
    confirmed（确认受灾）/ dismissed（证据消退，排除）/ inconclusive（仍存疑），
    连同"不确定性下降量"一并返回，由上层写回语义地图 candidate_goals 层与报告。

本模块不做任何 IO / 模型调用 / geo 之外的依赖；几何换算复用 semantic_map.offset_from_norm。
便于单测：assess() 是确定性纯函数式接口（仅内部维护按位置去重的复核状态）。
"""

from __future__ import annotations

import math
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


def best_evidence(detections: Optional[list[dict]]) -> tuple[float, str, Optional[list]]:
    """受灾相关检测里 conf 最高的 (conf, class_name, bbox)；无则 (0.0, '', None)。"""
    best_conf, best_cls, best_bbox = 0.0, "", None
    for det in detections or []:
        cls = det.get("class_name", "")
        if cls not in EVIDENCE_CLASSES:
            continue
        conf = float(det.get("conf", 0.0))
        if conf >= best_conf:
            best_conf = conf
            best_cls = cls
            best_bbox = det.get("bbox") or det.get("bbox_xyxy")
    return best_conf, best_cls, best_bbox


def uncertainty_score(risk_level: str, evidence_conf: float, has_evidence: bool) -> float:
    """不确定性评分 ∈ [0,1]：risk 暧昧度与证据低置信度各占一半。"""
    if not has_evidence:
        return 0.0
    risk_unc = _RISK_UNCERTAINTY.get(risk_level, 0.5)
    return round(0.5 * risk_unc + 0.5 * (1.0 - _clamp(evidence_conf, 0.0, 1.0)), 3)


@dataclass
class RecheckConfig:
    conf_threshold: float = 0.5      # 证据置信度"够格"阈值
    trigger: float = 0.5             # 触发复核的不确定性阈值
    descend_step_m: float = 20.0     # 每次复核下降的高度
    alt_min_m: float = 30.0          # 高度下限（防止贴地）
    max_rechecks: int = 2            # 同一位置最多复核次数
    recenter_max_m: float = 40.0     # 复核单步水平居中的最大位移
    cell_m: float = 20.0             # 复核去重的位置量化格


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

    def _key(self, lat: float, lon: float) -> tuple[int, int]:
        # 用经纬度的粗量化做去重（episode 内百米级，误差无所谓）。
        scale = self.config.cell_m / 111_000.0  # 约略：1 度纬度 ≈ 111km
        return (int(round(lat / scale)), int(round(lon / scale)))

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
        conf, label, bbox = best_evidence(detections)
        has_detection_evidence = bool(label)
        has_evidence = has_detection_evidence or (risk_level not in ("none", ""))
        # 仅分割出水体等、无检测框时，conf 视为低（更不确定）
        eff_conf = conf if has_detection_evidence else 0.3
        unc = uncertainty_score(risk_level, eff_conf, has_evidence)

        key = self._key(lat, lon)
        rec = self._state.get(key)

        # ── 1) 把握足够 / 无可疑目标 ────────────────────────────────
        if not has_evidence or unc < cfg.trigger:
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
