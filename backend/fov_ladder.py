"""backend/fov_ladder.py — 高度 → 视场 → 有效 GSD 阶梯 (改动一)

替代 `gsd_ladder.py` 的几何部分。核心区别：

    旧: 裁一小块 → 加高斯模糊 → 撤销模糊 = 「降高」
        (review2 B2: "下降在你的仿真里等于撤销你自己刚加的高斯模糊")

    新: 固定传感器 + 固定视场角，降高 = 收缩地面足迹 → 在固定输出像素数下
        重采样比下降 → 单位目标像素数真实提升。分辨率损失来自真实下采样比，
        不是人工模糊。

物理模型（全部参数由此推出，不是独立设定的）::

    footprint(alt) = 2 · alt · tan(FOV/2)          [m]
    eff_gsd(alt)   = footprint(alt) / SENSOR_PX    [m/px]

xBD 瓦片 1024 px @ 0.5 m/px = 512 m。约束「最小视场 = 恰好一整张瓦片」
唯一确定下限高度::

    alt_min = TILE_SPAN_M / (2 · tan(FOV/2)) = 443.4 m   → eff_gsd = 0.500 m/px  (原生)
    alt_max = 3 · TILE_SPAN_M / (2 · tan(FOV/2)) = 1330 m → eff_gsd = 1.500 m/px

于是信息天花板 (0.5 m/px) 是**推导出来的**，不是像旧阶梯那样被设定的。
443–1330 m 也正好是固定翼应急侦察平台的真实作业高度，
在这个高度上 0.5 m/px 物理自洽 —— 直接消掉 review2 B2 的后半句
"真实 10 m 飞行给的是厘米级 GSD"。

阶梯定义在**瓦片跨度**而非米上，使「下限 = 恰好一整瓦片」成为不变量而非巧合。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

# ── 传感器与几何常量 ─────────────────────────────────────────────────────────

SENSOR_PX = 1024          # 渲染输出边长（像素）= 模型输入尺寸
FOV_DEG = 60.0            # 水平视场角
NATIVE_GSD_M = 0.5        # xBD 原生地面分辨率
TILE_PX = 1024            # xBD 瓦片边长（像素）
TILE_SPAN_M = TILE_PX * NATIVE_GSD_M   # = 512.0 m

SPAN_TILES_MIN = 1.0      # 硬不变量：最小视场 = 恰好一整张瓦片
SPAN_TILES_MAX = 3.0      # 巡航视场 = 3×3 瓦片

CLASS_NAMES = ("no-damage", "minor-damage", "major-damage", "destroyed")

# 2·tan(FOV/2)：footprint = _FOV_FACTOR · alt
_FOV_FACTOR = 2.0 * math.tan(math.radians(FOV_DEG) / 2.0)


# ── 高度 ↔ 视场 ↔ GSD ────────────────────────────────────────────────────────

def span_m_for_alt(alt_m: float) -> float:
    """高度 → 地面足迹边长（米）。"""
    return _FOV_FACTOR * float(alt_m)


def alt_for_span_m(span_m: float) -> float:
    """地面足迹边长（米）→ 高度。"""
    return float(span_m) / _FOV_FACTOR


def alt_for_span_tiles(span_tiles: float) -> float:
    """视场跨度（以 xBD 瓦片为单位）→ 高度。

    ``alt_for_span_tiles(1.0)`` 即下限高度，此时视场恰好一整张瓦片、
    有效 GSD 等于原生 GSD。
    """
    return alt_for_span_m(float(span_tiles) * TILE_SPAN_M)


def span_tiles_for_alt(alt_m: float) -> float:
    """高度 → 视场跨度（瓦片数）。"""
    return span_m_for_alt(alt_m) / TILE_SPAN_M


def eff_gsd_for_alt(alt_m: float, sensor_px: int = SENSOR_PX) -> float:
    """高度 → 有效地面分辨率（m/px）。

    这是渲染窗口重采样到 ``sensor_px`` 之后的真实每像素地面尺寸。
    在 ``alt_min`` 处严格等于 ``NATIVE_GSD_M``。
    """
    return span_m_for_alt(alt_m) / float(sensor_px)


def alt_min_m(sensor_px: int = SENSOR_PX) -> float:
    """下限高度：视场 = 一整张瓦片。"""
    return alt_for_span_tiles(SPAN_TILES_MIN)


def alt_cruise_m() -> float:
    """巡航高度：视场 = SPAN_TILES_MAX × SPAN_TILES_MAX 瓦片。"""
    return alt_for_span_tiles(SPAN_TILES_MAX)


def clamp_alt(alt_m: float) -> float:
    """把高度夹到 [alt_min, alt_cruise]。"""
    return max(alt_min_m(), min(alt_cruise_m(), float(alt_m)))


def resample_ratio(alt_m: float, sensor_px: int = SENSOR_PX) -> float:
    """渲染窗口的源像素数 / 输出像素数。

    >1 表示下采样（信息被压缩）；==1 表示原生。等价于 eff_gsd / native_gsd。
    这是旧 `gsd_ladder.effective_scale` 的真实对应物，但它现在描述的是
    真实重采样比，而不是人工模糊强度。
    """
    return eff_gsd_for_alt(alt_m, sensor_px) / NATIVE_GSD_M


def ladder_points(n: int = 3, sensor_px: int = SENSOR_PX) -> list[dict]:
    """从巡航到下限的 n 档阶梯，等间隔于**瓦片跨度**。

    n=3 → span_tiles 3.0 / 2.0 / 1.0，对应 1330 / 887 / 443 m，
    有效 GSD 1.50 / 1.00 / 0.50 m/px。
    """
    if n < 2:
        n = 2
    out: list[dict] = []
    for i in range(n):
        # i=0 → SPAN_TILES_MAX, i=n-1 → SPAN_TILES_MIN
        t = i / (n - 1)
        span_tiles = SPAN_TILES_MAX + t * (SPAN_TILES_MIN - SPAN_TILES_MAX)
        alt = alt_for_span_tiles(span_tiles)
        out.append({
            "span_tiles": round(span_tiles, 4),
            "alt_m": round(alt, 3),
            "span_m": round(span_m_for_alt(alt), 3),
            "gsd_m": round(eff_gsd_for_alt(alt, sensor_px), 6),
            "resample_ratio": round(resample_ratio(alt, sensor_px), 4),
        })
    return out


def descend_step_m(n_steps: int = 2) -> float:
    """把 [alt_min, alt_cruise] 均分成 n_steps 步的单步下降量（米）。"""
    if n_steps < 1:
        n_steps = 1
    return (alt_cruise_m() - alt_min_m()) / float(n_steps)


def roi_pixel_fraction(alt_m: float) -> float:
    """ROI（一整张瓦片）在当前视场中占的**线性**比例。

    巡航 (3 瓦片跨度) → 1/3；下限 → 1.0。ROI 的像素面积占比是它的平方。
    这是 ROI-scoped 设定下「降高带来多少真实信息增益」的直接度量。
    """
    return min(1.0, SPAN_TILES_MIN / max(span_tiles_for_alt(alt_m), 1e-9))


# ── 期望熵表（沿用旧接口，但必须用新阶梯重新离线拟合）─────────────────────────

class ExpectedEntropyTable:
    r"""查表 \(\hat E[U | GSD, \hat y]\)，在事件不相交裁块上离线拟合。

    接口与 `gsd_ladder.ExpectedEntropyTable` 保持一致，供 `recheck.py` 的
    info_gain 模式复用。

    警告: 旧的拟合结果是在**合成高斯模糊**上得到的，与新的视场收缩阶梯
    不是同一个降质过程，**不得直接复用**，必须重新拟合。
    """

    SCHEMA = "fov-ladder-entropy/1.0"

    def __init__(self, payload: Optional[dict] = None):
        self.payload = payload or {}
        self._rows: list[dict] = list(self.payload.get("bins") or [])
        self._rows.sort(key=lambda r: float(r.get("gsd_m") or 0.0))

    @classmethod
    def load(cls, path: str | Path) -> "ExpectedEntropyTable":
        p = Path(path)
        if not p.is_file():
            return cls({})
        payload = json.loads(p.read_text(encoding="utf-8"))
        schema = payload.get("schema")
        if schema and schema != cls.SCHEMA:
            raise ValueError(
                f"entropy table schema mismatch: got {schema!r}, expected {cls.SCHEMA!r}. "
                "旧的 gsd_ladder 熵表是在合成模糊上拟合的，不能用于视场收缩阶梯。"
            )
        return cls(payload)

    def expected_entropy(self, gsd_m: float, pred_class: str) -> Optional[float]:
        if not self._rows:
            return None
        pred = pred_class if pred_class in CLASS_NAMES else "no-damage"
        row = min(self._rows, key=lambda r: abs(float(r.get("gsd_m", 0.0)) - float(gsd_m)))
        by_cls = row.get("by_pred_class") or {}
        stats = by_cls.get(pred) or by_cls.get("all") or {}
        mean_u = stats.get("mean_entropy")
        return float(mean_u) if mean_u is not None else None

    def info_gain(
        self,
        entropy_now: float,
        alt_now_m: float,
        descend_step_m: float,
        alt_min_m: float,
        pred_class: str,
    ) -> Optional[float]:
        alt_after = max(float(alt_now_m) - float(descend_step_m), float(alt_min_m))
        expected_after = self.expected_entropy(eff_gsd_for_alt(alt_after), pred_class)
        if expected_after is None:
            return None
        return max(0.0, float(entropy_now) - float(expected_after))
