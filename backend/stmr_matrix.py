"""
backend/stmr_matrix.py — STMR 文字矩阵（P1）

借鉴 STMR (arXiv:2410.08500) 的关键洞察：
    直接把俯视地图"图片"丢给 VLM 做空间推理效果差；把它转成"文字矩阵"喂 LLM 更稳。

本模块把 semantic_map.SemanticMap 的局部窗口渲染成 LLM 友好的文本：
    - 一张以无人机为中心、window_m×window_m、grid_n×grid_n 的语义网格（数字码 + 图例）；
    - UAV 所在格标 'U'；行=南北（上北下南），列=东西（左西右东）；
    - 附"目标方位摘要"（每个目标相对 UAV 的方位 + 距离），便于 LLM 直接做 OROI 常识推理。

DisasterClaw 比 STMR 更省：我们有真实地理参考，矩阵直接来自 2D 地理语义地图，
无需深度反投影。
"""

from __future__ import annotations

import math
from typing import Any

from semantic_map import CLASS_CODE, SemanticMap

# 数字码 → 名称（渲染图例用）
_CODE_NAME: dict[int, str] = {v: k for k, v in CLASS_CODE.items()}


def _bearing_name(north_m: float, east_m: float) -> str:
    """(north, east) 偏移 → 八方位中文名。"""
    if abs(north_m) < 1e-6 and abs(east_m) < 1e-6:
        return "正下方"
    ang = math.degrees(math.atan2(east_m, north_m)) % 360.0  # 0=北，顺时针
    names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    idx = int((ang + 22.5) // 45) % 8
    return names[idx]


def object_bearing_summary(
    smap: SemanticMap, uav_lat: float, uav_lon: float, max_items: int = 12
) -> list[dict]:
    """每个已知目标相对 UAV 的方位 + 距离摘要（按距离升序）。"""
    from geo import latlon_to_meters

    items: list[dict] = []
    for obj in smap.objects():
        n, e = latlon_to_meters(uav_lat, uav_lon, obj["lat"], obj["lon"])
        dist = math.hypot(n, e)
        items.append({
            "class_name": obj.get("class_name", ""),
            "layer": obj.get("layer", ""),
            "bearing": _bearing_name(n, e),
            "dist_m": round(dist, 1),
            "conf": round(float(obj.get("conf", 0.0)), 2),
            "lat": obj["lat"],
            "lon": obj["lon"],
        })
    items.sort(key=lambda x: x["dist_m"])
    return items[:max_items]


def build_stmr(
    smap: SemanticMap,
    uav_lat: float,
    uav_lon: float,
    window_m: float = 100.0,
    grid_n: int = 20,
) -> dict[str, Any]:
    """渲染 STMR 文字矩阵 + 方位摘要。

    返回 {text, matrix, grid_n, cell_m, window_m, objects}。
    text 即可直接放进 LLM prompt。
    """
    local = smap.to_local_matrix(uav_lat, uav_lon, window_m=window_m, grid_n=grid_n)
    matrix = local["matrix"]
    cell_m = local["cell_m"]

    ci, cj = grid_n // 2, grid_n // 2  # UAV 在中心格

    # 出现过的类别码（用于精简图例）
    present = sorted({c for row in matrix for c in row})
    legend_bits = [f"{code}={_CODE_NAME.get(code, '?')}" for code in present]
    legend_bits.append("U=无人机当前位置")

    # 渲染网格：中心格标 U
    lines: list[str] = []
    for r in range(grid_n):
        cells = []
        for c in range(grid_n):
            cells.append("U" if (r == ci and c == cj) else str(matrix[r][c]))
        lines.append(" ".join(cells))
    grid_text = "\n".join(lines)

    objs = object_bearing_summary(smap, uav_lat, uav_lon)
    if objs:
        obj_lines = [
            f"  - {o['class_name']}：{o['bearing']} 约 {o['dist_m']:.0f}m（conf {o['conf']:.2f}）"
            for o in objs
        ]
        obj_text = "已发现目标（相对无人机）：\n" + "\n".join(obj_lines)
    else:
        obj_text = "已发现目标：暂无（当前已观测区域内未识别到目标）。"

    text = (
        f"[局部语义地图] 以无人机为中心 {window_m:.0f}m×{window_m:.0f}m，"
        f"每格约 {cell_m:.1f}m；{grid_n}×{grid_n} 网格。\n"
        f"方向：行从上到下 = 由北到南，列从左到右 = 由西到东。\n"
        f"图例：{'，'.join(legend_bits)}\n"
        f"网格：\n{grid_text}\n"
        f"{obj_text}"
    )

    return {
        "text": text,
        "matrix": matrix,
        "grid_n": grid_n,
        "cell_m": cell_m,
        "window_m": window_m,
        "objects": objs,
    }
