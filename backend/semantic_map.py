"""
backend/semantic_map.py — 2D 地理语义栅格地图（P0 地基）

定位（对应 docs/vln_rescue_agent_实施计划.md 的 P0）：
    在 DisasterClaw 的 2D 俯视 + 真实地理参考之上，维护一张"会一直累积"的地图，
    记录无人机飞过哪、每格看到了什么。后续 STMR 文字矩阵（P1）、灾情驱动复核
    （P2）、记忆拓扑图（P3）都建在它上面。

图层（对齐 CityNav 的 Geographic Semantic Map）：
    - current_fov          : 当前这一步的视场覆盖（每步覆盖刷新）
    - explored             : 累积已观测区域（视场并集）
    - landmarks            : 建筑类（无损伤 / 轻微 / 严重 / 完全损毁）
    - surrounding_objects  : 车辆 / 积水等非建筑目标
    - candidate_goals      : 预留给 P2（灾情可疑、待复核的目标）

坐标与栅格：
    以 episode 起点 (origin_lat, origin_lon) 为原点，用 geo.latlon_to_meters 做
    局部线性化（单 episode 百米~公里级，误差可忽略），按 cell_size_m 栅格化。
    这样即便途中 active_tile / world.anchor 改变，地图坐标也不漂移。

投影（检测框 → 经纬度）：
    复用 vln_navigator 的几何约定——俯视 patch 半边 ≈ patch_radius_m。
    检测框中心的归一化坐标 (nx, ny)（左上 0,0 / 右下 1,1）换算：
        east_m  = (nx - 0.5) * 2 * radius_m
        north_m = -(ny - 0.5) * 2 * radius_m   # 图像 y 向下为南
    再用 geo.meters_to_latlon 相对无人机位置还原 lat/lon。
    退化视场（degraded，裁到整图 / 贴边 clamp）几何不可信，只记 explored、跳过投影。

本模块不做任何 IO / socket / 模型加载；线程安全（内部一把锁）。
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

from geo import latlon_to_meters, meters_to_latlon


# ── 图层名（对齐 CityNav GSM）──────────────────────────────────────────
LAYER_CURRENT_FOV = "current_fov"
LAYER_EXPLORED = "explored"
LAYER_LANDMARKS = "landmarks"
LAYER_OBJECTS = "surrounding_objects"
LAYER_CANDIDATE_GOALS = "candidate_goals"

_OBJECT_LAYERS = (LAYER_LANDMARKS, LAYER_OBJECTS, LAYER_CANDIDATE_GOALS)

# 建筑类 → landmarks；其余已知目标 → surrounding_objects。
# 标签需与 perception.YOLO_LABEL_MAP 的中文标签一致。
_BUILDING_CLASSES = {
    "无损伤建筑",
    "轻微损伤建筑",
    "严重损伤建筑",
    "完全损毁建筑",
}
_OBJECT_CLASSES = {
    "车辆",
    "水池/积水区域",
}

# STMR 文字矩阵用的数字标签（P1 会用到；0 = 空 / 未探索）。
CLASS_CODE: dict[str, int] = {
    "未探索": 0,
    "已探索空地": 1,
    "无损伤建筑": 2,
    "轻微损伤建筑": 3,
    "严重损伤建筑": 4,
    "完全损毁建筑": 5,
    "车辆": 6,
    "水池/积水区域": 7,
}
# 同一格多个类别时，数字越大优先级越高（损毁/目标盖过空地）。
_CODE_PRIORITY = {code: code for code in CLASS_CODE.values()}


def layer_for_class(class_name: str) -> Optional[str]:
    if class_name in _BUILDING_CLASSES:
        return LAYER_LANDMARKS
    if class_name in _OBJECT_CLASSES:
        return LAYER_OBJECTS
    return None


def offset_from_norm(
    norm_xy: tuple[float, float], radius_m: float
) -> tuple[float, float]:
    """归一化中心 (x,y) → 相对无人机的 (north_m, east_m)。

    与 vln_navigator.VlnNavigator._offset_from_norm 同一套约定，保证导航与建图一致。
    """
    nx, ny = norm_xy
    east_m = (nx - 0.5) * 2.0 * radius_m
    north_m = -(ny - 0.5) * 2.0 * radius_m
    return north_m, east_m


class SemanticMap:
    """单架 UAV、单 episode 累积的 2D 地理语义地图。"""

    def __init__(
        self,
        origin_lat: float,
        origin_lon: float,
        cell_size_m: float = 5.0,
        instruction: str = "",
    ) -> None:
        self._lock = threading.RLock()
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)
        self.cell_size_m = float(cell_size_m) if cell_size_m > 0 else 5.0
        self.instruction = instruction
        self.created_at = time.time()
        self.step_count = 0

        # 每个 object 图层： (gi, gj) -> {class_name, conf, count, risk, lat, lon, last_step}
        self._object_cells: dict[str, dict[tuple[int, int], dict[str, Any]]] = {
            LAYER_LANDMARKS: {},
            LAYER_OBJECTS: {},
            LAYER_CANDIDATE_GOALS: {},
        }
        # 覆盖类图层用 cell 集合。
        self._explored: set[tuple[int, int]] = set()
        self._current_fov: set[tuple[int, int]] = set()

    # ── 坐标 ↔ 栅格 ──────────────────────────────────────────────────
    def cell_of(self, lat: float, lon: float) -> tuple[int, int]:
        north, east = latlon_to_meters(self.origin_lat, self.origin_lon, lat, lon)
        gi = int(math.floor(north / self.cell_size_m))
        gj = int(math.floor(east / self.cell_size_m))
        return gi, gj

    def cell_center(self, gi: int, gj: int) -> tuple[float, float]:
        north = (gi + 0.5) * self.cell_size_m
        east = (gj + 0.5) * self.cell_size_m
        return meters_to_latlon(self.origin_lat, self.origin_lon, north, east)

    # ── 写入 ─────────────────────────────────────────────────────────
    def mark_observation(
        self,
        uav_lat: float,
        uav_lon: float,
        radius_m: float,
        detections: Optional[list[dict]] = None,
        degraded: bool = False,
        risk_level: str = "none",
        patch_width: int = 0,
        patch_height: int = 0,
    ) -> dict:
        """记录一次观测：标 explored / current_fov，并（非退化时）投影检测框。

        返回本次写入的小结，便于上层日志 / socket。
        """
        radius_m = max(float(radius_m or 0.0), self.cell_size_m)
        with self._lock:
            self.step_count += 1
            # 1) 视场覆盖（圆盘内的格子）
            fov_cells = self._disk_cells(uav_lat, uav_lon, radius_m)
            self._current_fov = set(fov_cells)
            self._explored.update(fov_cells)

            # 2) 检测框投影（退化视场几何不可信，跳过）
            added = 0
            if not degraded and detections and patch_width > 0 and patch_height > 0:
                for det in detections:
                    cell_rec = self._project_detection(
                        uav_lat, uav_lon, radius_m, det, patch_width, patch_height,
                        risk_level,
                    )
                    if cell_rec is not None:
                        added += 1

            return {
                "step": self.step_count,
                "fov_cells": len(fov_cells),
                "explored_cells": len(self._explored),
                "objects_added": added,
                "degraded": bool(degraded),
            }

    def add_candidate_goal(
        self, lat: float, lon: float, label: str, conf: float = 0.0, risk: str = "none"
    ) -> None:
        """P2 复核接口预留：把"灾情可疑、待复核"的点写入 candidate_goals 层。"""
        with self._lock:
            cell = self.cell_of(lat, lon)
            self._object_cells[LAYER_CANDIDATE_GOALS][cell] = {
                "class_name": label,
                "conf": float(conf),
                "count": 1,
                "risk": risk,
                "lat": float(lat),
                "lon": float(lon),
                "last_step": self.step_count,
            }

    def _project_detection(
        self,
        uav_lat: float,
        uav_lon: float,
        radius_m: float,
        det: dict,
        patch_width: int,
        patch_height: int,
        risk_level: str,
    ) -> Optional[dict]:
        class_name = det.get("class_name")
        layer = layer_for_class(class_name)
        if layer is None:
            return None
        bbox = det.get("bbox") or det.get("bbox_xyxy")
        if not bbox or len(bbox) < 4:
            return None
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        nx = cx / patch_width
        ny = cy / patch_height
        north_m, east_m = offset_from_norm((nx, ny), radius_m)
        lat, lon = meters_to_latlon(uav_lat, uav_lon, north_m, east_m)
        cell = self.cell_of(lat, lon)
        conf = float(det.get("conf", 0.0))

        bucket = self._object_cells[layer]
        prev = bucket.get(cell)
        if prev is None or conf >= prev.get("conf", 0.0):
            bucket[cell] = {
                "class_name": class_name,
                "conf": conf,
                "count": (prev.get("count", 0) if prev else 0) + 1,
                "risk": risk_level,
                "lat": float(lat),
                "lon": float(lon),
                "last_step": self.step_count,
            }
        else:
            prev["count"] = prev.get("count", 0) + 1
            prev["last_step"] = self.step_count
        return bucket[cell]

    def _disk_cells(
        self, lat: float, lon: float, radius_m: float
    ) -> list[tuple[int, int]]:
        """无人机当前视场圆盘覆盖到的格子（按 origin 栅格）。"""
        cn, ce = latlon_to_meters(self.origin_lat, self.origin_lon, lat, lon)
        cell = self.cell_size_m
        r_cells = int(math.ceil(radius_m / cell))
        gi0 = int(math.floor(cn / cell))
        gj0 = int(math.floor(ce / cell))
        out: list[tuple[int, int]] = []
        r2 = radius_m * radius_m
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                gi = gi0 + di
                gj = gj0 + dj
                # 格中心到无人机的距离（米）
                cell_n = (gi + 0.5) * cell
                cell_e = (gj + 0.5) * cell
                if (cell_n - cn) ** 2 + (cell_e - ce) ** 2 <= r2:
                    out.append((gi, gj))
        return out

    # ── 读取 ─────────────────────────────────────────────────────────
    def stats(self) -> dict:
        with self._lock:
            return {
                "step_count": self.step_count,
                "explored_cells": len(self._explored),
                "current_fov_cells": len(self._current_fov),
                "landmarks": len(self._object_cells[LAYER_LANDMARKS]),
                "surrounding_objects": len(self._object_cells[LAYER_OBJECTS]),
                "candidate_goals": len(self._object_cells[LAYER_CANDIDATE_GOALS]),
                "cell_size_m": self.cell_size_m,
            }

    def objects(self, layer: Optional[str] = None) -> list[dict]:
        """返回检测点列表（含 lat/lon/class/conf），供 socket / 调试。"""
        with self._lock:
            layers = [layer] if layer else list(_OBJECT_LAYERS)
            out: list[dict] = []
            for lyr in layers:
                for (gi, gj), rec in self._object_cells.get(lyr, {}).items():
                    out.append({
                        "layer": lyr,
                        "gi": gi,
                        "gj": gj,
                        **rec,
                    })
            return out

    def frontier_score(
        self,
        lat: float,
        lon: float,
        bearing_vec: tuple[float, float],
        probe_m: float = 60.0,
        n_samples: int = 4,
    ) -> float:
        """C3（HSPM 运动层工程改进，OROI 打分融合用）：给定当前位置与方向单位向量
        (north,east)，估计沿该方向探索能带来的"未探索覆盖增益" ∈ [0,1]。

        做法：沿方向等距采样 n_samples 个探测点（距离从 probe_m/n_samples 到
        probe_m），统计落入未探索格子的比例——分越高说明这个方向"新地方"越多，
        越值得去看，用于替代"LLM 自由选一个方位"里缺失的空间覆盖信号。
        全地图尚无探索记录时（刚起飞）任何方向都视为全新，返回 1.0。
        """
        with self._lock:
            if not self._explored:
                return 1.0
            dn, de = bearing_vec
            norm = math.hypot(dn, de) or 1.0
            dn, de = dn / norm, de / norm
            cn0, ce0 = latlon_to_meters(self.origin_lat, self.origin_lon, lat, lon)
            cell = self.cell_size_m
            unexplored = 0
            for i in range(1, n_samples + 1):
                dist = probe_m * i / n_samples
                cn = cn0 + dn * dist
                ce = ce0 + de * dist
                gi = int(math.floor(cn / cell))
                gj = int(math.floor(ce / cell))
                if (gi, gj) not in self._explored:
                    unexplored += 1
            return unexplored / n_samples

    def explored_bounds(self) -> Optional[dict]:
        """已探索区的经纬度包围盒（用于前端聚焦 / 调试）。"""
        with self._lock:
            if not self._explored:
                return None
            gis = [c[0] for c in self._explored]
            gjs = [c[1] for c in self._explored]
            lat_a, lon_a = self.cell_center(min(gis), min(gjs))
            lat_b, lon_b = self.cell_center(max(gis), max(gjs))
            return {
                "south": min(lat_a, lat_b),
                "north": max(lat_a, lat_b),
                "west": min(lon_a, lon_b),
                "east": max(lon_a, lon_b),
            }

    def to_local_matrix(
        self,
        center_lat: float,
        center_lon: float,
        window_m: float = 100.0,
        grid_n: int = 20,
    ) -> dict:
        """STMR 文字矩阵（P1 主用，这里给最小可用版）。

        以 (center_lat, center_lon) 为中心、window_m×window_m 的局部窗口切成
        grid_n×grid_n，每格取该窗口内最高优先级的语义类别码（语义 max-pooling）。
        返回矩阵（grid_n 行 × grid_n 列，row 0 = 最北）+ 元信息。
        """
        with self._lock:
            r = window_m / grid_n  # 每格边长（米）
            cn, ce = latlon_to_meters(
                self.origin_lat, self.origin_lon, center_lat, center_lon
            )
            half = window_m / 2.0
            # 行 r0 对应最北（north 最大），列 c0 对应最西（east 最小）
            matrix = [[0 for _ in range(grid_n)] for _ in range(grid_n)]

            def _put(north: float, east: float, code: int) -> None:
                row = int((cn + half - north) / r)
                col = int((east - (ce - half)) / r)
                if 0 <= row < grid_n and 0 <= col < grid_n:
                    if _CODE_PRIORITY.get(code, code) > _CODE_PRIORITY.get(matrix[row][col], matrix[row][col]):
                        matrix[row][col] = code

            # 1) explored 空地铺底
            for (gi, gj) in self._explored:
                north = (gi + 0.5) * self.cell_size_m
                east = (gj + 0.5) * self.cell_size_m
                _put(north, east, CLASS_CODE["已探索空地"])
            # 2) 目标盖上去
            for lyr in (LAYER_LANDMARKS, LAYER_OBJECTS):
                for (gi, gj), rec in self._object_cells[lyr].items():
                    north = (gi + 0.5) * self.cell_size_m
                    east = (gj + 0.5) * self.cell_size_m
                    code = CLASS_CODE.get(rec.get("class_name", ""), 0)
                    if code:
                        _put(north, east, code)

            return {
                "matrix": matrix,
                "grid_n": grid_n,
                "cell_m": r,
                "window_m": window_m,
                "center": {"lat": center_lat, "lon": center_lon},
                "legend": dict(CLASS_CODE),
            }

    def snapshot(self, max_objects: int = 200) -> dict:
        """供 socket 推送的精简地图状态（限量，避免 payload 过大）。"""
        with self._lock:
            objs = self.objects()
            objs.sort(key=lambda o: o.get("conf", 0.0), reverse=True)
            return {
                "origin": {"lat": self.origin_lat, "lon": self.origin_lon},
                "cell_size_m": self.cell_size_m,
                "stats": self.stats(),
                "explored_bounds": self.explored_bounds(),
                "objects": objs[:max_objects],
                "instruction": self.instruction,
            }

    def to_dict(self) -> dict:
        """完整序列化（调试 / 落盘用）。"""
        with self._lock:
            return {
                "origin": {"lat": self.origin_lat, "lon": self.origin_lon},
                "cell_size_m": self.cell_size_m,
                "instruction": self.instruction,
                "created_at": self.created_at,
                "stats": self.stats(),
                "explored": sorted(self._explored),
                "current_fov": sorted(self._current_fov),
                "objects": {
                    lyr: {f"{gi},{gj}": rec for (gi, gj), rec in cells.items()}
                    for lyr, cells in self._object_cells.items()
                },
            }
