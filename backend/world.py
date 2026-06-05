from __future__ import annotations

import copy
import threading
import time
import uuid
from typing import Any

from geo import latlon_to_meters


DEFAULT_BASEMAP = {
    "provider": "esri-world-imagery",
    "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "attribution": "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    "max_zoom": 19,
    "alternatives": [
        {
            "provider": "osm",
            "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attribution": "© OpenStreetMap contributors",
            "max_zoom": 19,
        },
    ],
}


class WorldModel:
    def __init__(
        self,
        anchor_lat: float,
        anchor_lon: float,
        hover_altitude_m: float = 30.0,
        anchor_label: str = "DisasterClaw Demo Anchor",
        basemap: dict | None = None,
    ):
        self._lock = threading.Lock()
        self._anchor_lat = anchor_lat
        self._anchor_lon = anchor_lon
        self._hover_altitude_m = hover_altitude_m
        self._state = {
            "robots": {},
            "targets": [],
            "map": {
                "anchor": {
                    "label": anchor_label,
                    "lat": anchor_lat,
                    "lon": anchor_lon,
                },
                "active_tile_id": None,
                "active_tile": None,
                "latlon_bounds": None,
                "corner_coordinates": None,
                "geo_features": [],
                "reports": [],
                "basemap": copy.deepcopy(basemap or DEFAULT_BASEMAP),
            },
            "timestamp": time.time(),
        }

    # ── properties ──────────────────────────────────────────────
    @property
    def anchor_lat(self) -> float:
        return self._anchor_lat

    @property
    def anchor_lon(self) -> float:
        return self._anchor_lon

    # ── robots ──────────────────────────────────────────────────
    def register_default_uav(self, robot_id: str = "UAV_1") -> None:
        with self._lock:
            self._state["robots"][robot_id] = {
                "robot_type": "UAV",
                "status": "airborne",
                "task_state": "idle",
                "battery": 100.0,
                "in_air": True,
                "heading_deg": 0.0,
                "speed_mps": 0.0,
                "position": {
                    "lat": self._anchor_lat,
                    "lon": self._anchor_lon,
                    "alt": self._hover_altitude_m,
                    "north_m": 0.0,
                    "east_m": 0.0,
                    "down_m": -self._hover_altitude_m,
                },
            }
            self._state["timestamp"] = time.time()

    def update_robot(self, robot_id: str, data: dict) -> None:
        with self._lock:
            robot = self._state["robots"].setdefault(robot_id, {})
            robot.update(data)
            position = robot.get("position") or {}
            if position and "lat" in position and "lon" in position:
                north_m, east_m = latlon_to_meters(
                    self._anchor_lat,
                    self._anchor_lon,
                    position["lat"],
                    position["lon"],
                )
                position["north_m"] = north_m
                position["east_m"] = east_m
                position["down_m"] = -float(position.get("alt", self._hover_altitude_m))
                robot["position"] = position
            self._state["timestamp"] = time.time()

    # ── targets / reports ───────────────────────────────────────
    def add_target(
        self,
        label: str,
        lat: float,
        lon: float,
        alt: float = 0.0,
        kind: str = "poi",
        confidence: float = 1.0,
        source: str = "manual",
    ) -> dict:
        north_m, east_m = latlon_to_meters(self._anchor_lat, self._anchor_lon, lat, lon)
        target = {
            "target_id": f"target-{uuid.uuid4().hex[:8]}",
            "label": label,
            "kind": kind,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "north_m": north_m,
            "east_m": east_m,
            "confidence": confidence,
            "source": source,
        }
        with self._lock:
            self._state["targets"].append(target)
            self._state["timestamp"] = time.time()
        return target

    def add_report(
        self,
        content: str,
        lat: float,
        lon: float,
        level: str = "info",
        source: str = "ai",
    ) -> dict:
        north_m, east_m = latlon_to_meters(self._anchor_lat, self._anchor_lon, lat, lon)
        report = {
            "id": f"report-{uuid.uuid4().hex[:8]}",
            "content": content,
            "level": level,
            "source": source,
            "lat": lat,
            "lon": lon,
            "north_m": north_m,
            "east_m": east_m,
            "timestamp": time.time(),
        }
        with self._lock:
            self._state["map"]["reports"].append(report)
            self._state["timestamp"] = time.time()
        return report

    # ── xBD active tile ─────────────────────────────────────────
    def set_active_tile(self, entry: dict[str, Any] | None) -> dict:
        """
        激活某张 xBD 瓦片：
            1. 把 world.map.active_tile / bounds / corners / geo_features 写成该瓦片信息
            2. 把锚点换成瓦片中心（所有米换算都在这个局部再线性化）
            3. 重算已存在 robots / targets / reports 的 north_m / east_m
        """
        with self._lock:
            if entry is None:
                self._state["map"]["active_tile_id"] = None
                self._state["map"]["active_tile"] = None
                self._state["map"]["latlon_bounds"] = None
                self._state["map"]["corner_coordinates"] = None
                self._state["map"]["geo_features"] = []
                self._state["timestamp"] = time.time()
                return copy.deepcopy(self._state)

            bounds = entry.get("bounds")
            corners = entry.get("corner_coordinates")
            tile_id = entry.get("tile_id")

            anchor_lat = self._anchor_lat
            anchor_lon = self._anchor_lon
            anchor_label = self._state["map"]["anchor"].get("label") or "xBD Anchor"
            if bounds:
                anchor_lat = (float(bounds["north"]) + float(bounds["south"])) * 0.5
                anchor_lon = (float(bounds["east"]) + float(bounds["west"])) * 0.5
                anchor_label = f"{entry.get('disaster') or 'xBD'} · {tile_id}"
                self._anchor_lat = anchor_lat
                self._anchor_lon = anchor_lon

            tile_summary = {
                "tile_id": tile_id,
                "group_id": entry.get("group_id"),
                "split": entry.get("split"),
                "stage": entry.get("stage"),
                "disaster": entry.get("disaster"),
                "disaster_type": entry.get("disaster_type"),
                "sensor": entry.get("sensor"),
                "capture_date": entry.get("capture_date"),
                "catalog_id": entry.get("catalog_id"),
                "gsd": entry.get("gsd"),
                "width": entry.get("width"),
                "height": entry.get("height"),
                "has_georef": entry.get("has_georef"),
                "transform_source": entry.get("transform_source"),
                "paired_tile_id": entry.get("paired_tile_id"),
                "paired_tile_ids": entry.get("paired_tile_ids"),
                "fit": entry.get("fit"),
                "bounds": bounds,
                "corner_coordinates": corners,
                "pixel_to_geo": entry.get("pixel_to_geo"),
                "geo_to_pixel": entry.get("geo_to_pixel"),
                "image_url": f"/api/xbd/images/{tile_id}" if tile_id else None,
                "annotations_url": f"/api/xbd/annotations/{tile_id}" if tile_id else None,
            }

            self._state["map"]["anchor"] = {
                "label": anchor_label,
                "lat": anchor_lat,
                "lon": anchor_lon,
                "source": "xbd_tile_center" if bounds else "fallback",
            }
            self._state["map"]["active_tile_id"] = tile_id
            self._state["map"]["active_tile"] = tile_summary
            self._state["map"]["latlon_bounds"] = bounds
            self._state["map"]["corner_coordinates"] = corners
            self._state["map"]["geo_features"] = [{
                "type": "xbd_tile",
                "tile_id": tile_id,
                "bounds": bounds,
                "stage": entry.get("stage"),
                "disaster": entry.get("disaster"),
                "disaster_type": entry.get("disaster_type"),
            }]

            # Recompute NED offsets for anything that has lat/lon already stored
            for robot in self._state["robots"].values():
                position = robot.get("position")
                if not position or "lat" not in position or "lon" not in position:
                    continue
                north_m, east_m = latlon_to_meters(
                    anchor_lat,
                    anchor_lon,
                    position["lat"],
                    position["lon"],
                )
                position["north_m"] = north_m
                position["east_m"] = east_m
                position["down_m"] = -float(position.get("alt", self._hover_altitude_m))

            for target in self._state["targets"]:
                if "lat" in target and "lon" in target:
                    n, e = latlon_to_meters(anchor_lat, anchor_lon, target["lat"], target["lon"])
                    target["north_m"] = n
                    target["east_m"] = e

            for report in self._state["map"]["reports"]:
                if "lat" in report and "lon" in report:
                    n, e = latlon_to_meters(anchor_lat, anchor_lon, report["lat"], report["lon"])
                    report["north_m"] = n
                    report["east_m"] = e

            self._state["timestamp"] = time.time()
            return copy.deepcopy(self._state)

    # ── read ────────────────────────────────────────────────────
    def get_world_state(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._state)
