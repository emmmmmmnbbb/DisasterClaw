from __future__ import annotations

import threading
import time

from geo import haversine_m, latlon_to_meters, meters_to_latlon


class MockAdapter:
    def __init__(self, anchor_lat: float, anchor_lon: float, hover_altitude_m: float = 30.0):
        self._lock = threading.RLock()
        self._anchor_lat = anchor_lat
        self._anchor_lon = anchor_lon
        self._lat = anchor_lat
        self._lon = anchor_lon
        self._alt = hover_altitude_m
        self._heading_deg = 0.0
        self._battery = 100.0
        self._in_air = True
        self._connected = True
        self._speed_mps = 0.0

    def reset_origin(self, lat: float, lon: float, alt: float | None = None) -> dict:
        """
        切换锚点到新灾区后，把 UAV 的位姿一并拉到新锚点。
        不调此函数的话，激活新瓦片后 UAV 仍会停在老坐标、越过半个地球。
        """
        with self._lock:
            self._anchor_lat = float(lat)
            self._anchor_lon = float(lon)
            self._lat = float(lat)
            self._lon = float(lon)
            if alt is not None:
                self._alt = float(alt)
            self._speed_mps = 0.0
            self._heading_deg = 0.0
            return self.snapshot()

    def snapshot(self) -> dict:
        with self._lock:
            north_m, east_m = latlon_to_meters(self._anchor_lat, self._anchor_lon, self._lat, self._lon)
            return {
                "lat": self._lat,
                "lon": self._lon,
                "alt": self._alt,
                "heading_deg": self._heading_deg,
                "battery": self._battery,
                "in_air": self._in_air,
                "speed_mps": self._speed_mps,
                "north_m": north_m,
                "east_m": east_m,
                "down_m": -self._alt,
            }

    def hover(self, duration: float = 3.0, update_callback=None, stop_event=None) -> dict:
        duration = max(0.5, float(duration))
        ticks = max(1, int(duration / 0.25))
        with self._lock:
            self._speed_mps = 0.0
            start = self.snapshot()
        if update_callback:
            update_callback(start)
        for _ in range(ticks):
            if stop_event and stop_event.is_set():
                return {"success": False, "message": "悬停已中止"}
            time.sleep(duration / ticks)
        if update_callback:
            update_callback(self.snapshot())
        return {"success": True, "message": f"悬停 {duration:.1f}s"}

    def fly_to_geo(self, lat: float, lon: float, alt: float | None = None, speed: float = 14.0, update_callback=None, stop_event=None) -> dict:
        with self._lock:
            start_lat = self._lat
            start_lon = self._lon
            start_alt = self._alt
        target_alt = start_alt if alt is None else float(alt)
        distance = haversine_m(start_lat, start_lon, lat, lon)
        vertical = abs(target_alt - start_alt)
        total_distance = distance + vertical
        speed = max(4.0, float(speed))
        duration = max(0.8, min(total_distance / speed, 8.0))
        steps = max(8, int(duration / 0.12))

        for index in range(1, steps + 1):
            if stop_event and stop_event.is_set():
                with self._lock:
                    self._speed_mps = 0.0
                return {"success": False, "message": "飞行已中止"}
            ratio = index / steps
            with self._lock:
                self._lat = start_lat + (lat - start_lat) * ratio
                self._lon = start_lon + (lon - start_lon) * ratio
                self._alt = start_alt + (target_alt - start_alt) * ratio
                self._speed_mps = speed
                self._heading_deg = _bearing_deg(start_lat, start_lon, lat, lon)
                snap = self.snapshot()
            if update_callback:
                update_callback(snap)
            time.sleep(duration / steps)

        with self._lock:
            self._speed_mps = 0.0
            snap = self.snapshot()
        if update_callback:
            update_callback(snap)
        return {
            "success": True,
            "message": f"已飞抵 ({lat:.6f}, {lon:.6f}) @ {target_alt:.1f}m",
            "data": snap,
        }

    def fly_relative(self, north_m: float, east_m: float, up_m: float = 0.0, speed: float = 12.0, update_callback=None, stop_event=None) -> dict:
        with self._lock:
            target_lat, target_lon = meters_to_latlon(self._lat, self._lon, north_m, east_m)
            target_alt = max(5.0, self._alt + up_m)
        return self.fly_to_geo(
            target_lat,
            target_lon,
            alt=target_alt,
            speed=speed,
            update_callback=update_callback,
            stop_event=stop_event,
        )


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta = math.radians(lon2 - lon1)
    y = math.sin(delta) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
