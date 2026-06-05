from __future__ import annotations

import math

TILE_SIZE = 256
EARTH_RADIUS_M = 6378137.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def meters_to_latlon(origin_lat: float, origin_lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    lat = origin_lat + (north_m / 110540.0)
    lon = origin_lon + (east_m / (111320.0 * math.cos(math.radians(origin_lat))))
    return lat, lon


def latlon_to_meters(origin_lat: float, origin_lon: float, lat: float, lon: float) -> tuple[float, float]:
    north = (lat - origin_lat) * 110540.0
    east = (lon - origin_lon) * 111320.0 * math.cos(math.radians(origin_lat))
    return north, east


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def normalize_tile_x(z: int, x: int) -> int:
    tile_count = 1 << z
    return ((x % tile_count) + tile_count) % tile_count


def latlon_to_world_pixels(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    scale = TILE_SIZE * (1 << zoom)
    lat = clamp(lat, -85.05112878, 85.05112878)
    x = (lon + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(lat))
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    return x, y


def world_pixels_to_latlon(x: float, y: float, zoom: int) -> tuple[float, float]:
    scale = TILE_SIZE * (1 << zoom)
    lon = x / scale * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * y / scale
    lat = math.degrees(math.atan(math.sinh(n)))
    return lat, lon


def tile_bounds(z: int, x: int, y: int) -> dict[str, float]:
    left = x * TILE_SIZE
    top = y * TILE_SIZE
    right = (x + 1) * TILE_SIZE
    bottom = (y + 1) * TILE_SIZE
    north, west = world_pixels_to_latlon(left, top, z)
    south, east = world_pixels_to_latlon(right, bottom, z)
    return {
        "north": north,
        "south": south,
        "west": west,
        "east": east,
    }

