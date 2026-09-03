from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import config as llm_config
import xbd_store
from ai_planner import TaskPlanner
from geo import latlon_to_meters, meters_to_latlon
from mock_adapter import MockAdapter
from perception import (
    PERCEPTION_OUTPUT_DIR,
    PerceptionResult,
    get_perception,
    level_for_risk,
)
from semantic_map import SemanticMap
from stmr_matrix import build_stmr
from hspm_planner import HspmConfig, HspmNavigator, OroiScoreWeights
from recheck import EVIDENCE_CLASSES, RecheckConfig, RecheckController
from memory_graph import MemoryGraph, text_match_scorer
from vlm_analyzer import VLMAnalyzer
from agent_vqa import (
    AgentVqaConfig,
    AgentVqaController,
    QuestionSpec,
    build_evidence_from_perception,
    parse_question,
    parse_vlm_json_output,
)
from vln_navigator import (
    GroundHit,
    Observation,
    VlnConfig,
    VlnNavigator,
    ground_with_yolo,
    parse_ground_xy,
)
import fov_ladder
from world import DEFAULT_BASEMAP, WorldModel
from xbd_map import build_annotation_geojson

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("disasterclaw.app")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
FALLBACK_ANCHOR = {"lat": 31.2304, "lon": 121.4737, "label": "Shanghai Fallback Anchor"}
# 改动一（视场收缩式降高）重标定高度尺度：固定 1024 px 传感器 + 60° 视场角下，
# 「最小视场 = 恰好一整张 xBD 瓦片(512 m)」唯一确定下限高度 443.4 m（原生 0.5 m/px），
# 3×3 瓦片巡航对应 1330.2 m（1.5 m/px）。旧的 30 m 巡航 / 0.5 m·px⁻¹ 在物理上差两个
# 数量级（review2 B2），重标定后 0.5 m/px 与 443 m 自洽。
DEFAULT_HOVER_ALTITUDE_M = float(
    os.getenv("DEFAULT_HOVER_ALTITUDE_M", str(fov_ladder.alt_cruise_m()))
)
# 默认初始瓦片：
#   xbd 模式     → palu-tsunami_00000118_post_disaster  (~1540 destroyed，1024x1024 卫星)
#   rescuenet 模式 → rescuenet_12215_post_disaster       (4 destroyed + 14 major，4000x3000 无人机高清)
# 可通过 XBD_DEFAULT_TILE 环境变量显式覆盖。
_DATASET_MODE = (os.getenv("DATASET_MODE") or "xbd").strip().lower() or "xbd"
if _DATASET_MODE == "rescuenet":
    _MODE_DEFAULT_TILE = "rescuenet_12215_post_disaster"
    _MODE_DEFAULT_DISASTER = "hurricane-michael"
else:
    _MODE_DEFAULT_TILE = "palu-tsunami_00000118_post_disaster"
    _MODE_DEFAULT_DISASTER = "palu-tsunami"

DEFAULT_TILE_ID = os.getenv("XBD_DEFAULT_TILE", _MODE_DEFAULT_TILE)
DEFAULT_DISASTER = os.getenv("XBD_DEFAULT_DISASTER", _MODE_DEFAULT_DISASTER)
DEFAULT_STAGE = os.getenv("XBD_DEFAULT_STAGE", "post")
ELEVATION_URL = os.getenv("XBD_ELEVATION_URL", "https://api.open-meteo.com/v1/elevation")
ELEVATION_TIMEOUT = float(os.getenv("XBD_ELEVATION_TIMEOUT", "4"))
ELEVATION_DISABLED = os.getenv("XBD_ELEVATION_DISABLE", "0").lower() in {"1", "true", "yes", "on"}

app = Flask(__name__, static_folder=str(FRONTEND_DIST), static_url_path="")
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "disasterclaw-dev")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "12")) * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/socket.io/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", allow_unsafe_werkzeug=True)


def _resolved_module_settings(module_name: str) -> dict:
    module_cfg = llm_config.MODULE_CONFIG.get(module_name, {})
    provider = module_cfg.get("provider") or llm_config.ACTIVE_PROVIDER
    model = module_cfg.get("model") or llm_config.PROVIDERS.get(provider, {}).get("default_model", "")
    return {"provider": provider, "model": model}


def _pick_initial_anchor() -> tuple[dict, dict | None]:
    """
    返回 (anchor_dict, initial_tile_entry)。

    优先级：
      1. DEFAULT_TILE_ID（XBD_DEFAULT_TILE 环境变量）指定的瓦片，必须 has_georef 才使用
      2. manifest 的默认灾区 (DEFAULT_DISASTER/STAGE) 首张 has_georef 瓦片
      3. 任意 has_georef POST 瓦片
      4. 回落上海（FALLBACK_ANCHOR）
    """
    entry: dict | None = None
    try:
        if DEFAULT_TILE_ID:
            candidate = xbd_store.get_entry(DEFAULT_TILE_ID)
            if candidate and candidate.get("has_georef"):
                # POST_ONLY 下仍要保证候选是 POST，防止用户误配 PRE id
                if (not xbd_store.POST_ONLY_MODE) or xbd_store._is_post(candidate):
                    entry = candidate
                else:
                    logger.warning(
                        "XBD_DEFAULT_TILE %s is pre_disaster; POST_ONLY_MODE on, falling back.",
                        DEFAULT_TILE_ID,
                    )
            elif candidate:
                logger.warning(
                    "XBD_DEFAULT_TILE %s lacks georef, falling back.", DEFAULT_TILE_ID,
                )
            else:
                logger.warning(
                    "XBD_DEFAULT_TILE %s not found in manifest, falling back.", DEFAULT_TILE_ID,
                )
        if entry is None:
            entry = xbd_store.first_georef_entry(DEFAULT_DISASTER, DEFAULT_STAGE)
        if entry is None:
            entry = xbd_store.first_georef_entry()
    except Exception as exc:
        logger.warning("unable to load xBD manifest for initial anchor: %s", exc)
        entry = None

    if entry and entry.get("bounds"):
        bounds = entry["bounds"]
        return {
            "lat": (float(bounds["north"]) + float(bounds["south"])) * 0.5,
            "lon": (float(bounds["east"]) + float(bounds["west"])) * 0.5,
            "label": f"{entry.get('disaster') or 'xBD'} · {entry.get('tile_id')}",
        }, entry
    return dict(FALLBACK_ANCHOR), None


class AppState:
    def __init__(self):
        self.mode = "manual"
        self.current_robot = "UAV_1"
        self.is_executing = False
        self.initialized = True
        self.hover_altitude_m = DEFAULT_HOVER_ALTITUDE_M

        initial_anchor, initial_tile = _pick_initial_anchor()
        self.world = WorldModel(
            initial_anchor["lat"],
            initial_anchor["lon"],
            self.hover_altitude_m,
            anchor_label=initial_anchor["label"],
            basemap=DEFAULT_BASEMAP,
        )
        self.world.register_default_uav(self.current_robot)
        self.adapter = MockAdapter(initial_anchor["lat"], initial_anchor["lon"], self.hover_altitude_m)
        self.planner = TaskPlanner(self.hover_altitude_m)
        self.log_buffer: list[dict] = []
        self._log_lock = threading.Lock()
        self._task_lock = threading.Lock()
        self._stop_event = threading.Event()
        # P0：当前 episode 的 2D 地理语义地图（run_vln_episode 开始时重建；None 表示无活动地图）
        self.semantic_map: SemanticMap | None = None
        self._sync_world_from_adapter()

        summary = xbd_store.summary()
        if summary:
            self.push_log(
                "success",
                f"xBD manifest loaded: {summary.get('tiles', 0)} tiles, "
                f"{summary.get('has_georef', 0)} georef",
            )
        else:
            self.push_log(
                "warn",
                "xBD manifest not found — map will use fallback anchor only "
                "(build manifest.json under backend/data/xbd/).",
            )

        if initial_tile:
            self.world.set_active_tile(initial_tile)
            self._sync_world_from_adapter()
            self.push_log(
                "info",
                f"Initial xBD tile: {initial_tile.get('tile_id')} "
                f"({initial_tile.get('disaster')} / {initial_tile.get('stage')})",
            )

        self.push_log("success", "DisasterClaw ready: xBD-backed real-world map + mock hover")
        planner_cfg = _resolved_module_settings("planner")
        vlm_cfg = _resolved_module_settings("vlm")
        self.push_log("info", f"Planner LLM: {planner_cfg['provider']} / {planner_cfg['model']}")
        self.push_log("info", f"Vision VLM: {vlm_cfg['provider']} / {vlm_cfg['model']}")

        # 在任何 warmup 线程启动前，主线程同步完整导入一次重型 ML 库，避免
        # qwen-vl-warmup 与 perception-warmup 两线程并发首次 import 时撞上
        # accelerate 的循环导入（"partially initialized module"），导致 VLM /
        # SegFormer 永久加载失败。
        self._eager_import_ml_libs()

        self._prime_ml_imports()
        self._warmup_local_qwen_vl([planner_cfg, vlm_cfg])
        self._warmup_perception()

    def _prime_ml_imports(self) -> None:
        """Resolve transformers/accelerate lazy submodules once, synchronously,
        in the main thread.

        transformers 5.x uses lazy `_LazyModule` loading. When the Qwen-VL and
        perception warmup threads both perform the *first* import concurrently,
        Python hits a "partially initialized module" circular import inside
        `accelerate.hooks` and both model stacks fail to load. Importing the heavy
        symbols here (before any warmup thread starts) makes the threaded imports
        pure cache hits, eliminating the race."""
        if os.getenv("PRIME_ML_IMPORTS", "1").lower() in {"0", "false", "no", "off"}:
            return
        try:
            import accelerate  # noqa: F401
            from accelerate.hooks import AlignDevicesHook, add_hook_to_module  # noqa: F401
            import transformers  # noqa: F401
            from transformers import (  # noqa: F401
                AutoModelForImageTextToText,
                AutoProcessor,
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
        except Exception as exc:
            logger.warning("ML import priming failed (warmups may race): %s", exc)

    def _eager_import_ml_libs(self) -> None:
        """主线程同步预导入 transformers / accelerate，消除并发首次导入的循环导入竞态。

        逐项触发会形成循环依赖链的子模块（accelerate.hooks / big_modeling、
        transformers.generation），确保它们在 sys.modules 中被完整初始化。
        失败只记录告警，不阻断启动（YOLO 仍可用）。
        """
        t0 = time.time()
        try:
            import accelerate  # noqa: F401
            from accelerate.hooks import AlignDevicesHook, add_hook_to_module  # noqa: F401
            from accelerate.big_modeling import dispatch_model  # noqa: F401
            import transformers  # noqa: F401
            from transformers.generation import GenerationMixin  # noqa: F401
            from transformers import (  # noqa: F401
                AutoModelForImageTextToText,
                AutoProcessor,
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
            # 同步修正 segformer_tool 可能在早期并发下捕获到的 HAS_TRANSFORMERS=False
            try:
                import sys as _sys
                _seg = _sys.modules.get("segformer_tool")
                if _seg is not None:
                    _seg.HAS_TRANSFORMERS = True
            except Exception:
                pass
            self.push_log(
                "success",
                f"ML libs preloaded (transformers {transformers.__version__}, "
                f"accelerate {accelerate.__version__}) in {time.time() - t0:.1f}s",
                {"module": "startup"},
            )
        except Exception as exc:
            logger.exception("eager ML lib preload failed")
            self.push_log(
                "warn",
                f"预导入 transformers/accelerate 失败（VLM/SegFormer 可能不可用，YOLO 仍可用）: {exc}",
                {"module": "startup"},
            )

    def _warmup_perception(self) -> None:
        """YOLO + SegFormer 懒加载线程。
        transformers 惰性加载已由 `_prime_ml_imports` 在主线程预解析，
        此处线程内导入均为缓存命中，不再与 Qwen-VL warmup 竞争。"""
        if os.getenv("PERCEPTION_WARMUP", "1").lower() in {"0", "false", "no", "off"}:
            return

        warmup_delay = float(os.getenv("PERCEPTION_WARMUP_DELAY_S", "15"))

        def _worker() -> None:
            if warmup_delay > 0:
                time.sleep(warmup_delay)
            perception = get_perception()
            try:
                self.push_log("info", "Loading perception models (YOLO + SegFormer)")
                perception.load()
                self.push_log(
                    "success",
                    "Perception ready: YOLO + SegFormer + SceneDescriptor",
                    {"module": "perception"},
                )
            except Exception as exc:
                logger.exception("perception warmup failed")
                self.push_log(
                    "error",
                    f"Perception 模型加载失败（detect_disaster 将不可用）: {exc}",
                    {"module": "perception"},
                )

        threading.Thread(target=_worker, daemon=True, name="perception-warmup").start()

    def _warmup_local_qwen_vl(self, module_cfgs: list[dict]) -> None:
        """If any active module uses the local Qwen-VL backend, load it in a
        background thread so the first AI task doesn't freeze the socket for
        ~2 minutes."""
        wants_local = any(cfg.get("provider") == "qwen_vl_local" for cfg in module_cfgs)
        if not wants_local:
            return

        def _worker() -> None:
            try:
                provider_cfg = dict(llm_config.PROVIDERS["qwen_vl_local"])
                from local_qwen_vl import get_local_qwen_vl_backend
                backend = get_local_qwen_vl_backend(provider_cfg)
                self.push_log("info", f"Loading local Qwen-VL: {provider_cfg.get('model_id')}")
                backend.load()
                self.push_log(
                    "success",
                    f"Local Qwen-VL ready on {backend.device}",
                    {"module": "llm"},
                )
            except Exception as exc:
                logger.exception("local Qwen-VL warmup failed")
                self.push_log("error", f"Local Qwen-VL load failed: {exc}")

        threading.Thread(target=_worker, daemon=True, name="qwen-vl-warmup").start()

    def push_log(self, level: str, msg: str, extra: dict | None = None) -> None:
        entry = {
            "ts": round(time.time() * 1000),
            "level": level,
            "msg": msg,
            **(extra or {}),
        }
        with self._log_lock:
            self.log_buffer.append(entry)
            if len(self.log_buffer) > 300:
                self.log_buffer.pop(0)
        socketio.emit("log", entry)

    def system_status(self) -> dict:
        return {
            "initialized": self.initialized,
            "mode": self.mode,
            "current_robot": self.current_robot,
            "is_executing": self.is_executing,
            "hover_altitude_m": self.hover_altitude_m,
            "anchor": self.world.get_world_state()["map"]["anchor"],
        }

    def world_state(self) -> dict:
        return self.world.get_world_state()

    def _sync_world_from_adapter(self) -> None:
        snap = self.adapter.snapshot()
        self.world.update_robot(
            self.current_robot,
            {
                "status": "airborne",
                "task_state": "busy" if self.is_executing else "idle",
                "battery": snap["battery"],
                "in_air": snap["in_air"],
                "heading_deg": snap["heading_deg"],
                "speed_mps": snap["speed_mps"],
                "position": {
                    "lat": snap["lat"],
                    "lon": snap["lon"],
                    "alt": snap["alt"],
                },
            },
        )

    def activate_xbd_tile(self, entry: dict) -> dict:
        """切换地图活动瓦片：更新 world model、adapter 原点、UAV 位置。"""
        self.world.set_active_tile(entry)
        bounds = entry.get("bounds") or {}
        anchor_lat = self.world.anchor_lat
        anchor_lon = self.world.anchor_lon
        if bounds:
            anchor_lat = (float(bounds["north"]) + float(bounds["south"])) * 0.5
            anchor_lon = (float(bounds["east"]) + float(bounds["west"])) * 0.5
        self.adapter.reset_origin(anchor_lat, anchor_lon, alt=self.hover_altitude_m)
        self._sync_world_from_adapter()
        return self.world_state()

    def emit_world(self) -> None:
        socketio.emit("world_state", self.world_state())

    def emit_system_status(self) -> None:
        socketio.emit("system_status", self.system_status())


state = AppState()


def on_position_update(_snap: dict) -> None:
    state._sync_world_from_adapter()
    state.emit_world()


def start_background_job(target, *args):
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


_PARAM_ALIASES = {
    "latitude": "lat",
    "lat_deg": "lat",
    "longitude": "lon",
    "lng": "lon",
    "long": "lon",
    "altitude": "alt",
    "altitude_m": "alt",
    "height": "alt",
    "height_m": "alt",
    "duration_s": "duration",
    "duration_sec": "duration",
    "seconds": "duration",
    "north": "north_m",
    "east": "east_m",
    "up": "up_m",
    "description": "content",
    "report": "content",
    "message": "content",
    "name": "label",
    "severity": "level",
}


def _normalize_params(params: dict | None) -> dict:
    if not isinstance(params, dict):
        return {}
    normalized: dict = {}
    for key, value in params.items():
        target = _PARAM_ALIASES.get(key, key)
        normalized.setdefault(target, value)
    return normalized


def _align_active_tile_for(
    lat: float,
    lon: float,
    *,
    source: str = "manual",
    strict_post: bool = False,  # 保留参数兼容旧调用，POST_ONLY 下恒等于 True
) -> dict | None:
    """
    确保目标点 (lat, lon) 落在激活的 POST 灾后瓦片内。

    POST_ONLY_MODE 下全流程只搜 POST 瓦片，未命中直接返回 None（不再 fallback PRE）。

    返回最终激活瓦片 entry（或 None 表示未命中/无覆盖）。
    """
    try:
        world = state.world_state()
    except Exception:
        world = {}
    active = (world.get("map") or {}).get("active_tile") or {}

    if active and xbd_store.tile_contains(active, lat, lon):
        return active

    entry = xbd_store.find_tile_containing(
        lat, lon, stage_priority=("post_disaster",)
    )
    if entry is None:
        state.push_log(
            "warn",
            f"目标 ({lat:.6f}, {lon:.6f}) 不在任何 POST 灾后瓦片覆盖范围内。",
            {"module": "tile_align", "source": source},
        )
        return None

    if entry.get("tile_id") != active.get("tile_id"):
        try:
            state.activate_xbd_tile(entry)
        except Exception as exc:  # 防御性：切瓦片失败时保留原状态
            logger.warning("activate_xbd_tile failed: %s", exc)
            state.push_log(
                "warn",
                f"自动切换瓦片失败: {exc}",
                {"module": "tile_align", "source": source},
            )
            return None

        state.push_log(
            "info",
            f"自动切换到瓦片 {entry['tile_id']} ({entry.get('stage')}) 以覆盖目标点。",
            {"module": "tile_align", "source": source},
        )
    return entry


def execute_action(action: str, params: dict, source: str = "manual") -> dict:
    params = _normalize_params(params)
    if action == "hover":
        return state.adapter.hover(
            duration=float(params.get("duration", 3.0)),
            update_callback=on_position_update,
            stop_event=state._stop_event,
        )
    if action == "fly_to_geo":
        target_lat = float(params["lat"])
        target_lon = float(params["lon"])
        _align_active_tile_for(target_lat, target_lon, source=source)
        return state.adapter.fly_to_geo(
            lat=target_lat,
            lon=target_lon,
            alt=float(params.get("alt", state.hover_altitude_m)),
            speed=float(params.get("speed", 14.0)),
            update_callback=on_position_update,
            stop_event=state._stop_event,
        )
    if action == "fly_relative":
        north_m = float(params.get("north_m", 0.0))
        east_m = float(params.get("east_m", 0.0))
        up_m = float(params.get("up_m", 0.0))
        snap = state.adapter.snapshot()
        target_lat, target_lon = meters_to_latlon(
            float(snap["lat"]), float(snap["lon"]), north_m, east_m
        )
        _align_active_tile_for(target_lat, target_lon, source=source)
        return state.adapter.fly_relative(
            north_m=north_m,
            east_m=east_m,
            up_m=up_m,
            speed=float(params.get("speed", 12.0)),
            update_callback=on_position_update,
            stop_event=state._stop_event,
        )
    if action == "mark_target":
        target = state.world.add_target(
            label=params.get("label", "Map Marker"),
            lat=float(params.get("lat", state.adapter.snapshot()["lat"])),
            lon=float(params.get("lon", state.adapter.snapshot()["lon"])),
            alt=float(params.get("alt", 0.0)),
            kind=params.get("kind", "poi"),
            source=source,
        )
        state.emit_world()
        return {"success": True, "message": f"已标记 {target['label']}", "data": target}
    if action == "report_observation":
        snap = state.adapter.snapshot()
        report = state.world.add_report(
            content=params.get("content", "Mock observation report"),
            lat=float(params.get("lat", snap["lat"])),
            lon=float(params.get("lon", snap["lon"])),
            level=params.get("level", "info"),
            source=source,
        )
        state.emit_world()
        return {"success": True, "message": "已写入观察报告", "data": report}
    if action == "detect_disaster":
        return _execute_detect_disaster(params, source)
    return {"success": False, "message": f"未知动作: {action}"}


def _execute_detect_disaster(params: dict, source: str) -> dict:
    """
    从活动 xBD 瓦片按 UAV 当前 (lat,lon,alt) 裁视场 →
    YOLO + SegFormer + SceneDescriptor → 可选 VLM 总结 →
    push_log + add_report + socket perception_result。
    """
    perception = get_perception()

    snap = state.adapter.snapshot()
    # 严格 POST-only：detect_disaster 要求 UAV 当前位置落在 POST 灾后瓦片覆盖范围内。
    # 不允许静默回退到 PRE（否则 YOLO/Seg 看不到任何损伤）。
    aligned = _align_active_tile_for(
        float(snap["lat"]),
        float(snap["lon"]),
        source=source,
        strict_post=True,
    )
    world = state.world_state()
    active_tile = (world.get("map") or {}).get("active_tile") or {}

    # manifest 里 stage 是 "post"/"pre"，必须用 alias-aware 比较，避免静默 bug
    if aligned is None or not xbd_store._is_post(active_tile):
        state.push_log(
            "error",
            f"detect_disaster 拒绝执行：UAV 当前位置 "
            f"({snap['lat']:.6f}, {snap['lon']:.6f}) 不在任何 POST 灾后瓦片覆盖范围内。"
            " 请先在地图上选中灾后覆盖区（POST 足迹）内的点。",
            {"module": "perception", "source": source, "no_post_coverage": True},
        )
        return {
            "success": False,
            "message": "当前位置无 POST 灾后瓦片覆盖，detect_disaster 已中止。",
        }

    if not active_tile or not active_tile.get("tile_id"):
        state.push_log("error", "detect_disaster: 当前未激活任何 xBD 瓦片，无法裁视场。")
        return {"success": False, "message": "未激活任何 xBD 瓦片"}

    if not perception.is_available:
        try:
            perception.load()
        except Exception as exc:
            state.push_log("error", f"detect_disaster: 视觉模型未就绪 — {exc}")
            return {"success": False, "message": f"视觉模型未就绪: {exc}"}

    patch_id = f"uav-{int(time.time() * 1000)}"
    state.push_log(
        "info",
        f"detect_disaster: 在 ({snap['lat']:.6f}, {snap['lon']:.6f}) @ "
        f"{snap['alt']:.1f}m 裁视场并运行 YOLO + SegFormer ...",
    )
    try:
        result: PerceptionResult = perception.perceive_at(
            lat=float(snap["lat"]),
            lon=float(snap["lon"]),
            alt=float(snap["alt"]),
            active_tile=active_tile,
            patch_id=patch_id,
        )
    except Exception as exc:
        logger.exception("perception pipeline failed")
        state.push_log("error", f"detect_disaster 失败: {exc}")
        return {"success": False, "message": f"感知失败: {exc}"}

    if result.degraded and result.degraded_reason:
        state.push_log("warn", f"detect_disaster: {result.degraded_reason}")

    det_counts = result.detection.get("class_counts") or {}
    seg_stats = result.segmentation.get("stats") or {}
    top_seg = sorted(seg_stats.items(), key=lambda kv: kv[1], reverse=True)[:3]
    seg_top_text = ", ".join(f"{k} {v}" for k, v in top_seg) if top_seg else "无"
    det_text = (
        ", ".join(f"{k}:{v}" for k, v in det_counts.items())
        if det_counts
        else "无目标"
    )
    state.push_log(
        "info",
        f"视觉感知: YOLO {result.detection.get('num_objects', 0)} 个目标 ({det_text}); "
        f"SegFormer top3: {seg_top_text}",
    )
    state.push_log(level_for_risk(result.risk_level), f"灾情判定: {result.risk_summary}")

    # P0：若当前有活动语义地图（如 VLN episode 中的人工巡检），也把这次观测并入。
    smap = getattr(state, "semantic_map", None)
    if smap is not None:
        try:
            smap.mark_observation(
                uav_lat=float(snap["lat"]),
                uav_lon=float(snap["lon"]),
                radius_m=float(result.patch_radius_m),
                detections=result.detection.get("detections", []),
                degraded=bool(result.degraded),
                risk_level=result.risk_level,
                patch_width=int(result.patch_width),
                patch_height=int(result.patch_height),
            )
            socketio.emit("semantic_map", smap.snapshot())
        except Exception as exc:
            logger.warning("semantic_map update (detect_disaster) failed: %s", exc)

    # 可选：Qwen-VL 基于 patch + scene_text 出自然语言结论
    vlm_text = ""
    use_vlm = str(params.get("use_vlm_summary", True)).lower() not in {"0", "false", "no"}
    if use_vlm:
        try:
            with open(result.patch_path, "rb") as f:
                img_bytes = f.read()
            prompt = (
                "以下是 UAV 当前视场的结构化感知结果，请结合图像用中文两三句话概括："
                "场景整体情况、是否受灾、建议动作。如果结果不足以判断，请明示。\n\n"
                + result.scene_text
            )
            vlm_result = VLMAnalyzer().analyze_image_bytes(
                image_bytes=img_bytes, mime_type="image/png", prompt=prompt, max_tokens=360
            )
            vlm_text = (vlm_result.get("analysis") or "").strip()
            if vlm_text:
                state.push_log(
                    "success",
                    f"VLM 结论: {vlm_text[:140]}{'…' if len(vlm_text) > 140 else ''}",
                    {"module": "vlm"},
                )
        except Exception as exc:
            logger.warning("VLM summary after detect_disaster failed: %s", exc)
            state.push_log("warn", f"VLM 结论生成失败: {exc}")

    # 写入 world report，让地图出一个标记点
    report_content = result.risk_summary
    if vlm_text:
        report_content = f"{report_content} | VLM: {vlm_text[:160]}"
    if result.degraded:
        report_content = f"[degraded] {report_content}"
    report = state.world.add_report(
        content=report_content,
        lat=float(snap["lat"]),
        lon=float(snap["lon"]),
        level=level_for_risk(result.risk_level),
        source="perception",
    )

    payload = {
        "patch_id": result.patch_id,
        "patch_url": result.patch_url,
        "overlay_url": result.overlay_url,
        "detection_url": result.detection_url,
        "patch_width": result.patch_width,
        "patch_height": result.patch_height,
        "radius_m": result.patch_radius_m,
        "risk_level": result.risk_level,
        "risk_summary": result.risk_summary,
        "damaged_buildings": result.damaged_buildings,
        "intact_buildings": result.intact_buildings,
        "vehicles": result.vehicles,
        "detection": {
            "num_objects": result.detection.get("num_objects", 0),
            "class_counts": det_counts,
            "detections": result.detection.get("detections", [])[:50],
        },
        "segmentation": {
            "num_labels": result.segmentation.get("num_labels", 0),
            "stats": seg_stats,
        },
        "scene": result.scene_dict,
        "scene_text": result.scene_text,
        "vlm_summary": vlm_text,
        "position": {
            "lat": snap["lat"],
            "lon": snap["lon"],
            "alt": snap["alt"],
        },
        "report_id": report.get("id"),
        "degraded": result.degraded,
        "degraded_reason": result.degraded_reason or "",
        "active_tile_id": active_tile.get("tile_id"),
        "active_tile_stage": active_tile.get("stage"),
        "source": source,
        "timestamp": int(time.time() * 1000),
        "timings": dict(result.extras or {}),
    }
    socketio.emit("perception_result", payload)
    state.emit_world()
    return {
        "success": True,
        "message": result.risk_summary,
        "data": payload,
    }


def run_action_sequence(label: str, steps: list[dict], source: str, raw_task: str = "") -> None:
    if not state._task_lock.acquire(blocking=False):
        state.push_log("warn", "当前已有任务在执行，忽略新的请求")
        return

    try:
        state.is_executing = True
        state._stop_event.clear()
        state._sync_world_from_adapter()
        state.emit_system_status()
        state.emit_world()
        state.push_log("info", f"开始执行任务: {label}")

        # bench: emit a task_started event so external benchmarks can align
        # the wall clock with `ai_execution_report` without parsing logs.
        task_started_ns = time.time_ns()
        socketio.emit(
            "task_started",
            {
                "label": label,
                "raw_task": raw_task,
                "source": source,
                "step_count": len(steps),
                "ts_ns": task_started_ns,
                "ts_ms": task_started_ns // 1_000_000,
            },
        )

        executed = []
        for index, step in enumerate(steps, start=1):
            if state._stop_event.is_set():
                state.push_log("warn", "任务已停止")
                break
            action = step.get("action", "")
            params = step.get("params", {})
            reason = step.get("reason", "")
            state.push_log("info", f"[{index}/{len(steps)}] {action} — {reason}")
            socketio.emit(
                "ai_thought",
                {
                    "iteration": index,
                    "skill": action,
                    "thinking": reason,
                    "progress": f"step {index}/{len(steps)}",
                },
            )
            socketio.emit(
                "ai_thinking",
                {
                    "phase": "executing",
                    "detail": f"{action}",
                    "iteration": index,
                    "action": {"skill": action, "parameters": params},
                    "decision": reason,
                },
            )
            step_start_ns = time.time_ns()
            result = execute_action(action, params, source=source)
            step_end_ns = time.time_ns()
            step_wall_ms = (step_end_ns - step_start_ns) // 1_000_000
            executed.append({
                "action": action,
                "params": params,
                "result": result,
                "wall_ms": step_wall_ms,
                "step_start_ns": step_start_ns,
                "step_end_ns": step_end_ns,
            })
            socketio.emit(
                "action_result",
                {
                    "action": action,
                    "params": params,
                    "result": result,
                    "wall_ms": step_wall_ms,
                    "step_index": index,
                    "step_start_ns": step_start_ns,
                    "step_end_ns": step_end_ns,
                },
            )
            level = "success" if result.get("success") else "error"
            state.push_log(level, f"{action}: {result.get('message', '')}")
            if not result.get("success"):
                break

        final_snap = state.adapter.snapshot()
        succeeded = not state._stop_event.is_set() and all(
            step["result"].get("success") for step in executed
        )
        status_txt = "已完成" if succeeded else ("已停止" if state._stop_event.is_set() else "部分失败")
        final_summary = (
            f"任务{status_txt}: {label}；共执行 {len(executed)}/{len(steps)} 步；"
            f"UAV 当前位于 ({final_snap['lat']:.6f}, {final_snap['lon']:.6f}) "
            f"@ {final_snap['alt']:.1f}m"
        )
        state.push_log("success" if succeeded else "warn", final_summary)
        finished_ns = time.time_ns()
        report = {
            "ok": succeeded,
            "task": raw_task or label,
            "summary": final_summary,
            "steps": executed,
            "ts_ns": finished_ns,
            "ts_ms": finished_ns // 1_000_000,
            "task_started_ns": task_started_ns,
            "wall_ms": (finished_ns - task_started_ns) // 1_000_000,
        }
        # benchmarks (manual + ai source) need the report; UI only listens for ai
        if source == "ai":
            socketio.emit("ai_execution_report", report)
            socketio.emit("ai_thinking", {"phase": "idle", "detail": ""})
        else:
            # mirror to a generic channel for benchmark / manual tooling
            socketio.emit("execution_report", report)
    finally:
        state.is_executing = False
        state._sync_world_from_adapter()
        state.emit_system_status()
        state.emit_world()
        state._task_lock.release()


# ───────────────────────── VLN 语言目标导航闭环 ──────────────────────────
#
# 与 run_action_sequence（一次性出 plan 再顺序执行）不同，VLN 是"依观测决策"
# 的闭环：每步 perceive_at 裁俯视视场 → VlnNavigator 对指令目标做 grounding →
# 朝目标质心步进飞行 → 直到到达或步数预算耗尽。复用同一套 socket 事件，
# 前端感知/轨迹/日志/报告面板无需改动。

VLN_STEP_BUDGET = int(os.getenv("VLN_STEP_BUDGET", "12"))
VLN_ARRIVAL_RADIUS_M = float(os.getenv("VLN_ARRIVAL_RADIUS_M", "35"))
VLN_MAX_STEP_M = float(os.getenv("VLN_MAX_STEP_M", "80"))
VLN_EXPLORE_STEP_M = float(os.getenv("VLN_EXPLORE_STEP_M", "90"))
VLN_USE_LLM_STOP = os.getenv("VLN_USE_LLM_STOP", "0").lower() in {"1", "true", "yes", "on"}
# grounding 后端：vlm（默认，用 VLM 判读，开放词汇）/ yolo（仅 YOLO 类别）/ hybrid（YOLO 命中优先，否则 VLM）。
VLN_GROUNDER = (os.getenv("VLN_GROUNDER", "vlm") or "vlm").strip().lower()
# P1：规划器后端。legacy（默认，贪心朝质心 + 螺旋探索）/ hspm（CityNavAgent 式
# landmark→OROI→motion 三层 + STMR 文字矩阵驱动的 LLM 常识推理）。
VLN_PLANNER = (os.getenv("VLN_PLANNER", "legacy") or "legacy").strip().lower()
# C3（非 headline，P4.5「B1 + OROI-Score」消融行）：HSPM 未见子目标时的方位选择从
# "LLM 自由选一个"换成"LLM打分+方向先验+未探索区域增益"三路信号融合，默认关。
VLN_OROI_SCORE = os.getenv("VLN_OROI_SCORE", "0").strip().lower() in {"1", "true", "yes", "on"}
# E12 扩展：三路信号权重可调（默认对齐 OroiScoreWeights 的 0.5/0.2/0.3）。把三个权重
# 设成 (0,0,1) 即退化成经典 Frontier-Based Exploration（Yamauchi 1997：永远朝"未探索
# 覆盖增益"最大的方向走）——作为 E12 除"自由选择/打分融合"外的第三档、来自主流探索
# 文献的标准基线，而不是自造的对照。
VLN_OROI_W_LLM = float(os.getenv("VLN_OROI_W_LLM", "0.5"))
VLN_OROI_W_PRIOR = float(os.getenv("VLN_OROI_W_PRIOR", "0.2"))
VLN_OROI_W_FRONTIER = float(os.getenv("VLN_OROI_W_FRONTIER", "0.3"))
# HSPM 的 STMR 文字矩阵窗口（米）与网格数；越大视野越广但 token 越多。
HSPM_STMR_WINDOW_M = float(os.getenv("HSPM_STMR_WINDOW_M", "200"))
HSPM_STMR_GRID_N = int(os.getenv("HSPM_STMR_GRID_N", "20"))
# P2：灾情不确定性驱动主动复核（默认关，VLN_RECHECK=1 开）。
VLN_RECHECK = os.getenv("VLN_RECHECK", "0").lower() in {"1", "true", "yes", "on"}
# 已知问题修复（E11 实测发现，2026-07）：巡航高度若等于复核高度下限，recheck.py 里
# `alt <= alt_min_m` 从 episode 一开始就恒真——复核一旦触发，第一次 assess() 就会因为
# "已到高度下限"直接 resolve，永远拿不到 kind="recheck" 的真实"降高+居中再观测"机动，
# 且这次 resolve 的 before/after 是同一次观测（reduction 恒为 0）。
#
# 改动一重标定后该不变量由 fov_ladder 保证并有单测锁死
# （test_fov_ladder.py::test_recheck_altitude_invariant_holds）：
# 巡航 1330.2 m > 下限 443.4 m，单步 443.4 m，两步到底。
VLN_RECHECK_DESCEND_M = float(
    os.getenv("VLN_RECHECK_DESCEND_M", str(fov_ladder.descend_step_m(2)))
)
VLN_RECHECK_ALT_MIN_M = float(
    os.getenv("VLN_RECHECK_ALT_MIN_M", str(fov_ladder.alt_min_m()))
)
VLN_RECHECK_MAX = int(os.getenv("VLN_RECHECK_MAX", "2"))          # 同一位置最多复核次数
VLN_RECHECK_MAX_TOTAL = int(os.getenv("VLN_RECHECK_MAX_TOTAL", "8"))  # 单 episode 复核机动总上限
# P5（升级接口落地，仍是 C2 的实现细节，非新增贡献）：
#   VLN_UNCERTAINTY_MODE：U_t 用 heuristic 查表（默认，向后兼容）还是校准熵（entropy）。
#   VLN_RECHECK_TRIGGER：复核触发用固定阈值（默认）还是信息增益 argmax（info_gain）。
VLN_UNCERTAINTY_MODE = (os.getenv("VLN_UNCERTAINTY_MODE", "heuristic") or "heuristic").strip().lower()
VLN_RECHECK_TEMPERATURE = float(os.getenv("VLN_RECHECK_TEMPERATURE", "1.0"))
VLN_RECHECK_TRIGGER = (os.getenv("VLN_RECHECK_TRIGGER", "threshold") or "threshold").strip().lower()
VLN_RECHECK_THRESHOLD = float(os.getenv("VLN_RECHECK_THRESHOLD", "0.5"))
VLN_RECHECK_MIN_INFO_GAIN = float(os.getenv("VLN_RECHECK_MIN_INFO_GAIN", "0.05"))
VLN_ENTROPY_TABLE = os.getenv("VLN_ENTROPY_TABLE", str(BASE_DIR / "data" / "fov_entropy_table.json"))
VLN_RECHECK_MOTION_MODE = (
    os.getenv("VLN_RECHECK_MOTION_MODE", "descend_center") or "descend_center"
).strip().lower()
VLN_CONFORMAL_QHAT = float(os.getenv("VLN_CONFORMAL_QHAT", "0.9"))
VLN_CONFORMAL_ALPHA = float(os.getenv("VLN_CONFORMAL_ALPHA", "0.1"))
VLN_ORACLE_NAV = os.getenv("VLN_ORACLE_NAV", "0").lower() in {"1", "true", "yes", "on"}
VLN_ORACLE_GROUNDING = os.getenv("VLN_ORACLE_GROUNDING", "0").lower() in {"1", "true", "yes", "on"}
VLN_ORACLE_GOAL: dict | None = None  # set per-episode by the headless bench
# E11 对照基线专用（trigger_mode="random" 时才生效）：固定复核概率 + 可复现种子。
VLN_RECHECK_RANDOM_PROB = float(os.getenv("VLN_RECHECK_RANDOM_PROB", "0.5"))
VLN_RECHECK_RANDOM_SEED = int(os.getenv("VLN_RECHECK_RANDOM_SEED", "0"))
# P3：记忆拓扑图 + LM-Nav 图搜索（默认关，VLN_MEMORY=1 开）。
VLN_MEMORY = os.getenv("VLN_MEMORY", "0").lower() in {"1", "true", "yes", "on"}
VLN_MEMORY_PATH = os.getenv("VLN_MEMORY_PATH", str(BASE_DIR / "outputs" / "memory_graph.json"))
VLN_MEMORY_MERGE_M = float(os.getenv("VLN_MEMORY_MERGE_M", "15"))
VLN_MEMORY_MIN_SCORE = float(os.getenv("VLN_MEMORY_MIN_SCORE", "0.34"))
VLN_MEMORY_MAX_HOPS = int(os.getenv("VLN_MEMORY_MAX_HOPS", "6"))
# 地理门控：匹配到的记忆目标节点离起点超过此距离则视为"别区域同名地标"，不预飞（防跨灾种乱飞）。
VLN_MEMORY_MAX_DIST_M = float(os.getenv("VLN_MEMORY_MAX_DIST_M", "1500"))

# Agent-VQA 配置 (D3, 计划 7.3)。与 VLN 共享感知/搜索/复核底座，但问答闭环独立。
AGENT_VQA_CONFIDENCE_THRESHOLD = float(os.getenv("AGENT_VQA_CONFIDENCE_THRESHOLD", "0.5"))
AGENT_VQA_MAX_SEARCH_STEPS = int(os.getenv("AGENT_VQA_MAX_SEARCH_STEPS", "6"))
AGENT_VQA_MAX_REOBSERVATIONS = int(os.getenv("AGENT_VQA_MAX_REOBSERVATIONS", "2"))
AGENT_VQA_VLM_MAX_TOKENS = int(os.getenv("AGENT_VQA_VLM_MAX_TOKENS", "300"))
AGENT_VQA_ORACLE = os.getenv("AGENT_VQA_ORACLE", "0").lower() in {"1", "true", "yes", "on"}
# 证据层级 (计划 9.1): raw=仅图像, struct=图像+结构化感知, state=struct+STMR/历史。
# 控制下发给 VLM 的 evidence_text 是否包含结构化检测/状态信息。
AGENT_VQA_EVIDENCE_LEVEL = (os.getenv("AGENT_VQA_EVIDENCE_LEVEL", "struct") or "struct").strip().lower()

# 进程内缓存的记忆图（懒加载，跨 episode/任务复用并落盘）。
_memory_graph: MemoryGraph | None = None


def get_memory_graph() -> MemoryGraph:
    global _memory_graph
    if _memory_graph is None:
        _memory_graph = MemoryGraph.load(VLN_MEMORY_PATH, merge_radius_m=VLN_MEMORY_MERGE_M)
    return _memory_graph
VLN_VLM_MAX_TOKENS = int(os.getenv("VLN_VLM_MAX_TOKENS", "200"))
# VLM grounding 裁剪复核（实验开关，默认关）：粗命中后裁目标周边小窗"数字放大"再精定位一次，
# 映射回原 patch 坐标。本意纠正"偏中心/低精度"，但 A/B 实测（B1×8 题）系统性变差：
# 中位 NE 99m→151m（5 题更差/1 题更好）——高空 patch 本身低分辨率，放大只放大模糊，
# 且二次 VLM 易改选裁剪窗内的相邻建筑。故默认 0；置 VLN_VLM_REFINE=1 可复现该实验。
VLN_VLM_REFINE = os.getenv("VLN_VLM_REFINE", "0").lower() not in {"0", "false", "no", "off"}
VLN_VLM_REFINE_WIN = float(os.getenv("VLN_VLM_REFINE_WIN", "0.34"))  # 裁剪窗口边长占原图比例
VLN_VLM_REFINE_UPSCALE = int(os.getenv("VLN_VLM_REFINE_UPSCALE", "512"))  # 裁剪后放大到的边长(px)
# P0：2D 地理语义地图栅格边长（米）；与 STMR 默认 5m/格对齐。设 0/off 可关闭建图。
SEMANTIC_MAP_CELL_M = float(os.getenv("SEMANTIC_MAP_CELL_M", "5"))
SEMANTIC_MAP_ENABLED = os.getenv("SEMANTIC_MAP", "1").lower() not in {"0", "false", "no", "off"}

# grounding 提示（已实测）：不要让小 VLM 输出 present 布尔——它会无视自身描述默认 false。
# 改为"看得到就给坐标、看不到就回‘没有’"，由是否解析出坐标来判定 present。
_VLN_GROUND_SYS_PROMPT = (
    "你是无人机俯视(nadir)视场的视觉判读器，擅长在卫星/航拍俯视图里定位目标。"
    "给你一张无人机正下方的俯视影像和一个要找的目标。"
    "只要画面里能看到该目标（哪怕只是疑似、或只露出一部分）就算看到，"
    "要给出它中心的归一化坐标，格式严格为 x,y 两个 0~1 之间的小数"
    "（x：最左0 最右1；y：最上0 最下1），例如 0.32,0.78。"
    "如果画面里完全没有该目标，就只回答两个字：没有。"
    "不要输出任何坐标或‘没有’以外的文字、解释或标点。"
)


def _vlm_refine_xy(
    patch_path: str, target: str, x0: float, y0: float
) -> tuple[float, float] | None:
    """裁剪复核：以粗命中点 (x0,y0) 为中心裁一小窗 + 放大，让 VLM 在清晰大图上精定位一次，
    再把窗内坐标映射回原 patch 的归一化坐标。失败/未命中返回 None（调用方回退粗定位）。
    """
    try:
        import io
        from PIL import Image
        img = Image.open(patch_path).convert("RGB")
        W, H = img.size
        if W <= 0 or H <= 0:
            return None
        ww = max(64, int(W * VLN_VLM_REFINE_WIN))
        wh = max(64, int(H * VLN_VLM_REFINE_WIN))
        cx, cy = x0 * W, y0 * H
        left = int(min(max(cx - ww / 2.0, 0), max(W - ww, 0)))
        top = int(min(max(cy - wh / 2.0, 0), max(H - wh, 0)))
        crop = img.crop((left, top, left + ww, top + wh))
        up = VLN_VLM_REFINE_UPSCALE
        crop = crop.resize((up, up))  # 数字放大，让小目标占更大比例
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        out = VLMAnalyzer().analyze_image_bytes(
            image_bytes=buf.getvalue(),
            mime_type="image/png",
            prompt=(
                f"这是放大后的局部俯视影像。要找的目标：{target}\n"
                "看得到就只输出它中心的坐标 x,y；完全看不到就只回答：没有。"
            ),
            system_prompt=_VLN_GROUND_SYS_PROMPT,
            max_tokens=VLN_VLM_MAX_TOKENS,
        )
        xyc = parse_ground_xy((out.get("analysis") or "").strip())
    except Exception as exc:
        logger.debug("VLM refine failed: %s", exc)
        return None
    if xyc is None:
        return None
    # 窗内归一化 → 原 patch 归一化
    gx = (left + xyc[0] * ww) / W
    gy = (top + xyc[1] * wh) / H
    return (min(max(gx, 0.0), 1.0), min(max(gy, 0.0), 1.0))


def _vln_vlm_ground(parsed: dict, obs: Observation) -> GroundHit | None:
    """用 VLM 对当前俯视 patch 做开放词汇 grounding（坐标-or-没有 范式）。"""
    patch_path = getattr(obs, "patch_path", "") or ""
    if not patch_path or not os.path.exists(patch_path):
        return GroundHit(present=False, reason="VLM grounder: 无可用 patch 图像", source="vlm")
    # 用开放词汇短语（保留"蓝色"等修饰），回退到 target_label / 原指令。
    target = (
        parsed.get("target_phrase")
        or parsed.get("target_label")
        or parsed.get("raw", "")
    )
    prompt = (
        f"要找的目标：{target}\n"
        "看得到就只输出坐标 x,y；完全看不到就只回答：没有。"
    )
    try:
        with open(patch_path, "rb") as f:
            img_bytes = f.read()
        out = VLMAnalyzer().analyze_image_bytes(
            image_bytes=img_bytes,
            mime_type="image/png",
            prompt=prompt,
            system_prompt=_VLN_GROUND_SYS_PROMPT,
            max_tokens=VLN_VLM_MAX_TOKENS,
        )
        text = (out.get("analysis") or "").strip()
    except Exception as exc:
        logger.warning("VLN VLM grounder failed: %s", exc)
        return GroundHit(present=False, reason=f"VLM 调用失败: {exc}", source="vlm")

    xy = parse_ground_xy(text)
    if xy is None:
        return GroundHit(present=False, reason=f"VLM: 未发现「{target}」({text[:30]})", source="vlm")
    # 裁剪复核：在目标周边小窗放大后再精定位一次，纠正"偏中心/低精度"导致的假到达。
    reason = f"VLM 命中「{target}」于 ({xy[0]:.2f},{xy[1]:.2f})"
    if VLN_VLM_REFINE:
        refined = _vlm_refine_xy(patch_path, target, xy[0], xy[1])
        if refined is not None:
            reason = (
                f"VLM 命中「{target}」粗({xy[0]:.2f},{xy[1]:.2f})→"
                f"精({refined[0]:.2f},{refined[1]:.2f})"
            )
            xy = refined
    # 到达交给导航器按距离判定（VLM 自报到达不可靠），这里只给位置。
    return GroundHit(
        present=True,
        norm_xy=xy,
        arrived=False,
        label=target or "目标",
        conf=1.0,
        reason=reason,
        source="vlm",
    )


def _post_covered(lat: float, lon: float) -> bool:
    """目标点是否落在某张 POST 灾后瓦片覆盖内（VLN 防越界用）。"""
    try:
        return xbd_store.find_tile_containing(
            lat, lon, stage_priority=("post_disaster",)
        ) is not None
    except Exception:
        return False


def _make_hspm_navigator() -> HspmNavigator:
    """构造 P1 HSPM 分层规划器：复用 VLN 的 grounder + STMR(地图) + planner LLM。"""
    # planner LLM（landmark 拆解 / OROI 推理）；不可用则 HSPM 回退到短语 + 方向先验。
    llm_chat = None
    try:
        from llm_client import get_client
        _client = get_client(module="planner")
        llm_chat = _client.chat  # 签名 (messages, temperature, max_tokens) 与 HSPM 期望一致
    except Exception as exc:
        logger.warning("HSPM planner LLM 不可用，退化为短语+方向先验：%s", exc)

    def _stmr_provider(snap: dict):
        smap = getattr(state, "semantic_map", None)
        if smap is None:
            return None
        try:
            return build_stmr(
                smap, float(snap["lat"]), float(snap["lon"]),
                window_m=HSPM_STMR_WINDOW_M, grid_n=HSPM_STMR_GRID_N,
            )
        except Exception as exc:
            logger.debug("build_stmr failed: %s", exc)
            return None

    # HSPM grounder 用开放词汇短语，默认走 VLM（landmark 是自由短语，YOLO 类别覆盖不到）。
    grounder = _make_vln_grounder("vlm" if VLN_GROUNDER == "yolo" else VLN_GROUNDER)
    return HspmNavigator(
        config=HspmConfig(
            step_budget=VLN_STEP_BUDGET,
            arrival_radius_m=VLN_ARRIVAL_RADIUS_M,
            max_step_m=VLN_MAX_STEP_M,
            explore_step_m=VLN_EXPLORE_STEP_M,
            use_oroi_score=VLN_OROI_SCORE,
            oroi_weights=OroiScoreWeights(
                llm=VLN_OROI_W_LLM, prior=VLN_OROI_W_PRIOR, frontier=VLN_OROI_W_FRONTIER,
            ),
        ),
        grounder=grounder,
        llm_chat=llm_chat,
        stmr_provider=_stmr_provider,
        semantic_map_provider=lambda: getattr(state, "semantic_map", None),
    )


def _vln_memory_prefly(landmarks: list[str], source: str) -> bool:
    """P3：相似指令命中记忆图时，沿"熟路"航点预飞到已知目标附近。

    返回是否实际预飞过（用于日志）。预飞不消耗 navigator 步数预算，飞完照常进入
    grounding 精定位循环。
    """
    graph = get_memory_graph()
    if graph.stats()["nodes"] == 0:
        return False
    snap = state.adapter.snapshot()
    plan = graph.plan(
        landmarks, float(snap["lat"]), float(snap["lon"]),
        scorer=text_match_scorer, min_score=VLN_MEMORY_MIN_SCORE,
        max_dist_m=VLN_MEMORY_MAX_DIST_M,
    )
    if not plan:
        state.push_log("info", "[VLN 记忆] 记忆图中无匹配熟路，转入常规探索。")
        return False

    wps = plan["waypoints"]
    state.push_log(
        "info",
        f"[VLN 记忆] 命中记忆图（{plan['mode']}，目标「{plan['target_label']}」匹配分 "
        f"{plan['target_score']}），沿熟路预飞最多 {min(len(wps), VLN_MEMORY_MAX_HOPS)} 跳。",
    )
    flown = 0
    for i, wp in enumerate(wps):
        if state._stop_event.is_set() or flown >= VLN_MEMORY_MAX_HOPS:
            break
        cur = state.adapter.snapshot()
        n, e = latlon_to_meters(float(cur["lat"]), float(cur["lon"]), wp["lat"], wp["lon"])
        hop_d = (n * n + e * e) ** 0.5
        if hop_d < 20.0:
            continue  # 跳过离当前太近的航点（如起点节点本身）
        if hop_d > VLN_MEMORY_MAX_DIST_M:
            state.push_log("warn", f"[VLN 记忆] 熟路航点距当前 {hop_d:.0f}m 超门控，跳过（防跨区域乱飞）。")
            continue
        if not _post_covered(wp["lat"], wp["lon"]):
            continue  # 熟路航点已脱离当前 POST 覆盖，跳过
        res = execute_action(
            "fly_to_geo",
            {
                "lat": wp["lat"], "lon": wp["lon"],
                "alt": wp.get("alt") or state.hover_altitude_m, "speed": 14.0,
            },
            source=source,
        )
        if not res.get("success"):
            break
        flown += 1
        socketio.emit(
            "ai_thought",
            {
                "iteration": 0,
                "skill": "memory_fly",
                "thinking": f"沿记忆图熟路飞往航点 {i + 1}/{len(wps)}（{plan['mode']}）",
                "progress": f"memory {flown}/{min(len(wps), VLN_MEMORY_MAX_HOPS)}",
                "matched": False,
            },
        )
    if flown:
        state.push_log("success", f"[VLN 记忆] 熟路预飞完成 {flown} 跳，转入精定位。")
    return flown > 0


def _oracle_ground(parsed: dict, obs: Observation) -> GroundHit | None:
    goal = VLN_ORACLE_GOAL or {}
    try:
        glat, glon = float(goal["lat"]), float(goal["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    snap = state.adapter.snapshot()
    north_m, east_m = latlon_to_meters(float(snap["lat"]), float(snap["lon"]), glat, glon)
    radius = float(obs.patch_radius_m or 0.0)
    if radius <= 0:
        return None
    nx = max(0.0, min(1.0, 0.5 + east_m / (2.0 * radius)))
    ny = max(0.0, min(1.0, 0.5 - north_m / (2.0 * radius)))
    dist = (north_m * north_m + east_m * east_m) ** 0.5
    return GroundHit(
        present=True,
        norm_xy=(nx, ny),
        arrived=dist <= VLN_ARRIVAL_RADIUS_M,
        label=str(goal.get("class") or parsed.get("target_label") or ""),
        conf=1.0,
        reason=f"oracle grounding ({dist:.1f}m)",
        source="oracle",
    )


def _make_vln_grounder(mode: str):
    """按模式构造 grounder：yolo→None(导航器内置)，vlm→VLM，hybrid→YOLO 优先否则 VLM。"""
    if VLN_ORACLE_GROUNDING:
        return _oracle_ground
    if mode == "yolo":
        return None
    if mode == "vlm":
        return _vln_vlm_ground

    def _hybrid(parsed: dict, obs: Observation) -> GroundHit | None:
        hit = ground_with_yolo(obs, parsed.get("target_classes") or [])
        if hit is not None and hit.present:
            hit.source = "hybrid:yolo"
            return hit
        vh = _vln_vlm_ground(parsed, obs)
        if vh is not None:
            vh.source = "hybrid:vlm"
        return vh

    return _hybrid


def _vln_perceive(source: str) -> tuple[PerceptionResult | None, dict, dict]:
    """
    在 UAV 当前位姿裁俯视视场并跑感知，发射 perception_result 供前端面板更新。

    返回 (PerceptionResult|None, snapshot, active_tile)。
    None 表示当前位置无 POST 瓦片覆盖或感知失败，导航应停止。
    """
    perception = get_perception()
    snap = state.adapter.snapshot()

    aligned = _align_active_tile_for(
        float(snap["lat"]), float(snap["lon"]), source=source, strict_post=True
    )
    world = state.world_state()
    active_tile = (world.get("map") or {}).get("active_tile") or {}
    if aligned is None or not xbd_store._is_post(active_tile) or not active_tile.get("tile_id"):
        return None, snap, active_tile

    if not perception.is_available:
        try:
            perception.load()
        except Exception as exc:
            state.push_log("error", f"VLN: 视觉模型未就绪 — {exc}")
            return None, snap, active_tile

    patch_id = f"vln-{int(time.time() * 1000)}"
    try:
        result: PerceptionResult = perception.perceive_at(
            lat=float(snap["lat"]),
            lon=float(snap["lon"]),
            alt=float(snap["alt"]),
            active_tile=active_tile,
            patch_id=patch_id,
        )
    except Exception as exc:
        logger.exception("VLN perception failed")
        state.push_log("error", f"VLN 感知失败: {exc}")
        return None, snap, active_tile

    det_counts = result.detection.get("class_counts") or {}
    seg_stats = result.segmentation.get("stats") or {}
    payload = {
        "patch_id": result.patch_id,
        "patch_url": result.patch_url,
        "overlay_url": result.overlay_url,
        "detection_url": result.detection_url,
        "patch_width": result.patch_width,
        "patch_height": result.patch_height,
        "radius_m": result.patch_radius_m,
        "risk_level": result.risk_level,
        "risk_summary": result.risk_summary,
        "damaged_buildings": result.damaged_buildings,
        "intact_buildings": result.intact_buildings,
        "vehicles": result.vehicles,
        "detection": {
            "num_objects": result.detection.get("num_objects", 0),
            "class_counts": det_counts,
            "detections": result.detection.get("detections", [])[:50],
        },
        "segmentation": {
            "num_labels": result.segmentation.get("num_labels", 0),
            "stats": seg_stats,
        },
        "scene": result.scene_dict,
        "scene_text": result.scene_text,
        "vlm_summary": "",
        "position": {"lat": snap["lat"], "lon": snap["lon"], "alt": snap["alt"]},
        "degraded": result.degraded,
        "degraded_reason": result.degraded_reason or "",
        "active_tile_id": active_tile.get("tile_id"),
        "active_tile_stage": active_tile.get("stage"),
        "source": source,
        "timestamp": int(time.time() * 1000),
        "timings": dict(result.extras or {}),
    }
    socketio.emit("perception_result", payload)

    # P0：把这次观测写入 2D 地理语义地图（探索区 + 检测框投影），并推送精简地图状态。
    smap = getattr(state, "semantic_map", None)
    if smap is not None:
        try:
            written = smap.mark_observation(
                uav_lat=float(snap["lat"]),
                uav_lon=float(snap["lon"]),
                radius_m=float(result.patch_radius_m),
                detections=result.detection.get("detections", []),
                degraded=bool(result.degraded),
                risk_level=result.risk_level,
                patch_width=int(result.patch_width),
                patch_height=int(result.patch_height),
            )
            socketio.emit("semantic_map", smap.snapshot())
            logger.debug("semantic_map updated: %s", written)
        except Exception as exc:
            logger.warning("semantic_map update failed: %s", exc)

    state.emit_world()
    return result, snap, active_tile


def _vln_llm_stop(instruction: str, scene_text: str, candidate: dict) -> bool | None:
    """到达候选时让 planner LLM 复核"这是否就是指令描述的目标"。失败回退 None。"""
    try:
        from llm_client import get_client

        client = get_client(module="planner")
        content = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你在为无人机做视觉语言导航的到达确认。只输出 JSON："
                        '{"arrived": true|false}。给定一句导航指令、当前俯视视场的结构化'
                        "感知描述、以及一个候选目标，判断无人机是否已到达指令描述的目标上空。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": instruction,
                            "candidate": {
                                "class_name": candidate.get("class_name"),
                                "conf": candidate.get("conf"),
                                "dist_m": round(float(candidate.get("offset", (0, 0, 0))[2]), 1),
                            },
                            "scene": scene_text[:600],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=64,
        )
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            return None
        return bool(json.loads(match.group(0)).get("arrived"))
    except Exception as exc:
        logger.warning("VLN LLM stop-confirm failed: %s", exc)
        return None


def run_vln_episode(instruction: str, source: str = "ai") -> dict | None:
    """运行一次 VLN episode。

    返回最终 report dict（也通过 socket 广播）；忙时返回 {"ok": False, "error": "busy"}，
    异常时返回带 error 字段的 report。无头评测（run_vln_episode_headless）依赖该返回值。
    """
    if not state._task_lock.acquire(blocking=False):
        state.push_log("warn", "当前已有任务在执行，忽略新的 VLN 指令")
        return {"ok": False, "error": "busy", "task": instruction}

    if VLN_PLANNER == "hspm":
        navigator = _make_hspm_navigator()
    else:
        navigator = VlnNavigator(
            config=VlnConfig(
                step_budget=VLN_STEP_BUDGET,
                arrival_radius_m=VLN_ARRIVAL_RADIUS_M,
                max_step_m=VLN_MAX_STEP_M,
                explore_step_m=VLN_EXPLORE_STEP_M,
                use_llm_stop=VLN_USE_LLM_STOP,
            ),
            grounder=_make_vln_grounder(VLN_GROUNDER),
            llm_stop_fn=_vln_llm_stop if VLN_USE_LLM_STOP else None,
        )
    parsed = navigator.reset(instruction)

    # P0：为本次 episode 建一张全新的 2D 地理语义地图，以当前 UAV 位置为原点累积。
    if SEMANTIC_MAP_ENABLED:
        try:
            init_snap = state.adapter.snapshot()
            state.semantic_map = SemanticMap(
                origin_lat=float(init_snap["lat"]),
                origin_lon=float(init_snap["lon"]),
                cell_size_m=SEMANTIC_MAP_CELL_M,
                instruction=instruction,
            )
        except Exception as exc:
            logger.warning("semantic_map init failed: %s", exc)
            state.semantic_map = None
    else:
        state.semantic_map = None

    # P2：灾情不确定性驱动复核控制器（按位置去重、带预算）。
    rechecker: RecheckController | None = None
    if VLN_RECHECK:
        rechecker = RecheckController(
            RecheckConfig(
                descend_step_m=VLN_RECHECK_DESCEND_M,
                alt_min_m=VLN_RECHECK_ALT_MIN_M,
                max_rechecks=VLN_RECHECK_MAX,
                uncertainty_mode=VLN_UNCERTAINTY_MODE,
                entropy_temperature=VLN_RECHECK_TEMPERATURE,
                trigger=VLN_RECHECK_THRESHOLD,
                trigger_mode=VLN_RECHECK_TRIGGER,
                min_info_gain=VLN_RECHECK_MIN_INFO_GAIN,
                entropy_table_path=VLN_ENTROPY_TABLE,
                conformal_qhat=VLN_CONFORMAL_QHAT,
                conformal_alpha=VLN_CONFORMAL_ALPHA,
                random_prob=VLN_RECHECK_RANDOM_PROB,
                random_seed=VLN_RECHECK_RANDOM_SEED,
                motion_mode=VLN_RECHECK_MOTION_MODE,
            )
        )
    recheck_total = 0  # 本 episode 已执行的复核机动数（全局上限防失控）
    recheck_horizontal_m = 0.0  # 复核机动带来的额外水平距离（按实际下发参数）
    recheck_vertical_m = 0.0  # 复核机动带来的累计垂直距离
    recheck_motion_m = 0.0  # 复核机动三维距离（水平居中 + 降高）
    evidence_observation_count = 0  # 独立于策略触发，供 NONE 等配置公平分层

    arrived = False
    executed: list[dict] = []
    trajectory: list[dict] = []  # P3：本 episode 观测轨迹（成功后写入记忆图）
    last_good_pos: tuple[float, float, float] | None = None  # 最近一次成功观测的 (lat,lon,alt)
    oob_recover = 0  # 连续脱离 POST 覆盖的次数
    path_len_m = 0.0  # 累计实际飞行水平路径长（SPL 用）
    report: dict | None = None
    task_started_ns = time.time_ns()
    try:
        state.is_executing = True
        state._stop_event.clear()
        state._sync_world_from_adapter()
        state.emit_system_status()
        state.emit_world()

        # 路径长度累计起点（含 P3 记忆预飞段）
        _ep_start = state.adapter.snapshot()
        prev_pos = (float(_ep_start["lat"]), float(_ep_start["lon"]))

        if VLN_ORACLE_NAV and isinstance(VLN_ORACLE_GOAL, dict):
            try:
                execute_action(
                    "fly_to_geo",
                    {
                        "lat": float(VLN_ORACLE_GOAL["lat"]),
                        "lon": float(VLN_ORACLE_GOAL["lon"]),
                        "alt": float(_ep_start.get("alt", state.hover_altitude_m)),
                    },
                    source=source,
                )
            except Exception as exc:
                logger.warning("oracle nav fly_to_geo failed: %s", exc)

        target_label = parsed.get("target_label")
        dir_name = parsed.get("direction_name")
        landmarks = parsed.get("landmarks") or []
        goal_desc = (" → ".join(landmarks)) if landmarks else target_label
        plan_summary = (
            f"VLN 语言目标导航（{VLN_PLANNER}）：寻找「{goal_desc}」"
            + (f"（方向先验：{dir_name}）" if dir_name else "")
            + f"，grounding={VLN_GROUNDER}，步数预算 {navigator.config.step_budget}。"
        )
        state.push_log("info", f"收到 VLN 指令: {instruction}")
        state.push_log("info", plan_summary)
        socketio.emit(
            "task_started",
            {
                "label": plan_summary,
                "raw_task": instruction,
                "source": source,
                "step_count": navigator.config.step_budget,
                "ts_ns": task_started_ns,
                "ts_ms": task_started_ns // 1_000_000,
            },
        )
        socketio.emit(
            "ai_plan_result",
            {
                "summary": plan_summary,
                "steps": [
                    {"action": "vln_loop", "reason": f"目标类别: {', '.join(parsed.get('target_classes') or []) or '未识别'}"}
                ],
                "vln": True,
                "submit_ns": task_started_ns,
            },
        )
        socketio.emit("ai_thinking", {"phase": "planning", "detail": plan_summary})

        # ── P3：记忆拓扑图 LM-Nav 预飞 ───────────────────────────────
        # 相似指令命中记忆图 → 先沿"熟路"飞到已知目标节点附近，再进入精定位循环。
        if VLN_MEMORY:
            try:
                _vln_memory_prefly(
                    landmarks=landmarks or [parsed.get("target_phrase") or instruction],
                    source=source,
                )
            except Exception as exc:
                logger.warning("memory prefly failed: %s", exc)

        while not navigator.budget_exhausted():
            if state._stop_event.is_set():
                state.push_log("warn", "VLN 已停止")
                break

            result, snap, active_tile = _vln_perceive(source)
            if result is None:
                # 脱离 POST 覆盖（或感知不可用）。不再直接中止：
                #   - 起点就没覆盖 → 确实无法导航，结束；
                #   - 否则飞回上一个有效观测点重试，连续多次才结束。
                if last_good_pos is None:
                    state.push_log(
                        "error",
                        "VLN 中止：起点不在任何 POST 灾后瓦片覆盖范围内，无法导航。",
                    )
                    break
                oob_recover += 1
                if oob_recover > 3:
                    state.push_log("warn", "VLN：多次脱离 POST 覆盖，结束本次导航。")
                    break
                state.push_log(
                    "warn",
                    f"VLN：当前位置脱离 POST 覆盖，返回上一个有效观测点 "
                    f"({last_good_pos[0]:.6f}, {last_good_pos[1]:.6f}) 重试。",
                )
                execute_action(
                    "fly_to_geo",
                    {"lat": last_good_pos[0], "lon": last_good_pos[1], "alt": last_good_pos[2]},
                    source=source,
                )
                continue

            oob_recover = 0
            last_good_pos = (float(snap["lat"]), float(snap["lon"]), float(snap["alt"]))

            # SPL：累计相邻观测点之间的水平飞行距离（含预飞/复核机动）。
            _cur = (float(snap["lat"]), float(snap["lon"]))
            _n, _e = latlon_to_meters(prev_pos[0], prev_pos[1], _cur[0], _cur[1])
            path_len_m += (_n * _n + _e * _e) ** 0.5
            prev_pos = _cur

            # P3：记录轨迹观测点（成功到达后写入记忆图）。
            _dn, _de = latlon_to_meters(
                float(_ep_start["lat"]), float(_ep_start["lon"]),
                float(snap["lat"]), float(snap["lon"]),
            )
            trajectory.append({
                "lat": float(snap["lat"]),
                "lon": float(snap["lon"]),
                "alt": float(snap["alt"]),
                "labels": dict(result.detection.get("class_counts", {})),
                "risk": result.risk_level,
                "summary": result.risk_summary,
                "pipeline": (result.extras or {}).get("pipeline") or result.detection.get("pipeline"),
                "effective_gsd_m": (result.extras or {}).get("effective_gsd_m"),
                "gsd_scale": (result.extras or {}).get("gsd_scale"),
                "start_dist_m": round((_dn * _dn + _de * _de) ** 0.5, 2),
            })

            obs = Observation.from_perception(result)
            if any(
                det.get("class_name") in EVIDENCE_CLASSES
                for det in (result.detection.get("detections") or [])
            ):
                evidence_observation_count += 1

            # ── P2：灾情不确定性驱动主动复核 ──────────────────────────
            # 看到疑似受灾目标但没把握 → 先降高+飞近再确认，不急着往下走。
            if rechecker is not None:
                rc = rechecker.assess(
                    lat=float(snap["lat"]),
                    lon=float(snap["lon"]),
                    alt=float(snap["alt"]),
                    risk_level=result.risk_level,
                    detections=result.detection.get("detections", []),
                    patch_radius_m=float(result.patch_radius_m),
                    patch_width=int(result.patch_width),
                    patch_height=int(result.patch_height),
                    degraded=bool(result.degraded),
                    # 总预算耗尽后仍要把最后一次复核后的观测交给控制器，否则
                    # pending 闭环看不到 U_after，episode 结束时 ΔU 会退化为 0。
                    allow_recheck=recheck_total < VLN_RECHECK_MAX_TOTAL,
                )
                if rc.kind == "recheck" and rc.params is not None:
                    recheck_total += 1
                    # 可疑目标位置（相对 UAV 偏移投影）写入 candidate_goals（待复核）。
                    off = rc.target_offset_m or (0.0, 0.0)
                    susp_lat, susp_lon = meters_to_latlon(
                        float(snap["lat"]), float(snap["lon"]), off[0], off[1]
                    )
                    smap = getattr(state, "semantic_map", None)
                    if smap is not None:
                        try:
                            smap.add_candidate_goal(
                                susp_lat, susp_lon, rc.label or "疑似受灾目标",
                                conf=1.0 - rc.uncertainty, risk=result.risk_level,
                            )
                            socketio.emit("semantic_map", smap.snapshot())
                        except Exception as exc:
                            logger.debug("candidate_goal write failed: %s", exc)
                    # 防越界：若居中目标飞出 POST 覆盖，则只降高、不水平移动。
                    p = dict(rc.params)
                    dlat, dlon = meters_to_latlon(
                        float(snap["lat"]), float(snap["lon"]),
                        float(p.get("north_m", 0.0)), float(p.get("east_m", 0.0)),
                    )
                    if not _post_covered(dlat, dlon):
                        p["north_m"], p["east_m"] = 0.0, 0.0
                    _re_n = float(p.get("north_m", 0.0))
                    _re_e = float(p.get("east_m", 0.0))
                    _re_u = float(p.get("up_m", 0.0))
                    _re_horizontal = (_re_n * _re_n + _re_e * _re_e) ** 0.5
                    recheck_horizontal_m += _re_horizontal
                    recheck_vertical_m += abs(_re_u)
                    recheck_motion_m += (_re_horizontal * _re_horizontal + _re_u * _re_u) ** 0.5
                    state.push_log("warn", f"[VLN 复核 {recheck_total}] {rc.reason}")
                    socketio.emit(
                        "ai_thought",
                        {
                            "iteration": navigator.step_index,
                            "skill": "recheck",
                            "thinking": rc.reason,
                            "progress": f"recheck {recheck_total}/{VLN_RECHECK_MAX_TOTAL}",
                            "matched": False,
                            "uncertainty": rc.uncertainty,
                        },
                    )
                    res_exec = execute_action("fly_relative", p, source=source)
                    executed.append({
                        "action": "recheck", "params": p, "result": res_exec,
                        "reason": rc.reason, "uncertainty": rc.uncertainty,
                    })
                    if not res_exec.get("success"):
                        state.push_log("error", f"VLN 复核机动失败，转常规导航: {res_exec.get('message','')}")
                    else:
                        continue  # 降高后重新感知，再决策
                elif rc.kind == "resolve":
                    smap = getattr(state, "semantic_map", None)
                    if smap is not None:
                        try:
                            smap.add_candidate_goal(
                                float(snap["lat"]), float(snap["lon"]),
                                f"{rc.label or '受灾目标'}[{rc.status}]",
                                conf=1.0 - rc.uncertainty, risk=result.risk_level,
                            )
                            socketio.emit("semantic_map", smap.snapshot())
                        except Exception as exc:
                            logger.debug("candidate_goal resolve write failed: %s", exc)
                    state.push_log(
                        level_for_risk(result.risk_level),
                        f"[VLN 复核定论] {rc.reason}",
                    )

            decision = navigator.step(obs, snap)
            step_idx = navigator.step_index

            # 防越界：若这一步会飞出 POST 覆盖，改朝当前瓦片中心回拉，保持在覆盖内继续搜索。
            if decision.action == "fly_relative":
                nm = float(decision.params.get("north_m", 0.0))
                em = float(decision.params.get("east_m", 0.0))
                dlat, dlon = meters_to_latlon(float(snap["lat"]), float(snap["lon"]), nm, em)
                if not _post_covered(dlat, dlon):
                    rn, re_ = latlon_to_meters(
                        float(snap["lat"]), float(snap["lon"]),
                        state.world.anchor_lat, state.world.anchor_lon,
                    )
                    rn, re_ = VlnNavigator._clamp_step(rn, re_, navigator.config.max_step_m)
                    decision.params["north_m"] = round(rn, 1)
                    decision.params["east_m"] = round(re_, 1)
                    note = f"原方向将飞出 POST 覆盖，改朝瓦片中心回拉 N{rn:+.0f}/E{re_:+.0f}m。"
                    decision.reason = note + " " + decision.reason
                    decision.thought = f"step{step_idx}: 防越界，{note}"

            state.push_log("info", f"[VLN {step_idx}/{navigator.config.step_budget}] {decision.thought}")
            socketio.emit(
                "ai_thought",
                {
                    "iteration": step_idx,
                    "skill": decision.action,
                    "thinking": decision.reason,
                    "progress": f"step {step_idx}/{navigator.config.step_budget}",
                    "matched": decision.matched,
                    "target_dist_m": decision.target_dist_m,
                },
            )
            socketio.emit(
                "ai_thinking",
                {
                    "phase": "executing",
                    "detail": decision.action,
                    "iteration": step_idx,
                    "action": {"skill": decision.action, "parameters": decision.params},
                    "decision": decision.reason,
                },
            )

            if decision.action == "stop" or decision.arrived:
                arrived = True
                # X0 修复：判到达但目标不在正下方时，先平移到目标上方再停。
                # 否则 final NE 恒等于"看见目标那一刻的水平距离"，判到达半径
                # (35 m) 大于成功半径 (25 m) 时会系统性早停。
                off = decision.target_offset_m
                if off is not None:
                    dist = float(decision.target_dist_m
                                 or (off[0] ** 2 + off[1] ** 2) ** 0.5)
                    if dist > 3.0:
                        fp = {"north_m": round(float(off[0]), 1),
                              "east_m": round(float(off[1]), 1),
                              "up_m": 0.0, "speed": 12.0}
                        res_final = execute_action("fly_relative", fp, source=source)
                        executed.append({
                            "action": "final_approach", "params": fp,
                            "result": res_final,
                            "reason": f"到达前对准目标（余距 {dist:.0f}m）",
                        })
                state.push_log("success", f"VLN 到达: {decision.reason}")
                break

            step_start_ns = time.time_ns()
            result_exec = execute_action(decision.action, decision.params, source=source)
            step_end_ns = time.time_ns()
            executed.append({
                "action": decision.action,
                "params": decision.params,
                "result": result_exec,
                "reason": decision.reason,
                "matched": decision.matched,
                "wall_ms": (step_end_ns - step_start_ns) // 1_000_000,
                "step_start_ns": step_start_ns,
                "step_end_ns": step_end_ns,
            })
            socketio.emit(
                "action_result",
                {
                    "action": decision.action,
                    "params": decision.params,
                    "result": result_exec,
                    "wall_ms": (step_end_ns - step_start_ns) // 1_000_000,
                    "step_index": step_idx,
                    "step_start_ns": step_start_ns,
                    "step_end_ns": step_end_ns,
                },
            )
            if not result_exec.get("success"):
                state.push_log("error", f"VLN 飞行失败，中止: {result_exec.get('message', '')}")
                break

        final_snap = state.adapter.snapshot()
        # 补最后一段（最后一次决策机动后到终点的位移）
        _fn, _fe = latlon_to_meters(
            prev_pos[0], prev_pos[1], float(final_snap["lat"]), float(final_snap["lon"])
        )
        path_len_m += (_fn * _fn + _fe * _fe) ** 0.5

        # P3：成功到达则把本次轨迹沉淀进记忆图并落盘（越用越熟）。
        if VLN_MEMORY and arrived and trajectory:
            try:
                graph = get_memory_graph()
                graph.add_trajectory(
                    trajectory,
                    instruction=instruction,
                    landmarks=(parsed.get("landmarks") or [parsed.get("target_phrase") or instruction]),
                    success=True,
                )
                graph.save(VLN_MEMORY_PATH)
                state.push_log("info", f"[VLN 记忆] 轨迹已沉淀进记忆图：{graph.stats()}")
            except Exception as exc:
                logger.warning("memory graph record failed: %s", exc)

        summary = navigator.summarize(arrived, final_snap)
        rstats = None
        if rechecker is not None:
            # episode 到这里就要收尾了：把仍"复核中"但没等到正式定论的位置（常见
            # 于到达终点/步数耗尽打断复核循环）按最新观测补记账，避免 ΔU 统计
            # 系统性地丢掉这部分样本（E11 实测发现，见 recheck.py 顶部说明）。
            rechecker.finalize()
            rstats = rechecker.stats()
            rstats.update({
                "extra_actions": recheck_total,
                "extra_horizontal_m": round(recheck_horizontal_m, 2),
                "extra_vertical_m": round(recheck_vertical_m, 2),
                "extra_motion_m": round(recheck_motion_m, 2),
                "has_evidence": evidence_observation_count > 0,
            })
            if rstats["resolved"] or recheck_total:
                summary += (
                    f" 复核 {recheck_total} 次机动、触发 {rstats['triggered']} 处、"
                    f"完成 {rstats['completed']} 处"
                    f"（确认 {rstats['confirmed']} / 排除 {rstats['dismissed']} / 存疑 {rstats['inconclusive']}"
                    f" / episode 结束时未收尾 {rstats['episode_end_pending']}），"
                    f"平均不确定性下降 {rstats['avg_uncertainty_reduction']}。"
                )
        # 收尾写一条 world report，让地图落一个标记点
        state.world.add_report(
            content=summary,
            lat=float(final_snap["lat"]),
            lon=float(final_snap["lon"]),
            level="success" if arrived else "warn",
            source="vln",
        )
        state.push_log("success" if arrived else "warn", summary)

        finished_ns = time.time_ns()
        report = {
            "ok": arrived,
            "task": instruction,
            "summary": summary,
            "steps": executed,
            "steps_executed": len(executed),
            "arrived": arrived,
            "vln_history": navigator.history,
            "recheck": rstats,
            "recheck_log": rechecker.resolved_log if rechecker is not None else [],
            "evidence_observations": evidence_observation_count,
            "memory": (get_memory_graph().stats() if VLN_MEMORY else None),
            # ── P4 评测所需字段 ────────────────────────────────────────
            "final_pos": {
                "lat": float(final_snap["lat"]),
                "lon": float(final_snap["lon"]),
                "alt": float(final_snap["alt"]),
            },
            "path_len_m": round(path_len_m, 2),
            "landmarks": landmarks,
            "target_label": target_label,
            "target_classes": parsed.get("target_classes") or [],
            "planner": VLN_PLANNER,
            "grounder": VLN_GROUNDER,
            "trajectory": trajectory,
            "config": {
                "step_budget": navigator.config.step_budget,
                "arrival_radius_m": navigator.config.arrival_radius_m,
                "recheck": bool(VLN_RECHECK),
                "memory": bool(VLN_MEMORY),
            },
            "ts_ns": finished_ns,
            "ts_ms": finished_ns // 1_000_000,
            "task_started_ns": task_started_ns,
            "wall_ms": (finished_ns - task_started_ns) // 1_000_000,
        }
        if source == "ai":
            socketio.emit("ai_execution_report", report)
            socketio.emit("ai_thinking", {"phase": "idle", "detail": ""})
        else:
            socketio.emit("execution_report", report)
    except Exception as exc:
        logger.exception("VLN episode crashed")
        state.push_log("error", f"VLN 异常: {exc}")
        socketio.emit("ai_thinking", {"phase": "idle", "detail": ""})
        report = {"ok": False, "task": instruction, "error": str(exc), "arrived": False}
    finally:
        state.is_executing = False
        state._sync_world_from_adapter()
        state.emit_system_status()
        state.emit_world()
        state._task_lock.release()

    return report


def run_vln_episode_headless(
    instruction: str,
    start: dict,
    source: str = "bench",
) -> dict:
    """无头评测入口：把 UAV 放到指定起点后同步跑一次 VLN episode，返回 report dict。

    与 run_vln_episode 共享同一套感知 / 规划 / 复核 / 记忆逻辑（真实模型），只是：
      - 调用方先指定起点 start={"lat","lon","alt"}，本函数负责对齐 POST 瓦片并定位 UAV；
      - 同步阻塞返回 report（含 final_pos / path_len_m / arrived / steps / recheck / memory），
        供 bench_vln_navigation.py 计算 NE / SR / SPL 等指标；
      - 不依赖前端：socket 广播在无客户端时为 no-op，互不影响。

    起点未被任何 POST 瓦片覆盖时，返回 {"ok": False, "error": "start_not_covered"}。
    """
    try:
        lat = float(start["lat"])
        lon = float(start["lon"])
        alt = float(start.get("alt", state.hover_altitude_m))
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"bad_start: {exc}", "task": instruction}

    entry = xbd_store.find_tile_containing(lat, lon, stage_priority=("post_disaster",))
    if entry is None:
        return {"ok": False, "error": "start_not_covered", "task": instruction,
                "start": {"lat": lat, "lon": lon, "alt": alt}}

    # 对齐活动瓦片（设置 world / adapter 原点到瓦片中心），再把 UAV 精确放到起点。
    state.activate_xbd_tile(entry)
    state.adapter.reset_origin(lat, lon, alt=alt)
    state._sync_world_from_adapter()

    report = run_vln_episode(instruction, source=source)
    if isinstance(report, dict):
        report.setdefault("tile_id", entry.get("tile_id"))
        report.setdefault("disaster", entry.get("disaster"))
        report["start"] = {"lat": lat, "lon": lon, "alt": alt}
    return report or {"ok": False, "error": "no_report", "task": instruction}


# ───────────────────────── Agent-VQA 闭环 (D3, 计划 7.3) ──────────────────────

def _mark_agent_vqa_target(image_bytes: bytes) -> bytes:
    """在 VQA 图像中心叠加可见十字；标记像素不进入感知或策略状态。"""
    with Image.open(BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    cx, cy = image.width // 2, image.height // 2
    arm = max(12, min(image.width, image.height) // 16)
    width = max(3, min(image.width, image.height) // 100)
    for color, extra in (("white", width + 4), ("#ff2d55", width)):
        draw.line((cx - arm, cy, cx + arm, cy), fill=color, width=extra)
        draw.line((cx, cy - arm, cx, cy + arm), fill=color, width=extra)
        draw.ellipse((cx - arm, cy - arm, cx + arm, cy + arm), outline=color, width=extra)
    out = BytesIO()
    image.save(out, format="JPEG", quality=95)
    return out.getvalue()


def _action_failed(result: object) -> bool:
    return isinstance(result, dict) and (
        result.get("success") is False or result.get("ok") is False
    )


def _make_agent_vqa_controller(source: str) -> AgentVqaController:
    """构造 Agent-VQA 控制器：复用 VLN 的感知/HSPM 搜索/复核底座，问答闭环独立。

    依赖注入使无模型环境也能跑（VLM/LLM 不可用时控制器自动规则回退）。
    在线决策只读当前观测；不读测试条目的 answer 或未来图像。
    """
    # VLM 结构化问答：读当前 patch 图像字节
    _vlm = VLMAnalyzer()

    def vlm_answer_fn(image_bytes, perception_result, spec, qid):
        if not image_bytes:
            raise RuntimeError("no_patch_bytes")
        if spec.question_type == "damage":
            image_bytes = _mark_agent_vqa_target(image_bytes)
        ev_text = ""
        # 证据层级 (计划 9.1): raw 不下发结构化证据; struct 下发检测计数; state 额外含 STMR/历史
        if AGENT_VQA_EVIDENCE_LEVEL != "raw" and perception_result is not None:
            det = (perception_result.detection or {}).get("class_counts", {})
            if det:
                ev_text = json.dumps(det, ensure_ascii=False)
            if AGENT_VQA_EVIDENCE_LEVEL == "state":
                smap = getattr(state, "semantic_map", None)
                if smap is not None:
                    try:
                        ev_text = ev_text + "||" + json.dumps(smap.snapshot(), ensure_ascii=False)
                    except Exception:
                        pass
        res = _vlm.answer_image_question(
            image_bytes=image_bytes,
            question=spec.raw,
            choices=_choices_for_spec(spec),
            evidence_text=ev_text,
            max_tokens=AGENT_VQA_VLM_MAX_TOKENS,
        )
        return res["raw"]

    def perceive_fn():
        result, snap, _tile = _vln_perceive(source)
        if result is None and _post_covered(float(snap["lat"]), float(snap["lon"])):
            raise RuntimeError("perception_backend_unavailable")
        return result

    def get_position_fn():
        snap = state.adapter.snapshot()
        return {"lat": float(snap["lat"]), "lon": float(snap["lon"]), "alt": float(snap["alt"])}

    def get_image_bytes_fn(result):
        p = getattr(result, "patch_path", "") or ""
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
        return b""

    # 搜索：一个 episode 只维护一个 HSPM 状态机；每步使用刚得到的真实观测。
    nav = _make_hspm_navigator()
    nav_ready = False

    def search_fn(spec, step, perception_result):
        nonlocal nav_ready
        try:
            if not nav_ready:
                nav.reset(spec.raw)
                nav_ready = True
            obs = Observation.from_perception(perception_result)
            snap = state.adapter.snapshot()
            decision = nav.step(obs, snap)
            if decision.action == "stop":
                return None
            if decision.action not in {"fly_relative", "fly_to_geo", "hover"}:
                raise RuntimeError(f"illegal_hspm_action:{decision.action}")
            params = dict(decision.params or {})
            executed = execute_action(decision.action, params, source=source)
            if _action_failed(executed):
                raise RuntimeError(str(executed.get("message") or "hspm_action_failed"))
            return params
        except Exception as exc:
            logger.warning("Agent-VQA search step failed: %s", exc)
            return None

    # 重观测：每个 episode 复用同一控制器，确保触发、次数和随机数状态连续。
    rechecker = RecheckController(RecheckConfig(
        descend_step_m=VLN_RECHECK_DESCEND_M,
        alt_min_m=VLN_RECHECK_ALT_MIN_M,
        max_rechecks=AGENT_VQA_MAX_REOBSERVATIONS,
        uncertainty_mode=VLN_UNCERTAINTY_MODE,
        entropy_temperature=VLN_RECHECK_TEMPERATURE,
        trigger=VLN_RECHECK_THRESHOLD,
        trigger_mode=VLN_RECHECK_TRIGGER,
        min_info_gain=VLN_RECHECK_MIN_INFO_GAIN,
        entropy_table_path=VLN_ENTROPY_TABLE,
        conformal_qhat=VLN_CONFORMAL_QHAT,
        conformal_alpha=VLN_CONFORMAL_ALPHA,
        random_prob=VLN_RECHECK_RANDOM_PROB,
        random_seed=VLN_RECHECK_RANDOM_SEED,
        motion_mode=VLN_RECHECK_MOTION_MODE,
    ))

    def reobserve_fn(perception_result, spec):
        snap = state.adapter.snapshot()
        dets = (perception_result.detection or {}).get("detections", []) if perception_result else []
        risk_level = getattr(perception_result, "risk_level", "low") or "low"
        if spec.question_type == "damage" and risk_level == "none" and dets:
            risk_level = "low"
        out = rechecker.assess(
            lat=float(snap["lat"]), lon=float(snap["lon"]), alt=float(snap["alt"]),
            risk_level=risk_level,
            detections=dets,
            patch_radius_m=float(getattr(perception_result, "patch_radius_m", 60.0)),
            patch_width=int(getattr(perception_result, "patch_width", 100)),
            patch_height=int(getattr(perception_result, "patch_height", 100)),
        )
        audit = {
            "kind": out.kind,
            "params": out.params,
            "reason": out.reason,
            "uncertainty": out.uncertainty,
            "label": out.label,
            "entropy_table_loaded": bool(getattr(rechecker, "entropy_table_loaded", False)),
            "entropy_fallback_used": False,
            "motion_mode": VLN_RECHECK_MOTION_MODE,
        }
        # Agent-VQA A2_ALWAYS 是“额外观测上限”对照：即使当前 detector 没有给出
        # 可疑目标，也应执行一次中心下降重观测，验证动作通道和额外图像本身的价值。
        if out.kind == "skip" and VLN_RECHECK_TRIGGER == "fixed":
            up_m = -min(VLN_RECHECK_DESCEND_M, max(0.0, float(snap["alt"]) - VLN_RECHECK_ALT_MIN_M))
            if VLN_RECHECK_MOTION_MODE not in {"descend_only", "descend_center"}:
                up_m = 0.0
            if VLN_RECHECK_MOTION_MODE != "hold" and (
                up_m < 0 or VLN_RECHECK_MOTION_MODE == "center_only"
            ):
                audit.update({
                    "kind": "recheck",
                    "params": {"north_m": 0.0, "east_m": 0.0, "up_m": round(up_m, 1), "speed": 10.0},
                    "reason": "A2_ALWAYS 固定基线：无论当前证据是否充分，强制获取一次更高分辨率观测。",
                })
        if audit["kind"] == "recheck" and audit["params"]:
            # 真正执行降高+居中，使下一步感知看到放大后的同一目标。
            # 同一分支同时覆盖策略原生 recheck 与 A2 fixed 强制对照。
            executed = execute_action("fly_relative", audit["params"], source=source)
            if _action_failed(executed):
                raise RuntimeError(str(executed.get("message") or "reobserve_motion_failed"))
            return audit
        return audit

    return AgentVqaController(
        config=AgentVqaConfig(
            confidence_threshold=AGENT_VQA_CONFIDENCE_THRESHOLD,
            max_search_steps=AGENT_VQA_MAX_SEARCH_STEPS,
            max_reobservations=AGENT_VQA_MAX_REOBSERVATIONS,
            oracle=AGENT_VQA_ORACLE,
            allow_target_leak=AGENT_VQA_ORACLE,
            evidence_level=AGENT_VQA_EVIDENCE_LEVEL,
        ),
        vlm_answer_fn=vlm_answer_fn,
        perceive_fn=perceive_fn,
        search_fn=search_fn,
        reobserve_fn=reobserve_fn,
        get_position_fn=get_position_fn,
        get_image_bytes_fn=get_image_bytes_fn,
        is_cancelled_fn=state._stop_event.is_set,
    )


def _choices_for_spec(spec: QuestionSpec) -> list[str] | None:
    from agent_vqa import (BEARING_CHOICES, COUNT_CHOICES, DAMAGE_CHOICES, PRESENCE_CHOICES)
    if spec.question_type == "presence":
        return PRESENCE_CHOICES
    if spec.question_type == "damage":
        return DAMAGE_CHOICES
    if spec.question_type == "count":
        return COUNT_CHOICES
    if spec.question_type == "spatial":
        return BEARING_CHOICES
    return None


def _execute_agent_vqa_action(decision: str, action: str, params: dict | None, source: str) -> None:
    """把 Agent-VQA 控制器的高层决策映射到底层动作 (计划 5.3)。"""
    if decision == "answer" or decision == "abstain":
        if action == "report_observation":
            execute_action("report_observation", {
                "content": f"Agent-VQA: {decision}",
                "level": "info" if decision == "answer" else "warn",
            }, source=source)
        # stop 是控制器的逻辑终止符，不下发给飞行适配器。
        return
    if decision in ("continue_search", "reobserve") and params:
        execute_action("fly_relative", params, source=source)
        if decision == "reobserve":
            execute_action("detect_disaster", {}, source=source)


def run_agent_vqa_episode(question: str, source: str = "ai", item: dict | None = None) -> dict | None:
    """运行一次 Agent-VQA episode (计划 7.3)。

    返回最终 report dict（含 answer / decision / trajectory / budget），也通过
    socket 广播 agent_query_started / agent_query_update / agent_query_result。
    """
    if not state._task_lock.acquire(blocking=False):
        state.push_log("warn", "当前已有任务在执行，忽略新的 Agent-VQA 问题")
        return {"ok": False, "error": "busy", "question": question}

    task_started_ns = time.time_ns()
    spec = parse_question(question)
    report: dict | None = None
    try:
        state.is_executing = True
        state._stop_event.clear()
        state._sync_world_from_adapter()
        if SEMANTIC_MAP_ENABLED:
            snap = state.adapter.snapshot()
            state.semantic_map = SemanticMap(
                origin_lat=float(snap["lat"]), origin_lon=float(snap["lon"]),
                cell_size_m=SEMANTIC_MAP_CELL_M, instruction=question,
            )
        else:
            state.semantic_map = None
        state.emit_system_status()
        state.emit_world()
        state.push_log("info", f"收到 Agent-VQA 问题: {question}")
        socketio.emit("agent_query_started", {
            "question": question, "question_type": spec.question_type,
            "source": source, "ts_ns": task_started_ns,
            "ts_ms": task_started_ns // 1_000_000,
        })

        ctl = _make_agent_vqa_controller(source)
        # 逐步广播轨迹（不阻塞闭环）；最终结果由 agent_query_result 统一发出
        def _on_step(rec: dict) -> None:
            socketio.emit("agent_query_update", rec)
        ans = ctl.run(
            question, question_id=f"agentvqa_{task_started_ns}", item=item, on_step=_on_step,
        )
        # 执行最终动作
        last = ctl.trajectory[-1] if ctl.trajectory else None
        if last:
            _execute_agent_vqa_action(last.decision, last.action, None, source)

        failed_reasons = {"cancelled", "execution_error", "out_of_coverage", "planner_unavailable"}
        report = {
            "ok": ans.reason_code not in failed_reasons,
            "cancelled": ans.reason_code == "cancelled",
            "question": question,
            "question_type": spec.question_type,
            "answer": ans.to_dict(),
            "trajectory": ctl.trajectory_dicts(),
            "fallback_used": ctl.fallback_used,
            "degraded_reason": ctl.degraded_reason,
            "n_steps": len(ctl.trajectory),
        }
        socketio.emit("agent_query_result", report)
        log_level = "warn" if ans.reason_code == "cancelled" else "info"
        state.push_log(log_level, f"Agent-VQA 完成: decision={ans.decision} reason={ans.reason_code}")
    except Exception as exc:
        logger.exception("Agent-VQA episode failed")
        report = {"ok": False, "error": f"execution_error: {exc}", "question": question}
        socketio.emit("agent_query_result", report)
        state.push_log("error", f"Agent-VQA 失败: {exc}")
    finally:
        state.is_executing = False
        state._task_lock.release()
        state.emit_system_status()
    return report


def run_agent_vqa_episode_headless(
    question: str,
    start: dict,
    item: dict | None = None,
    source: str = "bench",
) -> dict:
    """无头评测入口：把 UAV 放到指定起点后同步跑一次 Agent-VQA episode (计划 7.3)。

    与 run_agent_vqa_episode 共享同一套感知/问答/搜索/重观测逻辑。item 仅在
    oracle 配置下用于诊断；非 oracle 时忽略 item 的 answer/target，不泄漏在线信息。
    """
    try:
        lat = float(start["lat"])
        lon = float(start["lon"])
        alt = float(start.get("alt", state.hover_altitude_m))
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"bad_start: {exc}", "question": question}

    entry = xbd_store.find_tile_containing(lat, lon, stage_priority=("post_disaster",))
    if entry is None:
        return {"ok": False, "error": "start_not_covered", "question": question,
                "start": {"lat": lat, "lon": lon, "alt": alt}}

    state.activate_xbd_tile(entry)
    state.adapter.reset_origin(lat, lon, alt=alt)
    state._sync_world_from_adapter()

    report = run_agent_vqa_episode(question, source=source, item=item)
    if isinstance(report, dict):
        report.setdefault("tile_id", entry.get("tile_id"))
        report.setdefault("disaster", entry.get("disaster"))
        report["start"] = {"lat": lat, "lon": lon, "alt": alt}
        if item is not None:
            # 离线评分字段：记录 GT 供评测脚本计算 corrected/harmed，在线控制器不读
            report["gt_answer"] = item.get("answer")
            report["gt_target"] = item.get("target")
    return report or {"ok": False, "error": "no_report", "question": question}


# ───────────────────────────── REST API ────────────────────────────────

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(state.system_status())


@app.route("/api/world", methods=["GET"])
def api_world():
    return jsonify(state.world_state())


@app.route("/api/semantic_map", methods=["GET"])
def api_semantic_map():
    """P0：返回当前 episode 的 2D 地理语义地图（无活动地图时返回 null）。

    ?full=1 返回完整序列化（含每格），否则返回精简 snapshot。
    """
    smap = getattr(state, "semantic_map", None)
    if smap is None:
        return jsonify({"active": False, "map": None})
    full = request.args.get("full", "0").lower() in {"1", "true", "yes", "on"}
    return jsonify({"active": True, "map": smap.to_dict() if full else smap.snapshot()})


@app.route("/api/memory_graph", methods=["GET"])
def api_memory_graph():
    """P3：返回记忆拓扑图（VLN_MEMORY 关闭时 enabled=False）。

    ?full=1 返回完整节点/边，否则只返回统计。
    """
    if not VLN_MEMORY:
        return jsonify({"enabled": False, "stats": None})
    graph = get_memory_graph()
    full = request.args.get("full", "0").lower() in {"1", "true", "yes", "on"}
    return jsonify({
        "enabled": True,
        "stats": graph.stats(),
        "graph": graph.to_dict() if full else None,
    })


@app.route("/api/logs", methods=["GET"])
def api_logs():
    with state._log_lock:
        return jsonify(state.log_buffer[-200:])


@app.route("/api/init", methods=["POST"])
def api_init():
    return jsonify({"ok": True, "status": state.system_status()})


@app.route("/api/llm/config", methods=["GET"])
def api_llm_config():
    modules = {
        module_name: _resolved_module_settings(module_name)
        for module_name in llm_config.MODULE_CONFIG
    }

    return jsonify(
        {
            "active_provider": llm_config.ACTIVE_PROVIDER,
            "providers": {
                name: {
                    "api_type": provider_cfg.get("api_type", ""),
                    "base_url": provider_cfg.get("base_url", ""),
                    "default_model": provider_cfg.get("default_model", ""),
                    "model_id": provider_cfg.get("model_id", ""),
                    "configured": bool(provider_cfg.get("api_key") or provider_cfg.get("model_id")),
                    "image_input_mode": provider_cfg.get("image_input_mode"),
                }
                for name, provider_cfg in llm_config.PROVIDERS.items()
            },
            "modules": modules,
        }
    )


@app.route("/api/vlm/analyze", methods=["POST"])
def api_vlm_analyze():
    upload = request.files.get("image")
    if upload is None:
        return jsonify({"ok": False, "error": "missing image file field 'image'"}), 400

    image_bytes = upload.read()
    if not image_bytes:
        return jsonify({"ok": False, "error": "empty image file"}), 400

    prompt = str(request.form.get("prompt", "")).strip()
    mime_type = upload.mimetype or "image/jpeg"

    try:
        result = VLMAnalyzer().analyze_image_bytes(
            image_bytes=image_bytes,
            mime_type=mime_type,
            prompt=prompt,
        )
    except Exception as exc:
        state.push_log("error", f"VLM analysis failed: {exc}")
        return jsonify({"ok": False, "error": str(exc)}), 502

    summary = result["analysis"][:120].replace("\n", " ")
    state.push_log(
        "success",
        f"VLM analyzed image with {result['model']}: {summary}",
        {"module": "vlm"},
    )
    return jsonify({"ok": True, **result})


@app.route("/api/agent/query", methods=["POST"])
def api_agent_query():
    """Agent-VQA REST 入口 (计划 7.3)。同步返回最终 report；socket 同时广播轨迹。"""
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"ok": False, "error": "missing 'question'"}), 400
    start = data.get("start")  # 可选: {"lat","lon","alt"}; 缺省用当前 UAV 位姿
    if isinstance(start, dict) and {"lat", "lon"} <= set(start.keys()):
        report = run_agent_vqa_episode_headless(question, start, source="api")
    else:
        report = run_agent_vqa_episode(question, source="api")
    if not isinstance(report, dict):
        return jsonify({"ok": False, "error": "no_report"}), 502
    return jsonify(report)


# ───────────────────────────── xBD API ─────────────────────────────────

def _manifest_missing_response() -> Response:
    hint = (
        "Build manifest.json or symlink AerialClaw/data/xbd/manifest.json to "
        f"{xbd_store.get_manifest_path()}"
    )
    return jsonify({"ok": False, "error": "xBD manifest not found", "hint": hint}), 404


def _parse_int_arg(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = request.args.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


@app.route("/api/xbd/catalog", methods=["GET"])
def api_xbd_catalog():
    manifest, manifest_path = xbd_store.load_cached()
    if not manifest:
        return _manifest_missing_response()

    split = request.args.get("split") or None
    disaster = request.args.get("disaster") or None
    disaster_type = request.args.get("disaster_type") or None
    stage = request.args.get("stage") or None
    georef = xbd_store.parse_bool(request.args.get("georef"))
    offset = _parse_int_arg("offset", 0, minimum=0)
    limit = _parse_int_arg("limit", 200, minimum=1, maximum=2000)

    items, total = xbd_store.filter_catalog(
        split=split,
        disaster=disaster,
        disaster_type=disaster_type,
        stage=stage,
        georef=georef,
        offset=offset,
        limit=limit,
    )
    return jsonify({
        "ok": True,
        "total": total,
        "offset": offset,
        "limit": limit,
        "manifest_path": manifest_path,
        "dataset_root": manifest.get("dataset_root"),
        "summary": manifest.get("summary", {}),
        "items": items,
    })


@app.route("/api/xbd/tiles/<tile_id>", methods=["GET"])
def api_xbd_tile(tile_id: str):
    manifest, _ = xbd_store.load_cached()
    if not manifest:
        return _manifest_missing_response()
    entry = xbd_store.get_entry(tile_id)
    if not entry:
        return jsonify({"ok": False, "error": f"tile '{tile_id}' not found"}), 404
    return jsonify({"ok": True, "item": entry})


@app.route("/api/xbd/images/<tile_id>", methods=["GET"])
def api_xbd_image(tile_id: str):
    manifest, _ = xbd_store.load_cached()
    if not manifest:
        return _manifest_missing_response()
    entry = xbd_store.get_entry(tile_id)
    if not entry:
        return jsonify({"ok": False, "error": f"tile '{tile_id}' not found"}), 404

    image_path = os.path.join(manifest["dataset_root"], entry["image_relpath"])
    if not os.path.exists(image_path):
        return jsonify({"ok": False, "error": f"image missing: {image_path}"}), 404
    response = send_file(image_path)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


@app.route("/api/xbd/annotations/<tile_id>", methods=["GET"])
def api_xbd_annotations(tile_id: str):
    manifest, _ = xbd_store.load_cached()
    if not manifest:
        return _manifest_missing_response()
    entry = xbd_store.get_entry(tile_id)
    if not entry:
        return jsonify({"ok": False, "error": f"tile '{tile_id}' not found"}), 404

    label_path = os.path.join(manifest["dataset_root"], entry["label_relpath"])
    if not os.path.exists(label_path):
        return jsonify({"ok": False, "error": f"label missing: {label_path}"}), 404

    try:
        geojson = build_annotation_geojson(label_path, entry)
    except Exception as exc:
        logger.exception("failed to build annotation geojson for %s", tile_id)
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({
        "ok": True,
        "tile_id": tile_id,
        "item": entry,
        "geojson": geojson,
    })


@app.route("/api/xbd/find-tile", methods=["GET"])
def api_xbd_find_tile():
    """查询覆盖 (lat, lon) 的瓦片。用于前端在 Ask AI Inspect 前做预检查。

    参数：
        lat / lon : float, 必填
        stage     : post | pre | any （默认 post）
    响应：
        { ok: true, covered: bool, entry: {...} | null, stage: "post_disaster" | "pre_disaster" | null }
    """
    try:
        lat = float(request.args.get("lat", ""))
        lon = float(request.args.get("lon", ""))
    except ValueError:
        return jsonify({"ok": False, "error": "lat/lon 必须为数值"}), 400

    stage_arg = (request.args.get("stage") or "post").lower()
    if stage_arg == "post":
        stage_priority = ("post_disaster",)
    elif stage_arg == "pre":
        stage_priority = ("pre_disaster",)
    else:
        stage_priority = ("post_disaster", "pre_disaster")

    entry = xbd_store.find_tile_containing(lat, lon, stage_priority=stage_priority)

    nearest = None
    if entry is None:
        # 计算最近的 POST 瓦片（中心距离），供前端提示一键跳转。
        manifest, _ = xbd_store.load_cached()
        if manifest:
            best_d2 = None
            best = None
            for item in manifest.get("items", []):
                if not item.get("has_georef"):
                    continue
                if not xbd_store._is_post(item):
                    continue
                b = item.get("bounds") or {}
                try:
                    clat = (float(b["north"]) + float(b["south"])) / 2.0
                    clon = (float(b["east"]) + float(b["west"])) / 2.0
                except Exception:
                    continue
                dlat = clat - lat
                dlon = clon - lon
                d2 = dlat * dlat + dlon * dlon
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = (item, clat, clon)
            if best is not None:
                item, clat, clon = best
                # 粗略把度差转成公里：lat 1° ≈ 111km；lon 按 cos(lat) 缩放。
                import math as _math

                km = _math.sqrt(
                    ((clat - lat) * 111.0) ** 2
                    + ((clon - lon) * 111.0 * _math.cos(_math.radians(lat))) ** 2
                )
                nearest = {
                    "tile_id": item.get("tile_id"),
                    "disaster": item.get("disaster"),
                    "disaster_type": item.get("disaster_type"),
                    "center": {"lat": clat, "lon": clon},
                    "distance_km": round(km, 2),
                }

    return jsonify(
        {
            "ok": True,
            "covered": entry is not None,
            "entry": entry,
            "stage": entry.get("stage") if entry else None,
            "nearest": nearest,
        }
    )


def _damage_ranking_path() -> str:
    """跟着当前 DATASET_MODE 走：xbd 读 data/xbd/damage_ranking.json；
    rescuenet 模式读 data/rescuenet/damage_ranking.json。"""
    return str(xbd_store.resolve_output_dir() / "damage_ranking.json")



@app.route("/api/xbd/damage-ranking", methods=["GET"])
def api_xbd_damage_ranking():
    """读取 scripts/rank_damage_tiles.py 生成的排名 JSON。

    查询参数：
        limit     : int   默认 50，最多返回多少条
        disaster  : str   可选，按 disaster 名前缀过滤
        min_dest  : int   可选，最少 destroyed 栋数
    """
    ranking_path = _damage_ranking_path()
    if not os.path.isfile(ranking_path):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": (
                        f"damage_ranking.json not found at {ranking_path}; run "
                        "`python scripts/rank_damage_tiles.py` (xBD) or "
                        "`python scripts/build_rescuenet_dataset.py --force` (RescueNet) to generate it."
                    ),
                }
            ),
            404,
        )
    try:
        with open(ranking_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"failed to read ranking: {exc}"}), 500

    items = list(data.get("items") or [])

    disaster = (request.args.get("disaster") or "").strip().lower()
    if disaster:
        items = [row for row in items if disaster in str(row.get("disaster") or "").lower()]

    try:
        min_dest = int(request.args.get("min_dest") or 0)
    except ValueError:
        min_dest = 0
    if min_dest > 0:
        items = [row for row in items if int((row.get("counts") or {}).get("destroyed", 0)) >= min_dest]

    try:
        limit = int(request.args.get("limit") or 50)
    except ValueError:
        limit = 50
    limit = max(1, min(limit, 500))

    return jsonify(
        {
            "ok": True,
            "total": len(items),
            "limit": limit,
            "generated_from": data.get("generated_from"),
            "total_post_tiles": data.get("total_post_tiles"),
            "weights": data.get("weights"),
            "items": items[:limit],
        }
    )


@app.route("/api/xbd/activate/<tile_id>", methods=["POST"])
def api_xbd_activate(tile_id: str):
    manifest, _ = xbd_store.load_cached()
    if not manifest:
        return _manifest_missing_response()
    entry = xbd_store.get_entry(tile_id)
    if not entry:
        return jsonify({"ok": False, "error": f"tile '{tile_id}' not found"}), 404
    if not entry.get("has_georef"):
        return jsonify({"ok": False, "error": f"tile '{tile_id}' has no georef"}), 400
    if xbd_store.POST_ONLY_MODE and not xbd_store._is_post(entry):
        return jsonify({
            "ok": False,
            "error": (
                f"tile '{tile_id}' is pre_disaster; POST_ONLY_MODE is enabled "
                "(set XBD_POST_ONLY=0 to allow)."
            ),
        }), 400

    world_state = state.activate_xbd_tile(entry)
    state.push_log(
        "success",
        f"激活 xBD 瓦片 {tile_id} ({entry.get('disaster')} / {entry.get('stage')})",
        {"module": "map", "tile_id": tile_id},
    )
    socketio.emit("world_state", world_state)
    socketio.emit("system_status", state.system_status())
    return jsonify({"ok": True, "item": entry, "world_state": world_state})


@app.route("/api/perception/view/<path:fname>", methods=["GET"])
def api_perception_view(fname: str):
    """只读：serve outputs/uav_view 目录下的 patch / 可视化 PNG。"""
    safe = os.path.basename(fname)
    full = PERCEPTION_OUTPUT_DIR / safe
    if not full.exists():
        return jsonify({"ok": False, "error": f"perception file not found: {safe}"}), 404
    response = send_from_directory(str(PERCEPTION_OUTPUT_DIR), safe)
    response.headers["Cache-Control"] = "public, max-age=120"
    return response


@app.route("/api/xbd/footprints.geojson", methods=["GET"])
def api_xbd_footprints():
    path = xbd_store.get_footprints_path()
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": f"footprints file missing: {path}"}), 404

    # POST_ONLY_MODE 下需要把 PRE 灾前 features 过滤掉，并在服务端完成，
    # 前端无需再做 stage 分流。
    if xbd_store.POST_ONLY_MODE:
        try:
            with open(path, "r", encoding="utf-8") as fp:
                geojson = json.load(fp)
        except Exception as exc:
            return jsonify({"ok": False, "error": f"footprints parse failed: {exc}"}), 500

        raw_features = geojson.get("features") or []
        kept = [
            feat for feat in raw_features
            if str((feat.get("properties") or {}).get("stage") or "").lower()
            in {"post_disaster", "post"}
        ]
        geojson["features"] = kept
        response = jsonify(geojson)
        response.mimetype = "application/geo+json"
        # POST_ONLY_MODE 状态可能被切换；短缓存 + must-revalidate 避免浏览器
        # 继续吃老的含 PRE 的版本。
        response.headers["Cache-Control"] = "public, max-age=60, must-revalidate"
        return response

    response = send_file(path, mimetype="application/geo+json")
    response.headers["Cache-Control"] = "public, max-age=60, must-revalidate"
    return response


# ───────────────────────────── Elevation ───────────────────────────────

@app.route("/api/elevation", methods=["GET"])
def api_elevation():
    if ELEVATION_DISABLED:
        return jsonify({"ok": False, "elevation": None, "error": "elevation disabled"})

    try:
        lat = float(request.args.get("lat", ""))
        lon = float(request.args.get("lon", ""))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "elevation": None, "error": "invalid lat/lon"}), 400

    try:
        resp = requests.get(
            ELEVATION_URL,
            params={"latitude": lat, "longitude": lon},
            timeout=ELEVATION_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        values = payload.get("elevation")
        elevation = None
        if isinstance(values, list) and values:
            elevation = float(values[0])
        elif isinstance(values, (int, float)):
            elevation = float(values)
        return jsonify({
            "ok": elevation is not None,
            "elevation": elevation,
            "lat": lat,
            "lon": lon,
            "source": "open-meteo",
        })
    except Exception as exc:
        logger.warning("elevation lookup failed: %s", exc)
        return jsonify({
            "ok": False,
            "elevation": None,
            "lat": lat,
            "lon": lon,
            "error": str(exc),
        })


# ───────────────────────────── Socket.IO ───────────────────────────────

@socketio.on("connect")
def socket_connect():
    emit("system_status", state.system_status())
    emit("world_state", state.world_state())


@socketio.on("set_mode")
def on_set_mode(data):
    mode = (data or {}).get("mode", "manual")
    if mode not in {"manual", "ai"}:
        mode = "manual"
    state.mode = mode
    state.emit_system_status()


@socketio.on("select_robot")
def on_select_robot(data):
    robot_id = (data or {}).get("robot_id", "UAV_1")
    state.current_robot = robot_id
    state.emit_system_status()


@socketio.on("execute_action")
def on_execute_action(data):
    payload = data or {}
    action = payload.get("action", "")
    params = payload.get("params", {})
    start_background_job(run_action_sequence, action, [{"action": action, "params": params, "reason": "manual action"}], "manual", action)


def _dispatch_ai_task(task: str) -> None:
    submit_ns = time.time_ns()
    state.push_log("info", f"收到 AI 任务: {task}")
    socketio.emit(
        "ai_thinking",
        {
            "phase": "planning",
            "detail": "LLM 正在规划...",
            "submit_ns": submit_ns,
            "submit_ms": submit_ns // 1_000_000,
        },
    )
    try:
        plan = state.planner.plan(task, state.world_state())
    except Exception as exc:
        logger.exception("ai_task planner crashed")
        state.push_log("error", f"AI planner 异常: {exc}")
        socketio.emit("ai_thinking", {"phase": "idle", "detail": ""})
        return

    plan_done_ns = time.time_ns()
    plan_wall_ms = (plan_done_ns - submit_ns) // 1_000_000

    if "（LLM 不可用" in plan.get("summary", ""):
        state.push_log("warn", "LLM 不可用，已切换规则规划（详情见后端日志）")

    steps = plan.get("steps", []) or []
    plan_summary = plan.get("summary") or task
    action_chain = " → ".join(
        (step.get("action") or "?") for step in steps
    ) or "(无步骤)"
    state.push_log(
        "info",
        f"规划摘要: {plan_summary}（{len(steps)} 步: {action_chain}, 规划耗时 {plan_wall_ms} ms）",
    )

    plan_payload = dict(plan)
    plan_payload["plan_wall_ms"] = plan_wall_ms
    plan_payload["submit_ns"] = submit_ns
    plan_payload["plan_done_ns"] = plan_done_ns
    socketio.emit("ai_plan_result", plan_payload)
    socketio.emit("ai_thinking", {"phase": "planning", "detail": plan.get("summary", "")})
    run_action_sequence(plan_summary, steps, "ai", task)


@socketio.on("ai_task")
def on_ai_task(data):
    payload = data or {}
    task = str(payload.get("task", "")).strip()
    if not task:
        return
    state.mode = "ai"
    start_background_job(_dispatch_ai_task, task)


@socketio.on("vln_task")
def on_vln_task(data):
    payload = data or {}
    instruction = str(payload.get("instruction") or payload.get("task") or "").strip()
    if not instruction:
        return
    state.mode = "ai"
    start_background_job(run_vln_episode, instruction)


@socketio.on("agent_query")
def on_agent_query(data):
    """Agent-VQA Socket 入口 (计划 7.3 / 8.2)。"""
    payload = data or {}
    question = str(payload.get("question") or "").strip()
    if not question:
        return
    state.mode = "ai"
    start_background_job(run_agent_vqa_episode, question)


@socketio.on("stop_execution")
def on_stop_execution():
    state._stop_event.set()
    state.push_log("warn", "收到停止请求")


# ───────────────────────────── Frontend ────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    candidate = FRONTEND_DIST / path
    if path and candidate.exists():
        return send_from_directory(FRONTEND_DIST, path)
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return Response(
        "DisasterClaw backend is running. Build frontend in ./frontend first.",
        mimetype="text/plain",
    )


if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", os.getenv("DISASTERCLAW_PORT", "5011")))
    socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)
