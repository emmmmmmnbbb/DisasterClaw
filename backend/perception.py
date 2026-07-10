"""
backend/perception.py — UAV 视觉感知管线（YOLO + SegFormer + SceneDescriptor）

对齐 rs_agent_system.layers.L1_feature_extraction：
    - 复用 mars tools 里的 YOLOTool / SegFormerTool
    - 直接 import rs_agent_system.layers.L1_feature_extraction.SceneDescriptor
    - 进程级单例 + 懒加载 + 可选预热

核心入口：
    get_perception().perceive_at(lat, lon, alt, active_tile, patch_id)
返回结构化 PerceptionResult（含 patch_url / overlay_url / scene_dict / scene_text /
risk_level / risk_summary），供 app.execute_action 在 detect_disaster 分支中写
push_log + world.add_report + socket perception_result。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

import xbd_store
from xbd_map import geo_to_pixel

logger = logging.getLogger(__name__)


# ── 环境路径 ────────────────────────────────────────────────────────────

BACKEND_DIR = Path(__file__).resolve().parent

RS_AGENT_ROOT = Path(os.getenv("RS_AGENT_SYSTEM_PATH", "/home/lc/rs_agent_system"))
VISION_TOOLS_PATH = Path(
    os.getenv("VISION_TOOLS_PATH", "/home/lc/Langchain-Chatchat/tools/mars/tools")
)
VISION_RESULTS_PATH = Path(
    os.getenv(
        "VISION_RESULTS_PATH",
        "/home/lc/Langchain-Chatchat/tools/mars/results/yolo",
    )
)
YOLO_WEIGHTS = os.getenv(
    "YOLO_WEIGHTS",
    "/home/lc/Langchain-Chatchat/tools/mars/results/yolo/runs/train/mars_det_yolov8n4/weights/best.pt",
)
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
# 推理分辨率：xBD 域内权重在 1024² 卫星图上训练，小目标需大 imgsz；RescueNet 旧权重用 640。
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
SEGFORMER_MODEL = os.getenv(
    "SEGFORMER_MODEL", "nvidia/segformer-b2-finetuned-ade-512-512"
)
ADE20K_CLASSES_PATH = os.getenv(
    "ADE20K_CLASSES_PATH",
    "/home/lc/Langchain-Chatchat/tools/mars/tools/ade20k_classes.json",
)
_raw_output_dir = Path(
    os.getenv("PERCEPTION_OUTPUT_DIR", str(BACKEND_DIR / "outputs" / "uav_view"))
).expanduser()
if not _raw_output_dir.is_absolute():
    _raw_output_dir = (BACKEND_DIR.parent / _raw_output_dir).resolve()
PERCEPTION_OUTPUT_DIR = _raw_output_dir
PERCEPTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERCEPTION_DEVICE = os.getenv("PERCEPTION_DEVICE", "cuda")
PERCEPTION_VIEW_ALT_FACTOR = float(os.getenv("PERCEPTION_VIEW_ALT_FACTOR", "2.0"))
PERCEPTION_MIN_RADIUS_M = float(os.getenv("PERCEPTION_MIN_RADIUS_M", "20"))
PERCEPTION_MAX_RADIUS_M = float(os.getenv("PERCEPTION_MAX_RADIUS_M", "300"))
PERCEPTION_MIN_PATCH_PX = int(os.getenv("PERCEPTION_MIN_PATCH_PX", "256"))
PERCEPTION_MAX_PATCH_PX = int(os.getenv("PERCEPTION_MAX_PATCH_PX", "1024"))

# 检测器 raw class_name → 中文标签。兼容两套权重：
#   - RescueNet（低空斜拍）：type_* 索引名
#   - xBD 域内微调（卫星正射）：英文 damage subtype（gen_xbd_yolo_dataset.py 的类名）
# perception._detect 用 .get(raw, raw) 兜底，grounding 按中文类匹配，故只需在此加映射。
YOLO_LABEL_MAP: dict[str, str] = {
    "type_2": "无损伤建筑",
    "type_3": "轻微损伤建筑",
    "type_4": "严重损伤建筑",
    "type_5": "完全损毁建筑",
    "type_6": "车辆",
    "type_10": "水池/积水区域",
    # xBD 域内检测器类名
    "no-damage": "无损伤建筑",
    "minor-damage": "轻微损伤建筑",
    "major-damage": "严重损伤建筑",
    "destroyed": "完全损毁建筑",
}
ADE20K_LABELS_ZH: dict[str, str] = {
    "wall": "墙体",
    "building": "建筑",
    "sky": "天空",
    "floor": "地面",
    "tree": "树木",
    "road": "道路",
    "grass": "草地",
    "earth": "裸土",
    "plant": "植被",
    "water": "水体",
    "house": "房屋",
    "sea": "海面",
    "field": "农田",
    "car": "汽车",
    "person": "人员",
    "truck": "卡车",
    "bus": "公交车",
    "boat": "船只",
    "bridge": "桥梁",
    "river": "河流",
    "lake": "湖泊",
    "sand": "沙地",
    "rubble": "瓦砾",
    "rock": "岩石",
    "roof": "屋顶",
}


# ── 延迟 import：YOLOTool / SegFormerTool / SceneDescriptor ──────────

def _inject_sys_path() -> None:
    for p in (RS_AGENT_ROOT, VISION_TOOLS_PATH, VISION_RESULTS_PATH):
        s = str(p)
        if s and s not in sys.path:
            sys.path.insert(0, s)


_YOLO_CLS = None
_SEG_CLS = None
_DESC_CLS = None


def _lazy_import_backends() -> tuple[Any, Any, Any]:
    global _YOLO_CLS, _SEG_CLS, _DESC_CLS
    if _YOLO_CLS is not None and _SEG_CLS is not None and _DESC_CLS is not None:
        return _YOLO_CLS, _SEG_CLS, _DESC_CLS

    _inject_sys_path()
    errors: list[str] = []
    yolo_cls = seg_cls = desc_cls = None
    try:
        from yolo_tool import YOLOTool  # type: ignore
        yolo_cls = YOLOTool
    except Exception as exc:
        errors.append(f"YOLOTool import failed: {exc}")
    try:
        from segformer_tool import SegFormerTool  # type: ignore
        seg_cls = SegFormerTool
    except Exception as exc:
        errors.append(f"SegFormerTool import failed: {exc}")
    # SceneDescriptor 直接从 .py 文件按路径加载，避开 rs_agent_system 包里
    # `from .vision_tools import VisionTools` 触发的 `from config import ...`
    # （那个 `config` 会撞上 disasterclaw/backend/config.py）。
    try:
        import importlib.util

        sd_path = RS_AGENT_ROOT / "layers" / "L1_feature_extraction" / "scene_descriptor.py"
        if not sd_path.exists():
            raise FileNotFoundError(str(sd_path))
        spec = importlib.util.spec_from_file_location(
            "rs_agent_scene_descriptor", str(sd_path)
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        desc_cls = module.SceneDescriptor
    except Exception as exc:
        errors.append(f"SceneDescriptor import failed: {exc}")

    if errors:
        raise RuntimeError("; ".join(errors))

    _YOLO_CLS, _SEG_CLS, _DESC_CLS = yolo_cls, seg_cls, desc_cls
    return yolo_cls, seg_cls, desc_cls


# ── 结果数据结构 ───────────────────────────────────────────────────────

_RISK_LEVEL_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3}


@dataclass
class PerceptionResult:
    patch_id: str
    patch_path: str
    patch_url: str
    overlay_path: Optional[str]
    overlay_url: Optional[str]
    detection_path: Optional[str]
    detection_url: Optional[str]
    patch_width: int
    patch_height: int
    patch_radius_m: float
    detection: dict
    segmentation: dict
    scene_dict: dict
    scene_text: str
    risk_level: str
    risk_summary: str
    damaged_buildings: int
    intact_buildings: int
    vehicles: int
    water_pixels: int
    degraded: bool = False
    degraded_reason: str = ""
    extras: dict = field(default_factory=dict)


# ── 核心类 ─────────────────────────────────────────────────────────────


class DisasterPerception:
    """
    进程级单例，封装：
        - YOLOTool（RescueNet 6 类）
        - SegFormerTool（ADE20K 150 类）
        - rs_agent_system.SceneDescriptor
        - 从活动 xBD 瓦片按 (lat,lon,alt) 裁 patch
    """

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._yolo = None
        self._segformer = None
        self._descriptor = None
        self._ade20k_names: dict[int, str] = {}
        self._ready = False
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    #  加载 / 预热
    # ------------------------------------------------------------------ #

    @property
    def is_available(self) -> bool:
        return self._yolo is not None and self._segformer is not None

    def load(self) -> None:
        """同步加载模型。对 warmup 线程来说，失败时记录错误但不抛出。"""
        if self._ready:
            return
        with self._load_lock:
            if self._ready:
                return
            try:
                YOLOTool, SegFormerTool, SceneDescriptor = _lazy_import_backends()
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("perception backend import failed")
                raise

            try:
                with open(ADE20K_CLASSES_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._ade20k_names = {int(k): str(v).strip() for k, v in raw.items()}
            except Exception as exc:
                logger.warning("ADE20K 类别文件加载失败: %s (分割标签将退化为索引)", exc)
                self._ade20k_names = {}

            # Force-resolve transformers lazy imports before SegFormerTool
            # checks them — avoids a race with the parallel Qwen-VL warmup
            # thread that also imports transformers. Also patch the
            # already-imported segformer_tool module's HAS_TRANSFORMERS flag
            # in case it captured False during early concurrent startup.
            try:
                from transformers import (  # type: ignore
                    SegformerImageProcessor,
                    SegformerForSemanticSegmentation,
                )
                import segformer_tool as _seg_mod  # type: ignore
                _seg_mod.HAS_TRANSFORMERS = True
                if not hasattr(_seg_mod, "SegformerImageProcessor"):
                    _seg_mod.SegformerImageProcessor = SegformerImageProcessor
                if not hasattr(_seg_mod, "SegformerForSemanticSegmentation"):
                    _seg_mod.SegformerForSemanticSegmentation = (
                        SegformerForSemanticSegmentation
                    )
            except Exception as exc:
                logger.warning(
                    "eager transformers import failed in perception worker: %s", exc
                )

            device_arg = "0" if PERCEPTION_DEVICE.startswith("cuda") else "cpu"
            if PERCEPTION_DEVICE.startswith("cuda:"):
                device_arg = PERCEPTION_DEVICE.split(":", 1)[1]

            logger.info("[Perception] loading YOLO weights: %s", YOLO_WEIGHTS)
            self._yolo = YOLOTool(
                weights=YOLO_WEIGHTS,
                device=device_arg,
                conf=YOLO_CONF_THRESHOLD,
                imgsz=YOLO_IMGSZ,
            )
            logger.info("[Perception] loading SegFormer: %s", SEGFORMER_MODEL)
            self._segformer = SegFormerTool(
                model_name=SEGFORMER_MODEL,
                device=PERCEPTION_DEVICE if PERCEPTION_DEVICE else "cuda",
            )
            self._descriptor = SceneDescriptor()
            self._ready = True
            self._last_error = None
            logger.info("[Perception] ready")

    def last_error(self) -> Optional[str]:
        return self._last_error

    # ------------------------------------------------------------------ #
    #  裁 patch
    # ------------------------------------------------------------------ #

    def _resolve_tile_image_path(self, active_tile: dict) -> Optional[Path]:
        tile_id = active_tile.get("tile_id")
        if not tile_id:
            return None
        entry = xbd_store.get_entry(tile_id)
        manifest, _ = xbd_store.load_cached()
        if not entry or not manifest:
            return None
        relpath = entry.get("image_relpath")
        root = manifest.get("dataset_root")
        if not relpath or not root:
            return None
        path = Path(root) / relpath
        if not path.exists():
            return None
        return path

    def _crop_uav_view(
        self,
        lat: float,
        lon: float,
        alt: float,
        active_tile: dict,
        patch_id: str,
    ) -> tuple[Path, int, int, float, bool, str]:
        """
        根据 UAV 位置从活动瓦片裁剪视场 patch。

        返回 (patch_path, patch_w, patch_h, radius_m, degraded, degraded_reason)
        degraded=True 时表示拿不到仿射 / 瓦片图，回退到占位或整图。
        """
        image_path = self._resolve_tile_image_path(active_tile)
        if image_path is None:
            raise FileNotFoundError(
                f"active tile image not resolvable from manifest: {active_tile.get('tile_id')}"
            )

        transform = {
            "pixel_to_geo": active_tile.get("pixel_to_geo"),
            "geo_to_pixel": active_tile.get("geo_to_pixel"),
        }
        gsd = float(active_tile.get("gsd") or 0.5)  # 默认 xBD ≈ 0.5 m/px

        with Image.open(image_path) as im:
            im.load()
            img = im.convert("RGB")
        W, H = img.size

        degraded = False
        degraded_reason = ""
        radius_m = max(
            PERCEPTION_MIN_RADIUS_M,
            min(PERCEPTION_MAX_RADIUS_M, alt * PERCEPTION_VIEW_ALT_FACTOR),
        )

        if transform["geo_to_pixel"] is None:
            degraded = True
            degraded_reason = "tile lacks geo_to_pixel; using full image"
            patch = img.copy()
            patch_w, patch_h = patch.size
        else:
            cx, cy = geo_to_pixel(transform, lon, lat)
            radius_px = int(max(PERCEPTION_MIN_PATCH_PX // 2, radius_m / max(gsd, 1e-3)))
            radius_px = min(radius_px, PERCEPTION_MAX_PATCH_PX // 2, max(W, H) // 2)

            left = int(round(cx - radius_px))
            top = int(round(cy - radius_px))
            right = int(round(cx + radius_px))
            bottom = int(round(cy + radius_px))

            # UAV 落点像素中心是否在瓦片内？
            cx_in = 0 <= cx < W
            cy_in = 0 <= cy < H
            if not (cx_in and cy_in):
                # 把 UAV 落点 clamp 到瓦片边界内最近的合法像素，
                # 围绕这个 clamped 中心再裁一个 PERCEPTION_MIN_PATCH_PX 方块。
                # 相比"裁瓦片中心"的退化回退，这会尽量接近用户真实意图（贴边）。
                half = max(PERCEPTION_MIN_PATCH_PX // 2, radius_px)
                half = min(half, PERCEPTION_MAX_PATCH_PX // 2, max(W, H) // 2)
                half = max(1, min(half, min(W, H) // 2 - 1))

                cx_cl = max(half, min(W - half - 1, int(round(cx))))
                cy_cl = max(half, min(H - half - 1, int(round(cy))))
                left_c = cx_cl - half
                top_c = cy_cl - half
                right_c = cx_cl + half
                bottom_c = cy_cl + half
                degraded = True
                degraded_reason = (
                    f"UAV position (lat={lat:.6f}, lon={lon:.6f}) "
                    f"outside active tile; clamped to tile-edge pixel "
                    f"({cx_cl}, {cy_cl}) and cropped a {2 * half}x{2 * half} patch."
                )
            else:
                # 常规路径：UAV 落在瓦片内。clamp 半径，避免 crop 越界。
                left_c = max(0, min(W - 1, left))
                top_c = max(0, min(H - 1, top))
                right_c = max(left_c + 1, min(W, right))
                bottom_c = max(top_c + 1, min(H, bottom))
                if (right_c - left_c) < PERCEPTION_MIN_PATCH_PX // 2 or (
                    bottom_c - top_c
                ) < PERCEPTION_MIN_PATCH_PX // 2:
                    # 贴边时 clamp 之后 patch 过小 → 围绕 UAV 像素重新扩开一点
                    half = max(
                        PERCEPTION_MIN_PATCH_PX // 2,
                        max(1, min(W, H) // 2 - 1),
                    )
                    cx_i = int(round(cx))
                    cy_i = int(round(cy))
                    left_c = max(0, cx_i - half)
                    top_c = max(0, cy_i - half)
                    right_c = min(W, cx_i + half)
                    bottom_c = min(H, cy_i + half)
                    degraded = True
                    degraded_reason = (
                        f"UAV too close to tile edge; recentered on UAV pixel "
                        f"({cx_i}, {cy_i}) for minimum {PERCEPTION_MIN_PATCH_PX}px patch."
                    )

            patch = img.crop((left_c, top_c, right_c, bottom_c))
            patch_w, patch_h = patch.size

        patch_path = PERCEPTION_OUTPUT_DIR / f"{patch_id}.png"
        patch.save(patch_path, "PNG")
        return patch_path, patch_w, patch_h, radius_m, degraded, degraded_reason

    # ------------------------------------------------------------------ #
    #  YOLO / SegFormer 调度（与 rs_agent_system.VisionTools 等价）
    # ------------------------------------------------------------------ #

    def _detect(self, patch_path: Path) -> dict:
        out_dir = PERCEPTION_OUTPUT_DIR
        base = patch_path.stem
        vis_path = str(out_dir / f"{base}_det.png")
        json_path = str(out_dir / f"{base}_det.json")
        raw_detections = self._yolo.detect(
            image_path=str(patch_path),
            save_vis=vis_path,
            save_json=json_path,
        )

        detections: list[dict] = []
        for det in raw_detections:
            raw_cls = str(det.get("class_name") or "")
            mapped_cls = YOLO_LABEL_MAP.get(raw_cls, raw_cls)
            bbox_xyxy = det.get("bbox_xyxy") or det.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            detections.append(
                {
                    "class_id": det.get("class_id"),
                    "class_name": mapped_cls,
                    "raw_class_name": raw_cls,
                    "conf": float(det.get("conf", 0.0)),
                    "bbox": [float(v) for v in bbox_xyxy],
                    "bbox_xyxy": [float(v) for v in bbox_xyxy],
                }
            )

        class_counts: dict[str, int] = {}
        for det in detections:
            cls = det["class_name"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        return {
            "detections": detections,
            "num_objects": len(detections),
            "class_counts": class_counts,
            "visualization": vis_path if Path(vis_path).exists() else None,
            "json_file": json_path if Path(json_path).exists() else None,
        }

    def _segment(self, patch_path: Path) -> dict:
        out_dir = PERCEPTION_OUTPUT_DIR
        base = patch_path.stem
        out_prefix = str(out_dir / f"{base}_seg")
        result = self._segformer.segment(
            image=str(patch_path),
            return_mask=True,
            return_overlay=True,
            output_path=out_prefix,
            colormap="high_contrast",
        )

        raw_stats = result.get("stats", {}) or {}
        named_stats: dict[str, int] = {}
        for idx, px in raw_stats.items():
            idx_int = int(idx)
            en_name = self._ade20k_names.get(idx_int, str(idx_int))
            zh_name = ADE20K_LABELS_ZH.get(en_name, en_name)
            named_stats[zh_name] = named_stats.get(zh_name, 0) + int(px)

        total_px = sum(named_stats.values()) or 1
        named_stats = {
            k: v for k, v in named_stats.items() if v / total_px > 0.005
        }

        mask_path = f"{out_prefix}_mask.png"
        overlay_path = f"{out_prefix}_overlay.png"
        return {
            "mask_image": mask_path if Path(mask_path).exists() else None,
            "overlay_image": overlay_path if Path(overlay_path).exists() else None,
            "stats": named_stats,
            "raw_stats": {str(k): int(v) for k, v in raw_stats.items()},
            "num_labels": len(named_stats),
        }

    # ------------------------------------------------------------------ #
    #  风险汇总
    # ------------------------------------------------------------------ #

    @staticmethod
    def _summarise_risk(
        class_counts: dict[str, int],
        seg_stats: dict[str, int],
    ) -> tuple[str, str, int, int, int, int]:
        intact = class_counts.get("无损伤建筑", 0)
        minor = class_counts.get("轻微损伤建筑", 0)
        major = class_counts.get("严重损伤建筑", 0)
        destroyed = class_counts.get("完全损毁建筑", 0)
        damaged_total = minor + major + destroyed
        vehicles = class_counts.get("车辆", 0)

        water_px_ann = class_counts.get("水池/积水区域", 0)  # YOLO 像素数？其实是数量
        water_px_seg = 0
        for key in ("水体", "水池/积水区域", "水", "water"):
            water_px_seg += int(seg_stats.get(key, 0))

        if destroyed >= 1 or major >= 3:
            risk_level = "high"
        elif major >= 1 or damaged_total >= 3 or water_px_ann >= 2:
            risk_level = "moderate"
        elif minor >= 1 or (water_px_ann >= 1 and intact >= 1):
            risk_level = "low"
        else:
            risk_level = "none"

        bits: list[str] = []
        if damaged_total > 0:
            parts = []
            if destroyed:
                parts.append(f"完全损毁 {destroyed}")
            if major:
                parts.append(f"严重损伤 {major}")
            if minor:
                parts.append(f"轻微损伤 {minor}")
            bits.append("受损建筑 " + " / ".join(parts))
        if intact > 0 and not damaged_total:
            bits.append(f"{intact} 栋建筑外观完好")
        elif intact > 0:
            bits.append(f"另有 {intact} 栋外观完好")
        if vehicles > 0:
            bits.append(f"车辆 {vehicles} 辆")
        if water_px_ann > 0:
            bits.append(f"积水/水池目标 {water_px_ann} 处")
        elif water_px_seg > 0:
            bits.append("分割图中出现水体")

        risk_phrase = {
            "high": "判定：灾情较重",
            "moderate": "判定：局部受灾",
            "low": "判定：轻度受灾或可疑",
            "none": "判定：未发现明显受灾",
        }[risk_level]
        summary = (
            risk_phrase + "；" + "，".join(bits) + "。" if bits else risk_phrase + "。"
        )
        return risk_level, summary, damaged_total, intact, vehicles, water_px_seg

    # ------------------------------------------------------------------ #
    #  入口
    # ------------------------------------------------------------------ #

    def perceive_at(
        self,
        lat: float,
        lon: float,
        alt: float,
        active_tile: dict,
        patch_id: str,
    ) -> PerceptionResult:
        self.load()  # 幂等
        crop_t0 = time.perf_counter_ns()
        patch_path, pw, ph, radius_m, degraded, degraded_reason = self._crop_uav_view(
            lat=lat,
            lon=lon,
            alt=alt,
            active_tile=active_tile or {},
            patch_id=patch_id,
        )
        crop_ms = (time.perf_counter_ns() - crop_t0) // 1_000_000

        det_t0 = time.perf_counter_ns()
        detection = self._detect(patch_path)
        det_ms = (time.perf_counter_ns() - det_t0) // 1_000_000

        seg_t0 = time.perf_counter_ns()
        segmentation = self._segment(patch_path)
        seg_ms = (time.perf_counter_ns() - seg_t0) // 1_000_000

        desc_t0 = time.perf_counter_ns()
        perception_result = {
            "detection": {
                "detections": detection["detections"],
                "class_counts": detection["class_counts"],
            },
            "segmentation": {"stats": segmentation["stats"]},
        }
        scene = self._descriptor.generate(perception_result, image_width=pw, image_height=ph)
        scene_dict = scene.to_dict()
        scene_text = self._descriptor.format_for_llm(scene)
        desc_ms = (time.perf_counter_ns() - desc_t0) // 1_000_000

        (
            risk_level,
            risk_summary,
            damaged_total,
            intact,
            vehicles,
            water_px,
        ) = self._summarise_risk(detection["class_counts"], segmentation["stats"])

        patch_url = f"/api/perception/view/{patch_path.name}"
        overlay_path = segmentation.get("overlay_image")
        det_vis_path = detection.get("visualization")

        overlay_url = (
            f"/api/perception/view/{Path(overlay_path).name}"
            if overlay_path and Path(overlay_path).exists()
            else None
        )
        det_url = (
            f"/api/perception/view/{Path(det_vis_path).name}"
            if det_vis_path and Path(det_vis_path).exists()
            else None
        )

        return PerceptionResult(
            patch_id=patch_id,
            patch_path=str(patch_path),
            patch_url=patch_url,
            overlay_path=overlay_path,
            overlay_url=overlay_url,
            detection_path=det_vis_path,
            detection_url=det_url,
            patch_width=pw,
            patch_height=ph,
            patch_radius_m=radius_m,
            detection=detection,
            segmentation=segmentation,
            scene_dict=scene_dict,
            scene_text=scene_text,
            risk_level=risk_level,
            risk_summary=risk_summary,
            damaged_buildings=damaged_total,
            intact_buildings=intact,
            vehicles=vehicles,
            water_pixels=water_px,
            degraded=degraded,
            degraded_reason=degraded_reason,
            extras={
                "crop_ms": int(crop_ms),
                "det_ms": int(det_ms),
                "seg_ms": int(seg_ms),
                "desc_ms": int(desc_ms),
                "total_ms": int(crop_ms + det_ms + seg_ms + desc_ms),
            },
        )


# ── 进程级单例 ─────────────────────────────────────────────────────────

_INSTANCE: Optional[DisasterPerception] = None
_INSTANCE_LOCK = threading.Lock()


def get_perception() -> DisasterPerception:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = DisasterPerception()
    return _INSTANCE


def level_for_risk(risk: str) -> str:
    """把 risk_level 映射成 push_log 的 level。"""
    return {
        "high": "error",
        "moderate": "warn",
        "low": "info",
        "none": "success",
    }.get(risk, "info")
