"""backend/detectors/base.py — 检测器后端的统一契约 (计划 §3.4)

所有后端（legacy_unet / xview2_first / changeos）必须返回同一种 `Detection`，
字段与 `perception._detect()` 既有输出对齐，以免上层需要分支。

**`class_probs` 是硬要求，不是可选项。** `recheck.py` 的 entropy / info_gain
触发模式与 Agent-VQA 的 A3_ENTROPY 策略全靠它；缺了它这些策略会静默退化成
常量策略，产生"看似不同实为噪声"的排序——正是计划 11.6 要防的情况。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from PIL import Image

# xBD 官方损伤标签顺序（create_submission.py: msk_dmg = preds[...,1:].argmax()+1）
DAMAGE_SUBTYPES = ("no-damage", "minor-damage", "major-damage", "destroyed")

SUBTYPE_TO_ZH = {
    "no-damage": "无损伤建筑",
    "minor-damage": "轻微损伤建筑",
    "major-damage": "严重损伤建筑",
    "destroyed": "完全损毁建筑",
}


@dataclass
class Detection:
    """一栋建筑实例。bbox 一律以 **post 视场像素** 为坐标系。"""
    bbox_xyxy: list[float]
    class_name: str                     # 中文损伤类，供 ground_with_yolo 匹配
    raw_class_name: str                 # xBD 英文 subtype
    conf: float                         # argmax 类的概率
    class_probs: dict[str, float]       # 4 类，和为 1 —— 必填
    loc_conf: float = 0.0               # 建筑存在性置信度（定位分支）
    area_px: int = 0
    proposer: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "class_id": DAMAGE_SUBTYPES.index(self.raw_class_name)
            if self.raw_class_name in DAMAGE_SUBTYPES else 0,
            "class_name": self.class_name,
            "raw_class_name": self.raw_class_name,
            "conf": round(float(self.conf), 6),
            "bbox": [round(float(v), 2) for v in self.bbox_xyxy],
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox_xyxy],
            "class_probs": {k: round(float(v), 6) for k, v in self.class_probs.items()},
            "loc_conf": round(float(self.loc_conf), 6),
            "area_px": int(self.area_px),
            "proposer": self.proposer,
            **({"extras": self.extras} if self.extras else {}),
        }


@runtime_checkable
class DetectorBackend(Protocol):
    """双时相建筑损伤检测器。

    `pre` 与 `post` 必须已经地理配准且同尺寸。谁负责重采样由调用方决定
    （见 perception 的 pre_scale 处理）。
    """

    name: str

    def detect(self, pre: Image.Image, post: Image.Image) -> list[Detection]:
        ...

    def is_available(self) -> bool:
        ...
