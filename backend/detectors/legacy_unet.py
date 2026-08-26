"""backend/detectors/legacy_unet.py — 现有自训练管线的 Detector 适配层

把 `building_localization.BuildingLocalizer`（ResNet34 U-Net 定位，跑在 pre 上）
+ `change_perception.ChangePerceptionModel`（Siamese 差分注意力四分类）包装成
统一的 `DetectorBackend`，使它能与 `xview2_first` 在**同一评测脚本、同一协议**下
逐项对照。

这是论文的对照基线（review2 B7：「剩余差距归因于架构/训练规模」是断言而非测量）。
它与 xview2_first 的关键区别不只是架构，还有**训练数据**：
  - 本后端在**事件不相交** train 划分上训练 → 协议干净，但跨事件不泛化
    （实测 val macro-F1 长期 0.21–0.25，轻微/严重损伤召回 0.000）。
  - xview2_first 在 xBD 官方 train+tier3 上训练 → 见过全部评测事件，leaky。
两者的差值同时包含「架构/训练规模」与「事件曝光」两个因素，
**不能单独归因于任一方**；分解需要轨 B（同架构、事件不相交重训）。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import DAMAGE_SUBTYPES, SUBTYPE_TO_ZH, Detection

logger = logging.getLogger(__name__)


class LegacyUnetDetector:
    """自训练 U-Net 定位 + Siamese 变化分类。"""

    name = "legacy_unet"
    leaky = False  # 事件不相交协议下训练

    def __init__(
        self,
        loc_ckpt: str | None = None,
        cp_ckpt: str | None = None,
        device: str = "cuda",
        **_ignored,
    ):
        self.loc_ckpt = loc_ckpt or os.getenv("BUILDING_LOC_CKPT", "")
        self.cp_ckpt = cp_ckpt or os.getenv("CHANGE_PERCEPTION_CKPT", "")
        self.device = device
        self._loc = None
        self._cp = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return bool(self.loc_ckpt) and Path(self.loc_ckpt).is_file()

    def load(self) -> None:
        if self._loc is not None:
            return
        with self._lock:
            if self._loc is not None:
                return
            import building_localization as bl

            self._loc = bl.load_building_localizer(
                Path(self.loc_ckpt), device=self.device,
            )
            try:
                import change_perception as cp

                self._cp = cp.get_change_perception()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[legacy_unet] change_perception 不可用: %s", exc)
                self._cp = None

    def detect(self, pre: Image.Image, post: Image.Image) -> list[Detection]:
        if pre.size != post.size:
            raise ValueError(f"pre/post 尺寸不一致 {pre.size} vs {post.size}")
        self.load()

        proposals = self._loc.propose(pre)
        if not proposals:
            return []

        import change_perception as cp

        out: list[Detection] = []
        for p in proposals:
            bbox = [float(v) for v in (p.get("bbox_xyxy") or p.get("bbox") or [0, 0, 0, 0])]
            probs: Optional[dict] = None
            if self._cp is not None:
                try:
                    pre_c = cp.crop_patch(pre, bbox, pre.width, pre.height)
                    post_c = cp.crop_patch(post, bbox, post.width, post.height)
                    pred = self._cp.predict(pre_c, post_c)
                    probs = dict(pred.class_probs)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[legacy_unet] class_probs 失败（跳过分级）: %s", exc)
            if not probs:
                # 没有四类概率就无法参与 entropy / A3_ENTROPY 策略；如实标为均匀分布
                # 而不是伪造一个尖峰分布。
                probs = {s: 0.25 for s in DAMAGE_SUBTYPES}
            vec = np.array([float(probs.get(s, 0.0)) for s in DAMAGE_SUBTYPES], dtype="float64")
            total = float(vec.sum())
            vec = vec / total if total > 1e-9 else np.full(4, 0.25)
            k = int(vec.argmax())
            subtype = DAMAGE_SUBTYPES[k]
            w = max(0.0, bbox[2] - bbox[0])
            h = max(0.0, bbox[3] - bbox[1])
            out.append(Detection(
                bbox_xyxy=bbox,
                class_name=SUBTYPE_TO_ZH[subtype],
                raw_class_name=subtype,
                conf=float(vec[k]),
                class_probs={s: float(v) for s, v in zip(DAMAGE_SUBTYPES, vec)},
                loc_conf=float(p.get("conf", 0.0)),
                area_px=int(w * h),
                proposer="legacy_unet",
                extras={"leaky": False},
            ))
        return out

    def predict_maps(self, pre: Image.Image, post: Image.Image):
        """像素口径掩码。定位用 U-Net 概率图阈值化；损伤类按实例填充。"""
        self.load()
        prob = self._loc.predict_proba(pre)
        loc_mask = (prob > 0.5).astype("uint8")
        dmg = np.zeros(loc_mask.shape, dtype="uint8")
        for d in self.detect(pre, post):
            x1, y1, x2, y2 = [int(round(v)) for v in d.bbox_xyxy]
            cid = DAMAGE_SUBTYPES.index(d.raw_class_name) + 1
            region = loc_mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] > 0
            dmg[max(0, y1):max(0, y2), max(0, x1):max(0, x2)][region] = cid
        return loc_mask, dmg, None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "leaky": self.leaky,
            "leaky_reason": "",
            "ensemble_id": "resnet34unet+siamese",
            "loc_ckpt": self.loc_ckpt,
            "cp_ckpt": self.cp_ckpt,
            "device": self.device,
        }
