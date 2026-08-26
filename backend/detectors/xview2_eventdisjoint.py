"""backend/detectors/xview2_eventdisjoint.py — 轨 B：同架构、事件不相交重训

这是 `xview2_first` 的**协议干净**版：同一套 xView2 冠军架构
（Res34_Unet_Loc + Res34_Unet_Double），但只在 `event_split.TRAIN_EVENTS` 上
训练、`VAL_EVENTS` 上验证，**从不接触 TEST/HOLDOUT**（训练脚本启动时硬校验）。

它存在的意义是完成那个三方分解（review2 B7）：

    legacy_unet          —— 小架构 + 事件不相交  → 协议干净、弱
    xview2_first         —— 大架构 + 事件曝光    → leaky、强
    xview2_eventdisjoint —— 大架构 + 事件不相交  → 协议干净、中等

于是：
    xview2_eventdisjoint − legacy          = 架构/训练规模的贡献（数据协议受控）
    xview2_first − xview2_eventdisjoint    = 事件曝光的贡献（架构受控）

推理逻辑、预处理、阈值、实例分离全部继承自 `XView2FirstDetector`，只换权重来源。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .base import Detection
from .xview2_first import LOC_THR, XView2FirstDetector

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent

DEFAULT_LOC_CKPT = Path(
    os.getenv(
        "XVIEW2_ED_LOC_CKPT",
        str(_HERE.parent / "outputs" / "xview2_eventdisjoint" / "res34_loc_seed0_best.pt"),
    )
).expanduser()
DEFAULT_CLS_CKPT = Path(
    os.getenv(
        "XVIEW2_ED_CLS_CKPT",
        str(_HERE.parent / "outputs" / "xview2_eventdisjoint" / "res34_cls_seed0_best.pt"),
    )
).expanduser()


class XView2EventDisjointDetector(XView2FirstDetector):
    """事件不相交重训的 xView2 冠军架构。协议干净，`leaky=False`。"""

    name = "xview2_eventdisjoint"
    leaky = False

    def __init__(
        self,
        loc_ckpt: str | os.PathLike[str] | None = None,
        cls_ckpt: str | os.PathLike[str] | None = None,
        device: str = "cuda",
        fp16: bool = True,
        min_area_px: int = 12,
        tta_flip: bool = False,
        watershed: bool = True,
        ws_min_distance: int = 6,
        split_area_px: int = 3600,
        **_ignored,
    ):
        self.loc_ckpt = Path(loc_ckpt or DEFAULT_LOC_CKPT).expanduser()
        self.cls_ckpt = Path(cls_ckpt or DEFAULT_CLS_CKPT).expanduser()
        # 复用父类推理字段，但跳过父类基于模板的加载
        super().__init__(
            weights_dir=self.loc_ckpt.parent,  # 占位，不会走模板路径
            archs=(), seeds=(), device=device, fp16=fp16,
            min_area_px=min_area_px, tta_flip=tta_flip,
            watershed=watershed, ws_min_distance=ws_min_distance,
            split_area_px=split_area_px,
        )

    def is_available(self) -> bool:
        return self.loc_ckpt.is_file() and self.cls_ckpt.is_file()

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            import torch

            from xview2_zoo import models as zoo

            loc = zoo.Res34_Unet_Loc(pretrained=None)
            sd = torch.load(self.loc_ckpt, map_location="cpu", weights_only=False)
            loc.load_state_dict(sd.get("state_dict", sd), strict=True)

            cls = zoo.Res34_Unet_Double(pretrained=None)
            sd2 = torch.load(self.cls_ckpt, map_location="cpu", weights_only=False)
            cls.load_state_dict(sd2.get("state_dict", sd2), strict=True)

            self._loc_models = [("eventdisjoint_loc", self._deploy(loc))]
            self._cls_models = [("eventdisjoint_cls", self._deploy(cls))]
            self._loaded = True
            logger.info(
                "[xview2_eventdisjoint] loaded loc=%s cls=%s",
                self.loc_ckpt.name, self.cls_ckpt.name,
            )

    def _deploy(self, model):
        import torch

        model = model.eval().to(self.device)
        if self.fp16:
            model = model.half()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def ensemble_id(self) -> str:
        return "eventdisjoint_res34_seed0"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "leaky": self.leaky,
            "leaky_reason": "",
            "ensemble_id": self.ensemble_id(),
            "arch": "res34 (xView2 1st place 架构)",
            "protocol": "event-disjoint (TRAIN_EVENTS only; TEST/HOLDOUT never seen)",
            "loc_ckpt": str(self.loc_ckpt),
            "cls_ckpt": str(self.cls_ckpt),
            "loc_thresholds": list(LOC_THR),
            "fp16": self.fp16,
            "watershed": self.watershed,
            "split_area_px": self.split_area_px,
        }
