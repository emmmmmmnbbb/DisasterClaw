"""ChangeOS backend with binary building-damage output.

The official ChangeOS TorchScript model consumes a co-registered RGB pre/post
pair and returns a building-localization logit plus a five-channel damage logit
(background, no damage, minor, major, destroyed).  DisasterClaw only needs the
binary question "damaged or not", so this adapter marginalizes the three
damaged classes while retaining a binary probability interface for the
reobservation controller.  The official pretrained weights are frozen and
shared by every agent/controller arm: ChangeOS is an external perception tool,
not a trainable contribution or a standalone generalization claim of this work.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import Detection

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = Path(
    os.getenv(
        "CHANGEOS_WEIGHTS",
        str(_HERE.parent / "outputs" / "changeos" / "changeos_r34.pt"),
    )
).expanduser()

FOUR_DAMAGE_CLASSES = ("no-damage", "minor-damage", "major-damage", "destroyed")
BINARY_DAMAGE_CLASSES = ("no-damage", "damaged")
BINARY_TO_ZH = {"no-damage": "无损伤建筑", "damaged": "受损建筑"}

_MEAN = (123.675, 116.28, 103.53, 123.675, 116.28, 103.53)
_STD = (58.395, 57.12, 57.375, 58.395, 57.12, 57.375)


def collapse_damage_probs(four_probs: np.ndarray) -> np.ndarray:
    """Collapse (..., 4) xBD probabilities to (..., 2) no-damage/damaged."""
    probs = np.asarray(four_probs, dtype=np.float64)
    if probs.shape[-1] != 4:
        raise ValueError(f"expected four damage channels, got shape={probs.shape}")
    out = np.stack((probs[..., 0], probs[..., 1:].sum(axis=-1)), axis=-1)
    denom = out.sum(axis=-1, keepdims=True)
    return np.divide(out, denom, out=np.full_like(out, 0.5), where=denom > 1e-12)


class ChangeOSDetector:
    """Official ChangeOS TorchScript inference exposed as binary instances."""

    name = "changeos"
    leaky = False
    frozen = True

    def __init__(
        self,
        weights: str | os.PathLike[str] | None = None,
        device: str = "cuda",
        input_size: int = 1024,
        loc_threshold: float = 0.5,
        damage_threshold: float = 0.5,
        temperature: float = 1.0,
        min_area_px: int = 12,
        watershed: bool = True,
        ws_min_distance: int = 6,
        split_area_px: int = 3600,
        **_ignored: Any,
    ):
        self.weights = Path(weights or DEFAULT_WEIGHTS).expanduser()
        self.device = device
        self.input_size = int(input_size)
        self.loc_threshold = float(loc_threshold)
        self.damage_threshold = float(damage_threshold)
        self.temperature = max(float(temperature), 1e-6)
        self.min_area_px = int(min_area_px)
        self.watershed = bool(watershed)
        self.ws_min_distance = int(ws_min_distance)
        self.split_area_px = int(split_area_px)
        self._model = None
        self._loaded = False
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return self.weights.is_file()

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if not self.weights.is_file():
                raise FileNotFoundError(f"ChangeOS weights not found: {self.weights}")
            import torch

            device = torch.device(self.device)
            self._model = torch.jit.load(str(self.weights), map_location=device)
            self._model.eval().to(device)
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)
            self._loaded = True
            logger.info("[changeos] loaded %s on %s", self.weights.name, device)

    def _input_tensor(self, pre: Image.Image, post: Image.Image):
        import torch

        size = (self.input_size, self.input_size)
        pre_r = pre.convert("RGB")
        post_r = post.convert("RGB")
        if pre_r.size != size:
            pre_r = pre_r.resize(size, Image.BILINEAR)
            post_r = post_r.resize(size, Image.BILINEAR)
        image = np.concatenate((np.asarray(pre_r), np.asarray(post_r)), axis=2).copy()
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
        mean = torch.tensor(_MEAN, dtype=tensor.dtype).view(1, 6, 1, 1)
        std = torch.tensor(_STD, dtype=tensor.dtype).view(1, 6, 1, 1)
        return ((tensor - mean) / std).to(self.device)

    def _damage_probs(self, damage_logits):
        """Return Bx4xHxW probabilities, dropping the optional background channel."""
        import torch

        if damage_logits.ndim != 4:
            raise ValueError(f"unexpected ChangeOS damage output: {tuple(damage_logits.shape)}")
        channels = int(damage_logits.shape[1])
        if channels == 5:
            logits = damage_logits[:, 1:5]
        elif channels == 4:
            logits = damage_logits
        else:
            raise ValueError(f"expected 4 or 5 damage channels, got {channels}")
        return torch.softmax(logits / self.temperature, dim=1)

    def _predict_prob_maps(self, pre: Image.Image, post: Image.Image):
        if pre.size != post.size:
            raise ValueError(
                f"pre/post 尺寸不一致 {pre.size} vs {post.size}；ChangeOS 要求严格配准且同尺寸"
            )
        self.load()
        import torch
        import torch.nn.functional as torch_f

        original_h, original_w = pre.height, pre.width
        with torch.no_grad():
            output = self._model(self._input_tensor(pre, post))
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise ValueError("ChangeOS TorchScript model must return (loc_logits, damage_logits)")
            loc_logits, damage_logits = output
            loc_prob = torch.sigmoid(loc_logits)
            four_prob = self._damage_probs(damage_logits)
            if tuple(loc_prob.shape[-2:]) != (original_h, original_w):
                target = (original_h, original_w)
                loc_prob = torch_f.interpolate(loc_prob, target, mode="bilinear", align_corners=False)
                four_prob = torch_f.interpolate(four_prob, target, mode="bilinear", align_corners=False)

        loc = loc_prob[0, 0].float().cpu().numpy().astype("float32")
        four = four_prob[0].permute(1, 2, 0).float().cpu().numpy().astype("float32")
        four /= np.maximum(four.sum(axis=2, keepdims=True), 1e-12)
        return loc, four

    def detect(self, pre: Image.Image, post: Image.Image) -> list[Detection]:
        loc_prob, four_prob = self._predict_prob_maps(pre, post)
        return self._instances(loc_prob, four_prob)

    def _instances(self, loc_prob: np.ndarray, four_prob: np.ndarray) -> list[Detection]:
        from scipy import ndimage

        mask = np.asarray(loc_prob) >= self.loc_threshold
        if not mask.any():
            return []
        labels = self._label_instances(mask)
        out: list[Detection] = []
        for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None:
                continue
            object_mask = labels[sl] == idx
            area = int(object_mask.sum())
            if area < self.min_area_px:
                continue
            four = four_prob[sl][object_mask].mean(axis=0).astype("float64")
            four /= max(float(four.sum()), 1e-12)
            binary = collapse_damage_probs(four)
            p_no, p_damage = float(binary[0]), float(binary[1])
            subtype = "damaged" if p_damage >= self.damage_threshold else "no-damage"
            confidence = p_damage if subtype == "damaged" else p_no
            ys, xs = sl[0], sl[1]
            out.append(Detection(
                bbox_xyxy=[float(xs.start), float(ys.start), float(xs.stop), float(ys.stop)],
                class_name=BINARY_TO_ZH[subtype],
                raw_class_name=subtype,
                conf=confidence,
                class_probs={"no-damage": p_no, "damaged": p_damage},
                loc_conf=float(loc_prob[sl][object_mask].mean()),
                area_px=area,
                proposer="changeos",
                extras={
                    "label_mode": "binary",
                    "damage_probability": p_damage,
                    "four_class_probs": {
                        name: float(value) for name, value in zip(FOUR_DAMAGE_CLASSES, four)
                    },
                    "external_tool": "ChangeOS",
                    "frozen": True,
                },
            ))
        return out

    def predict_maps(self, pre: Image.Image, post: Image.Image):
        """Return xBD-compatible hard maps plus dense four-class probabilities.

        The application consumes binary instances from :meth:`detect`; keeping
        the 1..4 hard map here preserves the existing official xBD pixel-metric
        evaluator used to validate an adapter against published ChangeOS scores.
        """
        loc_prob, four_prob = self._predict_prob_maps(pre, post)
        loc_mask = (loc_prob >= self.loc_threshold).astype("uint8")
        damage = (four_prob.argmax(axis=2) + 1).astype("uint8") * loc_mask
        damage = self._object_vote_four(loc_mask, damage)
        return loc_mask, damage, four_prob

    @staticmethod
    def _object_vote_four(loc_mask: np.ndarray, damage: np.ndarray) -> np.ndarray:
        """Match the official ChangeOS object vote for xBD hard-map scoring."""
        from scipy import ndimage

        labels, count = ndimage.label(np.asarray(loc_mask).astype(bool))
        refined = np.zeros_like(damage, dtype="uint8")
        weights = np.asarray((8.0, 38.0, 25.0, 11.0), dtype="float64")
        for idx in range(1, int(count) + 1):
            region = labels == idx
            counts = np.asarray([(damage[region] == cid).sum() for cid in range(1, 5)])
            cls_id = int(np.argmax(counts * weights)) + 1
            refined[region] = cls_id
        return refined

    def _label_instances(self, mask: np.ndarray) -> np.ndarray:
        """Split only unusually large merged components, matching existing backends."""
        from scipy import ndimage

        base, count = ndimage.label(mask)
        if count == 0 or not self.watershed:
            return base
        sizes = ndimage.sum(mask, base, index=np.arange(1, count + 1))
        large = {int(i + 1) for i, size in enumerate(sizes) if size > self.split_area_px}
        if not large:
            return base

        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        result = base.copy()
        next_label = int(count) + 1
        for component in large:
            component_mask = base == component
            distance = ndimage.distance_transform_edt(component_mask)
            coordinates = peak_local_max(
                distance,
                min_distance=self.ws_min_distance,
                labels=component_mask,
                exclude_border=False,
            )
            if len(coordinates) <= 1:
                continue
            markers = np.zeros(mask.shape, dtype=np.int32)
            markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
            markers, _ = ndimage.label(markers > 0)
            split = watershed(-distance, markers, mask=component_mask)
            if int(split.max()) <= 1:
                continue
            result[component_mask & (split == 1)] = component
            for part in range(2, int(split.max()) + 1):
                result[component_mask & (split == part)] = next_label
                next_label += 1
        return result

    def describe(self) -> dict:
        return {
            "name": self.name,
            "leaky": self.leaky,
            "leaky_reason": "",
            "evaluation_role": "fixed_external_perception_tool",
            "frozen": self.frozen,
            "policy_shared": True,
            "pretrained": True,
            "pretraining_dataset": "xBD",
            "standalone_generalization_claim": False,
            "architecture": self.weights.stem,
            "weights": str(self.weights),
            "input_size": self.input_size,
            "label_mode": "binary",
            "classes": list(BINARY_DAMAGE_CLASSES),
            "loc_threshold": self.loc_threshold,
            "damage_threshold": self.damage_threshold,
            "temperature": self.temperature,
            "watershed": self.watershed,
            "split_area_px": self.split_area_px,
        }
