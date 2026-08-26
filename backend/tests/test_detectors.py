"""backend/tests/test_detectors.py — 检测器后端契约测试 (计划 §3.4 / §3.5)

不需要 GPU / 权重的部分始终跑；需要权重的用 skipif 跳过，
这样 CI 与无卡环境仍能锁住契约。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors import get_detector  # noqa: E402
from detectors.base import DAMAGE_SUBTYPES, SUBTYPE_TO_ZH, Detection  # noqa: E402
from detectors.xview2_first import (  # noqa: E402
    LOC_THR,
    XView2FirstDetector,
    _pil_to_bgr,
    _remap_se_keys,
    preprocess_inputs,
)

WEIGHTS = Path(__file__).resolve().parents[1] / "outputs" / "xview2_first" / "weights"
needs_weights = pytest.mark.skipif(
    not (WEIGHTS / "res34_loc_0_1_best").is_file(),
    reason="xView2 weights not downloaded",
)


def test_damage_subtype_order_matches_xbd():
    """顺序必须与 xBD 官方一致：create_submission.py 里
    `msk_dmg = preds[...,1:].argmax(2)+1` 把 channel 1..4 映到这四类。"""
    assert DAMAGE_SUBTYPES == ("no-damage", "minor-damage", "major-damage", "destroyed")
    assert set(SUBTYPE_TO_ZH) == set(DAMAGE_SUBTYPES)


def test_preprocess_matches_reference():
    """utils.preprocess_inputs: x/127 - 1。差一点就复现不出 SOTA。"""
    x = np.array([[[0, 127, 254]]], dtype=np.uint8)
    out = preprocess_inputs(x)
    assert out.dtype == np.float32
    assert out.flatten().tolist() == pytest.approx([-1.0, 0.0, 1.0], abs=1e-6)


def test_pil_to_bgr_flips_channels():
    """参考实现用 cv2.imread → BGR。RGB 直接喂会掉分。"""
    img = Image.new("RGB", (2, 2), (10, 20, 30))
    arr = _pil_to_bgr(img)
    assert arr[0, 0].tolist() == [30, 20, 10]


def test_loc_thresholds_match_reference():
    assert LOC_THR == (0.38, 0.13, 0.14)


def test_remap_se_keys_is_noop_without_se():
    sd = {"conv1.0.weight": 1, "res.weight": 2}
    assert _remap_se_keys(sd) is sd


def test_remap_se_keys_renames_flat_se():
    """权重用扁平 se_fc1/se_fc2，仓库 master 的 senet.py 用 se_module.fc1/fc2。"""
    sd = {"conv3.0.se_fc1.weight": 1, "conv3.0.se_fc2.bias": 2, "other": 3}
    out = _remap_se_keys(sd)
    assert "conv3.0.se_module.fc1.weight" in out
    assert "conv3.0.se_module.fc2.bias" in out
    assert out["other"] == 3


def test_detection_class_probs_required_and_normalised():
    d = Detection(
        bbox_xyxy=[0, 0, 10, 10], class_name="完全损毁建筑", raw_class_name="destroyed",
        conf=0.7, class_probs={"no-damage": 0.1, "minor-damage": 0.1,
                               "major-damage": 0.1, "destroyed": 0.7},
    )
    dd = d.to_dict()
    assert set(dd["class_probs"]) == set(DAMAGE_SUBTYPES)
    assert sum(dd["class_probs"].values()) == pytest.approx(1.0, abs=1e-6)
    assert dd["class_id"] == 3


def test_xview2_backend_is_flagged_leaky():
    """权重见过全部评测事件；必须自报 leaky，否则会被误混入主表。"""
    d = get_detector("xview2_first", device="cpu")
    assert d.leaky is True
    desc = d.describe()
    assert desc["leaky"] is True
    assert "train+tier3" in desc["leaky_reason"]


def test_legacy_backend_is_not_leaky():
    d = get_detector("legacy_unet", device="cpu")
    assert d.leaky is False


def test_eventdisjoint_backend_is_not_leaky():
    """轨 B 是协议干净的主结果底座，绝不能带 leaky 标记。"""
    d = get_detector("xview2_eventdisjoint", device="cpu")
    assert d.leaky is False
    desc = d.describe()
    assert desc["leaky"] is False
    assert "event-disjoint" in desc["protocol"]


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown DETECTOR_BACKEND"):
        get_detector("no_such_backend")


def test_detect_rejects_mismatched_sizes():
    d = XView2FirstDetector(device="cpu")
    with pytest.raises(ValueError, match="尺寸不一致"):
        d.detect(Image.new("RGB", (64, 64)), Image.new("RGB", (32, 32)))


def test_adaptive_watershed_only_splits_large_blobs():
    """小于 split_area_px 的连通域必须保持单实例，避免在稀疏建成区过切分。"""
    d = XView2FirstDetector(device="cpu", split_area_px=3600, ws_min_distance=6)
    mask = np.zeros((200, 200), dtype=bool)
    mask[20:40, 20:40] = True          # 400 px，远小于阈值
    labels = d._label_instances(mask)
    assert int(labels.max()) == 1


def test_adaptive_watershed_splits_merged_cluster():
    """两个靠得很近的大方块连成一片时应被切开（palu-tsunami 密集聚落的情形）。"""
    d = XView2FirstDetector(device="cpu", split_area_px=1000, ws_min_distance=5)
    mask = np.zeros((200, 200), dtype=bool)
    mask[40:90, 40:90] = True
    mask[40:90, 88:138] = True         # 通过 2px 细颈粘连
    labels = d._label_instances(mask)
    assert int(labels.max()) >= 2


@needs_weights
def test_all_eight_architectures_load_strict():
    """4 架构 × {loc, cls} 全部能 strict 加载 —— 键名映射不能回退。"""
    import torch

    from detectors.xview2_first import ARCH_SPECS
    from xview2_zoo import models as zoo

    for arch, (loc_name, cls_name, loc_t, cls_t) in ARCH_SPECS.items():
        for name, tmpl in ((loc_name, loc_t), (cls_name, cls_t)):
            sd = torch.load(WEIGHTS / tmpl.format(seed=0), map_location="cpu",
                            weights_only=False)
            sd = sd.get("state_dict", sd)
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            model = getattr(zoo, name)(pretrained=None)
            model.load_state_dict(_remap_se_keys(sd), strict=True)


@needs_weights
def test_weights_available_for_default_config():
    d = get_detector("xview2_first", device="cpu")
    assert d.is_available()
