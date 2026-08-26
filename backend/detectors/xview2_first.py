"""backend/detectors/xview2_first.py — xView2 冠军方案（Durnov）封装

权重来源：`DIUx-xView/xView2_first_place` GitHub Release tag `final`，
`split-weights-a{a..e}` 合并后是 **tar.gz**（仓库文档写 zip，是错的），
解压得 24 个文件 = 4 架构 × {loc, cls} × 3 seed。

与参考实现严格对齐的四个细节（错一个就复现不出 0.80）：
  1. **BGR 通道序** —— 参考实现用 `cv2.imread(..., IMREAD_COLOR)`。
  2. 预处理 `x/127 - 1`（`utils.preprocess_inputs`）。
  3. 输出取 **sigmoid**，不是 softmax（`predict34cls.py`）。
  4. cls 是 6 通道输入的 siamese：`x[:, :3]`=pre，`x[:, 3:]`=post；
     5 通道输出里 **channel 1..4 才是四类损伤**，channel 0 是建筑性
     （`create_submission.py`: `msk_dmg = preds[..., 1:].argmax(2) + 1`）。

定位掩码沿用参考实现的复合阈值 `_thr = [0.38, 0.13, 0.14]`。

⚠️ **事件泄漏警告（计划 §3.2）**：这些权重在 xBD 官方 train+tier3 上训练，
而 xBD 是按瓦片而非按事件划分的，所以 paper_cja 的 test / holdout 事件
**全部**被这些权重见过。本后端只能作为「有事件曝光的参照上界」(leaky)，
与 `O_REF` 同级，**不得进入正式对照主表**。干净主结果须用轨 B
（同架构在事件不相交协议下重训）。`self.leaky = True` 会写进每次输出的 extras，
下游报告脚本据此拒绝把它混入主表。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import DAMAGE_SUBTYPES, SUBTYPE_TO_ZH, Detection

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DEFAULT_WEIGHTS_DIR = Path(
    os.getenv(
        "XVIEW2_WEIGHTS_DIR",
        str(_HERE.parent / "outputs" / "xview2_first" / "weights"),
    )
).expanduser()

# 参考实现 create_submission.py 的定位复合阈值
LOC_THR = (0.38, 0.13, 0.14)

# 架构 → (loc 类名, cls 类名, loc 权重前缀, cls 权重前缀)
ARCH_SPECS = {
    "res34": ("Res34_Unet_Loc", "Res34_Unet_Double", "res34_loc_{seed}_1_best", "res34_cls2_{seed}_tuned_best"),
    "res50": ("SeResNext50_Unet_Loc", "SeResNext50_Unet_Double", "res50_loc_{seed}_tuned_best", "res50_cls_cce_{seed}_tuned_best"),
    "dpn92": ("Dpn92_Unet_Loc", "Dpn92_Unet_Double", "dpn92_loc_{seed}_tuned_best", "dpn92_cls_cce_{seed}_tuned_best"),
    "se154": ("SeNet154_Unet_Loc", "SeNet154_Unet_Double", "se154_loc_{seed}_1_best", "se154_cls_cce_{seed}_tuned_best"),
}


def preprocess_inputs(x: np.ndarray) -> np.ndarray:
    """utils.preprocess_inputs：逐元素 x/127 - 1。"""
    x = np.asarray(x, dtype="float32")
    x /= 127.0
    x -= 1.0
    return x


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """PIL(RGB) → numpy(BGR)，对齐参考实现的 cv2.imread 通道序。"""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return arr[..., ::-1].copy()


def _remap_se_keys(sd: dict) -> dict:
    """se_resnext50 / senet154 权重的 SE 模块用扁平命名 `se_fc1/se_fc2`，
    而仓库 master 的 `senet.py` 用 `se_module.fc1/fc2`。权重与代码是两个时期的，
    需要重命名才能 strict 加载。只改键名，不动张量。
    """
    if not any(".se_fc1." in k or ".se_fc2." in k for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        k2 = k.replace(".se_fc1.", ".se_module.fc1.").replace(".se_fc2.", ".se_module.fc2.")
        out[k2] = v
    return out


class XView2FirstDetector:
    """loc + cls 双阶段集成。

    `archs` / `seeds` 控制集成规模：
      - 在线闭环建议 `archs=("res34",), seeds=(0,)`（1 loc + 1 cls），
        并**显式报告这是降配**，不得拿 12 模型的 0.803 描述在线系统。
      - 离线主表可用全部 4 架构 × 3 seed（4 loc + 12 cls），对标文献数字。
    """

    name = "xview2_first"
    leaky = True  # 见模块 docstring：权重见过全部评测事件

    def __init__(
        self,
        weights_dir: str | os.PathLike[str] | None = None,
        archs: tuple[str, ...] = ("res34",),
        seeds: tuple[int, ...] = (0,),
        device: str = "cuda",
        fp16: bool = True,
        min_area_px: int = 12,
        tta_flip: bool = False,
        watershed: bool = True,
        ws_min_distance: int = 6,
        split_area_px: int = 3600,
    ):
        self.weights_dir = Path(weights_dir or DEFAULT_WEIGHTS_DIR).expanduser()
        self.archs = tuple(a for a in archs if a in ARCH_SPECS)
        self.seeds = tuple(int(s) for s in seeds)
        self.device = device
        self.fp16 = bool(fp16)
        self.min_area_px = int(min_area_px)
        self.tta_flip = bool(tta_flip)
        self.watershed = bool(watershed)
        self.ws_min_distance = int(ws_min_distance)
        self.split_area_px = int(split_area_px)
        self._loc_models: list = []
        self._cls_models: list = []
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error = ""

    # ── 加载 ────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        if not self.weights_dir.is_dir():
            return False
        for arch in self.archs:
            _, _, loc_t, cls_t = ARCH_SPECS[arch]
            for s in self.seeds:
                if not (self.weights_dir / loc_t.format(seed=s)).is_file():
                    return False
                if not (self.weights_dir / cls_t.format(seed=s)).is_file():
                    return False
        return True

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            import torch

            from xview2_zoo import models as zoo

            loc_models, cls_models = [], []
            for arch in self.archs:
                loc_cls_name, cls_cls_name, loc_t, cls_t = ARCH_SPECS[arch]
                for s in self.seeds:
                    for cls_name, tmpl, bucket in (
                        (loc_cls_name, loc_t, loc_models),
                        (cls_cls_name, cls_t, cls_models),
                    ):
                        ckpt_path = self.weights_dir / tmpl.format(seed=s)
                        model = getattr(zoo, cls_name)(pretrained=None)
                        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                        sd = sd.get("state_dict", sd)
                        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                        sd = _remap_se_keys(sd)
                        model.load_state_dict(sd, strict=True)
                        model = model.eval().to(self.device)
                        if self.fp16:
                            model = model.half()
                        for p in model.parameters():
                            p.requires_grad_(False)
                        bucket.append((f"{arch}_s{s}", model))
                        logger.info("[xview2_first] loaded %s from %s", cls_name, ckpt_path.name)

            self._loc_models = loc_models
            self._cls_models = cls_models
            self._loaded = True

    # ── 推理 ────────────────────────────────────────────────────────────────

    def _to_tensor(self, arr: np.ndarray):
        import torch

        t = torch.from_numpy(preprocess_inputs(arr)).permute(2, 0, 1).unsqueeze(0)
        t = t.to(self.device)
        return t.half() if self.fp16 else t.float()

    def _run_ensemble(self, models: list, tensor) -> np.ndarray:
        """对模型列表求 sigmoid 均值。返回 (H, W, C) float32。"""
        import torch

        acc = None
        n = 0
        views = [tensor]
        if self.tta_flip:
            views.append(torch.flip(tensor, dims=[3]))
        with torch.no_grad():
            for _, model in models:
                for i, v in enumerate(views):
                    out = torch.sigmoid(model(v))
                    if i == 1:
                        out = torch.flip(out, dims=[3])
                    out = out.float().cpu().numpy()[0].transpose(1, 2, 0)
                    acc = out if acc is None else acc + out
                    n += 1
        return (acc / max(n, 1)).astype("float32")

    def predict_maps(self, pre: Image.Image, post: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (loc_mask uint8 0/1, dmg_mask uint8 0..4, cls_prob HxWx5)。

        严格复刻参考实现 `create_submission.py` 的后处理，包括 class 2 的膨胀，
        以便与文献的 loc/damage F1 数字直接对比。**像素级评测必须用这个，
        不能用 bbox 近似**——bbox 比建筑轮廓大得多，会制造大量假阳。
        """
        from skimage.morphology import dilation, square

        self.load()
        pre_bgr = _pil_to_bgr(pre)
        post_bgr = _pil_to_bgr(post)
        loc_prob = self._run_ensemble(self._loc_models, self._to_tensor(pre_bgr))[..., 0]
        cls_prob = self._run_ensemble(
            self._cls_models, self._to_tensor(np.concatenate([pre_bgr, post_bgr], axis=2))
        )

        dmg = cls_prob[..., 1:].argmax(axis=2) + 1
        t0, t1, t2 = LOC_THR
        loc_mask = (
            (loc_prob > t0)
            | ((loc_prob > t1) & (dmg > 1) & (dmg < 4))
            | ((loc_prob > t2) & (dmg > 1))
        ).astype("uint8")
        dmg = (dmg * loc_mask).astype("uint8")
        m2 = dmg == 2
        if m2.sum() > 0:
            m2d = dilation(m2, square(5))
            dmg[m2d & (dmg == 1)] = 2
        return loc_mask, dmg.astype("uint8"), cls_prob

    def detect(self, pre: Image.Image, post: Image.Image) -> list[Detection]:
        if pre.size != post.size:
            raise ValueError(
                f"pre/post 尺寸不一致 {pre.size} vs {post.size}；"
                "xView2 cls 是像素对齐的 siamese，调用方须先重采样到同尺寸。"
            )
        self.load()

        pre_bgr = _pil_to_bgr(pre)
        post_bgr = _pil_to_bgr(post)

        # loc 在 pre 上跑（参考实现 predict34_loc.py 读 *_pre_disaster.png）
        loc_prob = self._run_ensemble(self._loc_models, self._to_tensor(pre_bgr))[..., 0]
        # cls 是 6 通道 siamese：前 3 通道 pre，后 3 通道 post
        cls_prob = self._run_ensemble(
            self._cls_models, self._to_tensor(np.concatenate([pre_bgr, post_bgr], axis=2))
        )

        return self._instances(loc_prob, cls_prob)

    def _instances(self, loc_prob: np.ndarray, cls_prob: np.ndarray) -> list[Detection]:
        """定位概率 + 损伤概率 → 建筑实例。

        定位掩码沿用参考实现 create_submission.py 的复合阈值。

        **实例化用分水岭而非朴素连通域**：xView2 冠军方案是语义分割，不产生实例；
        在密集聚落（实测 palu-tsunami 6 张瓦片有 1661 栋 destroyed，约 277 栋/瓦片）
        相邻建筑轮廓会连成一片，朴素连通域把上千栋并成 146 个 blob，
        实例召回只有 0.044，count / spatial 类问题会系统性低估。
        距离变换 + 局部极大值做种子的分水岭能把粘连的建筑切开。
        """
        from scipy import ndimage

        dmg_argmax = cls_prob[..., 1:].argmax(axis=2) + 1  # 1..4
        t0, t1, t2 = LOC_THR
        mask = (
            (loc_prob > t0)
            | ((loc_prob > t1) & (dmg_argmax > 1) & (dmg_argmax < 4))
            | ((loc_prob > t2) & (dmg_argmax > 1))
        )
        if not mask.any():
            return []

        labels = self._label_instances(mask)
        n = int(labels.max())
        if n == 0:
            return []

        out: list[Detection] = []
        for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None:
                continue
            sub_mask = labels[sl] == idx
            area = int(sub_mask.sum())
            if area < self.min_area_px:
                continue
            probs = cls_prob[sl][..., 1:][sub_mask].mean(axis=0).astype("float64")
            total = float(probs.sum())
            probs = probs / total if total > 1e-9 else np.full(4, 0.25)
            k = int(probs.argmax())
            subtype = DAMAGE_SUBTYPES[k]
            ys, xs = sl[0], sl[1]
            out.append(Detection(
                bbox_xyxy=[float(xs.start), float(ys.start), float(xs.stop), float(ys.stop)],
                class_name=SUBTYPE_TO_ZH[subtype],
                raw_class_name=subtype,
                conf=float(probs[k]),
                class_probs={s: float(p) for s, p in zip(DAMAGE_SUBTYPES, probs)},
                loc_conf=float(loc_prob[sl][sub_mask].mean()),
                area_px=area,
                proposer="xview2_first",
                extras={"leaky": True, "ensemble": self.ensemble_id()},
            ))
        return out

    def _label_instances(self, mask: np.ndarray) -> np.ndarray:
        """把二值建筑掩码切成实例标签图。

        **自适应分水岭**：只对「明显大于单栋建筑」的连通域做切分。

        动机（实测）：固定策略在两类场景上结论相反 —— 稀疏建成区（TRAIN_EVENTS，
        socal-fire / joplin-tornado 等）里分水岭只会过切分，macro-F1 0.581→0.568；
        密集聚落（palu-tsunami，6 张瓦片 1661 栋 destroyed）里不切分则上千栋并成
        146 个 blob，实例召回 0.044。既然最优选择随建筑密度反转，就不能靠在
        test 上选超参（那是协议泄漏），只能让判据本身与密度无关：
        *面积远大于一栋建筑的连通域才可能是多栋建筑*。

        `split_area_px` 的物理含义：0.5 m/px 下一栋 10×15 m 民居约 600 px，
        默认 3600 px ≈ 6 栋，超过才尝试切分。
        """
        from scipy import ndimage

        base, n = ndimage.label(mask)
        if n == 0 or not self.watershed:
            return base

        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        sizes = ndimage.sum(mask, base, index=np.arange(1, n + 1))
        big = {int(i + 1) for i, s in enumerate(sizes) if s > self.split_area_px}
        if not big:
            return base

        out = base.copy()
        next_label = n + 1
        for comp in big:
            comp_mask = base == comp
            dist = ndimage.distance_transform_edt(comp_mask)
            coords = peak_local_max(
                dist, min_distance=self.ws_min_distance, labels=comp_mask,
                exclude_border=False,
            )
            if len(coords) <= 1:
                continue
            markers = np.zeros(mask.shape, dtype=np.int32)
            markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
            markers, _ = ndimage.label(markers > 0)
            split = watershed(-dist, markers, mask=comp_mask)
            n_parts = int(split.max())
            if n_parts <= 1:
                continue
            # 第 1 块沿用原标签，其余分配新标签
            out[comp_mask & (split == 1)] = comp
            for part in range(2, n_parts + 1):
                out[comp_mask & (split == part)] = next_label
                next_label += 1
        return out

    def ensemble_id(self) -> str:
        return f"{'+'.join(self.archs)}_seeds{''.join(str(s) for s in self.seeds)}"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "leaky": self.leaky,
            "leaky_reason": (
                "weights trained on xBD official train+tier3; xBD splits by tile not event, "
                "so all paper_cja test/holdout events were seen during training"
            ),
            "archs": list(self.archs),
            "seeds": list(self.seeds),
            "n_loc_models": len(self.archs) * len(self.seeds),
            "n_cls_models": len(self.archs) * len(self.seeds),
            "ensemble_id": self.ensemble_id(),
            "loc_thresholds": list(LOC_THR),
            "fp16": self.fp16,
            "tta_flip": self.tta_flip,
            "watershed": self.watershed,
            "ws_min_distance": self.ws_min_distance,
            "split_area_px": self.split_area_px,
            "weights_dir": str(self.weights_dir),
        }
