"""
backend/building_localization.py — xView2 风格建筑定位（ResNet34 U-Net）

只做定位脚手架：在 pre 图上预测建筑二值 mask，连通域 → bbox 提议，
损伤分级仍由 change_perception 完成。

训练：
    python backend/building_localization.py train \
        --data-dir /home/lc/datasets/xbd_loc_strict_v1 \
        --epochs 20 --batch-size 4 --device cuda:0 \
        --require-event-disjoint \
        --out backend/outputs/building_localization/resnet34_strict_v1.pt

推理：
    from building_localization import get_building_localizer
    dets = get_building_localizer().propose(pre_image_path_or_pil)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_PATH = Path(
    os.getenv(
        "BUILDING_LOC_CKPT",
        str(BACKEND_DIR / "outputs" / "building_localization" / "resnet34_strict_v1.pt"),
    )
)
LOC_CONF_THRESHOLD = float(os.getenv("BUILDING_LOC_CONF", "0.45"))
LOC_MIN_AREA_PX = int(os.getenv("BUILDING_LOC_MIN_AREA", "64"))
LOC_DILATE_K = int(os.getenv("BUILDING_LOC_DILATE", "3"))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TORCH_IMPORT_ERROR: Optional[str] = None
try:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models as tv_models
    from torchvision import transforms as tv_transforms
except Exception as exc:  # noqa: BLE001
    np = None  # type: ignore
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = Dataset = object  # type: ignore
    tv_models = tv_transforms = None  # type: ignore
    _TORCH_IMPORT_ERROR = str(exc)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(f"torch 不可用: {_TORCH_IMPORT_ERROR}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── U-Net (ResNet34 encoder) ─────────────────────────────────────────────────


class _ConvBNReLU(nn.Module):  # type: ignore[misc]
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _Up(nn.Module):  # type: ignore[misc]
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _ConvBNReLU(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNet34UNet(nn.Module):  # type: ignore[misc]
    """ResNet34 编码器 + U-Net 解码器，输出单通道 logits。"""

    def __init__(self, pretrained: bool = True):
        _require_torch()
        super().__init__()
        weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
        encoder = tv_models.resnet34(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # /2
        self.pool = encoder.maxpool  # /4
        self.layer1 = encoder.layer1  # /4, 64
        self.layer2 = encoder.layer2  # /8, 128
        self.layer3 = encoder.layer3  # /16, 256
        self.layer4 = encoder.layer4  # /32, 512

        self.up4 = _Up(512, 256, 256)
        self.up3 = _Up(256, 128, 128)
        self.up2 = _Up(128, 64, 64)
        self.up1 = _Up(64, 64, 64)  # skip = stem (64)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        s0 = self.stem(x)          # B,64,H/2,W/2
        x = self.pool(s0)          # B,64,H/4,W/4
        s1 = self.layer1(x)        # B,64,H/4,W/4
        s2 = self.layer2(s1)       # B,128,H/8,W/8
        s3 = self.layer3(s2)       # B,256,H/16,W/16
        s4 = self.layer4(s3)       # B,512,H/32,W/32
        x = self.up4(s4, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        return self.head(x)


def _align_logits_to_input(logits, size_hw: tuple[int, int]):
    if logits.shape[-2:] != size_hw:
        logits = nn.functional.interpolate(logits, size=size_hw, mode="bilinear", align_corners=False)
    return logits


# ── Losses ───────────────────────────────────────────────────────────────────


def dice_loss_with_logits(logits, targets, eps: float = 1e-6) -> "torch.Tensor":
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def focal_loss_with_logits(
    logits, targets, alpha: float = 0.25, gamma: float = 2.0,
) -> "torch.Tensor":
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets >= 0.5, probs, 1.0 - probs)
    loss = alpha * (1.0 - pt).pow(gamma) * bce
    return loss.mean()


def combined_loc_loss(logits, targets) -> "torch.Tensor":
    return dice_loss_with_logits(logits, targets) + focal_loss_with_logits(logits, targets)


def batch_dice(logits, targets, eps: float = 1e-6) -> float:
    probs = (torch.sigmoid(logits) > 0.5).float()
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return float(dice.mean().item())


# ── Dataset ──────────────────────────────────────────────────────────────────


class LocDataset(Dataset):  # type: ignore[misc]
    def __init__(self, data_dir: Path, subset: str, imgsz: int = 1024, augment: bool = False):
        _require_torch()
        self.img_dir = data_dir / subset / "images"
        self.mask_dir = data_dir / subset / "masks"
        self.imgsz = imgsz
        self.augment = augment
        self.paths = sorted(self.img_dir.glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"空子集: {self.img_dir}")
        self.tf = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        mask_path = self.mask_dir / img_path.name
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.imgsz and (image.size[0] != self.imgsz or image.size[1] != self.imgsz):
            image = image.resize((self.imgsz, self.imgsz), Image.BILINEAR)
            mask = mask.resize((self.imgsz, self.imgsz), Image.NEAREST)
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            k = random.choice([0, 1, 2, 3])
            if k:
                image = image.rotate(90 * k, expand=True)
                mask = mask.rotate(90 * k, expand=True)
                if image.size[0] != self.imgsz:
                    image = image.resize((self.imgsz, self.imgsz), Image.BILINEAR)
                    mask = mask.resize((self.imgsz, self.imgsz), Image.NEAREST)
        x = self.tf(image)
        y = torch.from_numpy((np.array(mask) > 127).astype("float32"))[None, ...]
        return x, y


def assert_event_disjoint(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少切分 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = manifest.get("split_audit") or {}
    if (
        manifest.get("split_strategy") != "strict_event"
        or not manifest.get("strict_event_split")
        or not audit.get("event_disjoint")
        or audit.get("overlaps")
    ):
        raise ValueError(f"数据集不是严格事件级无泄漏切分: {manifest_path}")
    return manifest


def assert_events_match_reference(manifest: dict, reference_manifest: Path) -> None:
    if not reference_manifest.exists():
        logger.warning("参考 YOLO manifest 不存在，跳过事件集合对齐检查: %s", reference_manifest)
        return
    ref = json.loads(reference_manifest.read_text(encoding="utf-8"))
    ref_events = (ref.get("split_audit") or {}).get("events") or {}
    cur_events = (manifest.get("split_audit") or {}).get("events") or {}
    for subset in ("train", "val", "test", "holdout"):
        a = set(ref_events.get(subset) or [])
        b = set(cur_events.get(subset) or [])
        if a != b:
            raise ValueError(
                f"定位数据集 {subset} 事件与参考 YOLO 切分不一致: "
                f"only_in_loc={sorted(b - a)} only_in_yolo={sorted(a - b)}"
            )


# ── Mask → boxes ─────────────────────────────────────────────────────────────


def mask_to_boxes(
    prob_mask: "np.ndarray",
    conf_threshold: float = LOC_CONF_THRESHOLD,
    min_area: int = LOC_MIN_AREA_PX,
    dilate_k: int = LOC_DILATE_K,
) -> list[dict]:
    """概率图 → 连通域外接矩形列表。"""
    import cv2

    binary = (prob_mask >= conf_threshold).astype("uint8") * 255
    if dilate_k and dilate_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        binary = cv2.dilate(binary, kernel, iterations=1)
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[dict] = []
    for lab in range(1, n_labels):
        x, y, w, h, area = stats[lab]
        if area < min_area:
            continue
        region = prob_mask[y : y + h, x : x + w][labels[y : y + h, x : x + w] == lab]
        conf = float(region.mean()) if region.size else float(conf_threshold)
        boxes.append(
            {
                "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "conf": conf,
                "class_name": "building",
                "raw_class_name": "building",
                "class_id": 0,
                "area": int(area),
            }
        )
    boxes.sort(key=lambda d: d["conf"], reverse=True)
    return boxes


# ── Inference wrapper ────────────────────────────────────────────────────────


@dataclass
class BuildingLocalizer:
    model: "nn.Module"
    device: str
    imgsz: int = 1024
    conf_threshold: float = LOC_CONF_THRESHOLD
    min_area: int = LOC_MIN_AREA_PX
    dilate_k: int = LOC_DILATE_K

    def __post_init__(self) -> None:
        _require_torch()
        self.model.eval()
        self.tf = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.no_grad()
    def predict_proba(self, image: Union[str, Path, Image.Image]) -> "np.ndarray":
        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        else:
            pil = image.convert("RGB")
        orig_w, orig_h = pil.size
        resized = pil.resize((self.imgsz, self.imgsz), Image.BILINEAR) if (
            orig_w != self.imgsz or orig_h != self.imgsz
        ) else pil
        tensor = self.tf(resized).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        logits = _align_logits_to_input(logits, (self.imgsz, self.imgsz))
        probs = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()
        if probs.shape != (orig_h, orig_w):
            probs_img = Image.fromarray((probs * 255).astype("uint8"))
            probs_img = probs_img.resize((orig_w, orig_h), Image.BILINEAR)
            probs = np.array(probs_img, dtype="float32") / 255.0
        return probs

    def propose(self, image: Union[str, Path, Image.Image]) -> list[dict]:
        probs = self.predict_proba(image)
        return mask_to_boxes(
            probs,
            conf_threshold=self.conf_threshold,
            min_area=self.min_area,
            dilate_k=self.dilate_k,
        )

    def save_overlay(self, image: Union[str, Path, Image.Image], out_path: Path) -> list[dict]:
        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        else:
            pil = image.convert("RGB")
        boxes = self.propose(pil)
        draw = ImageDraw.Draw(pil)
        for det in boxes:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_path)
        return boxes


_LOCALIZER: Optional[BuildingLocalizer] = None
_LOCALIZER_LOCK = threading.Lock()


def load_building_localizer(
    ckpt_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> BuildingLocalizer:
    _require_torch()
    path = Path(ckpt_path or DEFAULT_CKPT_PATH).expanduser().resolve()
    # 禁止加载冠军竞赛权重文件名，避免事件泄漏（先于 exists 检查，便于测试拦截）
    banned = ("xview2_1st", "xview2_first", "vdurnov")
    lowered = str(path).lower()
    if any(b in lowered for b in banned):
        raise ValueError(
            f"拒绝加载疑似 xView2 冠军公开权重（事件泄漏风险）: {path}"
        )
    if not path.exists():
        raise FileNotFoundError(f"建筑定位 checkpoint 不存在: {path}")
    device = device or os.getenv("PERCEPTION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location="cpu")
    state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    model = ResNet34UNet(pretrained=False)
    model.load_state_dict(state)
    model.to(device)
    return BuildingLocalizer(
        model=model,
        device=device,
        imgsz=int(blob.get("imgsz", 1024)) if isinstance(blob, dict) else 1024,
        conf_threshold=float(blob.get("conf_threshold", LOC_CONF_THRESHOLD)) if isinstance(blob, dict) else LOC_CONF_THRESHOLD,
    )


def get_building_localizer() -> BuildingLocalizer:
    global _LOCALIZER
    if _LOCALIZER is not None:
        return _LOCALIZER
    with _LOCALIZER_LOCK:
        if _LOCALIZER is None:
            _LOCALIZER = load_building_localizer()
        return _LOCALIZER


# ── Train loop ───────────────────────────────────────────────────────────────


def _run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            logits = _align_logits_to_input(logits, masks.shape[-2:])
            loss = combined_loc_loss(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_dice += batch_dice(logits.detach(), masks) * bs
        n += bs
    return total_loss / max(1, n), total_dice / max(1, n)


def train_main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="building_localization train")
    ap.add_argument("--data-dir", default="/home/lc/datasets/xbd_loc_strict_v1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="每个子集最多样本数（调试）")
    ap.add_argument("--require-event-disjoint", action="store_true")
    ap.add_argument(
        "--reference-yolo-manifest",
        default="/home/lc/datasets/xbd_yolo_strict_v1/manifest.json",
        help="可选：断言定位切分事件集合与 YOLO strict 一致",
    )
    ap.add_argument(
        "--out",
        default=str(BACKEND_DIR / "outputs" / "building_localization" / "resnet34_strict_v1.pt"),
    )
    args = ap.parse_args(argv)

    _require_torch()
    set_seed(args.seed)
    data_dir = Path(args.data_dir).expanduser().resolve()
    manifest = None
    if args.require_event_disjoint:
        manifest = assert_event_disjoint(data_dir)
        assert_events_match_reference(manifest, Path(args.reference_yolo_manifest))

    train_ds = LocDataset(data_dir, "train", imgsz=args.imgsz, augment=True)
    val_ds = LocDataset(data_dir, "val", imgsz=args.imgsz, augment=False)
    if args.limit:
        train_ds.paths = train_ds.paths[: args.limit]
        val_ds.paths = val_ds.paths[: max(1, args.limit // 4)]
    print(f"[loc-train] train={len(train_ds)} val={len(val_ds)} device={args.device}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    device = args.device
    model = ResNet34UNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_dice = -1.0
    best_state = None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = _run_epoch(model, train_loader, device, optimizer)
        val_loss, val_dice = _run_epoch(model, val_loader, device, optimizer=None)
        print(
            f"[loc-train] epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.3f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.3f}"
        )
        if val_dice >= best_dice:
            best_dice = val_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(
            {
                "model_state": best_state or model.state_dict(),
                "epoch": epoch,
                "best_val_dice": best_dice,
                "imgsz": args.imgsz,
                "conf_threshold": LOC_CONF_THRESHOLD,
                "seed": args.seed,
                "encoder": "resnet34",
                "event_disjoint": bool(args.require_event_disjoint),
                "split_events": (manifest or {}).get("split_audit", {}).get("events"),
            },
            out_path,
        )

    print(f"[loc-train] done best_val_dice={best_dice:.3f} → {out_path}")
    return 0


def predict_main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="building_localization predict")
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT_PATH))
    ap.add_argument("--device", default=os.getenv("PERCEPTION_DEVICE", "cuda:0"))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    loc = load_building_localizer(Path(args.ckpt), device=args.device)
    out = Path(args.out) if args.out else Path(args.image).with_suffix(".loc_overlay.png")
    boxes = loc.save_overlay(args.image, out)
    print(json.dumps({"n_boxes": len(boxes), "overlay": str(out), "boxes": boxes[:20]}, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(argv) if argv is not None else None
    import sys as _sys

    args = argv if argv is not None else _sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print("usage: building_localization.py {train,predict} ...")
        return 2
    cmd, rest = args[0], args[1:]
    if cmd == "train":
        return train_main(rest)
    if cmd == "predict":
        return predict_main(rest)
    raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
