"""
backend/change_perception.py — P5 校准的双时相变化感知（C2 正式实现，升级接口落地）

对应 docs/vln_rescue_agent_实施计划.md 第六节"升级接口"：
    U_t：启发式标量 → 分布熵 U_t = -Σ p_i log p_i（p 为温度校准后的 4 类损伤 softmax）。

本模块**不是新的独立贡献**，而是给 backend/recheck.py 的熵模式提供一个真正校准过的
概率分布来源，替代当前 backend/perception.py 里 YOLO 检测框的单一 top-1 conf。

设计：
    - Siamese 编码器（共享权重）分别提取 pre / post 两张配对 patch 的特征；
      拼接 [f_pre, f_post, f_post-f_pre] 后接两个头：
        damage_head  : 4 类损伤 logits（no/minor/major-damage, destroyed）
        change_head  : 1 个二分类 logit（该建筑是否发生变化）—— 辅助监督信号，
                        呼应 Change-Agent / ISPRS 2026 论文"检测+分类多任务联合"的思路。
    - 训练完成后在验证集上做温度标定（temperature scaling，Guo et al. 2017 的标准做法）：
      只学一个标量 T，用 p = softmax(logits / T) 让概率分布的置信度与经验正确率对齐。
    - 推理侧暴露 `ChangePerceptionModel.predict(pre_patch, post_patch) -> dict`，
      返回校准后的 `class_probs`（4 维概率向量），供 recheck.py 算熵 U_t。

训练数据来自 `scripts/training/gen_xbd_change_dataset.py` 产出的 JSONL 清单（不落盘裁剪
每栋建筑的 patch，训练时现场从原始 tile PNG 裁剪，避免生成成百上千的小文件）。

训练：
    python backend/change_perception.py train \
        --data-dir /home/lc/datasets/xbd_change --epochs 8 --device cuda:0

推理（供 perception.py 调用）：
    from change_perception import get_change_perception
    result = get_change_perception().predict(pre_patch_img, post_patch_img)
    # result["class_probs"] -> {"no-damage": 0.7, "minor-damage": 0.2, ...}
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
NUM_CLASSES = len(CLASS_NAMES)

CROP_SIZE = 96          # 送入编码器的正方形边长（px）
CONTEXT_MARGIN = 0.25   # bbox 四周各扩 25% 作为上下文，帮助模型看到建筑周边变化痕迹

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_PATH = Path(
    __import__("os").getenv(
        "CHANGE_PERCEPTION_CKPT",
        str(BACKEND_DIR / "outputs" / "change_perception" / "model.pt"),
    )
)

# ── 延迟 import torch：避免只想跑数据生成 / 纯 CPU 调试时被迫装 GPU 版 torch ──────
_TORCH_IMPORT_ERROR: Optional[str] = None
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models as tv_models
    from torchvision import transforms as tv_transforms
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = Dataset = object  # type: ignore
    tv_models = tv_transforms = None  # type: ignore
    _TORCH_IMPORT_ERROR = str(exc)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(f"torch 不可用，无法训练/推理 change_perception: {_TORCH_IMPORT_ERROR}")


# ── 几何：bbox → 裁剪 patch（复用 gen_xbd_change_dataset.py 的标注约定）──────────

def _expand_bbox(
    bbox: tuple[float, float, float, float], margin_frac: float, w: int, h: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    mx, my = bw * margin_frac, bh * margin_frac
    return (
        max(0.0, x1 - mx), max(0.0, y1 - my),
        min(float(w), x2 + mx), min(float(h), y2 + my),
    )


def crop_patch(
    image_path: str, bbox: tuple[float, float, float, float],
    image_w: int, image_h: int, out_size: int = CROP_SIZE,
) -> Image.Image:
    """按 bbox（+上下文边距）从原图裁一块正方形 patch，resize 到 out_size。"""
    x1, y1, x2, y2 = _expand_bbox(bbox, CONTEXT_MARGIN, image_w, image_h)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        left, top = int(round(x1)), int(round(y1))
        right, bottom = max(left + 1, int(round(x2))), max(top + 1, int(round(y2)))
        patch = im.crop((left, top, right, bottom))
        patch = patch.resize((out_size, out_size), Image.BILINEAR)
    return patch


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _make_transform():
    _require_torch()
    return tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ── 数据集：从 JSONL 清单现场裁剪配对 patch ──────────────────────────────────

class XbdChangeDataset(Dataset):  # type: ignore[misc]
    """读取 gen_xbd_change_dataset.py 产出的 JSONL，现场裁剪 pre/post 配对 patch。"""

    def __init__(self, jsonl_path: str | Path, augment: bool = False):
        _require_torch()
        self.jsonl_path = Path(jsonl_path)
        self.records: list[dict] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.augment = augment
        self._transform = _make_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        pre = crop_patch(rec["pre_image"], tuple(rec["bbox_pre"]), rec["image_width"], rec["image_height"])
        post = crop_patch(rec["post_image"], tuple(rec["bbox_post"]), rec["image_width"], rec["image_height"])
        if self.augment and random.random() < 0.5:
            pre = pre.transpose(Image.FLIP_LEFT_RIGHT)
            post = post.transpose(Image.FLIP_LEFT_RIGHT)
        pre_t = self._transform(pre)
        post_t = self._transform(post)
        return pre_t, post_t, int(rec["class_id"]), float(rec["changed"])


# ── 模型：Siamese 编码器 + 双任务头 ───────────────────────────────────────────

class _SiameseEncoder(nn.Module):  # type: ignore[misc]
    def __init__(self, out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, out_dim)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class DifferenceAttention(nn.Module):  # type: ignore[misc]
    """D2ANet（Difference-aware Attention Network，为 xBD 损伤分级设计）思路的轻量
    适配：DTA（双时态聚合门控）+ DA（差分注意力）。

    原论文在卷积特征图上做逐空间位置的通道注意力；这里的编码器已经把每张 patch
    压成一个全局特征向量（分类头本身就是全图级判断，不需要空间粒度），所以退化成
    向量通道维度的 SE-style（Squeeze-and-Excitation）门控，但保留核心思想不变：
        DTA：从 [f_pre, f_post] 联合学一组通道门控，分别重加权两个时刻的特征——
             不同通道对"这次到底发生了什么类型的变化"的敏感度不同（原论文里
             对应"哪些通道该重点比较"）；
        DA ：在门控后的差分特征上再学一层通道注意力，进一步放大变化敏感通道、
             抑制光照/视角等无关差异——这是本模型和"直接拼接 [f_pre,f_post,diff]"
             baseline 的核心区别：diff 不是简单相减，而是先门控再相减再门控。
    """

    def __init__(self, feat_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, feat_dim // reduction)
        self.dta_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim * 2), nn.Sigmoid(),
        )
        self.da_gate = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim), nn.Sigmoid(),
        )
        self.feat_dim = feat_dim

    def forward(self, f_pre, f_post):
        gate = self.dta_gate(torch.cat([f_pre, f_post], dim=1))
        g_pre, g_post = gate[:, : self.feat_dim], gate[:, self.feat_dim :]
        f_pre_w = f_pre * g_pre
        f_post_w = f_post * g_post
        diff = f_post_w - f_pre_w
        diff_w = diff * self.da_gate(diff)
        return torch.cat([f_pre_w, f_post_w, diff_w], dim=1)


class ChangeMultiTaskNet(nn.Module):  # type: ignore[misc]
    """Siamese 多任务头：4 类损伤 logits + 二值变化 logit。

    dropout_p>0 时在两个头的隐藏层后插入 Dropout——不是为了正则化本身，而是给
    MC-Dropout（Gal & Ghahramani 2016）不确定性估计基线留一个"训练时就要打开"的
    开关：MC-Dropout 要求推理时也保持 dropout 随机采样，因此模型结构必须原生带
    dropout 层，不能靠推理脚本临时打补丁。默认 0.0，不影响现有已训练模型的行为。

    use_diff_attention=True 时用 DifferenceAttention 替换原来"直接拼接
    [f_pre,f_post,f_post-f_pre]"的融合方式（对标 D2ANet baseline，见 E15）；
    默认 False，不影响现有已训练模型的行为/结构。
    """

    def __init__(
        self, feat_dim: int = 128, num_classes: int = NUM_CLASSES,
        pretrained: bool = True, dropout_p: float = 0.0, use_diff_attention: bool = False,
    ):
        super().__init__()
        self.encoder = _SiameseEncoder(feat_dim, pretrained=pretrained)
        self.dropout_p = dropout_p
        self.use_diff_attention = use_diff_attention
        self.diff_attn = DifferenceAttention(feat_dim) if use_diff_attention else None
        fused_dim = feat_dim * 3  # [f_pre, f_post, f_post - f_pre]（或门控版）
        self.damage_head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )
        self.change_head = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Linear(64, 1),
        )

    def features(self, pre, post):
        f_pre = self.encoder(pre)
        f_post = self.encoder(post)
        if self.diff_attn is not None:
            return self.diff_attn(f_pre, f_post)
        return torch.cat([f_pre, f_post, f_post - f_pre], dim=1)

    def forward(self, pre, post):
        fused = self.features(pre, post)
        damage_logits = self.damage_head(fused)
        change_logit = self.change_head(fused).squeeze(-1)
        return damage_logits, change_logit


# ── 温度标定（Guo et al. 2017《On Calibration of Modern Neural Networks》）──────

def fit_temperature(logits, labels, max_iter: int = 50) -> float:
    """在验证集 logits/labels 上拟合标量 T，最小化 NLL；返回校准后的温度。"""
    _require_torch()
    temperature = nn.Parameter(torch.ones(1, device=logits.device) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def _closure():
        optimizer.zero_grad()
        t = temperature.clamp(min=1e-3)
        loss = nll(logits / t, labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(temperature.clamp(min=1e-3).item())


def set_seed(seed: int) -> None:
    """训练 Deep Ensemble 成员时用不同 seed，才能得到真正独立的模型（而不是同一次
    随机初始化的多份拷贝）——这是 Lakshminarayanan et al. 2017 Deep Ensembles 方法
    的核心前提：多样性来自训练随机性（初始化 + shuffle），不是显式的贝叶斯建模。"""
    random.seed(seed)
    _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def mc_dropout_logits(model, pre, post, n_passes: int = 30) -> "torch.Tensor":
    """MC-Dropout（Gal & Ghahramani 2016）：保持 dropout 层随机采样、其余（尤其
    BatchNorm）仍用 eval 统计量，跑 n_passes 次前向，返回 [n_passes, B, num_classes]
    的 logits 堆叠，供上层在概率空间取平均（而不是在 logits 空间平均——softmax 非
    线性，MC-Dropout 定义的是对预测分布 p(y|x) 的平均，必须先各自 softmax 再平均）。

    模型必须是用 dropout_p>0 训练出来的（结构里有非零 Dropout），否则 n_passes
    次前向会完全相同，退化成普通单次推理（不报错，但失去 MC-Dropout 的意义）。
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()  # 只让 Dropout 子模块保持随机，BatchNorm 等其余层仍是 eval 统计量
    stacked = []
    for _ in range(n_passes):
        damage_logits, _ = model(pre, post)
        stacked.append(damage_logits)
    return torch.stack(stacked, dim=0)


# ── 推理封装：进程级单例，风格对齐 perception.get_perception() ─────────────────

@dataclass
class ChangePrediction:
    class_probs: dict[str, float]
    class_name: str
    confidence: float
    change_prob: float
    temperature: float


class ChangePerceptionModel:
    """进程级单例：加载 checkpoint（含温度），暴露校准过的 4 类概率分布。"""

    def __init__(self, ckpt_path: Optional[str | Path] = None, device: Optional[str] = None):
        self.ckpt_path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT_PATH
        self._device_arg = device
        self.model = None
        self.temperature: float = 1.0
        self._loaded = False
        self._load_lock = threading.Lock()
        self._transform = None

    @property
    def device(self) -> str:
        if self._device_arg:
            return self._device_arg
        if torch is not None and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    @property
    def is_available(self) -> bool:
        return self._loaded and self.model is not None

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            _require_torch()
            use_diff_attention = False
            if self.ckpt_path.exists():
                # 先探一眼 checkpoint 里的结构标志，再构造匹配的模型——
                # use_diff_attention 影响的是模块结构（多了 DifferenceAttention 子模块），
                # 不像 dropout_p 那样只影响一个已存在层的行为，必须在 load_state_dict 前定下来。
                probe_state = torch.load(self.ckpt_path, map_location=self.device)
                use_diff_attention = bool(probe_state.get("use_diff_attention", False))
            self.model = ChangeMultiTaskNet(
                pretrained=not self.ckpt_path.exists(), use_diff_attention=use_diff_attention,
            ).to(self.device)
            if self.ckpt_path.exists():
                state = torch.load(self.ckpt_path, map_location=self.device)
                self.model.load_state_dict(state["model_state"])
                self.temperature = float(state.get("temperature", 1.0))
                logger.info(
                    "[ChangePerception] loaded checkpoint %s (T=%.3f)",
                    self.ckpt_path, self.temperature,
                )
            else:
                logger.warning(
                    "[ChangePerception] checkpoint 不存在: %s，使用随机初始化权重"
                    "（仅供打通链路，不代表真实精度；先跑 `change_perception.py train`）",
                    self.ckpt_path,
                )
            self.model.eval()
            self._transform = _make_transform()
            self._loaded = True

    def predict(self, pre_patch: Image.Image, post_patch: Image.Image) -> ChangePrediction:
        self.load()
        with torch.no_grad():
            pre_t = self._transform(pre_patch.convert("RGB")).unsqueeze(0).to(self.device)
            post_t = self._transform(post_patch.convert("RGB")).unsqueeze(0).to(self.device)
            damage_logits, change_logit = self.model(pre_t, post_t)
            t = max(self.temperature, 1e-3)
            probs = torch.softmax(damage_logits / t, dim=-1)[0].cpu().numpy()
            change_prob = float(torch.sigmoid(change_logit)[0].item())
        cls_idx = int(probs.argmax())
        return ChangePrediction(
            class_probs={name: float(p) for name, p in zip(CLASS_NAMES, probs)},
            class_name=CLASS_NAMES[cls_idx],
            confidence=float(probs[cls_idx]),
            change_prob=change_prob,
            temperature=float(self.temperature),
        )


_INSTANCE: Optional[ChangePerceptionModel] = None
_INSTANCE_LOCK = threading.Lock()


def get_change_perception() -> ChangePerceptionModel:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = ChangePerceptionModel()
    return _INSTANCE


# ── 训练 CLI ─────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, device, optimizer=None, change_loss_weight: float = 0.5):
    training = optimizer is not None
    model.train(training)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    total_loss, n_correct, n_total = 0.0, 0, 0
    all_logits, all_labels = [], []
    for pre, post, cls_id, changed in loader:
        pre, post = pre.to(device), post.to(device)
        cls_id, changed = cls_id.to(device), changed.to(device).float()
        with torch.set_grad_enabled(training):
            damage_logits, change_logit = model(pre, post)
            loss = ce(damage_logits, cls_id) + change_loss_weight * bce(change_logit, changed)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * pre.size(0)
        n_correct += int((damage_logits.argmax(dim=-1) == cls_id).sum().item())
        n_total += pre.size(0)
        all_logits.append(damage_logits.detach())
        all_labels.append(cls_id.detach())
    avg_loss = total_loss / max(1, n_total)
    acc = n_correct / max(1, n_total)
    logits_cat = torch.cat(all_logits, dim=0) if all_logits else None
    labels_cat = torch.cat(all_labels, dim=0) if all_labels else None
    return avg_loss, acc, logits_cat, labels_cat


def train_main(args: argparse.Namespace) -> int:
    _require_torch()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    train_ds = XbdChangeDataset(data_dir / "train.jsonl", augment=True)
    val_path = data_dir / "val.jsonl"
    if not val_path.exists():
        raise FileNotFoundError(f"缺少 val.jsonl: {val_path}（先跑 gen_xbd_change_dataset.py）")
    val_ds = XbdChangeDataset(val_path, augment=False)
    if args.limit:
        train_ds.records = train_ds.records[: args.limit]
        val_ds.records = val_ds.records[: max(1, args.limit // 4)]
    print(f"[train] train={len(train_ds)} val={len(val_ds)} device={args.device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = args.device
    model = ChangeMultiTaskNet(
        pretrained=True, dropout_p=args.dropout, use_diff_attention=args.diff_attention,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_state = None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = _run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc, val_logits, val_labels = _run_epoch(model, val_loader, device, optimizer=None)
        print(
            f"[train] epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # 长时间训练（I/O bound，单 epoch 数分钟到十几分钟）在共享 GPU 服务器上有被
        # 其它进程挤占显存/被杀的风险；每个 epoch 都落一次盘，避免全白跑。
        torch.save({"model_state": best_state or model.state_dict(),
                    "temperature": 1.0, "epoch": epoch, "val_acc": best_val_acc,
                    "dropout_p": args.dropout, "seed": args.seed,
                    "use_diff_attention": args.diff_attention},
                   out_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    # 温度标定：在验证集 logits 上拟合（用最后一轮的 val_logits/labels 即可，
    # 若模型在训练中更新过、最后一轮不是 best epoch，重新跑一次 val 前向）
    _, _, val_logits, val_labels = _run_epoch(model, val_loader, device, optimizer=None)
    temperature = fit_temperature(val_logits, val_labels) if val_logits is not None else 1.0
    print(f"[train] 温度标定完成: T={temperature:.3f}（best val_acc={best_val_acc:.3f}）")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "temperature": temperature,
            "class_names": CLASS_NAMES,
            "best_val_acc": best_val_acc,
            "dropout_p": args.dropout,
            "seed": args.seed,
            "use_diff_attention": args.diff_attention,
        },
        out_path,
    )
    print(f"[train] checkpoint 已保存 → {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    train_ap = sub.add_parser("train", help="训练多任务头 + 温度标定")
    train_ap.add_argument("--data-dir", default="/home/lc/datasets/xbd_change")
    train_ap.add_argument("--epochs", type=int, default=8)
    train_ap.add_argument("--batch-size", type=int, default=64)
    train_ap.add_argument("--lr", type=float, default=1e-3)
    train_ap.add_argument("--workers", type=int, default=4)
    train_ap.add_argument("--device", default="cuda:0" if (torch and torch.cuda.is_available()) else "cpu")
    train_ap.add_argument("--out", default=str(DEFAULT_CKPT_PATH))
    train_ap.add_argument("--limit", type=int, default=0, help="调试用：截断训练/验证记录数")
    train_ap.add_argument("--seed", type=int, default=0, help="Deep Ensemble 用不同 seed 训练多个独立成员")
    train_ap.add_argument("--dropout", type=float, default=0.0, help="head 内 Dropout 概率，>0 才能支持 MC-Dropout 推理")
    train_ap.add_argument("--diff-attention", action="store_true", help="用 D2ANet 式差分注意力融合替代简单拼接（E15）")

    args = ap.parse_args()
    if args.command == "train":
        return train_main(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
