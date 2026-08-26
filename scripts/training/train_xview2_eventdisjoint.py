#!/usr/bin/env python3
"""P3b 轨 B：xView2 冠军架构在**事件不相交协议**下重训。

## 为什么需要这个

轨 A（现成权重）在 xBD 官方 train+tier3 上训练，而 xBD 按瓦片而非按事件划分，
所以 paper_cja 的 test / holdout 事件全部被见过 —— 它只能作 leaky 参照上界。

轨 B 用**同一架构**、但只在 `event_split.TRAIN_EVENTS` 上训练，
`VAL_EVENTS` 上验证，**从不接触 TEST/HOLDOUT**。于是：

  - 轨 B 本身 = 协议干净的主结果；
  - 轨 A − 轨 B 的差值 = 「事件曝光」贡献了多少（架构被控制住了）；
  - 轨 B − legacy 的差值 = 「架构/训练规模」贡献了多少（数据协议被控制住了）。

第三条正是 review2 **B7** 要求把「剩余差距归因于架构/训练规模」
从断言变成测量的那个实验。

## 泄漏断言

启动时硬校验：训练/验证集里不得出现任何 TEST/HOLDOUT 事件的瓦片，
违反直接 abort（不是警告）。

## 用法

    python scripts/training/train_xview2_eventdisjoint.py --stage loc  --epochs 40
    python scripts/training/train_xview2_eventdisjoint.py --stage cls  --epochs 20 \
        --loc-ckpt backend/outputs/xview2_eventdisjoint/res34_loc_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "detectors"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import xbd_map  # noqa: E402
from event_split import HOLDOUT_EVENTS, TEST_EVENTS, TRAIN_EVENTS, VAL_EVENTS  # noqa: E402
from xview2_zoo.losses import ComboLoss  # noqa: E402
from xview2_zoo.models import Res34_Unet_Double, Res34_Unet_Loc  # noqa: E402

_POLY_RE = re.compile(r"POLYGON\s*\(\((.*?)\)\)", re.DOTALL)
SUBTYPE_ID = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}
EVAL_EVENTS = frozenset(TEST_EVENTS) | frozenset(HOLDOUT_EVENTS)


def preprocess(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    x /= 127.0
    x -= 1.0
    return x


def _read_bgr(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)[..., ::-1].copy()


def _masks(label_path: Path, size) -> np.ndarray:
    """(H, W) uint8：0=背景，1..4=四类损伤。"""
    data = json.loads(label_path.read_text(encoding="utf-8"))
    img = Image.new("L", size, 0)
    dr = ImageDraw.Draw(img)
    for feat in (data.get("features") or {}).get("xy") or []:
        m = _POLY_RE.search(feat.get("wkt") or "")
        if not m:
            continue
        pts = []
        for pair in m.group(1).split(","):
            parts = pair.strip().split()
            if len(parts) >= 2:
                pts.append((float(parts[0]), float(parts[1])))
        sub = ((feat.get("properties") or {}).get("subtype") or "").strip()
        if pts and sub in SUBTYPE_ID:
            dr.polygon(pts, fill=SUBTYPE_ID[sub])
    return np.asarray(img, dtype=np.uint8)


class XbdPairs(Dataset):
    """事件不相交的 pre/post 配对样本。"""

    def __init__(self, entries, root: Path, items: dict, crop: int, train: bool, stage: str):
        self.entries = entries
        self.root = root
        self.items = items
        self.crop = int(crop)
        self.train = bool(train)
        self.stage = stage

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        post = self.entries[i]
        pre = self.items[post["paired_tile_id"]]
        pre_img = _read_bgr(self.root / pre["image_relpath"])
        post_img = _read_bgr(self.root / post["image_relpath"])
        msk = _masks(self.root / post["label_relpath"], (post_img.shape[1], post_img.shape[0]))

        H, W = post_img.shape[:2]
        c = self.crop
        if self.train:
            # 偏向有建筑的区域采样，否则大量全背景 crop 会淹没信号
            best = None
            for _ in range(6):
                y = random.randint(0, max(0, H - c))
                x = random.randint(0, max(0, W - c))
                s = int((msk[y:y + c, x:x + c] > 0).sum())
                if best is None or s > best[0]:
                    best = (s, y, x)
                if s > 0:
                    break
            _, y, x = best
        else:
            y, x = max(0, (H - c) // 2), max(0, (W - c) // 2)

        pre_c = pre_img[y:y + c, x:x + c]
        post_c = post_img[y:y + c, x:x + c]
        m_c = msk[y:y + c, x:x + c]

        if self.train and random.random() < 0.5:
            pre_c, post_c, m_c = pre_c[:, ::-1], post_c[:, ::-1], m_c[:, ::-1]
        pre_c, post_c, m_c = pre_c.copy(), post_c.copy(), m_c.copy()

        if self.stage == "loc":
            x_t = torch.from_numpy(preprocess(pre_c)).permute(2, 0, 1)
            y_t = torch.from_numpy((m_c > 0).astype("float32")).unsqueeze(0)
        else:
            x_t = torch.from_numpy(
                preprocess(np.concatenate([pre_c, post_c], axis=2))
            ).permute(2, 0, 1)
            ch = [(m_c > 0).astype("float32")]
            for cid in (1, 2, 3, 4):
                ch.append((m_c == cid).astype("float32"))
            y_t = torch.from_numpy(np.stack(ch, axis=0))
        return x_t, y_t


def _dice(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5) -> float:
    p = (torch.sigmoid(pred) > thr).float()
    inter = (p * gt).sum().item()
    s = p.sum().item() + gt.sum().item()
    return (2 * inter / s) if s > 0 else 1.0


def build_split(manifest, split_events):
    root = Path(manifest["dataset_root"])
    items = {e["tile_id"]: e for e in manifest["items"]}
    out = []
    for e in manifest["items"]:
        if e.get("stage") != "post":
            continue
        if e.get("disaster") not in split_events:
            continue
        if not (e.get("paired_tile_id") and e.get("label_relpath")):
            continue
        pre = items.get(e["paired_tile_id"])
        if not pre or not pre.get("image_relpath"):
            continue
        if not (root / e["image_relpath"]).is_file():
            continue
        if not (root / pre["image_relpath"]).is_file():
            continue
        if not (root / e["label_relpath"]).is_file():
            continue
        out.append(e)
    return out, items, root


def assert_no_eval_leak(entries, name: str) -> None:
    bad = sorted({e["disaster"] for e in entries if e.get("disaster") in EVAL_EVENTS})
    if bad:
        raise SystemExit(
            f"EVENT LEAKAGE in {name}: {bad}. 轨 B 的全部意义就是事件不相交，"
            "训练/验证集不得出现 TEST/HOLDOUT 事件。"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["loc", "cls"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loc-ckpt", default="")
    ap.add_argument("--manifest", default=str(ROOT / "backend/data/xbd/manifest.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "backend/outputs/xview2_eventdisjoint"))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = xbd_map.load_manifest(args.manifest)
    train_e, items, root = build_split(manifest, frozenset(TRAIN_EVENTS))
    val_e, _, _ = build_split(manifest, frozenset(VAL_EVENTS))
    assert_no_eval_leak(train_e, "train")
    assert_no_eval_leak(val_e, "val")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"res34_{args.stage}_seed{args.seed}"

    print(f"[{tag}] train tiles={len(train_e)} from {len(TRAIN_EVENTS)} events; "
          f"val tiles={len(val_e)} from {list(VAL_EVENTS)}", flush=True)
    print(f"[{tag}] leakage assertion passed: no TEST/HOLDOUT events present", flush=True)

    ds_tr = XbdPairs(train_e, root, items, args.crop, True, args.stage)
    ds_va = XbdPairs(val_e, root, items, args.crop, False, args.stage)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    dev = torch.device(args.device)
    if args.stage == "loc":
        model = Res34_Unet_Loc(pretrained=True)
    else:
        model = Res34_Unet_Double(pretrained=True)
        if args.loc_ckpt and Path(args.loc_ckpt).is_file():
            # 参考实现用 loc 权重初始化 cls 的共享编码器。
            # 注意：`strict=False` 只容忍**缺失/多余**的键，遇到**形状不一致**仍会抛错。
            # loc 的输出头 res 是 (1, 48)，cls 的是 (5, 96)（siamese 拼接后通道翻倍），
            # 所以必须按形状先过滤，否则加载直接崩。
            sd = torch.load(args.loc_ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            tgt = model.state_dict()
            keep, skipped = {}, []
            for k, v in sd.items():
                if k in tgt and tuple(tgt[k].shape) == tuple(v.shape):
                    keep[k] = v
                else:
                    skipped.append(k)
            missing, unexpected = model.load_state_dict(keep, strict=False)
            print(f"[{tag}] warm-start from loc: loaded={len(keep)} "
                  f"shape_skipped={len(skipped)} missing={len(missing)} "
                  f"unexpected={len(unexpected)}", flush=True)
            if skipped:
                print(f"[{tag}]   skipped (shape differs): {skipped}", flush=True)
    model = model.to(dev)

    loss_fn = ComboLoss({"dice": 1.0, "focal": 10.0}, per_image=False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(args.epochs * f) for f in (0.4, 0.65, 0.85)], gamma=0.4,
    )
    scaler = torch.amp.GradScaler("cuda")

    best = -1.0
    hist = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(xb)
                loss = sum(loss_fn(out[:, i], yb[:, i]) for i in range(yb.shape[1]))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item()); n += 1
        sched.step()

        model.eval()
        dices = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(dev), yb.to(dev)
                with torch.amp.autocast("cuda"):
                    out = model(xb)
                dices.append(_dice(out[:, 0].float(), yb[:, 0]))
        score = float(np.mean(dices)) if dices else 0.0
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_dice": score,
                     "lr": sched.get_last_lr()[-1], "sec": round(time.time() - t0, 1)})
        print(f"[{tag}] ep {ep+1}/{args.epochs} loss={tot/max(n,1):.4f} "
              f"val_dice={score:.4f} best={max(best,score):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        if score > best:
            best = score
            torch.save({"epoch": ep + 1, "best_score": score,
                        "state_dict": model.state_dict()},
                       out_dir / f"{tag}_best.pt")

    (out_dir / f"{tag}_history.json").write_text(json.dumps({
        "schema": "xview2-eventdisjoint-train/1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage, "seed": args.seed, "epochs": args.epochs,
        "crop": args.crop, "batch_size": args.batch_size, "lr": args.lr,
        "train_events": list(TRAIN_EVENTS), "val_events": list(VAL_EVENTS),
        "n_train_tiles": len(train_e), "n_val_tiles": len(val_e),
        "leakage_assertion": "no TEST/HOLDOUT events in train or val",
        "best_val_dice": best, "history": hist,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{tag}] done. best val dice={best:.4f} → {out_dir/f'{tag}_best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
