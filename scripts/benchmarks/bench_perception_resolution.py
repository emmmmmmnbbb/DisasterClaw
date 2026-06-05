#!/usr/bin/env python
"""
bench_perception_resolution.py — Bench-B: YOLO/SegFormer 推理延迟与显存峰值
随分辨率（768→2048）的变化。

进程内直接拿 DisasterPerception 单例，从 RescueNet 重灾头牌图
/home/lc/tune7b/train/train-org-img/12215.jpg (4000×3000) 做 center crop，
喂给 _detect / _segment 与 _descriptor.generate。每档分辨率：
    warmup=5 + runs=20，统计 mean / p50 / p95（毫秒）和 GPU max_memory_allocated。

输出: runs/benchmarks/<run-id>/resolution.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    SLA,
    dump_json,
    ensure_run_dir,
    env_snapshot,
    now_run_id,
    stat_summary,
)

# Ensure backend modules are importable (perception lives in backend/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))


DEFAULT_IMG = Path("/home/lc/tune7b/train/train-org-img/12215.jpg")
DEFAULT_RESOLUTIONS = (768, 1024, 1280, 1536, 1792, 2048)


def center_crop_square(img: Image.Image, size: int) -> Image.Image:
    W, H = img.size
    if size > min(W, H):
        # downscale path: resize so the shorter side >= size, then crop
        scale = size / float(min(W, H))
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        W, H = img.size
    left = (W - size) // 2
    top = (H - size) // 2
    return img.crop((left, top, left + size, top + size))


def gpu_peak_reset() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def gpu_peak_mb() -> float | None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return None


def run_resolution_bench(args: argparse.Namespace) -> dict:
    if not Path(args.image).is_file():
        raise FileNotFoundError(f"reference image not found: {args.image}")

    print(f"[bench-b] loading reference image {args.image}")
    base_img = Image.open(args.image).convert("RGB")
    print(f"   base size: {base_img.size}")

    # Pre-crop all resolutions and save under a tmp folder (perception expects a path)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    crops: dict[int, Path] = {}
    for r in args.resolutions:
        out_path = tmp_dir / f"bench_b_crop_{r}.jpg"
        crop = center_crop_square(base_img, r)
        crop.save(out_path, quality=95)
        crops[r] = out_path
        print(f"   prepared {r}x{r} -> {out_path}")

    # Lazy import perception (loads heavy deps)
    print("[bench-b] importing perception backend (this may take a few seconds)...")
    from perception import get_perception  # type: ignore

    engine = get_perception()
    print("[bench-b] loading YOLO + SegFormer models...")
    t0 = time.time()
    engine.load()
    print(f"[bench-b] models ready in {time.time()-t0:.1f}s, available={engine.is_available}")

    if not engine.is_available:
        raise RuntimeError(
            f"perception engine not available: {engine.last_error}"
        )

    results: dict[int, dict] = {}
    for r in args.resolutions:
        path = crops[r]
        print(f"\n--- resolution {r}x{r} ({path.name}) ---")
        det_lat: list[float] = []
        seg_lat: list[float] = []
        desc_lat: list[float] = []
        det_peak_mb: list[float] = []
        seg_peak_mb: list[float] = []

        for i in range(args.warmup + args.runs):
            tag = "warm" if i < args.warmup else f"run{i - args.warmup + 1}"

            gpu_peak_reset()
            t_a = time.perf_counter_ns()
            det_payload = engine._detect(path)
            t_b = time.perf_counter_ns()
            det_dt_ms = (t_b - t_a) / 1_000_000
            det_mem = gpu_peak_mb()

            gpu_peak_reset()
            t_c = time.perf_counter_ns()
            seg_payload = engine._segment(path)
            t_d = time.perf_counter_ns()
            seg_dt_ms = (t_d - t_c) / 1_000_000
            seg_mem = gpu_peak_mb()

            t_e = time.perf_counter_ns()
            scene = engine._descriptor.generate(
                {
                    "detection": {
                        "detections": det_payload["detections"],
                        "class_counts": det_payload["class_counts"],
                    },
                    "segmentation": {"stats": seg_payload["stats"]},
                },
                image_width=r,
                image_height=r,
            )
            t_f = time.perf_counter_ns()
            desc_dt_ms = (t_f - t_e) / 1_000_000

            if i >= args.warmup:
                det_lat.append(det_dt_ms)
                seg_lat.append(seg_dt_ms)
                desc_lat.append(desc_dt_ms)
                if det_mem is not None:
                    det_peak_mb.append(det_mem)
                if seg_mem is not None:
                    seg_peak_mb.append(seg_mem)

            if i == 0 or i == args.warmup:
                # log stats once per phase (warm + first measured)
                print(
                    f"   [{tag}] det={det_dt_ms:.1f}ms  seg={seg_dt_ms:.1f}ms  desc={desc_dt_ms:.2f}ms"
                    f"   det_peak={det_mem and round(det_mem,0)}MB seg_peak={seg_mem and round(seg_mem,0)}MB"
                )

        results[r] = {
            "resolution": r,
            "yolo_ms": stat_summary(det_lat),
            "segformer_ms": stat_summary(seg_lat),
            "descriptor_ms": stat_summary(desc_lat),
            "yolo_peak_mb": stat_summary(det_peak_mb),
            "segformer_peak_mb": stat_summary(seg_peak_mb),
            "total_pipeline_ms": stat_summary(
                [d + s + dx for d, s, dx in zip(det_lat, seg_lat, desc_lat)]
            ),
            "n_runs": len(det_lat),
        }
        print(
            f"   summary: yolo mean={results[r]['yolo_ms']['mean']:.1f}ms "
            f"seg mean={results[r]['segformer_ms']['mean']:.1f}ms "
            f"total mean={results[r]['total_pipeline_ms']['mean']:.1f}ms"
        )

    return {
        "bench": "perception_resolution",
        "args": {
            "image": str(args.image),
            "resolutions": list(args.resolutions),
            "warmup": args.warmup,
            "runs": args.runs,
            "tmp_dir": str(tmp_dir),
        },
        "sla_ms": SLA,
        "env": env_snapshot(),
        "results": {str(k): v for k, v in results.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=str(DEFAULT_IMG))
    ap.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=list(DEFAULT_RESOLUTIONS),
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--tmp-dir", default="/tmp/disasterclaw_bench_b")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or now_run_id()
    out_dir = ensure_run_dir(run_id)
    print(f"[bench-b] run_id={run_id} out_dir={out_dir}")

    result = run_resolution_bench(args)
    result["run_id"] = run_id

    out_path = out_dir / "resolution.json"
    dump_json(out_path, result)
    print(f"\n[bench-b] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
