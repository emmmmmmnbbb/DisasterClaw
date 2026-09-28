"""MESSI replay with the repository DisasterClaw loop and legacy YOLO/SegFormer.

No labels or future image are exposed to the controller. A lower frame is opened
only after AgentVqaController requests reobservation and RecheckController emits
the bounded descend action.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, "/home/lc/Langchain-Chatchat/tools/mars/tools")
sys.path.insert(0, "/home/lc/Langchain-Chatchat/tools/mars/results/yolo")

from agent_vqa import AgentVqaConfig, AgentVqaController  # noqa: E402
from perception import PerceptionResult  # noqa: E402
from recheck import RecheckConfig, RecheckController  # noqa: E402
from segformer_tool import SegFormerTool  # noqa: E402
from yolo_tool import YOLOTool  # noqa: E402

VEGETATION_IDS = (4, 9, 17, 72)  # ADE20K tree, grass, plant, palm
QUESTION = "是否存在植被？"
TRIGGER = 0.50
YOLO_WEIGHTS = "/home/lc/Langchain-Chatchat/tools/mars/results/yolo/runs/train/mars_det_yolov8n4/weights/best.pt"
SEGFORMER_ID = "nvidia/segformer-b2-finetuned-ade-512-512"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_provenance():
    hub = Path.home() / ".cache/huggingface/hub/models--nvidia--segformer-b2-finetuned-ade-512-512"
    revision = (hub / "refs/main").read_text().strip()
    snapshot = hub / "snapshots" / revision
    files = {}
    for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin", "model.safetensors"):
        path = snapshot / name
        if path.exists():
            files[name] = sha(path)
    return {
        "yolo_weights": YOLO_WEIGHTS,
        "yolo_weights_sha256": sha(YOLO_WEIGHTS),
        "segformer_model_id": SEGFORMER_ID,
        "segformer_hf_snapshot": revision,
        "segformer_files_sha256": files,
    }


def load_models(device):
    yolo = YOLOTool(YOLO_WEIGHTS, device=device, conf=0.25, iou=0.45, imgsz=640)
    seg = SegFormerTool("segformer-b2", device=device)
    import torch
    import torch.nn.functional as F
    return yolo, seg, torch, F


def infer(path, yolo, seg, torch, F):
    began = time.time()
    with Image.open(path) as im:
        image = im.convert("RGB")
    inputs = {k: v.to(seg.device) for k, v in seg.processor(images=image, return_tensors="pt").items()}
    with torch.no_grad():
        logits = seg.model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=1)[0]
        veg_soft = probs[list(VEGETATION_IDS)].sum(dim=0)
        hard = logits[0].argmax(dim=0)
        hard_veg = torch.zeros_like(hard, dtype=torch.bool)
        for class_id in VEGETATION_IDS:
            hard_veg |= hard == class_id
        veg_fraction = float(veg_soft.mean().item())
        hard_fraction = float(hard_veg.float().mean().item())
    # The occupancy-to-answer mapping uses the frozen MESSI mask boundaries;
    # it is an uncalibrated task score, not a probability calibration claim.
    p_yes = float(np.clip((veg_fraction - 0.005) / (0.05 - 0.005), 0.0, 1.0))
    with contextlib.redirect_stdout(io.StringIO()):
        yolo_raw = yolo.detect(str(path))
    detections = []
    label_map = {
        "type_2": "无损伤建筑", "type_3": "轻微损伤建筑",
        "type_4": "严重损伤建筑", "type_5": "完全损毁建筑",
        "type_6": "车辆", "type_10": "水池/积水区域",
    }
    for d in yolo_raw:
        detections.append({
            "class_id": d["class_id"], "class_name": d["class_name"],
            "task_label": label_map.get(d["class_name"], d["class_name"]),
            "conf": d["conf"], "bbox_xyxy": d["bbox_xyxy"],
        })
    return {
        "p_yes": p_yes, "vegetation_soft_fraction": veg_fraction,
        "vegetation_hard_fraction": hard_fraction,
        "predicted_pixel_count": int(hard_veg.sum().item()),
        "yolo_detections": detections,
        "latency_s": round(time.time() - began, 4),
    }


def to_perception(sample, level, output, key=None):
    key = key or ("high_crop" if level == 0 else "low_crop")
    path = sample["images"][key]
    with Image.open(path) as im:
        width, height = im.size
    p = output["p_yes"]
    result = PerceptionResult(
        patch_id=f'{sample["sample_id"]}_{"high" if level == 0 else "low"}',
        patch_path=str(path), patch_url="", overlay_path=None, overlay_url=None,
        detection_path=None, detection_url=None, patch_width=width, patch_height=height,
        patch_radius_m=0.0,
        detection={"detections": [
            {"class_name": d["task_label"], "conf": d["conf"],
             "bbox_xyxy": d["bbox_xyxy"]} for d in output["yolo_detections"]
        ]},
        segmentation={"model": "nvidia/segformer-b2-finetuned-ade-512-512",
                      "vegetation_ids": list(VEGETATION_IDS),
                      "soft_vegetation_fraction": output["vegetation_soft_fraction"],
                      "hard_vegetation_fraction": output["vegetation_hard_fraction"]},
        scene_dict={"yolo_detections": output["yolo_detections"]},
        scene_text="", risk_level="moderate", risk_summary="",
        damaged_buildings=0, intact_buildings=0, vehicles=0, water_pixels=0,
        extras={"task_evidence": {
            "task": "vegetation_presence",
            "class_probs": {"vegetation": p, "no_vegetation": 1.0 - p},
            "predicted_pixel_count": output["predicted_pixel_count"],
            "soft_fraction": output["vegetation_soft_fraction"],
            "hard_fraction": output["vegetation_hard_fraction"],
        }, "level": level, "yolo_detections": output["yolo_detections"]},
    )
    return result


def make_episode(sample, arm, models, first_prediction=None, second_view="low",
                 random_selected=None):
    yolo, seg, torch, F = models
    level = {"current": 0}
    events = []
    cache = {}
    native = RecheckController(RecheckConfig(
        trigger=TRIGGER, uncertainty_mode="entropy", trigger_mode="threshold",
        descend_step_m=max(0.0, float(sample["high_rel_alt"]) - float(sample["low_rel_alt"])),
        alt_min_m=float(sample["low_rel_alt"]), max_rechecks=1,
        motion_mode="descend_only", conf_threshold=0.5,
    ))

    def perceive():
        idx = level["current"]
        key = "high_crop" if idx == 0 or second_view == "high" else "low_crop"
        pred = (first_prediction if idx == 0 and first_prediction is not None
                else infer(sample["images"][key], yolo, seg, torch, F))
        cache[idx] = pred
        events.append({"event": "observation", "level": idx,
                       "altitude_m": sample["high_rel_alt"] if key == "high_crop" else sample["low_rel_alt"],
                       "image": key, "prediction": pred})
        return to_perception(sample, idx, pred, key=key)

    def reobserve(result, spec, ev):
        pred = cache[level["current"]]
        probs = {"vegetation": pred["p_yes"], "no_vegetation": 1-pred["p_yes"]}
        dets = [{"class_name": "vegetation", "conf": max(probs.values()),
                 "class_probs": probs}]
        altitude = float(sample["high_rel_alt"] if level["current"] == 0 or second_view == "high"
                          else sample["low_rel_alt"])
        out = native.assess(
            lat=0.0, lon=0.0, alt=altitude, risk_level="moderate", detections=dets,
            patch_radius_m=0.0, patch_width=result.patch_width, patch_height=result.patch_height,
            track_id=sample["sample_id"], allow_recheck=(arm in {"agent", "repeat_high"}),
        )
        if random_selected is not None:
            # Supplementary budget-matched policy: only the selection rule changes.
            # Selection is frozen outside the episode without labels or low views.
            out = SimpleNamespace(
                kind="recheck" if random_selected else "skip",
                reason="frozen_uniform_random_selection", uncertainty=out.uncertainty,
                label="vegetation", status=None,
                params={"north_m": 0.0, "east_m": 0.0,
                        "up_m": float(sample["low_rel_alt"])-float(sample["high_rel_alt"]),
                        "speed": 10.0} if random_selected else None,
            )
        event = {"kind": out.kind, "reason": out.reason,
                 "uncertainty": out.uncertainty, "label": out.label,
                 "params": out.params, "status": out.status,
                 "altitude_m": altitude}
        events.append({"event": "gate", **event})
        if arm in {"agent", "repeat_high"} and out.kind == "recheck":
            level["current"] = 1
            events.append({"event": "action", "action": "descend_only" if second_view == "low" else "repeat_high",
                           "executed_view": "low_crop" if second_view == "low" else "high_crop",
                           "from_m": sample["high_rel_alt"],
                           "to_m": sample["low_rel_alt"] if second_view == "low" else sample["high_rel_alt"],
                           "requested_params": out.params})
            return {"kind": "recheck", "params": out.params,
                    "uncertainty": out.uncertainty, "reason": out.reason}
        return {"kind": "skip", "uncertainty": out.uncertainty, "reason": out.reason}

    controller = AgentVqaController(
        config=AgentVqaConfig(max_search_steps=0, max_reobservations=1,
                              answer_mode="deterministic", evidence_level="struct"),
        perceive_fn=perceive,
        reobserve_fn=reobserve, get_image_bytes_fn=lambda result: "",
        get_position_fn=lambda: {"lat": 0.0, "lon": 0.0,
                                 "alt": float(sample["high_rel_alt"] if level["current"] == 0 else sample["low_rel_alt"])},
    )
    started = time.time()
    final = controller.run(QUESTION, question_id=sample["sample_id"])
    pred = "yes" if final.answer == "是" else "no"
    trajectory = controller.trajectory_dicts()
    return {
        "sample_id": sample["sample_id"], "arm": arm,
        "answer": pred, "confidence": final.confidence,
        "rechecks": int(level["current"] == 1),
        "stop_reason": trajectory[-1]["reason_code"] if trajectory else final.reason_code,
        "events": events, "trajectory": trajectory,
        "elapsed_s": round(time.time()-started, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["full_descend", "full_train_paths", "full_test_paths"], required=True)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--triggered-repeat-high", action="store_true",
                    help="Run a same-trigger, same-budget repeated-high control for agent-triggered sample IDs.")
    args = ap.parse_args()
    manifest = ROOT / "manifests" / f"{args.split}_samples.jsonl"
    samples = [json.loads(x) for x in manifest.read_text().splitlines()]
    trigger_source = None
    trigger_first_predictions = {}
    if args.triggered_repeat_high:
        trigger_source = ROOT / "runs" / f"single_temporal_{args.split}" / "episodes.jsonl"
        if not trigger_source.exists():
            raise FileNotFoundError(f"Missing completed agent run: {trigger_source}")
        trigger_rows = [json.loads(line) for line in trigger_source.read_text().splitlines() if line.strip()]
        triggered_ids = {r["sample_id"] for r in trigger_rows
                         if r.get("arm") == "agent" and r.get("rechecks") == 1}
        trigger_first_predictions = {
            r["sample_id"]: r["events"][0]["prediction"] for r in trigger_rows
            if r.get("arm") == "hold" and r["sample_id"] in triggered_ids
        }
        if set(trigger_first_predictions) != triggered_ids:
            raise RuntimeError("Triggered sample is missing its paired shared high-view prediction")
        samples = [s for s in samples if s["sample_id"] in triggered_ids]
        if not samples:
            raise RuntimeError("No action-triggered sample IDs found")
    if args.limit:
        samples = samples[:args.limit]
    config = {
        "controller": "repository AgentVqaController + RecheckController",
        "perception": {"yolo_weights": YOLO_WEIGHTS, "yolo_conf": .25, "yolo_imgsz": 640,
                       "segformer": SEGFORMER_ID,
                       "vegetation_ade_ids": list(VEGETATION_IDS),
                       "presence_mapping": "clip((soft_fraction-.005)/(.05-.005),0,1)"},
        "model_provenance": model_provenance(),
        "question": QUESTION, "trigger": TRIGGER,
        "answer_module": "AgentVqaController deterministic vegetation-presence adapter",
        "answer_mapping_threshold_p_yes": 0.5,
        "gate": "RecheckController normalized binary entropy, threshold mode",
        "action": ("one gated repeat of the same high ROI as a no-change control"
                   if args.triggered_repeat_high else
                   "one fixed registered real frame descend; no horizontal recenter"),
        "max_reobservations": 1, "max_search": 0,
        "input": "unmarked ROI crops only; second observation accessed after gate action",
        "split": args.split, "manifest_sha256": sha(manifest),
        "arm_mode": "triggered_repeat_high" if args.triggered_repeat_high else "paired_hold_agent",
    }
    if trigger_source is not None:
        config["trigger_source_sha256"] = sha(trigger_source)
        config["triggered_sample_ids"] = [s["sample_id"] for s in samples]
        config["control"] = "same triggered IDs, same controller and one-recheck budget, repeat high ROI instead of opening low ROI"
    out_name = (f"single_temporal_trigger_repeat_high_{args.split}" if args.triggered_repeat_high
                else f"single_temporal_{args.split}")
    outdir = ROOT / "runs" / out_name
    outdir.mkdir(parents=True, exist_ok=True)
    config_path, log_path = outdir / "config.json", outdir / "episodes.jsonl"
    done = {}
    if log_path.exists():
        if not args.resume:
            raise FileExistsError(log_path)
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("Cannot resume with changed config")
        for line in log_path.read_text().splitlines():
            row = json.loads(line); key = (row["sample_id"], row["arm"])
            if key in done: raise RuntimeError(f"duplicate episode {key}")
            done[key] = row
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
    models = load_models(args.device)
    try:
        with log_path.open("a") as f:
            for i, sample in enumerate(samples, 1):
                arms = ("repeat_high",) if args.triggered_repeat_high else ("hold", "agent")
                for arm in arms:
                    key = (sample["sample_id"], arm)
                    if key in done: continue
                    first_prediction = None
                    if arm == "agent":
                        first_prediction = done[(sample["sample_id"], "hold")]["events"][0]["prediction"]
                    elif arm == "repeat_high":
                        first_prediction = trigger_first_predictions[sample["sample_id"]]
                    row = make_episode(sample, arm, models, first_prediction=first_prediction,
                                       second_view="high" if arm == "repeat_high" else "low")
                    f.write(json.dumps(row, ensure_ascii=False)+"\n")
                    f.flush(); os.fsync(f.fileno())
                    done[key] = row
                if args.triggered_repeat_high:
                    print(i, "/", len(samples), sample["sample_id"],
                          "repeat_high", done[(sample["sample_id"], "repeat_high")]["answer"], flush=True)
                else:
                    print(i, "/", len(samples), sample["sample_id"],
                          "hold", done[(sample["sample_id"], "hold")]["answer"],
                          "agent", done[(sample["sample_id"], "agent")]["answer"],
                          "recheck", done[(sample["sample_id"], "agent")]["rechecks"], flush=True)
    finally:
        del models


if __name__ == "__main__":
    main()
