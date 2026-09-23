"""Offline, action-gated MESSI replay through the real AgentVqaController.

The MESSI adapter changes the perception and binary-target uncertainty source;
it deliberately does not claim to run the ChangeOS damage assessor or a UAV.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "backend"))
from agent_vqa import AgentVqaConfig, AgentVqaController  # noqa: E402
from local_qwen_vl import LocalQwenVLBackend  # noqa: E402
from recheck import task_conditioned_recheck_decision  # noqa: E402

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
QUESTION = "图中标记地面区域是否存在可见的树木或灌木冠层？"
VEGETATION_QUESTION = "图中标记地面区域是否存在可见植被（包括草、树木或灌木）？"
FIRST_PROMPT = ("Inspect only the ground region inside the red rectangle. "
                "Does it contain visible tree or shrub canopy? "
                "Reply with one JSON object only, using keys answer and confidence. "
                "answer must be YES or NO. confidence is your probability (0.50 to 1.00) "
                "that this answer is correct from this image. Use 0.50 when uncertain. "
                "Example format: {\"answer\":\"YES\",\"confidence\":0.75}.")
VEGETATION_FIRST_PROMPT = FIRST_PROMPT.replace(
    "visible tree or shrub canopy", "visible vegetation, including grass, trees, or shrubs")
SECOND_PROMPT = ("Reassess the same ground region using this newly acquired lower-altitude "
                 "observation; the entire image is the region. Reply with one JSON object "
                 "using the same answer and confidence keys and scale.")
MAX_NEW_TOKENS = 64


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_model(raw):
    try:
        value = json.loads(raw.strip())
        answer = value["answer"].strip().upper()
        conf = float(value["confidence"])
        if answer not in ("YES", "NO") or not math.isfinite(conf) or not 0.5 <= conf <= 1:
            raise ValueError("answer/confidence out of range")
        return answer, conf
    except (ValueError, KeyError, TypeError, AttributeError):
        return None, None


def entropy(conf):
    if conf is None:
        return None
    if conf in (0.0, 1.0):
        return 0.0
    return -(conf * math.log(conf) + (1 - conf) * math.log(1 - conf)) / math.log(2)


def message(prompt, image):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


class OfflineEnvironment:
    def __init__(self, sample, arm, model, first_cache, first_prompt,
                 uncertainty_trigger=0.5,
                 uncertainty_source="qwen_self_reported_binary_entropy"):
        self.sample, self.arm, self.model = sample, arm, model
        self.first_cache = first_cache
        self.first_prompt = first_prompt
        self.uncertainty_trigger = uncertainty_trigger
        self.uncertainty_source = uncertainty_source
        self.level = 0
        self.read_log = []
        self.model_log = []
        self.last_conf = None
        self.first_raw = None

    def position(self):
        return {"lat": None, "lon": None,
                "alt": self.sample["high_rel_alt"] if self.level == 0 else self.sample["low_rel_alt"]}

    def perceive(self):
        key = "high_marked" if self.level == 0 else "low_crop"
        path = self.sample["images"][key]
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != self.sample["image_sha256"][key]:
            raise RuntimeError("image hash mismatch: " + key)
        self.read_log.append({"image_key": key, "sha256": digest, "level": self.level})
        return SimpleNamespace(patch_id=self.sample["sample_id"] + ":" + key,
                               patch_bytes=data, detection={"detections": []},
                               extras={}, patch_width=1200 if self.level == 0 else 800,
                               patch_height=800)

    def vlm_answer(self, image_bytes, result, spec, qid, evidence, context):
        if self.level == 0 and self.first_cache is not None:
            raw, latency = self.first_cache["raw"], 0.0
            cache_hit = True
        else:
            prompt = self.first_prompt if self.level == 0 else SECOND_PROMPT
            if self.level == 0:
                messages = [message(prompt, self.sample["images"]["high_marked"])]
            else:
                messages = [message(self.first_prompt, self.sample["images"]["high_marked"]),
                            {"role": "assistant", "content": self.first_raw},
                            message(prompt, self.sample["images"]["low_crop"])]
            begin = time.time()
            raw = self.model.infer(messages, max_new_tokens=MAX_NEW_TOKENS,
                                   temperature=0.0, seed=0)
            latency = time.time() - begin
            cache_hit = False
        answer, conf = parse_model(raw)
        self.last_conf = conf
        if self.level == 0:
            self.first_raw = raw
        self.model_log.append({"level": self.level, "raw": raw, "answer": answer,
                               "confidence": conf, "latency_s": round(latency, 3),
                               "cache_hit": cache_hit})
        if answer is None:
            return raw  # controller records invalid_output and stops
        return json.dumps({"answer": "是" if answer == "YES" else "否",
                           "confidence": conf, "abstain": False,
                           "decision": "answer", "reason_code": "sufficient_evidence",
                           "evidence": {"source": "image", "norm_xy": None}})

    def reobserve(self, result, spec, evidence):
        unc = entropy(self.last_conf)
        if unc is None:
            return {"kind": "skip", "reason": "invalid model confidence"}
        high = self.sample["high_rel_alt"]
        low = self.sample["low_rel_alt"]
        gate = task_conditioned_recheck_decision(
            question_type="damage",  # explicit MESSI binary-target adapter
            uncertainty=unc, alt=high, descend_step_m=high-low,
            alt_min_m=low, roi_norm_bbox=None, target_visible=True,
            target_matched=True, recenter_horizontal_m=0.0,
            uncertainty_trigger=self.uncertainty_trigger, cost_weight=0.05,
            coverage_weight=1.0, cost_scale_s=60.0, min_utility=0.05)
        metrics = gate.to_dict()
        metrics.update({"uncertainty_source": self.uncertainty_source,
                        "actual_question_type": spec.question_type,
                        "policy_question_type_adapter": "damage_binary_target",
                        "altitude_kind": "relative_to_takeoff_m",
                        "candidate_low_frame": self.sample["low_frame_id"]})
        if not gate.allow:
            return {"kind": "skip", "reason": gate.reason,
                    "uncertainty": unc, "motion_mode": gate.motion_mode,
                    "policy_metrics": metrics}
        if self.level != 0:
            raise RuntimeError("no further observation available")
        # Only the action changes the state. perceive() opens the low image afterward.
        self.level = 1
        params = {"action": "offline_descend_to_marked_region", "from_rel_alt_m": high,
                  "to_rel_alt_m": low, "frame_id": self.sample["low_frame_id"]}
        return {"kind": "recheck", "params": params, "executed": params,
                "reason": gate.reason, "uncertainty": unc,
                "motion_mode": gate.motion_mode, "policy_metrics": metrics}


def run_arm(sample, arm, model, first_cache, question, first_prompt,
            env_class=OfflineEnvironment):
    env = env_class(sample, arm, model, first_cache, first_prompt)
    config = AgentVqaConfig(max_search_steps=0,
                            max_reobservations=0 if arm == "hold" else 1,
                            answer_mode="vlm", evidence_level="raw",
                            generation_base_seed=0)
    controller = AgentVqaController(
        config=config, vlm_answer_fn=env.vlm_answer, perceive_fn=env.perceive,
        reobserve_fn=env.reobserve, get_position_fn=env.position,
        get_image_bytes_fn=lambda result: result.patch_bytes)
    final = controller.run(question, question_id=sample["sample_id"])
    reads = [x["image_key"] for x in env.read_log]
    assert reads[0] == "high_marked"
    assert ("low_crop" in reads) == (arm == "agent" and env.level == 1)
    assert len(reads) == 1 + int(env.level == 1)
    return {"sample_id": sample["sample_id"], "arm": arm,
            "answer": {"是": "yes", "否": "no"}.get(final.answer, "invalid"),
            "final": final.to_dict(), "rechecks": env.level,
            "stop_reason": final.reason_code, "trajectory": controller.trajectory_dicts(),
            "image_reads": env.read_log, "model_calls": env.model_log,
            "first_cache": {"raw": env.first_raw}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["pilot", "formal", "exhaustive_descend", "exhaustive_paths"], required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    manifest = ROOT / "manifests" / f"{args.split}_samples.jsonl"
    samples = [json.loads(line) for line in manifest.read_text().splitlines()]
    if args.limit:
        samples = samples[:args.limit]
    question = VEGETATION_QUESTION if args.split in {"exhaustive_descend", "exhaustive_paths"} else QUESTION
    first_prompt = VEGETATION_FIRST_PROMPT if args.split in {"exhaustive_descend", "exhaustive_paths"} else FIRST_PROMPT
    config = {"model": MODEL, "revision": REVISION, "question": question,
              "first_prompt": first_prompt, "second_prompt": SECOND_PROMPT,
              "max_new_tokens": MAX_NEW_TOKENS, "temperature": 0,
              "uncertainty": "normalized binary entropy of self-reported answer confidence",
              "gate": "task_conditioned_recheck_decision; damage branch for MESSI canopy target",
              "trigger": 0.5, "cost_weight": 0.05, "cost_scale_s": 60,
              "min_utility": 0.05, "max_rechecks": 1, "max_search": 0,
              "manifest_sha256": sha(manifest), "split": args.split}
    out = ROOT / "runs" / f"agent_replay_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    log_path = out / "episodes.jsonl"
    completed = {}
    if log_path.exists():
        if not args.resume:
            raise FileExistsError(log_path)
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("Cannot resume with a changed config")
        for line in log_path.read_text().splitlines():
            episode = json.loads(line)
            key = (episode["sample_id"], episode["arm"])
            if key in completed:
                raise RuntimeError(f"Duplicate episode: {key}")
            completed[key] = episode
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    model = LocalQwenVLBackend(MODEL, device=args.device,
                               top_p=1.0, repetition_penalty=1.0)
    model.load()
    try:
        with log_path.open("a") as log:
            for sample in samples:
                sid = sample["sample_id"]
                if (sid, "agent") in completed:
                    if (sid, "hold") not in completed:
                        raise RuntimeError(f"Agent episode without hold: {sid}")
                    continue
                hold = completed.get((sid, "hold"))
                if hold is None:
                    hold = run_arm(sample, "hold", model, None, question, first_prompt)
                    hold.pop("first_cache")
                    log.write(json.dumps(hold, ensure_ascii=False) + "\n")
                    log.flush(); os.fsync(log.fileno())
                cache = {"raw": hold["model_calls"][0]["raw"]}
                agent = run_arm(sample, "agent", model, cache, question, first_prompt)
                assert hold["model_calls"][0]["raw"] == agent["model_calls"][0]["raw"]
                for episode in (agent,):
                    episode.pop("first_cache")
                    log.write(json.dumps(episode, ensure_ascii=False) + "\n")
                    log.flush(); os.fsync(log.fileno())
                print(sample["sample_id"], "hold", hold["answer"],
                      "agent", agent["answer"], "rechecks", agent["rechecks"], flush=True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
