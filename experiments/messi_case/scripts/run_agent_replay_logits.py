"""Exploratory MESSI replay with direct Qwen YES/NO token entropy.

The threshold is 0.90, chosen on four pilot items before this variant's
evaluation inference. This is a separate adapted policy from the frozen
self-reported-confidence replay.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "backend"))
from local_qwen_vl import LocalQwenVLBackend, _normalize_messages  # noqa: E402
from run_agent_replay import (MODEL, REVISION, VEGETATION_QUESTION,
                              OfflineEnvironment, run_arm, sha)  # noqa: E402

FIRST = ("Inspect only the ground region inside the red rectangle. "
         "Does it contain visible vegetation, including grass, trees, or shrubs? "
         "Answer with exactly one token: YES or NO.")
SECOND = ("Reassess the same ground region using this newly acquired lower-altitude "
          "observation; the entire image is the region. "
          "Answer with exactly one token: YES or NO.")
TRIGGER = 0.90


def message(prompt, image):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


class LogitEnvironment(OfflineEnvironment):
    def __init__(self, sample, arm, model, first_cache, first_prompt):
        super().__init__(sample, arm, model, first_cache, first_prompt,
                         uncertainty_trigger=TRIGGER,
                         uncertainty_source="qwen_yes_no_token_entropy")
        tokenizer = model.processor.tokenizer
        self.yes_id = tokenizer.encode("YES", add_special_tokens=False)
        self.no_id = tokenizer.encode("NO", add_special_tokens=False)
        if len(self.yes_id) != 1 or len(self.no_id) != 1:
            raise RuntimeError("YES/NO must each tokenize to one token")
        self.first_answer = None

    def _score(self, messages):
        normalized, images = _normalize_messages(messages)
        prompt = self.model.processor.apply_chat_template(
            normalized, tokenize=False, add_generation_prompt=True)
        inputs = self.model.processor(text=[prompt], images=images, padding=True,
                                      return_tensors="pt").to(self.model.device)
        torch = self.model._torch
        with torch.no_grad():
            logits = self.model.model(**inputs).logits[0,-1].float()
            pair = logits[[self.yes_id[0], self.no_id[0]]]
            p_yes = float(torch.softmax(pair, dim=0)[0].item())
        return p_yes

    def vlm_answer(self, image_bytes, result, spec, qid, evidence, context):
        began = time.time()
        if self.level == 0 and self.first_cache is not None:
            raw = self.first_cache["raw"]
            p_yes = float(json.loads(raw)["p_yes"])
            cache_hit = True
        else:
            if self.level == 0:
                messages = [message(FIRST, image_bytes)]
            else:
                messages = [message(FIRST, self.sample["images"]["high_marked"]),
                            {"role": "assistant", "content": self.first_answer},
                            message(SECOND, image_bytes)]
            p_yes = self._score(messages)
            raw = json.dumps({"p_yes": p_yes, "answer": "YES" if p_yes >= .5 else "NO"})
            cache_hit = False
        answer = "YES" if p_yes >= .5 else "NO"
        conf = max(p_yes, 1-p_yes)
        self.last_conf = conf
        if self.level == 0:
            self.first_raw = raw
            self.first_answer = answer
        self.model_log.append({"level": self.level, "raw": raw,
                               "answer": answer, "confidence": conf,
                               "p_yes": p_yes, "latency_s": round(time.time()-began,3),
                               "cache_hit": cache_hit})
        return json.dumps({"answer": "是" if answer == "YES" else "否",
                           "confidence": conf, "abstain": False,
                           "decision": "answer", "reason_code": "sufficient_evidence",
                           "evidence": {"source": "image", "norm_xy": None}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["pilot","exhaustive_descend","exhaustive_paths",
                                        "full_descend","full_train_paths","full_test_paths"], required=True)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    manifest = ROOT / "manifests" / f"{args.split}_samples.jsonl"
    samples = [json.loads(line) for line in manifest.read_text().splitlines()]
    if args.limit:
        samples = samples[:args.limit]
    config = {"model": MODEL, "revision": REVISION,
              "question": VEGETATION_QUESTION, "first_prompt": FIRST,
              "second_prompt": SECOND, "answer": "normalized YES/NO next-token logits",
              "temperature": 0, "threshold": TRIGGER,
              "gate": "task_conditioned_recheck_decision damage-target branch",
              "cost_weight": .05, "cost_scale_s": 60, "min_utility": .05,
              "max_rechecks": 1, "max_search": 0,
              "threshold_provenance": "chosen on four 100_0001 pilot items",
              "manifest_sha256": sha(manifest), "split": args.split}
    out = ROOT / "runs" / f"agent_replay_logits_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    config_path, log_path = out / "config.json", out / "episodes.jsonl"
    completed = {}
    if log_path.exists():
        if not args.resume:
            raise FileExistsError(log_path)
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("Cannot resume with changed config")
        for line in log_path.read_text().splitlines():
            episode = json.loads(line)
            key = (episode["sample_id"], episode["arm"])
            if key in completed:
                raise RuntimeError(f"Duplicate episode: {key}")
            completed[key] = episode
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
    model = LocalQwenVLBackend(MODEL, device=args.device,
                               top_p=1.0, repetition_penalty=1.0)
    model.load()
    try:
        with log_path.open("a") as log:
            for sample in samples:
                sid = sample["sample_id"]
                if (sid,"agent") in completed:
                    if (sid,"hold") not in completed:
                        raise RuntimeError(f"Agent episode without hold: {sid}")
                    continue
                hold = completed.get((sid,"hold"))
                if hold is None:
                    hold = run_arm(sample, "hold", model, None, VEGETATION_QUESTION,
                                   FIRST, env_class=LogitEnvironment)
                    hold.pop("first_cache")
                    log.write(json.dumps(hold,ensure_ascii=False)+"\n")
                    log.flush(); os.fsync(log.fileno())
                cache = {"raw": hold["model_calls"][0]["raw"]}
                agent = run_arm(sample, "agent", model, cache, VEGETATION_QUESTION,
                                FIRST, env_class=LogitEnvironment)
                agent.pop("first_cache")
                assert hold["model_calls"][0]["raw"] == agent["model_calls"][0]["raw"]
                log.write(json.dumps(agent,ensure_ascii=False)+"\n")
                log.flush(); os.fsync(log.fileno())
                print(sid, "hold",hold["answer"],"agent",agent["answer"],
                      "rechecks",agent["rechecks"],flush=True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
