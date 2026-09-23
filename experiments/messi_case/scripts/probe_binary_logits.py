"""Pilot-only check of direct Qwen YES/NO token probabilities on high images."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "backend"))
from local_qwen_vl import LocalQwenVLBackend, _normalize_messages  # noqa: E402

PROMPT = ("Inspect only the ground region inside the red rectangle. "
          "Does it contain visible vegetation, including grass, trees, or shrubs? "
          "Answer with exactly one token: YES or NO.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    samples = [json.loads(x) for x in (ROOT / "manifests/pilot_samples.jsonl").read_text().splitlines()]
    model = LocalQwenVLBackend("Qwen/Qwen2.5-VL-7B-Instruct", device=args.device,
                               top_p=1.0, repetition_penalty=1.0)
    model.load()
    try:
        tokenizer = model.processor.tokenizer
        options = {word: tokenizer.encode(word, add_special_tokens=False) for word in ("YES", "NO", " YES", " NO")}
        print("tokens", options, flush=True)
        for sample in samples:
            msg = [{"role": "user", "content": [
                {"type": "image", "image": sample["images"]["high_marked"]},
                {"type": "text", "text": PROMPT}]}]
            normalized, images = _normalize_messages(msg)
            prompt_text = model.processor.apply_chat_template(normalized, tokenize=False,
                                                               add_generation_prompt=True)
            inputs = model.processor(text=[prompt_text], images=images, padding=True,
                                     return_tensors="pt").to(model.device)
            with model._torch.no_grad():
                logits = model.model(**inputs).logits[0,-1].float()
            top_ids = model._torch.topk(logits, 5).indices.tolist()
            top_tokens = [tokenizer.decode([int(x)]) for x in top_ids]
            yes_ids, no_ids = options["YES"], options["NO"]
            if len(yes_ids) != 1 or len(no_ids) != 1:
                raise RuntimeError("YES/NO are not single tokens")
            p_yes = model._torch.softmax(logits[[yes_ids[0],no_ids[0]]],dim=0)[0].item()
            p_no = 1-p_yes
            conf = max(p_yes,p_no)
            entropy = -(p_yes*math.log(max(p_yes,1e-30))+p_no*math.log(max(p_no,1e-30)))/math.log(2)
            print(sample["sample_id"],"yes" if p_yes>=.5 else "no",
                  "p_yes",round(p_yes,4),"entropy",round(entropy,4),
                  "top_tokens",top_tokens,flush=True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
