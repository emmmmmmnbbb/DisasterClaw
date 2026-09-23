"""Generate fixed-view and action-gated inputs for public path frame pairs."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "exhaustive_paths"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    selection = json.loads((ROOT / "manifests" / f"{SPLIT}_selection.json").read_text())
    candidates = {x["id"]: x for x in map(json.loads,
                  (ROOT / "derived/public_paths/candidates.jsonl").read_text().splitlines())}
    output = ROOT / "derived" / "exhaustive_path_inputs"
    output.mkdir(parents=True, exist_ok=True)
    samples, truths = [], []
    current_group = None
    high_zip = low_zip = None
    high_frames = {}
    source_hashes = None
    try:
        for chosen in selection:
            item = candidates[chosen["candidate_id"]]
            group = item["group"]
            if group != current_group:
                if high_zip:
                    high_zip.close(); low_zip.close()
                current_group = group
                base = ROOT / "raw/public_paths" / group
                high_zip = zipfile.ZipFile(base / "100/images.zip")
                low_zip = zipfile.ZipFile(base / "30/images.zip")
                source_hashes = {"high_images": sha(base / "100/images.zip"),
                                 "low_images": sha(base / "30/images.zip"),
                                 "high_annotations": sha(base / "100/annotations.zip"),
                                 "low_annotations": sha(base / "30/annotations.zip")}
                high_frames = {}
            high_name, low_name = item["high_frame"], item["low_frame"]
            if high_name not in high_frames:
                high_frames[high_name] = Image.open(io.BytesIO(high_zip.read(f"100/{high_name}.JPG"))).convert("RGB")
            high = high_frames[high_name]
            low = Image.open(io.BytesIO(low_zip.read(f"30/{low_name}.JPG"))).convert("RGB")
            box_h, box_l = item["high_box"], item["low_box"]
            marked = high.copy()
            draw = ImageDraw.Draw(marked)
            for offset in range(-8, 9):
                draw.rectangle([box_h[0]+offset,box_h[1]+offset,box_h[2]-offset,box_h[3]-offset],
                               outline=(255,0,0),width=1)
            marked.thumbnail((1200,800), Image.Resampling.LANCZOS)
            images = {"high_marked": marked,
                      "high_crop": high.crop(tuple(box_h)).resize((800,800), Image.Resampling.LANCZOS),
                      "low_crop": low.crop(tuple(box_l)).resize((800,800), Image.Resampling.LANCZOS)}
            paths = {}
            for name, image in images.items():
                path = output / f"{item['id']}_{name}.png"
                image.save(path, compress_level=1)
                paths[name] = str(path)
            samples.append({"sample_id": item["id"], "sequence_id": item["sequence"],
                            "scene_group_id": chosen["scene_group_id"], "split": SPLIT,
                            "high_frame_id": high_name, "low_frame_id": low_name,
                            "high_rel_alt": item["high_rel_alt"], "low_rel_alt": item["low_rel_alt"],
                            "gps_distance_m": item["gps_distance_m"],
                            "high_box": box_h, "low_box": box_l,
                            "high_polygon": item["high_polygon"],
                            "registration_matches": item["registration_matches"],
                            "registration_inliers": item["registration_inliers"],
                            "question": chosen["question"], "images": paths,
                            "image_sha256": {key: sha(value) for key,value in paths.items()},
                            "source_sha256": source_hashes})
            truths.append({"sample_id": item["id"],
                           "answer": "yes" if chosen["answer"] == "positive" else "no",
                           "evidence": chosen["evidence"],
                           "high_label_fraction": item["high_label_fraction"],
                           "low_label_fraction": item["low_label_fraction"]})
            if len(samples) % 20 == 0:
                print("generated",len(samples),flush=True)
    finally:
        if high_zip:
            high_zip.close(); low_zip.close()
    sample_path = ROOT / "manifests" / f"{SPLIT}_samples.jsonl"
    truth_path = ROOT / "manifests" / f"{SPLIT}_ground_truth.jsonl"
    sample_path.write_text("".join(json.dumps(x)+"\n" for x in samples))
    truth_path.write_text("".join(json.dumps(x)+"\n" for x in truths))
    print(len(samples),sha(sample_path),sha(truth_path))


if __name__ == "__main__":
    main()
