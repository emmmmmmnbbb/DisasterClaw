"""Exhaustively pair public 30-m path frames to nearest 100-m frames.

One predeclared central 760x760-pixel ROI is evaluated per low frame.
Ground truth is derived from vegetation masks at both elevations.
"""
import csv
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from inspect_pairs import SIZE, map_box, register

ROOT = Path(__file__).resolve().parents[1]
LOW_BOX = [2356, 1444, 3116, 2204]
LABEL_ID = 12


def distance_m(a, b):
    lat = (float(a["lat"]) + float(b["lat"])) / 2
    return math.hypot((float(a["lat"])-float(b["lat"]))*111_000,
                      (float(a["long"])-float(b["long"]))*111_000*math.cos(math.radians(lat)))


def decode(z, altitude, name, mask=False):
    suffix = "png" if mask else "JPG"
    raw = z.read(f"{altitude}/{name}.{suffix}")
    if mask:
        image = Image.open(io.BytesIO(raw))
        if image.mode != "P" or image.size != SIZE:
            raise ValueError(f"Bad mask: {altitude}/{name}")
        return np.asarray(image)
    arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None or arr.shape[:2] != (SIZE[1], SIZE[0]):
        raise ValueError(f"Bad RGB: {altitude}/{name}")
    return arr


def main():
    all_rows, excluded = [], []
    for group in "ABC":
        base = ROOT / "raw" / "public_paths" / group
        high_meta = list(csv.DictReader((base / "100/6DOF.csv").open()))
        low_meta = list(csv.DictReader((base / "30/6DOF.csv").open()))
        with (zipfile.ZipFile(base / "100/images.zip") as high_zip,
              zipfile.ZipFile(base / "30/images.zip") as low_zip,
              zipfile.ZipFile(base / "100/annotations.zip") as high_ann,
              zipfile.ZipFile(base / "30/annotations.zip") as low_ann):
            high_cache = {}
            for row in low_meta:
                nearest = min(high_meta, key=lambda h: distance_m(h, row))
                distance = distance_m(nearest, row)
                low_name, high_name = row["ImageName"], nearest["ImageName"]
                sid = f"path_{group}_{low_name}"
                if distance > 12:
                    excluded.append({"sample_id": sid, "reason": "gps_distance_over_12m", "distance_m": round(distance,2)})
                    continue
                if high_name not in high_cache:
                    high_cache[high_name] = (decode(high_zip, "100", high_name),
                                              decode(high_ann, "100", high_name, True))
                high, high_mask = high_cache[high_name]
                low = decode(low_zip, "30", low_name)
                low_mask = decode(low_ann, "30", low_name, True)
                try:
                    homography, matches, inliers = register(low, high)
                    high_box, poly = map_box(homography, LOW_BOX)
                except (ValueError, cv2.error) as exc:
                    excluded.append({"sample_id": sid, "reason": "registration_failed", "detail": str(exc)})
                    continue
                a,b,c,d = high_box
                if min(a,b) < 0 or c > SIZE[0] or d > SIZE[1] or c-a < 20 or d-b < 20:
                    excluded.append({"sample_id": sid, "reason": "mapped_roi_out_of_bounds"})
                    continue
                low_frac = float(np.mean(low_mask[LOW_BOX[1]:LOW_BOX[3], LOW_BOX[0]:LOW_BOX[2]] == LABEL_ID))
                high_frac = float(np.mean(high_mask[b:d, a:c] == LABEL_ID))
                status = ("candidate_positive" if min(low_frac, high_frac) >= .05 else
                          "candidate_negative" if max(low_frac, high_frac) <= .005 else "review")
                item = {"id": sid, "sequence": f"path_{group}", "group": group,
                        "high_frame": high_name, "low_frame": low_name,
                        "high_rel_alt": float(nearest["rel_alt"]),
                        "low_rel_alt": float(row["rel_alt"]),
                        "gps_distance_m": round(distance,3), "low_box": LOW_BOX,
                        "high_box": high_box, "high_polygon": poly,
                        "registration_matches": matches, "registration_inliers": inliers,
                        "high_label_fraction": round(high_frac,6),
                        "low_label_fraction": round(low_frac,6), "status": status}
                all_rows.append(item)
                if status == "review":
                    excluded.append({"sample_id": sid, "reason": "annotation_fractions_ambiguous_or_discordant",
                                     "high_label_fraction": round(high_frac,6),
                                     "low_label_fraction": round(low_frac,6)})
        print(group, len(low_meta), dict(Counter(r["status"] for r in all_rows if r["group"] == group)), flush=True)
    out = ROOT / "derived" / "public_paths"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.jsonl").write_text("".join(json.dumps(x)+"\n" for x in all_rows))
    (out / "exclusions.jsonl").write_text("".join(json.dumps(x)+"\n" for x in excluded))
    print("total", len(all_rows), "excluded", len(excluded))


if __name__ == "__main__":
    main()
