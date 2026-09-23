"""Inspect MESSI high/low frame pairs before any model calls.

The output is a candidate inventory, not a frozen evaluation set. It deliberately
keeps the annotation-derived counts separate from model inputs.
"""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SIZE = (5472, 3648)
CLASS_IDS = {'vehicle': 13, 'vegetation': 12}  # Author's MessiDataset.CLASSES.


def load(zpath: Path, seq: str, frame: str, suffix: str, flags: int):
    with zipfile.ZipFile(zpath) as z:
        data = z.read(f"{seq}/{frame}.{suffix}")
    out = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if out is None or out.shape[:2] != (SIZE[1], SIZE[0]):
        raise ValueError(f"Invalid image: {zpath}:{frame}")
    return out


def load_mask(zpath: Path, seq: str, frame: str):
    with zipfile.ZipFile(zpath) as z:
        data = z.read(f"{seq}/{frame}.png")
    im = Image.open(io.BytesIO(data))
    if im.mode != 'P' or im.size != SIZE:
        raise ValueError(f'Expected indexed {SIZE} label image: {seq}/{frame}')
    return np.asarray(im)


def choose_frame(rows, height):
    return min(rows, key=lambda r: abs(float(r['rel_alt']) - height))


def register(low, high):
    scale = 0.25
    a = cv2.resize(low, None, fx=scale, fy=scale)
    b = cv2.resize(high, None, fx=scale, fy=scale)
    sift = cv2.SIFT_create(nfeatures=6000)
    k1, d1 = sift.detectAndCompute(a, None)
    k2, d2 = sift.detectAndCompute(b, None)
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [x for x, y in matches if x.distance < 0.72 * y.distance]
    p1 = np.float32([k1[x.queryIdx].pt for x in good])
    p2 = np.float32([k2[x.trainIdx].pt for x in good])
    h, inliers = cv2.findHomography(p1, p2, cv2.RANSAC, 4)
    if h is None or int(inliers.sum()) < 40:
        raise ValueError(f"Registration failed: {len(good)} matches")
    # Coordinates returned in the original 5472 x 3648 image system.
    h[0, 2] /= scale
    h[1, 2] /= scale
    h[2, :2] *= scale
    return h, len(good), int(inliers.sum())


def map_box(h, box):
    x0, y0, x1, y1 = box
    corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    mapped = cv2.perspectiveTransform(corners[None], h)[0]
    x0h, y0h = np.floor(mapped.min(axis=0)).astype(int)
    x1h, y1h = np.ceil(mapped.max(axis=0)).astype(int)
    return [int(x0h), int(y0h), int(x1h), int(y1h)], mapped.round(1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sequence')
    ap.add_argument('--class-name', choices=CLASS_IDS, default='vehicle')
    args = ap.parse_args()
    seq = args.sequence
    label_id = CLASS_IDS[args.class_name]
    raw = ROOT / 'raw' / seq
    rows = list(csv.DictReader((raw / '6DOF.csv').open()))
    high_r, low_r = choose_frame(rows, 100), choose_frame(rows, 30)
    high_f, low_f = high_r['ImageName'], low_r['ImageName']
    high = load(raw / 'images.zip', seq, high_f, 'JPG', cv2.IMREAD_COLOR)
    low = load(raw / 'images.zip', seq, low_f, 'JPG', cv2.IMREAD_COLOR)
    high_mask = load_mask(raw / 'annotations.zip', seq, high_f)
    low_mask = load_mask(raw / 'annotations.zip', seq, low_f)
    h, matches, inliers = register(low, high)
    records = []
    tiles = []
    for y0 in [160, 1030, 1900, 2770]:
        for x0 in [160, 1070, 1980, 2890, 3800, 4710]:
            low_box = [x0, y0, x0 + 760, y0 + 760]
            high_box, poly = map_box(h, low_box)
            a, b, c, d = high_box
            if min(a, b) < 0 or c > SIZE[0] or d > SIZE[1]:
                continue
            low_count = int(np.count_nonzero(low_mask[y0:y0+760, x0:x0+760] == label_id))
            high_count = int(np.count_nonzero(high_mask[b:d, a:c] == label_id))
            low_fraction = low_count / (760 * 760)
            high_fraction = high_count / ((d - b) * (c - a))
            status = ('candidate_positive' if min(low_fraction, high_fraction) >= .05 else
                      'candidate_negative' if max(low_fraction, high_fraction) <= .005 else 'review')
            rid = f'{seq}_r{len(records)+1:02d}'
            records.append(dict(id=rid, sequence=seq, high_frame=high_f, low_frame=low_f,
                                high_rel_alt=float(high_r['rel_alt']), low_rel_alt=float(low_r['rel_alt']),
                                low_box=low_box, high_box=high_box, high_polygon=poly,
                                class_name=args.class_name, label_id=label_id,
                                low_label_pixels=low_count, high_label_pixels=high_count,
                                low_label_fraction=round(low_fraction, 6),
                                high_label_fraction=round(high_fraction, 6),
                                status=status, matches=matches, inliers=inliers))
            if status != 'review':
                hcrop = cv2.cvtColor(high[b:d, a:c], cv2.COLOR_BGR2RGB)
                lcrop = cv2.cvtColor(low[y0:y0+760, x0:x0+760], cv2.COLOR_BGR2RGB)
                pair = Image.new('RGB', (440, 260), 'white')
                for j, arr in enumerate([hcrop, lcrop]):
                    im = Image.fromarray(arr)
                    im.thumbnail((210, 210))
                    pair.paste(im, (j * 220, 30))
                ImageDraw.Draw(pair).text((5, 5), f'{rid} {status} H{high_count} L{low_count}', fill='black')
                tiles.append(pair)
    out = ROOT / 'derived' / seq / args.class_name
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'candidates.jsonl').open('w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    cols = 3
    sheet = Image.new('RGB', (cols*440, ((len(tiles)+cols-1)//cols)*260), 'white')
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i%cols)*440, (i//cols)*260))
    sheet.save(out / 'candidates.jpg', quality=90)
    print(json.dumps(dict(sequence=seq, class_name=args.class_name, high_frame=high_f, low_frame=low_f,
                          high_rel_alt=high_r['rel_alt'], low_rel_alt=low_r['rel_alt'],
                          matches=matches, inliers=inliers, candidates=len(records),
                          positive=sum(x['status']=='candidate_positive' for x in records),
                          negative=sum(x['status']=='candidate_negative' for x in records))))


if __name__ == '__main__':
    main()
