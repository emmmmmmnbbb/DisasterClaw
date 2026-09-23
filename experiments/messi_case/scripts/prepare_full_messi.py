"""Inventory and freeze all eligible 100-to-30-m MESSI vegetation pairs.

Reads the author's unpacked full release without copying or modifying it.
The previously used 100_0001 sequence remains pilot-only. Training and test
paths are scored in separate strata. No model or answer is consulted here.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from inspect_pairs import SIZE, choose_frame, map_box, register

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/lc/datasets/MESSI')
BOX = [2356, 1444, 3116, 2204]
GRID = [[x, y, x + 760, y + 760]
        for y in [160, 1030, 1900, 2770]
        for x in [160, 1070, 1980, 2890, 3800, 4710]]
QUESTION = 'Does the indicated ground region contain visible vegetation, including grass, trees, or shrubs?'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def jsonlines(path, rows):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))


def rows(path):
    return list(csv.DictReader(path.open()))


def image_path(split, scene, kind, sequence, level, name):
    middle = Path(scene) / kind
    if kind == 'Descend':
        middle /= sequence
    elif split == 'Test':
        middle /= str(level)
    else:
        middle /= Path(sequence) / str(level)
    return DATA / split / 'images' / middle / f'{name}.JPG'


def mask_path(split, scene, kind, sequence, level, name):
    middle = Path(scene) / kind
    if kind == 'Descend':
        middle /= sequence
    elif split == 'Test':
        middle /= str(level)
    else:
        middle /= Path(sequence) / str(level)
    return DATA / split / 'annotations' / middle / f'{name}.png'


def read_rgb(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (SIZE[1], SIZE[0]):
        raise ValueError(f'Bad RGB: {path}')
    return im


def read_mask(path):
    im = Image.open(path)
    if im.mode != 'P' or im.size != SIZE:
        raise ValueError(f'Bad mask: {path}')
    return np.asarray(im)


def gps_distance(a, b):
    lat = (float(a['lat']) + float(b['lat'])) / 2
    return math.hypot((float(a['lat']) - float(b['lat'])) * 111_000,
                      (float(a['long']) - float(b['long'])) * 111_000 * math.cos(math.radians(lat)))


def collect_inventory():
    out = []
    csv_paths = sorted(DATA.rglob('6DOF.csv'))
    for csv_path in csv_paths:
        rel = csv_path.relative_to(DATA)
        split, _, scene, kind, *rest = rel.parts
        sequence = (rest[0] if kind == 'Descend' or split != 'Test' else scene)
        level = (None if kind == 'Descend' else rest[0] if split == 'Test' else rest[1])
        metadata = rows(csv_path)
        actual_images = {p.stem for p in (DATA / split / 'images' / scene / kind / Path(*rest[:-1])).glob('*.JPG')}
        actual_masks = {p.stem for p in (DATA / split / 'annotations' / scene / kind / Path(*rest[:-1])).glob('*.png')}
        named = [r['ImageName'] for r in metadata]
        if len(named) != len(set(named)) or set(named) != actual_images or set(named) != actual_masks:
            raise RuntimeError(f'Image/mask/pose mismatch: {rel}: {len(named)}, {len(actual_images)}, {len(actual_masks)}')
        out.append({'split': split, 'scene': scene, 'kind': kind, 'sequence': sequence,
                    'level': level, 'n': len(named), 'csv': str(csv_path)})
    if sum(x['n'] for x in out) != 2525:
        raise RuntimeError('Official full release image count differs from 2525')
    path = ROOT / 'manifests/full_dataset_inventory.json'
    path.write_text(json.dumps({'data_root': str(DATA), 'total_images': 2525, 'groups': out}, indent=2) + '\n')
    return out


def candidates_for_pair(split, scene, kind, sequence, hrow, lrow, boxes, gps_m=None):
    hname, lname = hrow['ImageName'], lrow['ImageName']
    hp = image_path(split, scene, kind, sequence, 100, hname)
    lp = image_path(split, scene, kind, sequence, 30, lname)
    hm = mask_path(split, scene, kind, sequence, 100, hname)
    lm = mask_path(split, scene, kind, sequence, 30, lname)
    high, low = read_rgb(hp), read_rgb(lp)
    high_mask, low_mask = read_mask(hm), read_mask(lm)
    homography, matches, inliers = register(low, high)
    records = []
    for index, low_box in enumerate(boxes, 1):
        high_box, poly = map_box(homography, low_box)
        a, b, c, d = high_box
        reason = None
        if min(a, b) < 0 or c > SIZE[0] or d > SIZE[1] or c-a < 20 or d-b < 20:
            reason = 'mapped_roi_out_of_bounds'
        if reason is None:
            lf = float(np.mean(low_mask[low_box[1]:low_box[3], low_box[0]:low_box[2]] == 12))
            hf = float(np.mean(high_mask[b:d, a:c] == 12))
            status = 'positive' if min(lf, hf) >= .05 else 'negative' if max(lf, hf) <= .005 else 'review'
        else:
            lf = hf = None
            status = 'excluded'
        sid = f'{sequence}_r{index:02d}' if kind == 'Descend' else f'{sequence}_{lname}'
        records.append({'sample_id': sid, 'source_split': split, 'scene': scene, 'kind': kind,
                        'sequence_id': sequence, 'high_frame': hname, 'low_frame': lname,
                        'high_rel_alt': float(hrow['rel_alt']), 'low_rel_alt': float(lrow['rel_alt']),
                        'high_box': high_box, 'low_box': low_box, 'high_polygon': poly,
                        'registration_matches': matches, 'registration_inliers': inliers,
                        'gps_distance_m': gps_m, 'high_label_fraction': hf,
                        'low_label_fraction': lf, 'status': status, 'exclusion_reason': reason,
                        'source_paths': {'high_image': str(hp), 'low_image': str(lp),
                                         'high_mask': str(hm), 'low_mask': str(lm)}})
    return records


def prepare(inventory):
    candidates, exclusions = [], []
    descend = [g for g in inventory if g['kind'] == 'Descend']
    for group in descend:
        seq = group['sequence']
        metadata = rows(Path(group['csv']))
        h, l = choose_frame(metadata, 100), choose_frame(metadata, 30)
        try:
            items = candidates_for_pair(group['split'], group['scene'], 'Descend', seq, h, l, GRID)
        except (ValueError, cv2.error, TypeError) as exc:
            exclusions.append({'sequence_id': seq, 'reason': 'pair_registration_failed', 'detail': str(exc), 'n_cells': 24})
            print(seq, 'pair_registration_failed', str(exc), flush=True)
            continue
        candidates += items
        print(seq, dict(Counter(x['status'] for x in items)), flush=True)
    paths = [g for g in inventory if g['kind'] == 'Path' and g['level'] == '30']
    for group in paths:
        split, scene, seq = group['split'], group['scene'], group['sequence']
        high_csv = DATA / split / '6DOF' / scene / 'Path'
        if split != 'Test':
            high_csv /= seq
        high_csv = high_csv / '100' / '6DOF.csv'
        if not high_csv.exists():
            exclusions.append({'sequence_id': seq, 'reason': 'no_100m_pair', 'n_frames': group['n']})
            continue
        hrows, lrows = rows(high_csv), rows(Path(group['csv']))
        for index, low in enumerate(lrows, 1):
            high = min(hrows, key=lambda h: gps_distance(h, low))
            dist = gps_distance(high, low)
            sid = f'{seq}_{low["ImageName"]}'
            if dist > 12:
                exclusions.append({'sample_id': sid, 'reason': 'gps_distance_over_12m', 'gps_distance_m': dist})
                continue
            try:
                items = candidates_for_pair(split, scene, 'Path', seq, high, low, [BOX], dist)
            except (ValueError, cv2.error, TypeError) as exc:
                exclusions.append({'sample_id': sid, 'reason': 'pair_registration_failed', 'detail': str(exc)})
                continue
            candidates += items
            if index % 25 == 0:
                print(seq, index, '/', len(lrows), flush=True)
    for c in candidates:
        if c['sequence_id'] == '100_0001':
            c['exclusion_reason'] = 'pilot_sequence'
        elif c['status'] == 'review':
            c['exclusion_reason'] = 'annotation_fractions_ambiguous_or_discordant'
        if c['exclusion_reason']:
            exclusions.append({'sample_id': c['sample_id'], 'reason': c['exclusion_reason']})
    selected = [c for c in candidates if c['status'] in {'positive','negative'} and not c['exclusion_reason']]
    if len({c['sample_id'] for c in candidates}) != len(candidates):
        raise RuntimeError('Duplicate sample IDs')
    out = ROOT / 'manifests'
    jsonlines(out / 'full_100_30_candidates.jsonl', candidates)
    jsonlines(out / 'full_100_30_exclusions.jsonl', exclusions)
    (out / 'full_100_30_selection.json').write_text(json.dumps(
        {'protocol': 'fixed 100-to-30-m vegetation replay', 'pilot': '100_0001',
         'thresholds': {'positive_min_each': .05, 'negative_max_each': .005,
                        'path_max_gps_m': 12, 'min_registration_inliers': 40},
         'selected': [c['sample_id'] for c in selected]}, indent=2) + '\n')
    print('selected', len(selected), 'by_kind', dict(Counter((x['kind'],x['scene']) for x in selected)),
          'excluded', len(exclusions), flush=True)
    build_inputs(selected)


def build_inputs(selected):
    out = ROOT / 'derived/full_100_30_inputs'
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    @lru_cache(maxsize=3)
    def rgb(path):
        return Image.open(path).convert('RGB')
    samples, truths = [], []
    for index, c in enumerate(selected, 1):
        source = c['source_paths']
        hpath, lpath = source['high_image'], source['low_image']
        high, low = rgb(hpath), rgb(lpath)
        hb, lb = c['high_box'], c['low_box']
        marked = high.copy()
        draw = ImageDraw.Draw(marked)
        for offset in range(-8, 9):
            draw.rectangle([hb[0]+offset, hb[1]+offset, hb[2]-offset, hb[3]-offset],
                           outline=(255,0,0), width=1)
        marked.thumbnail((1200,800), Image.Resampling.LANCZOS)
        ims = {'high_marked': marked,
               'high_crop': high.crop(tuple(hb)).resize((800,800), Image.Resampling.LANCZOS),
               'low_crop': low.crop(tuple(lb)).resize((800,800), Image.Resampling.LANCZOS)}
        image_files = {}
        for kind, im in ims.items():
            dest = out / f'{c["sample_id"]}_{kind}.png'
            im.save(dest, compress_level=1)
            image_files[kind] = str(dest)
        for path in source.values():
            if path not in hashes:
                hashes[path] = sha(path)
        split = 'full_descend' if c['kind'] == 'Descend' else ('full_test_paths' if c['source_split'] == 'Test' else 'full_train_paths')
        samples.append({'sample_id': c['sample_id'], 'scene_group_id': c['scene']+'_'+c['sequence_id'],
                        'sequence_id': c['sequence_id'], 'split': split,
                        'source_split': c['source_split'], 'high_frame_id': c['high_frame'],
                        'low_frame_id': c['low_frame'], 'high_rel_alt': c['high_rel_alt'],
                        'low_rel_alt': c['low_rel_alt'], 'high_box': hb, 'low_box': lb,
                        'high_polygon': c['high_polygon'], 'gps_distance_m': c['gps_distance_m'],
                        'registration_matches': c['registration_matches'],
                        'registration_inliers': c['registration_inliers'],
                        'question': QUESTION, 'images': image_files,
                        'image_sha256': {k: sha(v) for k, v in image_files.items()},
                        'source_sha256': {k: hashes[v] for k, v in source.items()}})
        truths.append({'sample_id': c['sample_id'], 'answer': 'yes' if c['status'] == 'positive' else 'no',
                       'evidence': 'MESSI vegetation class 12 mask agreement at both heights',
                       'high_label_fraction': c['high_label_fraction'],
                       'low_label_fraction': c['low_label_fraction']})
        if index % 25 == 0:
            print('inputs', index, '/', len(selected), flush=True)
    for split in sorted({s['split'] for s in samples}):
        ids = {s['sample_id'] for s in samples if s['split'] == split}
        sp = ROOT / 'manifests' / f'{split}_samples.jsonl'
        tp = ROOT / 'manifests' / f'{split}_ground_truth.jsonl'
        jsonlines(sp, [s for s in samples if s['split'] == split])
        jsonlines(tp, [t for t in truths if t['sample_id'] in ids])
        print(split, len(ids), sha(sp), sha(tp), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=DATA)
    args = p.parse_args()
    if args.data != DATA:
        DATA = args.data
    inventory = collect_inventory()
    prepare(inventory)
