"""Create model inputs from pre-reviewed MESSI candidate IDs."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('selection_json')
    ap.add_argument('--output-subdir', default='inputs')
    args = ap.parse_args()
    selections = json.loads(Path(args.selection_json).read_text())
    output = ROOT / 'derived' / args.output_subdir
    output.mkdir(parents=True, exist_ok=True)
    manifests = ROOT / 'manifests'
    manifests.mkdir(exist_ok=True)
    samples, truths = [], []
    cached_seq, cached_frames, cached_candidates, cached_source_hashes = None, None, None, None
    for selection in selections:
        seq, rid = selection['sequence'], selection['candidate_id']
        if seq != cached_seq:
            cached_seq = seq
            cached_candidates = {x['id']:x for x in map(json.loads, (ROOT/'derived'/seq/'vegetation'/'candidates.jsonl').read_text().splitlines())}
            first = next(iter(cached_candidates.values()))
            raw = ROOT / 'raw' / seq
            with zipfile.ZipFile(raw/'images.zip') as z:
                high = Image.open(io.BytesIO(z.read(f"{seq}/{first['high_frame']}.JPG"))).convert('RGB')
                low = Image.open(io.BytesIO(z.read(f"{seq}/{first['low_frame']}.JPG"))).convert('RGB')
            cached_frames = (high, low)
            cached_source_hashes = {k:sha(raw/f'{k}.zip') for k in ['images','annotations']}
        candidate = cached_candidates[rid]
        if candidate['status'] != f"candidate_{selection['answer']}":
            raise ValueError(f'Visual answer conflicts with both frame labels: {rid}')
        high, low = cached_frames
        first = next(iter(cached_candidates.values()))
        if candidate['high_frame'] != first['high_frame'] or candidate['low_frame'] != first['low_frame']:
            raise ValueError(f'Mixed frame pair within sequence: {seq}')
        box_h, box_l = candidate['high_box'], candidate['low_box']
        high_marked = high.copy()
        draw = ImageDraw.Draw(high_marked)
        for offset in range(-8, 9):
            draw.rectangle([box_h[0]+offset,box_h[1]+offset,box_h[2]-offset,box_h[3]-offset],outline=(255,0,0),width=1)
        high_marked.thumbnail((1200, 800), Image.Resampling.LANCZOS)
        high_crop = high.crop(tuple(box_h)).resize((800,800), Image.Resampling.LANCZOS)
        low_crop = low.crop(tuple(box_l)).resize((800,800), Image.Resampling.LANCZOS)
        paths = {}
        for typ,im in [('high_marked',high_marked),('high_crop',high_crop),('low_crop',low_crop)]:
            p = output / f'{rid}_{typ}.png'
            im.save(p, optimize=True)
            paths[typ] = str(p)
        samples.append({
            'sample_id':rid,'scene_group_id':selection['scene_group_id'],'sequence_id':seq,
            'split':selection['split'],'high_frame_id':candidate['high_frame'],
            'low_frame_id':candidate['low_frame'],'high_rel_alt':candidate['high_rel_alt'],
            'low_rel_alt':candidate['low_rel_alt'],'high_box':box_h,'low_box':box_l,
            'high_polygon':candidate['high_polygon'],'registration_matches':candidate['matches'],
            'registration_inliers':candidate['inliers'],
            'question':selection.get('question','Does the indicated ground region contain visible tree or shrub canopy?'),
            'images':paths,'image_sha256':{k:sha(v) for k,v in paths.items()},
            'source_sha256':cached_source_hashes,
        })
        truths.append({'sample_id':rid,'answer':'yes' if selection['answer']=='positive' else 'no',
                       'evidence':selection.get('evidence','Visual review of paired RGB crops plus MESSI vegetation (class 12) labels'),
                       'high_label_fraction':candidate['high_label_fraction'],
                       'low_label_fraction':candidate['low_label_fraction']})
    for split in sorted(set(x['split'] for x in samples)):
        sample_path = manifests / f'{split}_samples.jsonl'
        truth_path = manifests / f'{split}_ground_truth.jsonl'
        sample_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in samples if x['split']==split))
        truth_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in truths if next(s for s in samples if s['sample_id']==x['sample_id'])['split']==split))
        print(split,len([s for s in samples if s['split']==split]),sha(sample_path),sha(truth_path))


if __name__ == '__main__':
    main()
