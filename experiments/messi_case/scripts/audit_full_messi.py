"""Check the frozen full-release manifests, source hashes and image boundary."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('full_descend', 'full_train_paths', 'full_test_paths')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def readj(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    inventory = json.loads((ROOT/'manifests/full_dataset_inventory.json').read_text())
    assert inventory['total_images'] == 2525
    assert sum(g['n'] for g in inventory['groups']) == 2525
    candidates = readj(ROOT/'manifests/full_100_30_candidates.jsonl')
    exclusions = readj(ROOT/'manifests/full_100_30_exclusions.jsonl')
    selection = json.loads((ROOT/'manifests/full_100_30_selection.json').read_text())['selected']
    assert len(selection) == len(set(selection))
    expected = {c['sample_id'] for c in candidates if c['status'] in ('positive','negative')
                and not c['exclusion_reason']}
    assert set(selection) == expected
    sample_ids, cached = set(), {}
    for split in SPLITS:
        samples = readj(ROOT/'manifests'/f'{split}_samples.jsonl')
        truths = readj(ROOT/'manifests'/f'{split}_ground_truth.jsonl')
        truth = {t['sample_id']:t for t in truths}
        assert len(truth) == len(truths) == len(samples)
        for s in samples:
            sid = s['sample_id']
            assert sid not in sample_ids and sid in truth and s['split'] == split
            sample_ids.add(sid)
            for key, path in s['images'].items():
                assert sha(path) == s['image_sha256'][key], (sid, key)
            for key, path in next(c for c in candidates if c['sample_id']==sid)['source_paths'].items():
                if path not in cached:
                    cached[path] = sha(path)
                assert cached[path] == s['source_sha256'][key], (sid, key)
        print(split, len(samples), dict(Counter(t['answer'] for t in truths)), flush=True)
    assert sample_ids == expected
    print('PASS', len(sample_ids), 'regions', len(candidates), 'recorded candidates',
          len(exclusions), 'exclusion records', len(cached), 'unique source files')


if __name__ == '__main__':
    main()
