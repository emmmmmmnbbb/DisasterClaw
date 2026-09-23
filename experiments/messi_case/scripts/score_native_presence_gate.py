"""Score the exact-presence-gate MESSI diagnostic."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('full_descend', 'full_train_paths', 'full_test_paths')


def readj(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


gt = {x['sample_id']: x['answer'] for split in SPLITS for x in
      readj(ROOT/'manifests'/f'{split}_ground_truth.jsonl')}
episodes = readj(ROOT/'runs/native_presence_full/episodes.jsonl')
assert len(episodes) == len(gt) == 409
assert len({x['sample_id'] for x in episodes}) == len(episodes)
assert all(x['rechecks'] == 0 and [r['image_key'] for r in x['image_reads']] == ['high_marked']
           for x in episodes)
out = {'n': len(episodes), 'correct': sum(x['answer'] == gt[x['sample_id']] for x in episodes),
       'rechecks': 0, 'stop_reasons': dict(Counter(x['stop_reason'] for x in episodes)),
       'by_stratum': {split: {'n': sum(x['split'] == split for x in episodes),
                             'correct': sum(x['split'] == split and x['answer'] == gt[x['sample_id']]
                                            for x in episodes)} for split in SPLITS},
       'scope': 'original DisasterClaw presence gate with a MESSI first-view VLM adapter; not full ChangeOS/hybrid pipeline'}
path = ROOT/'results/native_presence_full_summary.json'
path.write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
