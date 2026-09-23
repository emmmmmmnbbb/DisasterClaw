"""Aggregate the frozen full-release strata and cluster descriptive intervals."""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('full_descend', 'full_train_paths', 'full_test_paths')


def paired_interval(rows, left, right, cluster, seed=20260923, reps=10000):
    groups = {}
    for row in rows:
        groups.setdefault(row[cluster], []).append(int(row[left]) - int(row[right]))
    entries = [(len(v), sum(v)) for v in groups.values()]
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        draw = [entries[rng.randrange(len(entries))] for _ in entries]
        samples.append(sum(x[1] for x in draw)/sum(x[0] for x in draw))
    samples.sort()
    return {'estimate': sum(x[1] for x in entries)/sum(x[0] for x in entries),
            'cluster_n': len(entries), 'lower_95': samples[int(.025*reps)],
            'upper_95': samples[int(.975*reps)], 'method': 'cluster bootstrap percentile, 10000 draws'}


def main():
    fixed, replay, fixed_rows, replay_rows = {}, {}, [], []
    for split in SPLITS:
        fr = list(csv.DictReader((ROOT/'results'/f'{split}_per_sample.csv').open()))
        score = json.loads((ROOT/'results'/f'{split}_comparison.json').read_text())
        rr = json.loads((ROOT/'results'/f'agent_replay_logits_{split}_summary.json').read_text())
        fixed[split] = {'n': len(fr), 'correct': {b: sum(int(x[f'{b}_correct']) for x in fr) for b in 'ABC'},
                        'C_B_help': score['C_B_help'], 'C_B_harm': score['C_B_harm']}
        replay[split] = {'n': rr['hold']['n'], 'hold_correct': rr['hold']['correct'],
                         'agent_correct': rr['agent']['correct'], 'rechecks': rr['agent']['rechecks'],
                         'corrected': rr['paired']['corrected'], 'harmed': rr['paired']['harmed'],
                         'by_truth_class': rr['by_truth_class'], 'by_sequence': rr['by_sequence']}
        fixed_rows.extend(fr)
        replay_rows.extend([{'sequence_id': r['sequence_id'], 'hold_correct': int(r['hold']==r['gt']),
                             'agent_correct': int(r['agent']==r['gt'])} for r in rr['per_sample']])
    n = sum(fixed[s]['n'] for s in SPLITS)
    out = {'scope': 'all eligible 100-to-30-m paired vegetation regions in the 2525-image release',
           'fixed_by_stratum': fixed,
           'fixed_total': {'n': n, 'correct': {b: sum(fixed[s]['correct'][b] for s in SPLITS) for b in 'ABC'},
                           'C_B_help': sum(fixed[s]['C_B_help'] for s in SPLITS),
                           'C_B_harm': sum(fixed[s]['C_B_harm'] for s in SPLITS),
                           'C_minus_B_cluster_interval': paired_interval(fixed_rows, 'C_correct', 'B_correct', 'sequence_id')},
           'replay_by_stratum': replay,
           'replay_total': {'n': n, 'hold_correct': sum(replay[s]['hold_correct'] for s in SPLITS),
                            'agent_correct': sum(replay[s]['agent_correct'] for s in SPLITS),
                            'rechecks': sum(replay[s]['rechecks'] for s in SPLITS),
                            'corrected': sum(replay[s]['corrected'] for s in SPLITS),
                            'harmed': sum(replay[s]['harmed'] for s in SPLITS),
                            'agent_minus_hold_cluster_interval': paired_interval(
                                replay_rows, 'agent_correct', 'hold_correct', 'sequence_id')}}
    assert n == len(fixed_rows) == len(replay_rows)
    path = ROOT/'results/full_messi_summary.json'
    path.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
