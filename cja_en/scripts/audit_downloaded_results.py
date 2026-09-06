#!/usr/bin/env python3
"""Read fixed run paths; write independent checks only beneath cja_en/review.

Uses recorded predictions, not model inference. Does not certify image labels,
checkpoint provenance, API identity, or a currently running download's completion.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = Path('runs/benchmarks/paper_cja_mech_v1')
VQA = Path('runs/benchmarks/cja_agent_vqa')
NAMES = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
SOURCES = {}


def read(path, jsonl=False):
    p = ROOT / path
    before = p.stat()
    raw = p.read_bytes()
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Input changing during read: {path}')
    SOURCES[str(path)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return [json.loads(x) for x in raw.splitlines() if x.strip()] if jsonl else json.loads(raw)


def entropy(p):
    return np.array([round(-sum(float(v / row.sum()) * math.log(float(v / row.sum()))
                               for v in row if v > 0) / math.log(4), 3) for row in p])


def metrics(y, p):
    pred = p.argmax(1)
    f = []
    for c in range(4):
        tp = int(((pred == c) & (y == c)).sum())
        den = int((pred == c).sum() + (y == c).sum())
        f.append(2 * tp / den if den else 0.)
    conf = p.max(1)
    ece = 0.
    for j in range(15):
        mask = (conf <= (j + 1) / 15) & ((conf >= 0) if j == 0 else (conf > j / 15))
        if mask.any():
            ece += float(mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean()))
    return {'accuracy': float((pred == y).mean()), 'macro_f1': float(np.mean(f)),
            'ece': ece, 'brier': float(((p - np.eye(4)[y]) ** 2).sum(1).mean()),
            'nll': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())}


def vectors(rows, view):
    p = np.array([[r['views'][view]['probs'][n] for n in NAMES] for r in rows])
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid probability rows')
    return p


def main():
    result = {'scope': 'Independent arithmetic and local artifact consistency; not submission clearance'}
    summary = read(RUN / 'budget_allocation.json')
    frozen = read(RUN / 'frozen_manifest.json')
    items = read(RUN / 'final_fov/fov_ladder_eval_items.jsonl', True)
    fit = read(RUN / 'fov_ladder_val_items.jsonl', True)
    ladder = read(RUN / 'final_fov/fov_ladder_eval.json')
    val_ladder = read(RUN / 'fov_ladder_val.json')
    y = np.array([r['y'] for r in items], dtype=int)
    cruise, floor = vectors(items, 'cruise'), vectors(items, 'floor')
    cp, fp = cruise.argmax(1), floor.argmax(1)
    correctable, harmful = (cp != y) & (fp == y), (cp == y) & (fp != y)
    result['offline_population'] = {
        'n': len(items), 'unique_uids': len({r['uid'] for r in items}),
        'unique_rois': len({r['tile_id'] for r in items}),
        'events': dict(Counter(r['disaster'] for r in items)),
        'n_flip': int((cp != fp).sum()), 'n_correctable': int(correctable.sum()),
        'n_harmful': int(harmful.sum()), 'frozen_backend': frozen['backend'],
        'frozen_leaky': frozen['leaky'],
        'allocation_fit_test_overlap': sorted({r['disaster'] for r in fit} & {r['disaster'] for r in items})}
    differences = []
    fov_checks = {}
    for view in ['cruise', 'mid', 'floor']:
        measured = metrics(y, vectors(items, view))
        reported = next(r for r in ladder['curve'] if r['view'] == view)
        fov_checks[view] = measured
        for name in ['accuracy', 'macro_f1', 'brier', 'nll']:
            if abs(measured[name] - reported[name]) > 1e-9:
                differences.append(f'final FOV {view}/{name}')
    result['final_fov_recomputed'] = fov_checks
    result['exported_fov_source'] = {'split': val_ladder['split'], 'n': val_ladder['n_buildings'],
                                    'note': 'Compare original generated table to validation summary, not final FOV'}
    fit_c, fit_f = vectors(fit, 'cruise'), vectors(fit, 'floor')
    fit_ent, fit_pred = entropy(fit_c), fit_c.argmax(1)
    gains = fit_ent - entropy(fit_f)
    edges = np.linspace(0, 1, 6)
    fit_bins = np.digitize(fit_ent, edges[1:-1])
    table = {}
    for c, name in enumerate(NAMES):
        for b in range(5):
            mask = (fit_pred == c) & (fit_bins == b)
            if mask.any():
                table[f'{name}|{b}'] = float(gains[mask].mean())
    stored = summary['expected_gain_table']
    result['expected_gain_table_max_abs_difference'] = max(abs(v - stored['by_pred_and_bin'][k]) for k, v in table.items())
    ce = entropy(cruise)
    raw = np.clip(cruise, 1e-12, 1) ** summary['temperature']
    raw /= raw.sum(1, keepdims=True)
    expected = []
    for c, e in zip(cp, ce):
        b = int(np.digitize(e, edges[1:-1]))
        cmask = fit_pred == c
        fallback = float(gains[cmask].mean()) if cmask.any() else float(gains.mean())
        expected.append(table.get(f'{NAMES[c]}|{b}', fallback))
    conf = np.array([int(np.searchsorted(np.cumsum(sorted(row, reverse=True)), summary['qhat'])) + 1 for row in cruise])
    scores = {'entropy_uncal': entropy(raw), 'entropy_cal': ce,
              'expected_gain': np.array(expected), 'conformal': conf,
              'oracle': correctable.astype(int) - harmful.astype(int)}
    rng = np.random.default_rng(summary['seed'])
    selected = {}
    checks = {}
    for strategy in ['none', 'random', *scores]:
        for budget in [0., .1, .25, .5, 1.]:
            k = round(budget * len(y))
            mask = np.zeros(len(y), dtype=bool)
            if strategy != 'none' and k:
                ids = rng.choice(len(y), size=k, replace=False) if strategy == 'random' else np.argsort(-scores[strategy], kind='stable')[:k]
                mask[ids] = True
            p = np.where(mask[:, None], floor, cruise)
            measured = metrics(y, p)
            measured.update(n_descend=int(mask.sum()), net_corrected=int((mask & correctable).sum() - (mask & harmful).sum()))
            reported = next(r for r in summary['curves'][strategy] if r['budget'] == budget)
            for name, value in measured.items():
                if abs(value - reported[name]) > 1e-9:
                    differences.append(f'budget {strategy}/{budget}/{name}: {value} vs {reported[name]}')
            if budget == .25:
                checks[strategy] = measured
                selected[strategy] = p
    result['budget_at_025_recomputed'] = checks
    result['offline_numeric_mismatches'] = differences

    episodes = []
    manifests = []
    for shard in range(2):
        path = VQA / f'paper_cja_mech_final_shard{shard}of2'
        episodes.extend(read(path / 'episodes.jsonl', True))
        manifests.append(read(path / 'manifest.json'))
    aggregate = read(VQA / 'paper_cja_mech_final_reports/aggregate.json')
    read(VQA / 'paper_cja_mech_final_reports/budget_audit.json')
    groups = defaultdict(list)
    for row in episodes:
        groups[row['config']].append(row)
    key_counts = Counter((r['config'], r['qid']) for r in episodes)
    result['vqa_duplicates'] = [list(k) for k, n in key_counts.items() if n > 1]
    if result['vqa_duplicates']:
        raise ValueError('Duplicate config/qid; cannot silently aggregate')
    result['vqa_record_count'] = len(episodes)
    base = {r['qid']: r for r in groups['A0_HOLD']}
    vqa_checks = {}
    hist = defaultdict(lambda: [0, 0])
    changes = []
    for cfg, rows in sorted(groups.items()):
        ids = {r['qid'] for r in rows}
        if ids != set(base):
            raise ValueError(f'Unpaired question sets: {cfg}')
        correct = sum(bool(r['correct']) for r in rows)
        gains_n = sum(bool(r['correct']) and not base[r['qid']]['correct'] for r in rows)
        harms_n = sum(not r['correct'] and bool(base[r['qid']]['correct']) for r in rows)
        tail = [r for r in rows if (r.get('confidence') or 0) >= .9]
        reobserved = [r for r in rows if r.get('n_reobservations', 0) > 0]
        vqa_checks[cfg] = {'n': len(rows), 'correct': correct, 'accuracy': correct / len(rows),
            'abstained': sum(bool(r['abstain']) for r in rows),
            'actions': sum(r.get('n_reobservations', 0) for r in rows),
            'reobserved_episodes': sum(r.get('n_reobservations', 0) > 0 for r in rows),
            'gained_vs_A0': gains_n, 'lost_vs_A0': harms_n,
            'high_conf_n': len(tail), 'high_conf_correct': sum(bool(r['correct']) for r in tail),
            'reobserved_correct': sum(bool(r['correct']) for r in reobserved),
            'reobserved_gt_groups': dict(Counter(f"{r['question_type']} / {r['gt_answer']}" for r in reobserved)),
            'reobserved_events': dict(Counter(r['disaster'] for r in reobserved)),
            'reported_accuracy_matches_rounding': abs(correct / len(rows) - aggregate[cfg]['accuracy']) <= .000051,
            'reported_actions_match': sum(r.get('n_reobservations', 0) for r in rows) == aggregate[cfg]['n_reobservations'],
            'events': dict(Counter(r['disaster'] for r in rows))}
        for r in rows:
            c = r.get('confidence') or 0
            if c > 0:
                hist[round(c, 2)][0] += 1
                hist[round(c, 2)][1] += bool(r['correct'])
            cs = [t['confidence'] for t in r.get('trajectory', []) if t.get('confidence') is not None]
            if r.get('n_reobservations', 0) > 0 and len(cs) >= 2:
                changes.append((cs[0], cs[-1]))
    result['vqa_recomputed'] = vqa_checks
    result['confidence_pooled_across_configs'] = {str(c): {'n': n, 'correct': k} for c, (n, k) in sorted(hist.items())}
    result['confidence_change_episode_summary'] = {
        'n': len(changes), 'up': sum(b > a + .001 for a, b in changes),
        'down': sum(b < a - .001 for a, b in changes),
        'same': sum(abs(b - a) <= .001 for a, b in changes),
        'initial_mean': sum(a for a, b in changes) / len(changes) if changes else None,
        'final_mean': sum(b for a, b in changes) / len(changes) if changes else None}
    frozen_hash = SOURCES[str(RUN / 'frozen_manifest.json')]['sha256']
    result['vqa_manifest_checks'] = [{
        'run_id': m['run_id'], 'git_commit': m['env'].get('git_commit'),
        'git_dirty': m['env'].get('git_dirty'), 'llm_model': m['env'].get('llm_model'),
        'vlm_model': m['env'].get('vlm_model'),
        'frozen_manifest_hash_matches': m.get('frozen_manifest_sha256_16') == frozen_hash,
        'testset_sha256': m.get('testset_sha256_16'),
        'n_execution_errors_reported': m.get('n_execution_errors')} for m in manifests]
    for path, record in SOURCES.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError(f'Input changed before audit completed: {path}')
    result['sources'] = SOURCES
    dest = ROOT / 'cja_en/review/downloaded_results_audit.json'
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['offline_population', 'offline_numeric_mismatches',
        'vqa_record_count', 'vqa_duplicates', 'confidence_change_episode_summary', 'vqa_manifest_checks']}, ensure_ascii=False, indent=2))
    print(f'Written: {dest}')


if __name__ == '__main__':
    main()
