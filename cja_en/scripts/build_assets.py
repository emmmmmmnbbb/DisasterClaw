#!/usr/bin/env python3
"""Generate manuscript assets from audited records; never run a model.

The exploratory bootstrap resamples ROIs within each observed event, holds
allocation masks fixed, and does not estimate generalization to new events.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_downloaded_results import ROOT, RUN, VQA, entropy, vectors, metrics

OUT = ROOT / 'cja_en'
AUDIT = json.loads((OUT / 'review/downloaded_results_audit.json').read_text())
LABELS = {'none': 'No reobservation', 'random': 'Random',
          'entropy_uncal': 'Raw entropy', 'entropy_cal': 'Calibrated entropy',
          'expected_gain': 'Expected entropy reduction', 'conformal': 'APS set size',
          'oracle': 'Label-informed diagnostic'}
ONLINE = {'A0_HOLD': 'Hold', 'A1_RANDOM_MATCHED': 'Random',
          'A2_FIXED_MATCHED': 'Fixed', 'A3U_RAW_ENTROPY': 'Raw entropy',
          'A3_ENTROPY': 'Calibrated entropy', 'A4_CONFORMAL': 'APS trigger',
          'A5_EXPECTED': 'Expected entropy reduction',
          'AB_CENTER': 'Center only', 'AB_DESCEND': 'Descend only', 'AB_FULL': 'Combined motion'}


def read(path, lines=False):
    raw = (ROOT / path).read_bytes()
    expected = AUDIT['sources'].get(str(path))
    if expected and hashlib.sha256(raw).hexdigest() != expected['sha256']:
        raise RuntimeError(f'Run audit again: source changed: {path}')
    return [json.loads(r) for r in raw.splitlines() if r.strip()] if lines else json.loads(raw)


def write(path, text):
    (OUT / path).write_text(text + '\n')


def tabular(path, columns, header, rows):
    write(path, '\\begin{tabular}{' + columns + '}\n\\toprule\n' + header +
          ' \\\\\n\\midrule\n' + '\n'.join(' & '.join(map(str, r)) + ' \\\\' for r in rows) +
          '\n\\bottomrule\n\\end{tabular}')


def f1(cm):
    den = cm.sum(axis=-1) + cm.sum(axis=-2)
    return np.divide(2 * np.diagonal(cm, axis1=-2, axis2=-1), den,
                     out=np.zeros_like(den, dtype=float), where=den != 0).mean(axis=-1)


def main():
    if AUDIT['offline_numeric_mismatches'] or AUDIT['vqa_duplicates']:
        raise RuntimeError('Unresolved numerical audit failure')
    d = read(RUN / 'budget_allocation.json')
    rows = read(RUN / 'final_fov/fov_ladder_eval_items.jsonl', True)
    y = np.array([r['y'] for r in rows])
    pc, pf = vectors(rows, 'cruise'), vectors(rows, 'floor')
    pred = pc.argmax(1)
    ent = entropy(pc)
    raw = np.clip(pc, 1e-12, 1) ** d['temperature']
    raw /= raw.sum(1, keepdims=True)
    n = len(rows)
    k = round(.25 * n)
    rng = np.random.default_rng(d['seed'])
    rng.choice(n, size=round(.1 * n), replace=False)  # source random curve order
    random_ids = rng.choice(n, size=k, replace=False)
    predicted = {'none': pred}
    for name, ids in [('random', random_ids),
                      ('entropy_cal', np.argsort(-ent, kind='stable')[:k]),
                      ('entropy_uncal', np.argsort(-entropy(raw), kind='stable')[:k])]:
        mask = np.zeros(n, dtype=bool)
        mask[ids] = True
        p = np.where(mask[:, None], pf, pc)
        if abs(metrics(y, p)['macro_f1'] - AUDIT['budget_at_025_recomputed'][name]['macro_f1']) > 1e-9:
            raise RuntimeError(f'Allocation reconstruction failed: {name}')
        predicted[name] = p.argmax(1)
    rois = sorted({r['tile_id'] for r in rows})
    missing = {v: sum(not r['views'][v]['detected'] for r in rows) for v in ['cruise', 'mid', 'floor']}
    common = np.array([all(r['views'][v]['detected'] for v in ['cruise', 'mid', 'floor']) for r in rows])
    common_metrics = {v: metrics(y[common], vectors(rows, v)[common]) for v in ['cruise', 'mid', 'floor']}
    detection_effects = Counter()
    floor_pred = pf.argmax(1)
    for i, row in enumerate(rows):
        group = 'both_detected' if row['views']['cruise']['detected'] and row['views']['floor']['detected'] else 'unmatched_involved'
        if pred[i] != y[i] and floor_pred[i] == y[i]:
            detection_effects['corrected_' + group] += 1
        if pred[i] == y[i] and floor_pred[i] != y[i]:
            detection_effects['harmed_' + group] += 1
    roi_lookup = {r: i for i, r in enumerate(rois)}
    event_rois = defaultdict(set)
    cm = {s: np.zeros((len(rois), 4, 4), dtype=int) for s in predicted}
    for i, row in enumerate(rows):
        j = roi_lookup[row['tile_id']]
        event_rois[row['disaster']].add(j)
        for s in predicted:
            cm[s][j, y[i], predicted[s][i]] += 1
    rng = np.random.default_rng(20260905)
    samples = np.concatenate([rng.choice(sorted(ids), (2000, len(ids)), replace=True)
                              for event, ids in sorted(event_rois.items())], axis=1)
    boots = {s: f1(c[samples].sum(axis=1)) for s, c in cm.items()}
    intervals = {}
    for a, b in [('entropy_cal', 'random'), ('entropy_cal', 'entropy_uncal')]:
        intervals[f'{a}_minus_{b}'] = {
            'delta_macro_f1': float(f1(cm[a].sum(axis=0)) - f1(cm[b].sum(axis=0))),
            'percentile_95': np.quantile(boots[a] - boots[b], [.025, .975]).tolist()}
    episodes = []
    for shard in range(2):
        episodes += read(VQA / f'paper_cja_mech_final_shard{shard}of2/episodes.jsonl', True)
    groups = defaultdict(list)
    for r in episodes:
        groups[r['config']].append(r)
    summaries = {}
    for cfg, eps in groups.items():
        bytype = {}
        for kind in sorted({r['question_type'] for r in eps}):
            subset = [r for r in eps if r['question_type'] == kind]
            bytype[kind] = {'n': len(subset), 'correct': sum(bool(r['correct']) for r in subset),
                            'actions': sum(r['n_reobservations'] for r in subset)}
        risk = []
        for threshold in [0., .8, .9, .95, .99, 1.]:
            accepted = [r for r in eps if not r['abstain'] and (r['confidence'] or 0) >= threshold]
            risk.append({'threshold': threshold, 'n_accepted': len(accepted),
                         'coverage': len(accepted) / len(eps),
                         'risk': sum(not r['correct'] for r in accepted) / len(accepted) if accepted else None})
        summaries[cfg] = {'by_question_type': bytype, 'posthoc_threshold_risk': risk}
    result = {'bootstrap': {'method': 'paired ROI bootstrap stratified within the three observed events',
                           'n_resamples': 2000, 'seed': 20260905, 'n_rois': len(rois),
                           'event_roi_counts': {e: len(ids) for e, ids in sorted(event_rois.items())},
                           'fixed_allocation': True, 'exploratory': True, 'comparisons': intervals},
              'unmatched_buildings': missing, 'detection_effects': dict(detection_effects),
              'common_matched_sensitivity': {'n': int(common.sum()), 'selection': 'detected in all three recorded views', 'metrics': common_metrics},
              'online': summaries, 'input_audit_sha256': hashlib.sha256((OUT / 'review/downloaded_results_audit.json').read_bytes()).hexdigest()}
    write('review/manuscript_analysis.json', json.dumps(result, indent=2))

    fov = AUDIT['final_fov_recomputed']
    tabular('tables/fov.tex', 'lrrrrr', 'View & GSD (m/px) & Acc. & Macro-F1 & Brier & NLL',
            [[v.capitalize(), f'{g:.2f}', *[f'{fov[v][m]:.4f}' for m in ['accuracy', 'macro_f1', 'brier', 'nll']]]
             for v, g in [('cruise', 1.5), ('mid', 1.), ('floor', .5)]])
    tabular('tables/common_matched.tex', 'lrrrr', 'View & Accuracy & Macro-F1 & Brier & NLL',
            [[v.capitalize(), *[f'{common_metrics[v][m]:.4f}' for m in ['accuracy', 'macro_f1', 'brier', 'nll']]]
             for v in ['cruise', 'mid', 'floor']])
    tabular('tables/allocation.tex', 'lrrrrr', 'Policy & Actions & Acc. & Macro-F1 & ECE & Net correct',
            [[LABELS[s], m['n_descend'], f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", f"{m['ece']:.4f}", m['net_corrected']]
             for s, m in AUDIT['budget_at_025_recomputed'].items()])
    tabular('tables/online.tex', 'lrrrrr', 'Configuration & Correct & Actions & Episodes & Gain & Loss',
            [[label, f"{AUDIT['vqa_recomputed'][s]['correct']}/160", *[AUDIT['vqa_recomputed'][s][m] for m in
                  ['actions', 'reobserved_episodes', 'gained_vs_A0', 'lost_vs_A0']]] for s, label in ONLINE.items()])
    tabular('tables/question_types.tex', 'lrrrr', 'Question type & Questions & Hold correct & Expected correct & Expected actions',
            [[kind.capitalize(), m['n'], m['correct'], summaries['A5_EXPECTED']['by_question_type'][kind]['correct'],
              summaries['A5_EXPECTED']['by_question_type'][kind]['actions']]
             for kind, m in summaries['A0_HOLD']['by_question_type'].items()])
    tabular('tables/cluster_intervals.tex', 'lrr', 'Paired comparison & Macro-F1 difference & 95\\% interval',
            [[LABELS[a] + ' -- ' + LABELS[b], f"{intervals[a+'_minus_'+b]['delta_macro_f1']:+.4f}",
              '[' + ', '.join(f'{v:+.4f}' for v in intervals[a+'_minus_'+b]['percentile_95']) + ']']
             for a, b in [('entropy_cal', 'random'), ('entropy_cal', 'entropy_uncal')]])
    lines = [r'\begin{tikzpicture}', r'\begin{axis}[width=.96\linewidth,height=6.5cm,xlabel={Reobservation fraction $b$},ylabel={Macro-F1},',
             r'xmin=0,xmax=1,ymin=.60,ymax=.71,xtick={0,.1,.25,.5,1},grid=major,',
             r'legend style={at={(.5,-.25)},anchor=north,draw=none,font=\footnotesize},legend columns=2]']
    styles = ['black,dashed,mark=none', 'gray,mark=square*', 'orange!85!black,mark=triangle*',
              'blue!70!black,mark=*', 'teal!80!black,mark=diamond*', 'violet,mark=x', 'red!70!black,dashed,mark=o']
    for (s, label), style in zip(LABELS.items(), styles):
        coords = ' '.join(f"({r['budget']},{r['macro_f1']:.8f})" for r in d['curves'][s])
        lines += [f'\\addplot+[{style},thick] coordinates {{{coords}}};', '\\addlegendentry{' + label + '}']
    lines += [r'\end{axis}', r'\end{tikzpicture}']
    write('figures/budget_curve.tex', '\n'.join(lines))
    lines = [r'\begin{tikzpicture}', r'\begin{axis}[width=.88\linewidth,height=6cm,xlabel={Fraction of questions answered},ylabel={Error among accepted answers},',
             r'xmin=0,xmax=1,ymin=0,ymax=1,grid=major,legend style={at={(.5,-.24)},anchor=north,draw=none},legend columns=3]']
    for cfg, style in [('A0_HOLD', 'black,mark=square*'), ('A5_EXPECTED', 'teal!80!black,mark=diamond*'), ('A4_CONFORMAL', 'violet,mark=*')]:
        rs = sorted((r for r in summaries[cfg]['posthoc_threshold_risk'] if r['risk'] is not None), key=lambda r: r['coverage'])
        coords = ' '.join(f"({r['coverage']:.8f},{r['risk']:.8f})" for r in rs)
        lines += [f'\\addplot+[{style},thick] coordinates {{{coords}}};', '\\addlegendentry{' + ONLINE[cfg] + '}']
    lines += [r'\end{axis}', r'\end{tikzpicture}']
    write('figures/risk_coverage.tex', '\n'.join(lines))
    print(json.dumps(result['bootstrap'], indent=2))
    print(json.dumps(result['common_matched_sensitivity'], indent=2))
    print('Generated six tables and two PGFPlots figures from audited inputs.')


if __name__ == '__main__':
    main()
