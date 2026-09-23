"""Score the frozen three-branch case from raw request records."""
import csv
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def readj(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def writecsv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def metrics(rows, group='overall'):
    n = len(rows)
    first_correct = sum(r['first_correct'] for r in rows)
    result = []
    for branch in 'ABC':
        complete = [r for r in rows if r[f'{branch}_status'] == 'ok']
        correct = sum(r[f'{branch}_correct'] for r in complete)
        corrected = sum(r['first_correct'] == 0 and r[f'{branch}_correct'] == 1 for r in complete)
        harmed = sum(r['first_correct'] == 1 and r[f'{branch}_correct'] == 0 for r in complete)
        answer_counts = Counter(r[f'{branch}_answer'] for r in rows)
        result.append({'group':group,'branch':branch,'planned_n':n,'completed_n':len(complete),
                       'first_correct':first_correct,'first_accuracy':round(first_correct/n,6) if n else '',
                       'correct':correct,'accuracy':round(correct/len(complete),6) if complete else '',
                       'corrected':corrected,'harmed':harmed,
                       'uncertain':answer_counts['uncertain'],'invalid':answer_counts['invalid'],
                       'run_failed':sum(r[f'{branch}_status'] != 'ok' for r in rows)})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['formal','exhaustive_descend','exhaustive_paths',
                                        'full_descend','full_train_paths','full_test_paths'], default='formal')
    args = ap.parse_args()
    samples = readj(ROOT/'manifests'/f'{args.split}_samples.jsonl')
    truth = {r['sample_id']:r['answer'] for r in readj(ROOT/'manifests'/f'{args.split}_ground_truth.jsonl')}
    raw = readj(ROOT/'runs'/args.split/'responses.jsonl')
    expected = {(s['sample_id'],b) for s in samples for b in ('first','A','B','C')}
    actual = [(r['sample_id'],r['branch']) for r in raw]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Missing or duplicate model requests')
    logs = {(r['sample_id'],r['branch']):r for r in raw}
    hashes = {r['config_hash'] for r in raw if 'config_hash' in r}
    if len(hashes) != 1:
        raise ValueError('Mixed model configurations')
    if any('config_hash' not in r and r.get('status') != 'skipped_first_failed' for r in raw):
        raise ValueError('Missing configuration hash on a model call')
    rows = []
    for s in samples:
        sid = s['sample_id']
        calls = {b:logs[(sid,b)] for b in ('first','A','B','C')}
        row = {'sample_id':sid,'sequence_id':s['sequence_id'],'scene_group_id':s['scene_group_id'],
               'truth':truth[sid],'first_answer':calls['first']['parsed_answer'],
               'first_status':calls['first']['status']}
        row['first_correct'] = int(row['first_status']=='ok' and row['first_answer']==row['truth'])
        for branch in 'ABC':
            row[f'{branch}_answer'] = calls[branch]['parsed_answer']
            row[f'{branch}_status'] = calls[branch]['status']
            row[f'{branch}_correct'] = int(row[f'{branch}_status']=='ok' and row[f'{branch}_answer']==row['truth'])
            row[f'{branch}_transition'] = ('corrected' if not row['first_correct'] and row[f'{branch}_correct'] else
                                          'harmed' if row['first_correct'] and not row[f'{branch}_correct'] else
                                          'unchanged')
        rows.append(row)
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    prefix = '' if args.split == 'formal' else f'{args.split}_'
    writecsv(out/f'{prefix}per_sample.csv', rows, list(rows[0]))
    fields = ['group','branch','planned_n','completed_n','first_correct','first_accuracy','correct',
              'accuracy','corrected','harmed','uncertain','invalid','run_failed']
    writecsv(out/f'{prefix}summary.csv', metrics(rows), fields)
    grouped = defaultdict(list)
    for r in rows: grouped[r['sequence_id']].append(r)
    writecsv(out/f'{prefix}per_scene.csv', [x for k,v in sorted(grouped.items()) for x in metrics(v,k)],fields)
    by_class = defaultdict(list)
    for r in rows: by_class[r['truth']].append(r)
    writecsv(out/f'{prefix}per_class.csv', [x for k,v in sorted(by_class.items()) for x in metrics(v,k)],fields)
    paired = [r for r in rows if all(r[f'{b}_status']=='ok' for b in 'ABC')]
    comparison = {'planned_n':len(rows),'complete_triplets':len(paired),
                  'C_minus_B_correct':sum(r['C_correct']-r['B_correct'] for r in paired),
                  'C_minus_A_correct':sum(r['C_correct']-r['A_correct'] for r in paired),
                  'C_B_help':sum(r['C_correct']==1 and r['B_correct']==0 for r in paired),
                  'C_B_harm':sum(r['C_correct']==0 and r['B_correct']==1 for r in paired),
                  'C_A_help':sum(r['C_correct']==1 and r['A_correct']==0 for r in paired),
                  'C_A_harm':sum(r['C_correct']==0 and r['A_correct']==1 for r in paired),
                  'config_hash':next(iter(hashes))}
    (out/f'{prefix}comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison))


if __name__ == '__main__': main()
