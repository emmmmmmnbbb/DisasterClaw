"""Score all frozen seeds; distinguish seed variation from generalization."""
import argparse, json, statistics, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'supplement_random_budget'
SPLITS=('full_descend','full_train_paths','full_test_paths')
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,default=ROOT/'supplement_random_budget_gpu')
    OUT=ap.parse_args().out_dir.resolve()
    selection=json.loads((OUT/'selection.json').read_text());base={};old={};truth={};triggered=set()
    high_lat=[];low_lat=[];hold_wall=[];agent_wall=[]
    for split in SPLITS:
        for r in rows(ROOT/'manifests'/f'{split}_ground_truth.jsonl'):truth[split,r['sample_id']]=r['answer']
        for r in rows(ROOT/'runs'/f'single_temporal_{split}'/'episodes.jsonl'):
            key=split,r['sample_id']
            if r['arm']=='hold':
                base[key]=r['answer'];high_lat.append(r['events'][0]['prediction']['latency_s']);hold_wall.append(r['elapsed_s'])
            else:
                agent_wall.append(r['elapsed_s'])
                if r['rechecks']:
                    triggered.add(key);old[key]=r['answer']
                    low_lat.append(r['events'][-1]['prediction']['latency_s'])
    selected={tuple(x) for v in selection['selected'].values() for x in v}
    records=rows(OUT/'episodes.jsonl');new={(r['split'],r['sample_id']):r for r in records}
    assert len(new)==len(records) and set(new)==selected|triggered
    assert len(base)==409 and len(triggered)==20
    def score(keys):
        assert len(keys)==20
        corrected=sum(base[k]!=truth[k] and new[k]['answer']==truth[k] for k in keys)
        harmed=sum(base[k]==truth[k] and new[k]['answer']!=truth[k] for k in keys)
        for k in keys:assert new[k]['rechecks']==1 and len(new[k]['trajectory'])==2
        return {'correct':sum(v==truth[k] for k,v in base.items())+corrected-harmed,
                'corrected':corrected,'harmed':harmed,'rechecks':20,'observations':429,'deterministic_answers':429}
    scores=[dict(seed=int(seed),**score({tuple(x) for x in keys})) for seed,keys in selection['selected'].items()]
    entropy=score(triggered);values=[r['correct'] for r in scores]
    latency={'high_fresh_calls':len(high_lat),'high_perception_s_sum':sum(high_lat),'high_perception_s_mean':statistics.mean(high_lat),
             'triggered_low_fresh_calls':len(low_lat),'low_perception_s_sum':sum(low_lat),'low_perception_s_mean':statistics.mean(low_lat),
             'hold_episode_wall_s_sum':sum(hold_wall),'agent_episode_wall_s_sum_excludes_cached_high':sum(agent_wall),
             'note':'Archived prediction latency includes YOLO and SegFormer inference and input preparation, not flight. Agent elapsed time excludes reused high inference, so hold vs agent wall times must not be compared as full policy runtime. No warmup calls excluded.'}
    result={'selection_sha256':hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest(),
       'episodes_sha256':hashlib.sha256((OUT/'episodes.jsonl').read_bytes()).hexdigest(),
       'baseline_correct':sum(v==truth[k] for k,v in base.items()),'n':len(base),'seeds':scores,
       'entropy_same_device':entropy,'entropy_disagrees_with_archived_answers':[list(k) for k in triggered if old[k]!=new[k]['answer']],
       'random_correct_mean':statistics.mean(values),'random_correct_sd':statistics.stdev(values),
       'random_correct_min':min(values),'random_correct_max':max(values),
       'random_corrected_mean':statistics.mean(r['corrected'] for r in scores),
       'random_harmed_mean':statistics.mean(r['harmed'] for r in scores),
       'random_seeds_at_least_entropy':sum(v>=entropy['correct'] for v in values),
       'scope':'Seed variation on same fixed regions, not independent trials or a generalization confidence interval.',
       'archived_messi_latency':latency}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
