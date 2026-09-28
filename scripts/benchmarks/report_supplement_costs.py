"""Read-only frozen episode audit; output derived summaries separately."""
import json, hashlib, statistics
from pathlib import Path
from collections import defaultdict
from audit_changeos_p7_costs import audit_row
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/messi_case/supplement_random_budget'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    task=ROOT/'backend/data/benchmarks/agent_vqa_final_candidate_20260915_binary_human_reviewed_validated.json'
    starts={x['id']:x['start'] for x in json.loads(task.read_text())['items']}
    grouped=defaultdict(list); inputs={}; seen=set()
    for repeat in range(3):
        for shard in range(4):
            folder=ROOT/f'runs/benchmarks/changeos_binary/p6_final_three_event_v1_20260915_repeat{repeat}_shard{shard}of4'
            result=json.loads((folder/'results.json').read_text())
            assert result['valid_for_analysis'] and result['n_execution_errors']==0
            assert result['testset_sha256_16']==sha(task)
            p=folder/'episodes.jsonl';inputs[str(p.relative_to(ROOT))]=sha(p)
            for line in p.open():
                r=json.loads(line);key=(repeat,r['config'],r['qid']);assert key not in seen;seen.add(key)
                a=audit_row(r,starts[r['qid']]);traj=r['trajectory']
                a.update(observations=len({s['observation_id'] for s in traj}),
                         answer_calls=sum(s.get('generation_seed') is not None for s in traj),
                         search_actions=sum(s['action']=='fly_relative' and s.get('reobserve_kind')!='recheck' for s in traj),
                         wall_s=r.get('wall_s'),repeat=repeat)
                assert a['answer_calls']==len(r['generation_seeds'])
                grouped[r['config']].append(a)
    summary={}
    for cfg,items in sorted(grouped.items()):
        assert len(items)==480
        complete=[r for r in items if r['trajectory_motion_complete']]
        walls=[r['wall_s'] for r in items if r['wall_s'] is not None]
        summary[cfg]={'n':len(items),'observations':sum(r['observations'] for r in items),
          'answer_calls':sum(r['answer_calls'] for r in items),
          'dedicated_rechecks':sum(r['reported_n_reobservations'] for r in items),
          'search_actions':sum(r['search_actions'] for r in items),
          'motion_complete_n':len(complete),'incomplete_terminal_motion_n':len(items)-len(complete),
          'horizontal_m_complete_mean':statistics.mean(r['total_horizontal_m'] for r in complete),
          'vertical_m_complete_mean':statistics.mean(r['total_vertical_m'] for r in complete),
          'wall_s_n':len(walls),'wall_s_total':sum(walls),'wall_s_mean':statistics.mean(walls),
          'wall_s_median':statistics.median(walls)}
    report={'scope':'Answer calls are logged generation attempts; observations are distinct per-episode observation IDs. Motion means use only complete trajectories, including search. Wall time is software episode time, not flight time or matched hardware throughput. Component costs not recorded.',
            'inputs_sha256':inputs,'task_sha256':sha(task),'by_config':summary}
    (OUT/'benchmark_costs.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
