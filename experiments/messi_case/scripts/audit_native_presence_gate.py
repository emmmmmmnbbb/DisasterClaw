"""Run the unmodified DisasterClaw presence gate on MESSI first observations.

This is an incompatibility diagnostic: the original presence policy explicitly
answers at the wide view. It reuses the frozen fixed-observation first answers
and invokes the actual AgentVqaController, with no lower image exposed.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_agent_replay import OfflineEnvironment, VEGETATION_QUESTION, run_arm  # noqa: E402
from run_case import FIRST_VEGETATION  # noqa: E402
from recheck import task_conditioned_recheck_decision  # noqa: E402

SPLITS = ('full_descend', 'full_train_paths', 'full_test_paths')


class NativePresenceEnvironment(OfflineEnvironment):
    def reobserve(self, result, spec, evidence):
        decision = task_conditioned_recheck_decision(
            question_type=spec.question_type,
            uncertainty=1.0,  # maximum possible uncertainty; presence still holds
            alt=self.sample['high_rel_alt'],
            descend_step_m=self.sample['high_rel_alt']-self.sample['low_rel_alt'],
            alt_min_m=self.sample['low_rel_alt'], roi_norm_bbox=None,
            target_visible=False, target_matched=False,
            uncertainty_trigger=0.5, cost_weight=0.05, min_utility=0.05)
        return {'kind': 'skip', 'reason': decision.reason,
                'uncertainty': 1.0, 'motion_mode': decision.motion_mode,
                'policy_metrics': decision.to_dict()}


def readj(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    out = ROOT/'runs/native_presence_full'
    out.mkdir(parents=True, exist_ok=True)
    path = out/'episodes.jsonl'
    if path.exists():
        raise FileExistsError(path)
    counts = Counter()
    with path.open('w') as log:
        for split in SPLITS:
            samples = readj(ROOT/'manifests'/f'{split}_samples.jsonl')
            first = {x['sample_id']:x for x in readj(ROOT/'runs'/split/'responses.jsonl')
                     if x['branch']=='first'}
            assert len(first) == len(samples)
            for sample in samples:
                sid = sample['sample_id']
                record = first[sid]
                answer = record['parsed_answer']
                if answer in ('yes','no'):
                    cached_raw = json.dumps({'answer': answer.upper(), 'confidence': 0.5})
                else:
                    cached_raw = record['raw_response']
                episode = run_arm(sample, 'agent', None, {'raw': cached_raw},
                                  VEGETATION_QUESTION, FIRST_VEGETATION,
                                  env_class=NativePresenceEnvironment)
                episode.pop('first_cache')
                episode['source_first_raw_sha256'] = hashlib.sha256(
                    record['raw_response'].encode()).hexdigest()
                episode['source_fixed_status'] = record['status']
                episode['source_fixed_parsed_answer'] = answer
                episode['split'] = split
                assert episode['rechecks'] == 0
                assert [x['image_key'] for x in episode['image_reads']] == ['high_marked']
                counts[(split, episode['stop_reason'])] += 1
                log.write(json.dumps(episode, ensure_ascii=False)+'\n')
            log.flush()
    (out/'summary.json').write_text(json.dumps({
        'n': sum(counts.values()), 'rechecks': 0,
        'stop_reasons_by_split': {s: dict(Counter({reason:n for (split,reason),n in counts.items()
                                                   if split == s})) for s in SPLITS},
        'interpretation': 'original presence gate skips descent even at maximum uncertainty',
    }, indent=2)+'\n')
    print((out/'summary.json').read_text())


if __name__ == '__main__':
    main()
