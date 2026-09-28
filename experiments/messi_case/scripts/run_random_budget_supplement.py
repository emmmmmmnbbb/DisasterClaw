"""Frozen random-budget controller replay; no truth is read during inference."""
import argparse, json, os, shutil
from pathlib import Path
import run_single_temporal_agent as m

ROOT = m.ROOT
OUT = ROOT / 'supplement_random_budget'
def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out-dir', type=Path, default=ROOT/'supplement_random_budget_gpu')
    args = ap.parse_args()
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = ROOT/'supplement_random_budget/selection.json'
    target = OUT/'selection.json'
    if not target.exists(): shutil.copyfile(frozen, target)
    assert m.sha(target) == m.sha(frozen), 'Selection differs from frozen manifest'
    import torch
    torch.set_num_threads(4)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this process; no CPU fallback performed')
    selection = json.loads((OUT/'selection.json').read_text())
    selected = {tuple(x) for items in selection['selected'].values() for x in items}
    samples, high, triggered = {}, {}, set()
    for split in ('full_descend','full_train_paths','full_test_paths'):
        path = ROOT/'manifests'/f'{split}_samples.jsonl'
        assert m.sha(path) == selection['manifest_sha256'][str(path.relative_to(m.REPO))]
        for row in rows(path): samples[split,row['sample_id']] = row
        for row in rows(ROOT/'runs'/f'single_temporal_{split}'/'episodes.jsonl'):
            key = split,row['sample_id']
            if row['arm']=='hold': high[key]=row['events'][0]['prediction']
            elif row['rechecks']: triggered.add(key)
    keys = sorted(selected | triggered)
    config={'selection_sha256':m.sha(OUT/'selection.json'), 'device':args.device,'threads':4,
            'model_provenance':m.model_provenance(), 'n_unique_selected':len(selected),
            'n_unique_with_original_triggered':len(keys),
            'method':'One controller episode per unique selected region; deterministic per-region outcomes reused across frozen seeds. Shared archived high prediction. Original triggered cohort re-inferred on same device as random comparator.',
            'source_sha256':m.sha(Path(m.__file__))}
    cfg=OUT/'inference_config.json'
    if cfg.exists(): assert json.loads(cfg.read_text())==config
    else: cfg.write_text(json.dumps(config,indent=2)+'\n')
    log=OUT/'episodes.jsonl'
    done={(r['split'],r['sample_id']) for r in rows(log)} if log.exists() else set()
    original=json.loads((ROOT/'runs/single_temporal_full_descend/config.json').read_text())
    assert config['model_provenance'] == original['model_provenance'], 'Model provenance changed'
    models=m.load_models(args.device)
    with log.open('a') as f:
        for index,key in enumerate(keys,1):
            if key in done: continue
            row=m.make_episode(samples[key], 'agent', models, first_prediction=high[key],
                               random_selected=True if key in selected else None)
            assert row['rechecks']==1 and len(row['trajectory'])==2
            row.update(split=key[0],supplement_policy='random_selected' if key in selected else 'original_triggered_cpu_check')
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            print(f'{index}/{len(keys)} {key} elapsed={row["elapsed_s"]}',flush=True)
if __name__=='__main__': main()
