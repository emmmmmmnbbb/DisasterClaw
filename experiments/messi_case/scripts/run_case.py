"""Run isolated, matched two-turn A/B/C Qwen branches for a frozen manifest."""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'backend'))
from local_qwen_vl import LocalQwenVLBackend  # noqa: E402

FIRST = ('Inspect only the ground region inside the red rectangle. '
         'Does it contain visible tree or shrub canopy? '
         'Reply with exactly one word: YES, NO, or UNCERTAIN.')
FIRST_VEGETATION = FIRST.replace('visible tree or shrub canopy',
                                 'visible vegetation, including grass, trees, or shrubs')
SECOND = ('Reassess the same ground region using this second observation. '
          'Use the red rectangle if one is shown; otherwise the entire image is the region. '
          'Reply with exactly one word: YES, NO, or UNCERTAIN.')
MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
MAX_NEW_TOKENS = 16
TEMPERATURE = 0.0


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def parse(raw):
    m = re.fullmatch(r'\s*(YES|NO|UNCERTAIN)\s*[.!]?\s*', raw, re.I)
    return m.group(1).lower() if m else 'invalid'


def msg(prompt, image):
    return {'role':'user','content':[{'type':'image','image':image},
                                     {'type':'text','text':prompt}]}


def write(log, record):
    log.write(json.dumps(record,ensure_ascii=False)+'\n')
    log.flush()
    os.fsync(log.fileno())


def call(model, messages, sample, branch, log, config_hash):
    started = time.time()
    try:
        raw = model.infer(messages,max_new_tokens=MAX_NEW_TOKENS,
                          temperature=TEMPERATURE,seed=0)
        status, error = 'ok', None
    except Exception as exc:
        raw, status, error = '', 'run_failed', repr(exc)
    record = {'run_id':log.name,'sample_id':sample['sample_id'],'branch':branch,
              'model_id':MODEL,'model_revision':REVISION,'config_hash':config_hash,
              'prompt':messages[-1]['content'][-1]['text'],
              'input_image_sha256':[sample['image_sha256'][k] for k in
                  (['high_marked'] if branch=='first' else
                   ['high_marked',{'A':'high_marked','B':'high_crop','C':'low_crop'}[branch]])],
              'raw_response':raw,'parsed_answer':parse(raw) if status=='ok' else 'run_failed',
              'latency_s':round(time.time()-started,3),'status':status,'error':error,
              'retry_index':0}
    write(log,record)
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',choices=['pilot','formal','exhaustive_descend','exhaustive_paths',
                                       'full_descend','full_train_paths','full_test_paths'],required=True)
    ap.add_argument('--device',default='cuda:1')
    ap.add_argument('--limit',type=int)
    ap.add_argument('--offset',type=int,default=0)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    manifest=ROOT/'manifests'/f'{args.split}_samples.jsonl'
    samples=[json.loads(x) for x in manifest.read_text().splitlines()]
    samples=samples[args.offset:]
    if args.limit: samples=samples[:args.limit]
    first_prompt = FIRST_VEGETATION if args.split in {'exhaustive_descend','exhaustive_paths',
                                                      'full_descend','full_train_paths','full_test_paths'} else FIRST
    config={'first_prompt':first_prompt,'second_prompt':SECOND,'model':MODEL,'revision':REVISION,
            'temperature':TEMPERATURE,'max_new_tokens':MAX_NEW_TOKENS,'seed':0,
            'device':args.device,'input_size_high':(1200,800),'input_size_crop':(800,800),
            'parse':'exact YES/NO/UNCERTAIN; invalid counted incorrect','retry':0,
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    out=ROOT/'runs'/args.split
    out.mkdir(parents=True,exist_ok=True)
    config_hash=digest(config)
    config_path=out/'config.json'
    log_path=out/(f'responses_{args.offset}.jsonl' if args.offset else 'responses.jsonl')
    completed={}
    if log_path.exists():
        if not args.resume: raise FileExistsError(log_path)
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError('Cannot resume with changed config')
        for line in log_path.read_text().splitlines():
            record=json.loads(line)
            key=(record['sample_id'],record['branch'])
            if key in completed: raise RuntimeError(f'Duplicate call: {key}')
            completed[key]=record
    else:
        config_path.write_text(json.dumps(config,indent=2)+'\n')
    model=LocalQwenVLBackend(MODEL,device=args.device,top_p=1.0,repetition_penalty=1.0)
    model.load()
    try:
        with log_path.open('a') as log:
            for sample in samples:
                images=sample['images']
                first=[msg(first_prompt,images['high_marked'])]
                one=completed.get((sample['sample_id'],'first'))
                if one is None:
                    one=call(model,first,sample,'first',log,config_hash)
                if one['status']!='ok':
                    for branch in 'ABC':
                        if (sample['sample_id'],branch) in completed: continue
                        write(log,{'sample_id':sample['sample_id'],'branch':branch,
                                   'status':'skipped_first_failed','parsed_answer':'run_failed'})
                    continue
                for branch,key in [('A','high_marked'),('B','high_crop'),('C','low_crop')]:
                    if (sample['sample_id'],branch) in completed: continue
                    history=first+[{'role':'assistant','content':one['raw_response']},
                                   msg(SECOND,images[key])]
                    call(model,history,sample,branch,log,config_hash)
                print(sample['sample_id'],'done',flush=True)
    finally:
        model.unload()


if __name__=='__main__': main()
