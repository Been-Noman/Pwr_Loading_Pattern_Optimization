#!/usr/bin/env python3
import argparse, json, os
from collections import Counter
from pathlib import Path
from qubo_and_reactor_model import validate_decoded_pattern

scratch = os.environ.get('SCRATCH', str(Path.home()))
default_run = str(Path(scratch)/'openmc2500/outputs/production_sa_sqa_2500')
p = argparse.ArgumentParser()
p.add_argument('--run-dir', default=default_run)
p.add_argument('--expected', type=int, default=2500)
p.add_argument('--chunk-size', type=int, default=25)
a = p.parse_args(); run = Path(a.run_dir)

bank = json.loads((run/'patterns_sa_sqa_2500.json').read_text())
bank_by_id = {int(r['id']): r for r in bank}
if len(bank_by_id) != a.expected:
    raise SystemExit(f'Pattern bank has {len(bank_by_id)} rows; expected {a.expected}')

rows=[]; invalid=[]
for chunk in range((a.expected+a.chunk_size-1)//a.chunk_size):
    path=run/'chunks'/f'chunk_{chunk}.jsonl'
    if not path.exists():
        print('MISSING FILE:', path)
        continue
    for n,line in enumerate(path.open(),1):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError as e: invalid.append(f'{path}:{n}: {e}')
if invalid: raise SystemExit('\n'.join(invalid))

by_id={}; duplicates=[]
for r in rows:
    sid=int(r['id'])
    if sid in by_id: duplicates.append(sid)
    else: by_id[sid]=r
expected=set(range(a.expected)); missing=sorted(expected-set(by_id)); extra=sorted(set(by_id)-expected)
wrong=[]; bad=[]
for sid,r in by_id.items():
    if sid in bank_by_id and r.get('pattern') != bank_by_id[sid].get('pattern'): wrong.append(sid)
    if not validate_decoded_pattern(r.get('pattern')): bad.append(sid)
summary={
    'expected':a.expected,'rows_read':len(rows),'unique_rows':len(by_id),
    'duplicates':sorted(set(duplicates)),'missing_ids':missing,'extra_ids':extra,
    'wrong_pattern_ids':wrong,'constraint_failure_ids':bad,
    'group_counts':{f'{s}|symmetry={y}':c for (s,y),c in Counter((r['solver'],bool(r['symmetry_enabled'])) for r in by_id.values()).items()}
}
(run/'validation_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if missing:
    print('Array tasks to rerun:', ','.join(map(str,sorted({i//a.chunk_size for i in missing}))))
if any([duplicates,missing,extra,wrong,bad]): raise SystemExit('Validation failed')

merged=[by_id[i] for i in range(a.expected)]
out=run/'pwr_sa_sqa_openmc_2500_merged.json'; tmp=out.with_suffix('.json.tmp')
with tmp.open('w') as f:
    json.dump(merged,f,indent=2); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
with (run/'pwr_sa_sqa_openmc_2500_merged.jsonl').open('w') as f:
    for r in merged: f.write(json.dumps(r,separators=(',',':'))+'\n')
print('Merged:',out)
