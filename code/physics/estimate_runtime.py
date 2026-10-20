#!/usr/bin/env python3
import argparse,re,subprocess
p=argparse.ArgumentParser(); p.add_argument('job_ids',nargs='+'); a=p.parse_args(); grand=0.0
for requested in a.job_ids:
    out=subprocess.check_output(['sacct','-X','-j',str(requested),'-n','-P','--format=JobIDRaw,State,ElapsedRaw,AllocTRES%300'],text=True)
    records=[]
    for line in out.splitlines():
        f=line.split('|')
        if len(f)>=4 and (f[0]==str(requested) or re.fullmatch(re.escape(str(requested))+r'_\d+',f[0])): records.append(f)
    arr=[r for r in records if '_' in r[0]]; chosen=arr or [r for r in records if r[0]==str(requested)]
    if not chosen: raise SystemExit(f'No records for {requested}')
    subtotal=0.0
    for jid,state,elapsed,tres,*_ in chosen:
        m=re.search(r'(?:^|,)billing=([0-9.]+)',tres)
        if not m: raise SystemExit(f'No billing TRES for {jid}: {tres}')
        cost=int(elapsed or 0)*float(m.group(1))/3600; subtotal+=cost
        print(f'job={jid} state={state} elapsed={elapsed}s billing={m.group(1)}/h cost={cost:.3f} billed-h')
    print(f'SUBTOTAL {requested}={subtotal:.3f} billed-h'); grand+=subtotal
print(f'TOTAL={grand:.3f} billed-h')
