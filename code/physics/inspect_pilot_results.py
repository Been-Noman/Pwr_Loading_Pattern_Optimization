#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('metrics_file')
a=p.parse_args()
m=json.loads(Path(a.metrics_file).read_text())
wall=float(m['total_wall_seconds_this_run'])
projected=wall*5.0
print(f'pilot_5_wall_hours={wall/3600:.3f}')
print(f'projected_25_sample_task_hours={projected/3600:.3f}')
print(f'12h_headroom_hours={(12*3600-projected)/3600:.3f}')
if projected > 12*3600:
    raise SystemExit('Projected 25-sample task exceeds 12h; do NOT submit production array yet.')
