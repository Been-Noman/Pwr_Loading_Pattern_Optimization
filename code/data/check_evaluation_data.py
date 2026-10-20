#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from geometry import canonical_d4_key, is_hard_feasible, normalized_peaking_objective, physics_violation_score, pkey, true_physics_feasible
from data_io import atomic_json, read_json, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--expected', type=int, default=10000)
    args = parser.parse_args()
    rows = read_json(args.dataset)
    exact = [pkey(r) for r in rows]
    d4 = [canonical_d4_key(r['pattern']) for r in rows]
    hard = np.asarray([is_hard_feasible(r['pattern']) for r in rows], dtype=bool)
    feasible = np.asarray([true_physics_feasible(r) for r in rows], dtype=bool)
    fa = np.asarray([float(r['peaking_assembly']) for r in rows])
    fq = np.asarray([float(r.get('peaking_pin_radial', r.get('peaking_pin'))) for r in rows])
    k = np.asarray([float(r['keff']) for r in rows])
    source = Counter(r.get('final_origin', 'unknown') for r in rows)
    report = {
        'dataset': str(Path(args.dataset).resolve()),
        'sha256': sha256_file(args.dataset),
        'rows': len(rows),
        'expected_rows': args.expected,
        'exact_unique': len(set(exact)),
        'd4_unique': len(set(d4)),
        'hard_feasible_rows': int(hard.sum()),
        'physics_feasible_rows': int(feasible.sum()),
        'physics_feasible_fraction': float(feasible.mean()),
        'source_counts': dict(source),
        'targets': {
            'keff': {'min': float(k.min()), 'mean': float(k.mean()), 'max': float(k.max())},
            'Fa': {'min': float(fa.min()), 'mean': float(fa.mean()), 'max': float(fa.max())},
            'Fq_r': {'min': float(fq.min()), 'mean': float(fq.mean()), 'max': float(fq.max())},
        },
        'physics_violation_score_quantiles': {
            str(q): float(np.quantile([physics_violation_score(r) for r in rows], q))
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
        },
        'normalized_peaking_objective_quantiles': {
            str(q): float(np.quantile([normalized_peaking_objective(r) for r in rows], q))
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
        },
    }
    failures = []
    if len(rows) != args.expected: failures.append('row count')
    if len(set(exact)) != len(rows): failures.append('exact duplicates')
    if len(set(d4)) != len(rows): failures.append('D4 duplicates')
    if not hard.all(): failures.append('hard constraints')
    if not np.isfinite(k).all() or not np.isfinite(fa).all() or not np.isfinite(fq).all(): failures.append('non-finite targets')
    report['failures'] = failures
    atomic_json(report, Path(args.dataset).with_name('final_tail_10000_audit.json'))
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit('Final dataset audit failed: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
