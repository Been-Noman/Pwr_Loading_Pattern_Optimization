#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from geometry import canonical_d4_key, is_hard_feasible, pkey, true_physics_feasible
from data_io import atomic_json, read_json, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='append', required=True, help='Existing constrained OpenMC merged JSON; repeat as needed.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--patterns-output', required=True)
    parser.add_argument('--expected', type=int, default=7500)
    args = parser.parse_args()

    merged = []
    source_summary = []
    for source_index, filename in enumerate(args.input):
        rows = read_json(filename)
        if not isinstance(rows, list):
            raise SystemExit(f'{filename}: expected a JSON list.')
        source_summary.append({'file': str(Path(filename).resolve()), 'rows': len(rows), 'sha256': sha256_file(filename)})
        for row in rows:
            item = dict(row)
            item['source_existing_file'] = Path(filename).name
            item['source_existing_file_index'] = source_index
            merged.append(item)

    if len(merged) != args.expected:
        raise SystemExit(f'Existing constrained bank has {len(merged)} rows; expected {args.expected}.')

    exact = [pkey(row) for row in merged]
    if len(set(exact)) != len(exact):
        raise SystemExit('Existing constrained OpenMC bank contains exact duplicate patterns.')

    bad = [i for i, row in enumerate(merged) if not is_hard_feasible(row['pattern'])]
    if bad:
        raise SystemExit(f'Existing constrained bank contains hard-infeasible rows: {bad[:20]}')

    for i, row in enumerate(merged):
        for field in ('keff', 'peaking_assembly'):
            if field not in row:
                raise SystemExit(f'Row {i} lacks {field}.')
        if 'peaking_pin' not in row and 'peaking_pin_radial' not in row:
            raise SystemExit(f'Row {i} lacks radial pin-peaking target.')
        row['legacy_id'] = int(row.get('id', i))
        row['id'] = i
        row['pattern_hash'] = __import__('geometry').pattern_hash(row['pattern'])
        row['d4_hash'] = __import__('geometry').d4_hash(row['pattern'])
        row['physics_feasible'] = bool(true_physics_feasible(row))

    d4_groups = [canonical_d4_key(row['pattern']) for row in merged]
    feasible_count = sum(bool(row['physics_feasible']) for row in merged)
    group_counts = Counter((row.get('solver', 'unknown'), bool(row.get('symmetry_enabled', False))) for row in merged)

    atomic_json(merged, args.output)
    pattern_rows = [
        {
            'id': i,
            'pattern': row['pattern'],
            'solver': row.get('solver'),
            'symmetry_enabled': row.get('symmetry_enabled'),
            'pattern_hash': row['pattern_hash'],
            'd4_hash': row['d4_hash'],
        }
        for i, row in enumerate(merged)
    ]
    atomic_json(pattern_rows, args.patterns_output)

    report = {
        'rows': len(merged),
        'exact_unique': len(set(exact)),
        'd4_unique': len(set(d4_groups)),
        'd4_equivalent_duplicates': len(merged) - len(set(d4_groups)),
        'physics_feasible': feasible_count,
        'physics_feasible_fraction': feasible_count / len(merged),
        'sources': source_summary,
        'generator_group_counts': {f'{k[0]}|symmetry={k[1]}': v for k, v in sorted(group_counts.items())},
        'output': str(Path(args.output).resolve()),
        'patterns_output': str(Path(args.patterns_output).resolve()),
    }
    atomic_json(report, Path(args.output).with_name('existing_7500_report.json'))
    print(report)


if __name__ == '__main__':
    main()
