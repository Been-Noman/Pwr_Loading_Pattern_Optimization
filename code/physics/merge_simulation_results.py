#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from geometry import canonical_d4_key, is_hard_feasible, pattern_hash, pkey, true_physics_feasible
from data_io import atomic_json, read_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave-dir', required=True)
    parser.add_argument('--selected-patterns', required=True)
    parser.add_argument('--previous-labeled', required=True)
    parser.add_argument('--output-labeled', required=True)
    parser.add_argument('--expected-wave', type=int, default=2500)
    parser.add_argument('--chunk-size', type=int, default=25)
    parser.add_argument('--wave', type=int, required=True)
    args = parser.parse_args()

    run = Path(args.wave_dir)
    selected = read_json(args.selected_patterns)
    if len(selected) != args.expected_wave:
        raise SystemExit(f'Selected wave has {len(selected)} rows, expected {args.expected_wave}.')
    selected_by_id = {int(r['id']): r for r in selected}
    if set(selected_by_id) != set(range(args.expected_wave)):
        raise SystemExit('Selected wave IDs must be exactly 0..2499.')

    rows = []
    missing_files = []
    invalid_json = []
    n_chunks = (args.expected_wave + args.chunk_size - 1) // args.chunk_size
    for chunk in range(n_chunks):
        path = run / 'chunks' / f'chunk_{chunk}.jsonl'
        if not path.exists():
            missing_files.append(str(path))
            continue
        with path.open(encoding='utf-8') as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    invalid_json.append(f'{path}:{line_no}: {exc}')
    if invalid_json:
        raise SystemExit('\n'.join(invalid_json))

    by_id = {}
    duplicate_ids = []
    for row in rows:
        sid = int(row['id'])
        if sid in by_id:
            duplicate_ids.append(sid)
        else:
            by_id[sid] = row

    expected_ids = set(range(args.expected_wave))
    missing_ids = sorted(expected_ids - set(by_id))
    extra_ids = sorted(set(by_id) - expected_ids)
    wrong_pattern = []
    hard_fail = []
    for sid, row in by_id.items():
        if sid in selected_by_id and pkey(row) != pkey(selected_by_id[sid]):
            wrong_pattern.append(sid)
        if not is_hard_feasible(row['pattern']):
            hard_fail.append(sid)
        for field in ('keff', 'peaking_assembly'):
            if field not in row:
                raise SystemExit(f'OpenMC row {sid} lacks {field}.')
        if 'peaking_pin' not in row and 'peaking_pin_radial' not in row:
            raise SystemExit(f'OpenMC row {sid} lacks radial pin peaking.')

    report = {
        'wave': args.wave,
        'expected': args.expected_wave,
        'rows_read': len(rows),
        'unique_ids': len(by_id),
        'duplicate_ids': sorted(set(duplicate_ids)),
        'missing_chunk_files': missing_files,
        'missing_ids': missing_ids,
        'extra_ids': extra_ids,
        'wrong_pattern_ids': wrong_pattern,
        'hard_constraint_failures': hard_fail,
    }
    atomic_json(report, run / 'wave_validation_summary.json')
    if any((duplicate_ids, missing_files, missing_ids, extra_ids, wrong_pattern, hard_fail)):
        if missing_ids:
            rerun = sorted({sid // args.chunk_size for sid in missing_ids})
            print('Array tasks to rerun:', ','.join(map(str, rerun)))
        raise SystemExit('OpenMC wave validation failed. See wave_validation_summary.json.')

    wave_merged = []
    for sid in range(args.expected_wave):
        result = dict(by_id[sid])
        source = selected_by_id[sid]
        # Retain acquisition provenance even if the OpenMC runner only preserved physics fields.
        for key in (
            'candidate_id', 'active_learning_wave', 'acquisition_category', 'acquisition_tail_score',
            'ensemble_mean_keff', 'ensemble_std_keff', 'ensemble_mean_Fa', 'ensemble_std_Fa',
            'ensemble_mean_Fq_r', 'ensemble_std_Fq_r', 'structural_cluster', 'cluster_distance',
            'solver', 'symmetry_enabled', 'qubo_energy', 'symmetry_mismatches', 'solver_seed',
        ):
            if key in source:
                result[key] = source[key]
        result['wave_local_id'] = sid
        result['pattern_hash'] = pattern_hash(result['pattern'])
        result['d4_hash'] = __import__('geometry').d4_hash(result['pattern'])
        result['physics_feasible'] = bool(true_physics_feasible(result))
        wave_merged.append(result)

    atomic_json(wave_merged, run / f'openmc_wave_{args.wave}_merged.json')
    write_jsonl(wave_merged, run / f'openmc_wave_{args.wave}_merged.jsonl')

    previous = read_json(args.previous_labeled)
    previous_exact = {pkey(r) for r in previous}
    previous_d4 = {canonical_d4_key(r['pattern']) for r in previous}
    new_exact = [pkey(r) for r in wave_merged]
    new_d4 = [canonical_d4_key(r['pattern']) for r in wave_merged]
    if len(set(new_exact)) != len(new_exact):
        raise SystemExit('Wave contains exact duplicate patterns.')
    if len(set(new_d4)) != len(new_d4):
        raise SystemExit('Wave contains D4-equivalent duplicate patterns.')
    if set(new_exact) & previous_exact:
        raise SystemExit('Wave repeats a previously OpenMC-labelled exact pattern.')
    if set(new_d4) & previous_d4:
        raise SystemExit('Wave repeats a previously OpenMC-labelled D4-equivalent pattern.')

    cumulative = [dict(r) for r in previous] + wave_merged
    for i, row in enumerate(cumulative):
        row['id'] = i
    atomic_json(cumulative, args.output_labeled)
    write_jsonl(cumulative, Path(args.output_labeled).with_suffix('.jsonl'))

    report.update({
        'wave_physics_feasible': sum(bool(r['physics_feasible']) for r in wave_merged),
        'wave_physics_feasible_fraction': sum(bool(r['physics_feasible']) for r in wave_merged) / len(wave_merged),
        'previous_labeled_rows': len(previous),
        'cumulative_labeled_rows': len(cumulative),
        'output_labeled': str(Path(args.output_labeled).resolve()),
    })
    atomic_json(report, run / 'wave_validation_summary.json')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
