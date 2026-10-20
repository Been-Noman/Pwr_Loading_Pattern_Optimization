#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

from pattern_diversity import fit_diversity_model, select_cluster_representatives, transform_diversity_model
from geometry import canonical_d4_key, d4_hash, is_hard_feasible, pattern_hash, pkey
from data_io import atomic_json, read_json

GROUP_ORDER = [('sa', True), ('sa', False), ('sqa', True), ('sqa', False)]


def group_name(key):
    return f'{key[0]}|symmetry={bool(key[1])}'


def row_rank(row):
    # Only a deterministic tie-breaker among duplicate/equivalent solver records.
    return (
        float(row.get('qubo_energy', np.inf)),
        int(row.get('symmetry_mismatches', 10**9)),
        int(row.get('solver_seed', 10**18)),
        int(row.get('generation_shard', 10**9)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-root', required=True)
    parser.add_argument('--n-shards', type=int, default=100)
    parser.add_argument('--exclude-pattern-file', required=True)
    parser.add_argument('--target-total', type=int, default=50000)
    parser.add_argument('--target-per-group', type=int, default=12500)
    parser.add_argument('--clusters', type=int, default=512)
    parser.add_argument('--projection-components', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--model-output', required=True)
    args = parser.parse_args()

    if args.target_total != 4 * args.target_per_group:
        raise SystemExit('target-total must equal 4 * target-per-group.')

    old_rows = read_json(args.exclude_pattern_file)
    old_exact = {pkey(r) for r in old_rows}
    old_d4 = {canonical_d4_key(r['pattern']) for r in old_rows}

    candidates = defaultdict(list)
    raw_rows = 0
    shard_summary = []
    for shard in range(args.n_shards):
        path = Path(args.shard_root) / f'shard_{shard}' / f'candidate_shard_{shard}.json'
        if not path.exists():
            raise SystemExit(f'Missing shard output: {path}')
        rows = read_json(path)
        raw_rows += len(rows)
        local_exact = set()
        bad = 0
        for row in rows:
            key = pkey(row)
            if key in local_exact:
                raise SystemExit(f'Shard {shard} has an exact duplicate.')
            local_exact.add(key)
            if not is_hard_feasible(key):
                bad += 1
                continue
            if key in old_exact:
                continue
            dkey = canonical_d4_key(key)
            if dkey in old_d4:
                continue
            g = (row.get('solver'), bool(row.get('symmetry_enabled')))
            if g not in GROUP_ORDER:
                raise SystemExit(f'Unexpected generator group {g}.')
            item = dict(row)
            item['pattern_hash'] = pattern_hash(key)
            item['d4_hash'] = d4_hash(key)
            candidates[g].append(item)
        shard_summary.append({'shard': shard, 'rows': len(rows), 'hard_constraint_failures': bad})

    # D4-deduplicate inside each generator group. If multiple solver records represent the
    # same physical pattern under D4, keep the deterministic best solver record only.
    group_unique = {}
    for group in GROUP_ORDER:
        best = {}
        for row in candidates[group]:
            dkey = canonical_d4_key(row['pattern'])
            prev = best.get(dkey)
            if prev is None or row_rank(row) < row_rank(prev):
                best[dkey] = row
        group_unique[group] = list(best.values())

    # Fit one global diversity model on one representative per D4-equivalence class.
    global_best = {}
    for group in GROUP_ORDER:
        for row in group_unique[group]:
            dkey = canonical_d4_key(row['pattern'])
            prev = global_best.get(dkey)
            if prev is None or row_rank(row) < row_rank(prev):
                global_best[dkey] = row
    global_rows = list(global_best.values())
    if len(global_rows) < args.target_total:
        raise SystemExit(
            f'Only {len(global_rows)} globally D4-unique candidates remain after old-bank exclusion; '
            f'need at least {args.target_total}. Generate more shards.'
        )

    patterns = np.asarray([r['pattern'] for r in global_rows], dtype=np.int8)
    model, global_clusters, global_distances = fit_diversity_model(
        patterns,
        n_clusters=args.clusters,
        projection_components=args.projection_components,
        seed=args.seed,
    )
    for row, cluster, distance in zip(global_rows, global_clusters, global_distances):
        row['structural_cluster'] = int(cluster)
        row['cluster_distance'] = float(distance)
    cluster_by_d4 = {
        canonical_d4_key(row['pattern']): (row['structural_cluster'], row['cluster_distance'])
        for row in global_rows
    }

    for group in GROUP_ORDER:
        group_patterns = np.asarray([r['pattern'] for r in group_unique[group]], dtype=np.int8)
        if len(group_patterns):
            clusters, distances = transform_diversity_model(model, group_patterns)
            for row, cluster, distance in zip(group_unique[group], clusters, distances):
                row['structural_cluster'] = int(cluster)
                row['cluster_distance'] = float(distance)

    # Select scarcest groups first so cross-group D4 collisions cannot starve a sparse group.
    selection_order = sorted(GROUP_ORDER, key=lambda g: (len(group_unique[g]), GROUP_ORDER.index(g)))
    selected_by_group = {}
    used_d4 = set()
    shortage = {}
    for group in selection_order:
        available = [r for r in group_unique[group] if canonical_d4_key(r['pattern']) not in used_d4]
        if len(available) < args.target_per_group:
            shortage[group_name(group)] = args.target_per_group - len(available)
            selected_by_group[group] = available
            used_d4.update(canonical_d4_key(r['pattern']) for r in available)
            continue
        chosen = select_cluster_representatives(available, args.target_per_group)
        selected_by_group[group] = chosen
        used_d4.update(canonical_d4_key(r['pattern']) for r in chosen)

    if shortage:
        report = {'shortage': shortage, 'unique_per_group': {group_name(g): len(group_unique[g]) for g in GROUP_ORDER}}
        atomic_json(report, Path(args.output_file).with_name('candidate_bank_failure_report.json'))
        raise SystemExit(f'Group shortage after global D4 deduplication: {shortage}')

    final = []
    for i in range(args.target_per_group):
        for group in GROUP_ORDER:
            row = dict(selected_by_group[group][i])
            row['candidate_id'] = len(final)
            row['id'] = len(final)
            row['candidate_bank_version'] = 1
            final.append(row)

    exact_keys = [pkey(r) for r in final]
    d4_keys = [canonical_d4_key(r['pattern']) for r in final]
    if len(final) != args.target_total:
        raise SystemExit('Final bank row count mismatch.')
    if len(set(exact_keys)) != len(final):
        raise SystemExit('Final bank contains exact duplicates.')
    if len(set(d4_keys)) != len(final):
        raise SystemExit('Final bank contains D4-equivalent duplicates.')
    if set(exact_keys) & old_exact:
        raise SystemExit('Final bank overlaps existing 7500 exactly.')
    if set(d4_keys) & old_d4:
        raise SystemExit('Final bank overlaps existing 7500 under D4 symmetry.')
    if any(not is_hard_feasible(r['pattern']) for r in final):
        raise SystemExit('Final bank contains a hard-infeasible pattern.')

    atomic_json(final, args.output_file)
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)

    counts = Counter((r['solver'], bool(r['symmetry_enabled'])) for r in final)
    cluster_counts = Counter(int(r['structural_cluster']) for r in final)
    report = {
        'raw_rows': raw_rows,
        'n_shards': args.n_shards,
        'old_exact_count': len(old_exact),
        'old_d4_count': len(old_d4),
        'global_d4_unique_after_exclusion': len(global_rows),
        'group_d4_unique_after_exclusion': {group_name(g): len(group_unique[g]) for g in GROUP_ORDER},
        'selected_rows': len(final),
        'selected_exact_unique': len(set(exact_keys)),
        'selected_d4_unique': len(set(d4_keys)),
        'selected_group_counts': {group_name(g): counts[g] for g in GROUP_ORDER},
        'n_structural_clusters_used': len(cluster_counts),
        'cluster_size_min': min(cluster_counts.values()),
        'cluster_size_max': max(cluster_counts.values()),
        'cluster_size_median': float(np.median(list(cluster_counts.values()))),
        'diversity_method': {
            'categorical_geometry': 'sparse random projection of one-hot loading patterns; Euclidean distance is based on Hamming-equivalent one-hot representation',
            'explicit_features': 'zone/layer fuel occupancy, radial moments, adjacency pair frequencies, D4 symmetry tendency',
            'clusters': args.clusters,
            'projection_components': args.projection_components,
            'seed': args.seed,
        },
        'output_file': str(Path(args.output_file).resolve()),
        'model_output': str(Path(args.model_output).resolve()),
    }
    atomic_json(report, Path(args.output_file).with_name('candidate_bank_report.json'))
    print(report)


if __name__ == '__main__':
    main()
