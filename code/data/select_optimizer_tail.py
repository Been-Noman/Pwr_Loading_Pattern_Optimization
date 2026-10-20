#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

from pattern_diversity import transform_diversity_model
from geometry import (
    canonical_d4_key, normalized_peaking_objective, physics_violation_score,
    true_physics_feasible,
)
from data_io import atomic_json, read_json, write_jsonl


def d4_representative_rank(row):
    # Prefer genuinely feasible, then smaller physical violation, then lower peaking objective.
    return (
        0 if true_physics_feasible(row) else 1,
        float(physics_violation_score(row)),
        float(normalized_peaking_objective(row)),
        str(row.get('pattern_hash', '')),
    )


def cluster_balanced_ranked_pick(ranked_rows, target, cluster_key='final_structural_cluster', base_cluster_counts=None):
    if target <= 0:
        return []
    counts = Counter(base_cluster_counts or {})
    by_cluster_available = Counter(int(r[cluster_key]) for r in ranked_rows)
    n_clusters = max(1, len(by_cluster_available))
    # Soft cap: enough room for the requested target while still preventing one cluster
    # from dominating. If necessary a second pass relaxes the cap.
    cap = max(3, int(np.ceil(target / n_clusters)) * 3)
    chosen = []
    chosen_ids = set()
    for row in ranked_rows:
        c = int(row[cluster_key])
        if counts[c] >= cap:
            continue
        chosen.append(row)
        chosen_ids.add(id(row))
        counts[c] += 1
        if len(chosen) == target:
            return chosen
    for row in ranked_rows:
        if id(row) in chosen_ids:
            continue
        chosen.append(row)
        if len(chosen) == target:
            return chosen
    raise RuntimeError(f'Could select only {len(chosen)}/{target}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled-17500', required=True)
    parser.add_argument('--diversity-model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', type=int, default=10000)
    parser.add_argument('--diversity-fraction-of-nonfeasible-fill', type=float, default=0.20)
    args = parser.parse_args()

    rows = read_json(args.labeled_17500)
    if len(rows) != 17500:
        raise SystemExit(f'Expected 17,500 cumulative OpenMC rows, found {len(rows)}.')

    # D4-equivalent rows are redundant for CNN training. Keep the physically more useful
    # representative; source age (old/new) is intentionally not part of the ranking.
    best_by_d4 = {}
    d4_members = defaultdict(list)
    for row in rows:
        key = canonical_d4_key(row['pattern'])
        d4_members[key].append(row)
        prev = best_by_d4.get(key)
        if prev is None or d4_representative_rank(row) < d4_representative_rank(prev):
            best_by_d4[key] = row
    unique_rows = [dict(r) for r in best_by_d4.values()]
    if len(unique_rows) < args.target:
        raise SystemExit(f'Only {len(unique_rows)} D4-unique labelled rows; cannot select {args.target}.')

    model = joblib.load(args.diversity_model)
    patterns = np.asarray([r['pattern'] for r in unique_rows], dtype=np.int8)
    clusters, distances = transform_diversity_model(model, patterns)
    for row, cluster, distance in zip(unique_rows, clusters, distances):
        row['final_structural_cluster'] = int(cluster)
        row['final_cluster_distance'] = float(distance)
        row['physics_feasible'] = bool(true_physics_feasible(row))
        row['physics_violation_score'] = float(physics_violation_score(row))
        row['normalized_peaking_objective'] = float(normalized_peaking_objective(row))

    feasible = [r for r in unique_rows if r['physics_feasible']]
    infeasible = [r for r in unique_rows if not r['physics_feasible']]
    feasible.sort(key=lambda r: (r['normalized_peaking_objective'], r['final_cluster_distance'], str(r.get('pattern_hash', ''))))

    if len(feasible) >= args.target:
        selected = cluster_balanced_ranked_pick(feasible, args.target)
        selection_mode = 'feasible_only_more_than_target'
        near_count = 0
        diversity_count = 0
    else:
        selected = list(feasible)
        remaining = args.target - len(selected)
        diversity_count = int(round(remaining * args.diversity_fraction_of_nonfeasible_fill))
        near_count = remaining - diversity_count

        # Most of the non-feasible fill comes from the smallest true OpenMC constraint
        # violations, i.e. the just-outside-boundary / near-tail region.
        near_ranked = sorted(
            infeasible,
            key=lambda r: (
                r['physics_violation_score'],
                r['normalized_peaking_objective'],
                -r['final_cluster_distance'],
                str(r.get('pattern_hash', '')),
            ),
        )
        base_counts = Counter(int(r['final_structural_cluster']) for r in selected)
        near_selected = cluster_balanced_ranked_pick(near_ranked, near_count, base_cluster_counts=base_counts)
        selected.extend(near_selected)

        used = {canonical_d4_key(r['pattern']) for r in selected}
        remainder = [r for r in infeasible if canonical_d4_key(r['pattern']) not in used]
        current_counts = Counter(int(r['final_structural_cluster']) for r in selected)
        # Diversity fill prioritizes structurally underrepresented regions but still prefers
        # smaller physics violations when coverage ties.
        remainder.sort(key=lambda r: (
            current_counts[int(r['final_structural_cluster'])],
            -r['final_cluster_distance'],
            r['physics_violation_score'],
            r['normalized_peaking_objective'],
            str(r.get('pattern_hash', '')),
        ))
        diversity_selected = cluster_balanced_ranked_pick(
            remainder, diversity_count, base_cluster_counts=current_counts
        )
        selected.extend(diversity_selected)
        selection_mode = 'all_feasible_plus_near_boundary_and_structural_diversity'

    if len(selected) != args.target:
        raise SystemExit(f'Final selection has {len(selected)} rows; expected {args.target}.')
    d4 = [canonical_d4_key(r['pattern']) for r in selected]
    if len(set(d4)) != len(selected):
        raise SystemExit('Final selected dataset contains D4-equivalent duplicates.')

    old_count = 0
    new_count = 0
    for i, row in enumerate(selected):
        row['id'] = i
        row['final_tail_dataset'] = True
        row['final_selection_rank'] = i
        if 'source_existing_file' in row:
            old_count += 1
            row['final_origin'] = 'existing_7500'
        else:
            new_count += 1
            row['final_origin'] = f"active_wave_{row.get('active_learning_wave', 'unknown')}"

    atomic_json(selected, args.output)
    write_jsonl(selected, Path(args.output).with_suffix('.jsonl'))

    all_source_counts = Counter(
        'existing_7500' if 'source_existing_file' in r else f"active_wave_{r.get('active_learning_wave', 'unknown')}"
        for r in selected
    )
    report = {
        'input_rows': len(rows),
        'input_d4_unique_rows': len(unique_rows),
        'input_d4_redundant_rows_removed': len(rows) - len(unique_rows),
        'target_rows': args.target,
        'input_physics_feasible_d4_unique': len(feasible),
        'selected_physics_feasible': sum(bool(r['physics_feasible']) for r in selected),
        'selection_mode': selection_mode,
        'near_boundary_nonfeasible_count_requested': near_count,
        'structural_diversity_fill_count_requested': diversity_count,
        'existing_old_rows_selected': old_count,
        'new_active_rows_selected': new_count,
        'source_counts': dict(all_source_counts),
        'selected_structural_clusters': len({int(r['final_structural_cluster']) for r in selected}),
        'rule_freeze_note': 'This 10,000-row selection is completed before the final D4-safe train/validation/calibration/test split. The final test set is not used for model fitting, seed selection, or uncertainty calibration.',
        'interpretation_note': 'Performance on the final test set estimates performance on this deliberately tail-enriched legal-pattern distribution, not on arbitrary random loading patterns.',
    }
    atomic_json(report, Path(args.output).with_name('final_tail_10000_selection_report.json'))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
