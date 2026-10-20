#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from pattern_diversity import transform_diversity_model
from geometry import FA_LIMIT, FQR_LIMIT, KEFF_HI, KEFF_LO
from data_io import atomic_json, read_json

DEFAULT_TAIL = 1500
DEFAULT_UNCERTAIN = 625
DEFAULT_EXPLORATION = 375
MAX_PER_CLUSTER = 12
MAX_PER_GENERATOR_GROUP = 900


def load_scores(path):
    records = {}
    with Path(path).open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            cid = int(row['candidate_id'])
            records[cid] = {
                'candidate_id': cid,
                'mean_keff': float(row['mean_keff']), 'std_keff': float(row['std_keff']),
                'mean_Fa': float(row['mean_Fa']), 'std_Fa': float(row['std_Fa']),
                'mean_Fq_r': float(row['mean_Fq_r']), 'std_Fq_r': float(row['std_Fq_r']),
            }
    return records


def tail_score(record):
    k = record['mean_keff']
    kv = max(0.0, KEFF_LO - k) / 0.02 + max(0.0, k - KEFF_HI) / 0.02
    return 0.5 * record['mean_Fa'] / FA_LIMIT + 0.5 * record['mean_Fq_r'] / FQR_LIMIT + 4.0 * kv


def robust_uncertainty(records):
    matrix = np.asarray([[r['std_keff'], r['std_Fa'], r['std_Fq_r']] for r in records], dtype=float)
    scale = np.median(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, np.maximum(np.mean(matrix, axis=0), 1e-12))
    return np.sum(matrix / scale, axis=1)


def group_key(row):
    return f"{row.get('solver')}|symmetry={bool(row.get('symmetry_enabled'))}"


def pick_ranked(ranked, n, chosen_ids, cluster_counts, group_counts, category):
    picked = []
    # First pass with diversity caps.
    for score, row in ranked:
        cid = int(row['candidate_id'])
        if cid in chosen_ids:
            continue
        cluster = int(row.get('structural_cluster', -1))
        group = group_key(row)
        if cluster_counts[cluster] >= MAX_PER_CLUSTER:
            continue
        if group_counts[group] >= MAX_PER_GENERATOR_GROUP:
            continue
        picked.append((score, row, category))
        chosen_ids.add(cid)
        cluster_counts[cluster] += 1
        group_counts[group] += 1
        if len(picked) == n:
            return picked
    # Second pass relaxes caps only if the configured mix cannot otherwise be filled.
    for score, row in ranked:
        cid = int(row['candidate_id'])
        if cid in chosen_ids:
            continue
        cluster = int(row.get('structural_cluster', -1))
        group = group_key(row)
        picked.append((score, row, category + '_cap_relaxed'))
        chosen_ids.add(cid)
        cluster_counts[cluster] += 1
        group_counts[group] += 1
        if len(picked) == n:
            return picked
    raise RuntimeError(f'Could only pick {len(picked)}/{n} for {category}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-bank', required=True)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--labeled-dataset', required=True)
    parser.add_argument('--diversity-model', required=True)
    parser.add_argument('--history', required=True)
    parser.add_argument('--wave', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tail-count', type=int, default=DEFAULT_TAIL)
    parser.add_argument('--uncertain-count', type=int, default=DEFAULT_UNCERTAIN)
    parser.add_argument('--exploration-count', type=int, default=DEFAULT_EXPLORATION)
    args = parser.parse_args()

    total = args.tail_count + args.uncertain_count + args.exploration_count
    if total != 2500:
        raise SystemExit(f'Acquisition components must total 2500, got {total}.')

    bank = read_json(args.candidate_bank)
    by_id = {int(r['candidate_id']): r for r in bank}
    scores = load_scores(args.scores)
    if set(by_id) != set(scores):
        raise SystemExit('Candidate bank and score IDs do not match.')

    history_path = Path(args.history)
    if history_path.exists():
        history = read_json(history_path)
    else:
        history = {'selected_candidate_ids': [], 'waves': []}
    previously_selected = {int(x) for x in history.get('selected_candidate_ids', [])}
    if len(previously_selected) != len(history.get('selected_candidate_ids', [])):
        raise SystemExit('Selection history contains duplicate candidate IDs.')

    available = []
    for cid, row in by_id.items():
        if cid in previously_selected:
            continue
        item = dict(row)
        item.update(scores[cid])
        item['tail_score'] = float(tail_score(item))
        available.append(item)
    if len(available) < 2500:
        raise SystemExit(f'Only {len(available)} un-simulated candidates remain.')

    # Map the already labelled OpenMC set into the same structural clusters. Exploration
    # then targets clusters with less existing physics coverage.
    diversity_model = joblib.load(args.diversity_model)
    labeled = read_json(args.labeled_dataset)
    labeled_patterns = np.asarray([r['pattern'] for r in labeled], dtype=np.int8)
    labeled_clusters, _ = transform_diversity_model(diversity_model, labeled_patterns)
    labeled_cluster_counts = Counter(map(int, labeled_clusters))

    chosen_ids = set(previously_selected)
    wave_cluster_counts = Counter()
    wave_group_counts = Counter()

    # 1) Exploitation: lowest predicted peaking objective, with a soft keff-range penalty.
    tail_ranked = sorted(
        ((float(r['tail_score']), r) for r in available),
        key=lambda item: (item[0], int(item[1]['candidate_id'])),
    )
    selected = pick_ranked(
        tail_ranked, args.tail_count, chosen_ids, wave_cluster_counts, wave_group_counts, 'tail'
    )

    # 2) Epistemic uncertainty only within the promising half of the current predicted tail.
    remaining = [r for r in available if int(r['candidate_id']) not in chosen_ids]
    cutoff = float(np.quantile([r['tail_score'] for r in available], 0.50))
    uncertainty_pool = [r for r in remaining if r['tail_score'] <= cutoff]
    if len(uncertainty_pool) < args.uncertain_count:
        uncertainty_pool = remaining
    unc = robust_uncertainty(uncertainty_pool)
    uncertainty_ranked = sorted(
        ((-float(u), r) for u, r in zip(unc, uncertainty_pool)),
        key=lambda item: (item[0], float(item[1]['tail_score']), int(item[1]['candidate_id'])),
    )
    selected.extend(pick_ranked(
        uncertainty_ranked, args.uncertain_count, chosen_ids,
        wave_cluster_counts, wave_group_counts, 'tail_uncertainty'
    ))

    # 3) Exploration: under-observed structural clusters, while rejecting the worst 20%
    # of predicted tail scores so the wave remains relevant to the optimization objective.
    remaining = [r for r in available if int(r['candidate_id']) not in chosen_ids]
    tail80 = float(np.quantile([r['tail_score'] for r in available], 0.80))
    exploration_pool = [r for r in remaining if r['tail_score'] <= tail80]
    if len(exploration_pool) < args.exploration_count:
        exploration_pool = remaining

    def exploration_priority(row):
        cluster = int(row.get('structural_cluster', -1))
        covered = labeled_cluster_counts[cluster] + wave_cluster_counts[cluster]
        # Lower is better: under-covered clusters first; within a cluster prefer an edge
        # representative, then better predicted tail.
        return (
            covered,
            -float(row.get('cluster_distance', 0.0)),
            float(row['tail_score']),
            int(row['candidate_id']),
        )

    exploration_ranked = [(0.0, r) for r in sorted(exploration_pool, key=exploration_priority)]
    selected.extend(pick_ranked(
        exploration_ranked, args.exploration_count, chosen_ids,
        wave_cluster_counts, wave_group_counts, 'structural_exploration'
    ))

    if len(selected) != 2500:
        raise SystemExit(f'Wave selection produced {len(selected)} rows, expected 2500.')

    output_rows = []
    category_counts = Counter()
    for wave_id, (ranking_score, row, category) in enumerate(selected):
        item = {k: v for k, v in row.items() if k not in {
            'mean_keff', 'std_keff', 'mean_Fa', 'std_Fa', 'mean_Fq_r', 'std_Fq_r', 'tail_score'
        }}
        item['id'] = wave_id  # OpenMC runner consumes 0..2499 IDs for each wave.
        item['candidate_id'] = int(row['candidate_id'])
        item['active_learning_wave'] = int(args.wave)
        item['acquisition_category'] = category
        item['acquisition_tail_score'] = float(row['tail_score'])
        item['ensemble_mean_keff'] = float(row['mean_keff'])
        item['ensemble_std_keff'] = float(row['std_keff'])
        item['ensemble_mean_Fa'] = float(row['mean_Fa'])
        item['ensemble_std_Fa'] = float(row['std_Fa'])
        item['ensemble_mean_Fq_r'] = float(row['mean_Fq_r'])
        item['ensemble_std_Fq_r'] = float(row['std_Fq_r'])
        output_rows.append(item)
        category_counts[category] += 1

    atomic_json(output_rows, args.output)
    history['selected_candidate_ids'] = sorted(chosen_ids)
    history.setdefault('waves', []).append({
        'wave': int(args.wave),
        'selected_count': 2500,
        'categories': dict(category_counts),
        'generator_groups': dict(wave_group_counts),
        'structural_clusters_used': len(wave_cluster_counts),
        'labeled_dataset_rows_before_wave': len(labeled),
        'output': str(Path(args.output).resolve()),
    })
    atomic_json(history, history_path)
    report = {
        'wave': args.wave,
        'selected': len(output_rows),
        'previously_selected': len(previously_selected),
        'remaining_after_selection': len(bank) - len(chosen_ids),
        'category_counts': dict(category_counts),
        'generator_group_counts': dict(wave_group_counts),
        'structural_clusters_used': len(wave_cluster_counts),
        'tail_score_min': min(r['acquisition_tail_score'] for r in output_rows),
        'tail_score_median': float(np.median([r['acquisition_tail_score'] for r in output_rows])),
        'tail_score_max': max(r['acquisition_tail_score'] for r in output_rows),
        'principle': 'CNN predictions prioritize where to spend OpenMC budget; no candidate is treated as physically feasible until OpenMC labels it.',
    }
    atomic_json(report, Path(args.output).with_name('selection_report.json'))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
