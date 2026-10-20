#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from geometry import (
    FA_LIMIT, FQR_LIMIT, KEFF_HI, KEFF_LO, N_CELLS,
    canonical_d4_key, is_hard_feasible, patterns_to_onehot_grid, true_physics_feasible,
)
from data_io import atomic_json, read_json, sha256_file

SPLIT_NAMES = ('train', 'validation', 'calibration', 'test')
SPLIT_FRACTIONS = {'train': 0.70, 'validation': 0.075, 'calibration': 0.075, 'test': 0.15}
SPLIT_SEED = 42


def choose_groups_exact(group_to_indices, target_size, rng):
    target_size = int(target_size)
    if target_size == 0:
        return set()
    keys = list(group_to_indices)
    rng.shuffle(keys)
    reachable = {0}
    parent = {}
    for key in keys:
        group_size = len(group_to_indices[key])
        for total in sorted(tuple(reachable), reverse=True):
            new_total = total + group_size
            if new_total <= target_size and new_total not in reachable:
                reachable.add(new_total)
                parent[new_total] = (total, key)
        if target_size in reachable:
            break
    if target_size not in reachable:
        raise RuntimeError(f'No exact group-preserving subset of size {target_size} exists.')
    selected = set()
    total = target_size
    while total:
        previous, key = parent[total]
        selected.add(key)
        total = previous
    return selected


def balanced_integer_allocation(stratum_counts, split_sizes):
    row_totals = np.asarray(stratum_counts, dtype=int)
    col_totals = np.asarray(split_sizes, dtype=int)
    expected = np.outer(row_totals, col_totals) / float(row_totals.sum())
    allocation = np.floor(expected).astype(int)
    row_deficit = row_totals - allocation.sum(axis=1)
    col_deficit = col_totals - allocation.sum(axis=0)
    fractional = expected - np.floor(expected)
    while int(row_deficit.sum()) > 0:
        candidates = [
            (fractional[i, j], -i, -j, i, j)
            for i in range(len(row_totals)) if row_deficit[i] > 0
            for j in range(len(col_totals)) if col_deficit[j] > 0
        ]
        if not candidates:
            raise RuntimeError('Unable to balance exact stratum/split allocation.')
        _, _, _, i, j = max(candidates)
        allocation[i, j] += 1
        row_deficit[i] -= 1
        col_deficit[j] -= 1
    return allocation


def split_sizes_for_n(n):
    # Preserve the notebook's 70/7.5/7.5/15 proportions as closely as integer
    # counts allow, keeping the 15% test size fixed by rounding first.
    train = int(round(0.70 * n))
    test = int(round(0.15 * n))
    calibration = int(round(0.075 * n))
    validation = n - train - calibration - test
    return {'train': train, 'validation': validation, 'calibration': calibration, 'test': test}


def grouped_split(group_keys, strata, split_sizes, seed=SPLIT_SEED):
    stratum_names = tuple(sorted(set(map(str, strata))))
    group_to_indices = defaultdict(list)
    for index, key in enumerate(group_keys):
        group_to_indices[key].append(index)
    groups_by_stratum = {name: {} for name in stratum_names}
    for key, indices in group_to_indices.items():
        labels = {str(strata[index]) for index in indices}
        if len(labels) != 1:
            raise RuntimeError('A D4-equivalent group crosses physics strata. Inspect OpenMC labels.')
        groups_by_stratum[labels.pop()][key] = indices
    counts = [sum(len(v) for v in groups_by_stratum[name].values()) for name in stratum_names]
    allocation = balanced_integer_allocation(counts, [split_sizes[name] for name in SPLIT_NAMES])
    split_indices = {name: [] for name in SPLIT_NAMES}
    for si, stratum in enumerate(stratum_names):
        remaining = dict(groups_by_stratum[stratum])
        for split_index, split_name in enumerate(('validation', 'calibration', 'test'), start=1):
            target = int(allocation[si, split_index])
            rng = np.random.default_rng(int(seed) + 1009 * si + 97 * split_index)
            chosen = choose_groups_exact(remaining, target, rng)
            for key in chosen:
                split_indices[split_name].extend(remaining[key])
            remaining = {k: v for k, v in remaining.items() if k not in chosen}
        split_indices['train'].extend(i for indices in remaining.values() for i in indices)
    for name in SPLIT_NAMES:
        split_indices[name] = np.asarray(sorted(split_indices[name]), dtype=int)
        if len(split_indices[name]) != split_sizes[name]:
            raise RuntimeError(f'{name} split mismatch: {len(split_indices[name])} vs {split_sizes[name]}')
    return split_indices, stratum_names, allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--round-name', required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_json(args.dataset)
    n = len(rows)
    if n < 1000:
        raise SystemExit(f'Dataset unexpectedly small: {n}')

    patterns = np.asarray([r['pattern'] for r in rows], dtype=np.int8)
    if patterns.shape != (n, N_CELLS):
        raise SystemExit(f'Unexpected pattern matrix shape: {patterns.shape}')
    bad = [i for i, p in enumerate(patterns) if not is_hard_feasible(p)]
    if bad:
        raise SystemExit(f'Active-learning CNN dataset must be entirely hard-constrained; bad rows: {bad[:20]}')

    keff = np.asarray([float(r['keff']) for r in rows], dtype=float)
    fa = np.asarray([float(r['peaking_assembly']) for r in rows], dtype=float)
    fqr = np.asarray([float(r.get('peaking_pin_radial', r.get('peaking_pin'))) for r in rows], dtype=float)
    if not (np.isfinite(keff).all() and np.isfinite(fa).all() and np.isfinite(fqr).all()):
        raise SystemExit('Non-finite OpenMC target found.')

    physics_feasible = np.asarray([true_physics_feasible(r) for r in rows], dtype=bool)
    strata = np.where(physics_feasible, 'physics_feasible', 'physics_infeasible')
    d4_keys = [canonical_d4_key(p) for p in patterns]
    split_sizes = split_sizes_for_n(n)
    splits, stratum_names, allocation = grouped_split(d4_keys, strata, split_sizes)

    group_sets = {name: {d4_keys[i] for i in idx} for name, idx in splits.items()}
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1:]:
            if group_sets[left] & group_sets[right]:
                raise SystemExit(f'D4 leakage between {left} and {right}.')

    x = patterns_to_onehot_grid(patterns)
    train = splits['train']
    val = splits['validation']
    cal = splits['calibration']
    test = splits['test']

    scalers = {
        'keff': StandardScaler().fit(keff[train].reshape(-1, 1)),
        'fa': StandardScaler().fit(fa[train].reshape(-1, 1)),
        'fqr': StandardScaler().fit(fqr[train].reshape(-1, 1)),
    }
    for name, scaler in scalers.items():
        joblib.dump(scaler, out / f'{name}_scaler.pkl')

    np.savez_compressed(
        out / 'round_arrays.npz',
        X_train=x[train], X_val=x[val], X_calibration=x[cal], X_test=x[test],
        yk_train=scalers['keff'].transform(keff[train, None]),
        yk_val=scalers['keff'].transform(keff[val, None]),
        yfa_train=scalers['fa'].transform(fa[train, None]),
        yfa_val=scalers['fa'].transform(fa[val, None]),
        yfqr_train=scalers['fqr'].transform(fqr[train, None]),
        yfqr_val=scalers['fqr'].transform(fqr[val, None]),
        yk_calibration_raw=keff[cal], yfa_calibration_raw=fa[cal], yfqr_calibration_raw=fqr[cal],
        yk_test_raw=keff[test], yfa_test_raw=fa[test], yfqr_test_raw=fqr[test],
        calibration_patterns=patterns[cal], test_patterns=patterns[test],
        train_indices=train, validation_indices=val, calibration_indices=cal, test_indices=test,
    )
    np.savez_compressed(out / 'all_patterns_targets.npz', patterns=patterns, keff=keff, fa=fa, fqr=fqr)

    split_hash = hashlib.sha256(b''.join(np.asarray(splits[name], dtype=np.int64).tobytes() for name in SPLIT_NAMES)).hexdigest()
    metadata = {
        'round_name': args.round_name,
        'dataset': str(Path(args.dataset).resolve()),
        'dataset_sha256': sha256_file(args.dataset),
        'dataset_size': n,
        'split_seed': SPLIT_SEED,
        'split_sha256': split_hash,
        'split_sizes': split_sizes,
        'split_fractions': SPLIT_FRACTIONS,
        'd4_canonical_groups': len(set(d4_keys)),
        'd4_equivalent_duplicates': n - len(set(d4_keys)),
        'physics_feasible_count': int(physics_feasible.sum()),
        'physics_feasible_fraction': float(physics_feasible.mean()),
        'strata': list(stratum_names),
        'stratum_allocation': {
            stratum_names[i]: {SPLIT_NAMES[j]: int(allocation[i, j]) for j in range(4)}
            for i in range(len(stratum_names))
        },
        'cnn_settings': {
            'epochs': 200,
            'batch_size': 32,
            'patience': 15,
            'seeds': [42, 314, 2718, 31415, 161803],
            'architecture': 'same residual-attention multi-output CNN as current notebook',
            'initial_learning_rate': 3e-4,
            'learning_rate_schedule': 'CosineDecay over CNN_EPOCHS * steps_per_epoch',
            'loss': 'equal-weight MSE for keff, Fa, Fq_r',
        },
        'physics_limits': {'keff_low': KEFF_LO, 'keff_high': KEFF_HI, 'Fa': FA_LIMIT, 'Fq_r': FQR_LIMIT},
    }
    atomic_json(metadata, out / 'round_metadata.json')
    print(metadata)


if __name__ == '__main__':
    main()
