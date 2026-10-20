#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection

from geometry import (
    ADJ_PAIRS, D4_ORBITS, INNER, LAYERS, MIDDLE, N_CELLS, PERIPHERAL, POS,
)

LAYER_VALUES = tuple(sorted(set(LAYERS)))
PAIR_TYPES = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))


def canonical_pattern_array(patterns):
    return np.asarray(patterns, dtype=np.int8)


def onehot_sparse(patterns):
    patterns = np.asarray(patterns, dtype=np.int8)
    n = len(patterns)
    rows = np.repeat(np.arange(n, dtype=np.int32), N_CELLS)
    cols = np.tile(np.arange(N_CELLS, dtype=np.int32) * 3, n) + (patterns.ravel() - 1)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, N_CELLS * 3), dtype=np.float32)


def structural_features(patterns):
    patterns = np.asarray(patterns, dtype=np.int8)
    features = []
    layer_array = np.asarray(LAYERS, dtype=int)
    zones = [np.asarray(PERIPHERAL, dtype=int), np.asarray(MIDDLE, dtype=int), np.asarray(INNER, dtype=int)]
    xy = np.asarray([POS[i] for i in range(N_CELLS)], dtype=float)
    center = 7.0
    radius = np.sqrt((xy[:, 0] - center) ** 2 + (xy[:, 1] - center) ** 2)

    for pattern in patterns:
        row = []
        # Fuel occupancy by peripheral/middle/inner zone.
        for fuel in (1, 2, 3):
            for zone in zones:
                row.append(float(np.mean(pattern[zone] == fuel)))
        # Fuel occupancy by graph-peeling radial layer.
        for fuel in (1, 2, 3):
            for layer in LAYER_VALUES:
                idx = np.where(layer_array == layer)[0]
                row.append(float(np.mean(pattern[idx] == fuel)))
        # Radial moments for each fuel class.
        for fuel in (1, 2, 3):
            r = radius[pattern == fuel]
            row.extend([float(np.mean(r)), float(np.std(r)), float(np.min(r)), float(np.max(r))])
        # Unordered adjacency-type frequencies.
        pair_counts = {pair: 0 for pair in PAIR_TYPES}
        for i, j in ADJ_PAIRS:
            pair = tuple(sorted((int(pattern[i]), int(pattern[j]))))
            pair_counts[pair] += 1
        denom = max(1, len(ADJ_PAIRS))
        row.extend(pair_counts[pair] / denom for pair in PAIR_TYPES)
        # Fraction of D4 orbits that are homogeneous; this captures symmetry tendency
        # without requiring the pattern to be exactly symmetric.
        homogeneous = sum(len({int(pattern[i]) for i in orbit}) == 1 for orbit in D4_ORBITS)
        row.append(homogeneous / max(1, len(D4_ORBITS)))
        features.append(row)
    return np.asarray(features, dtype=np.float32)


def fit_diversity_model(patterns, n_clusters=512, projection_components=64, seed=42):
    patterns = np.asarray(patterns, dtype=np.int8)
    onehot = onehot_sparse(patterns)
    projection = SparseRandomProjection(
        n_components=int(projection_components),
        dense_output=True,
        random_state=int(seed),
    )
    projected = projection.fit_transform(onehot).astype(np.float32)
    struct = structural_features(patterns)
    scaler = StandardScaler().fit(struct)
    struct_scaled = scaler.transform(struct).astype(np.float32)
    # Random-projection block approximates categorical Hamming geometry; the explicit
    # structural block ensures radial, zonal, adjacency, and symmetry variation matter.
    embedding = np.concatenate([projected, struct_scaled], axis=1)
    kmeans = MiniBatchKMeans(
        n_clusters=min(int(n_clusters), len(patterns)),
        random_state=int(seed),
        batch_size=min(4096, max(256, len(patterns))),
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    cluster = kmeans.fit_predict(embedding)
    distance = np.linalg.norm(embedding - kmeans.cluster_centers_[cluster], axis=1)
    return {
        'projection': projection,
        'struct_scaler': scaler,
        'kmeans': kmeans,
        'seed': int(seed),
        'projection_components': int(projection_components),
    }, cluster.astype(int), distance.astype(float)


def transform_diversity_model(model, patterns):
    patterns = np.asarray(patterns, dtype=np.int8)
    projected = model['projection'].transform(onehot_sparse(patterns)).astype(np.float32)
    struct = model['struct_scaler'].transform(structural_features(patterns)).astype(np.float32)
    embedding = np.concatenate([projected, struct], axis=1)
    cluster = model['kmeans'].predict(embedding).astype(int)
    distance = np.linalg.norm(embedding - model['kmeans'].cluster_centers_[cluster], axis=1)
    return cluster, distance.astype(float)


def proportional_cluster_quota(cluster_ids, target):
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    unique, counts = np.unique(cluster_ids, return_counts=True)
    expected = counts * (float(target) / len(cluster_ids))
    quota = np.floor(expected).astype(int)
    remainder = int(target - quota.sum())
    fractions = expected - quota
    order = np.argsort(-fractions, kind='stable')
    for idx in order[:remainder]:
        quota[idx] += 1
    return {int(c): int(q) for c, q in zip(unique, quota)}


def select_cluster_representatives(rows, target, cluster_key='structural_cluster', distance_key='cluster_distance'):
    """Deterministically cover clusters and within-cluster center-to-edge quantiles."""
    if target >= len(rows):
        return list(rows)
    clusters = np.asarray([int(r[cluster_key]) for r in rows], dtype=int)
    quotas = proportional_cluster_quota(clusters, target)
    by_cluster = {}
    for row in rows:
        by_cluster.setdefault(int(row[cluster_key]), []).append(row)
    selected = []
    for cluster in sorted(by_cluster):
        group = sorted(
            by_cluster[cluster],
            key=lambda r: (float(r.get(distance_key, 0.0)), str(r.get('pattern_hash', ''))),
        )
        q = min(quotas.get(cluster, 0), len(group))
        if q <= 0:
            continue
        if q == len(group):
            picks = group
        elif q == 1:
            picks = [group[len(group) // 2]]
        else:
            # Evenly sample the centroid-distance ordering: representatives span
            # typical and edge members of each Hamming/structural cluster.
            indices = np.unique(np.round(np.linspace(0, len(group) - 1, q)).astype(int))
            if len(indices) < q:
                used = set(indices.tolist())
                for idx in range(len(group)):
                    if idx not in used:
                        indices = np.append(indices, idx)
                        used.add(idx)
                        if len(indices) == q:
                            break
            picks = [group[int(i)] for i in sorted(indices[:q])]
        selected.extend(picks)
    if len(selected) < target:
        chosen = {r['pattern_hash'] for r in selected}
        remainder = sorted(
            (r for r in rows if r['pattern_hash'] not in chosen),
            key=lambda r: (-float(r.get(distance_key, 0.0)), str(r.get('pattern_hash', ''))),
        )
        selected.extend(remainder[: target - len(selected)])
    return selected[:target]
