#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from itertools import groupby

import numpy as np

GRID_SIDE = 15
N_CELLS = 193
N_FRESH = 65
N_ONCE = 64
N_TWICE = 64
KEFF_LO = 1.0
KEFF_HI = 1.2
FA_LIMIT = 2.0
FQR_LIMIT = 2.5


def compute_layers(adj, n_cells):
    degree = {i: len(adj[i]) for i in range(n_cells)}
    layer = [-1] * n_cells
    remaining = set(range(n_cells))
    current_layer = 0
    while remaining:
        max_degree = max(degree[i] for i in remaining)
        boundary = [i for i in remaining if degree[i] < max_degree]
        if not boundary:
            for i in remaining:
                layer[i] = current_layer
            break
        for i in boundary:
            layer[i] = current_layer
            remaining.remove(i)
            for j in adj[i]:
                if j in remaining:
                    degree[j] -= 1
        current_layer += 1
    return layer


def find_grid_side(n_cells):
    side = int(np.ceil(np.sqrt(n_cells)))
    remainder = n_cells % 4
    if remainder in (0, 3) and side % 2:
        side += 1
    elif remainder in (1, 2) and side % 2 != 1:
        side += 1
    return side


def generate_core_geometry(n_cells=N_CELLS):
    side = find_grid_side(n_cells)
    all_points = [(x, y) for x in range(side) for y in range(side)]
    center = (side - 1) / 2.0
    distances = {
        point: round(np.hypot(point[0] - center, point[1] - center), 4)
        for point in all_points
    }

    def sort_key(point):
        x, y = point
        rx, ry = abs(x - center), abs(y - center)
        return distances[point], -min(rx, ry), max(rx, ry), x, y

    sorted_points = sorted(all_points, key=sort_key)
    rings = [list(group) for _, group in groupby(sorted_points, key=lambda p: distances[p])]
    selected = []
    selected_set = set()
    cumulative = 0
    for ring in rings:
        if cumulative + len(ring) <= n_cells:
            selected.extend(ring)
            selected_set.update(ring)
            cumulative += len(ring)
            continue
        need = n_cells - cumulative
        ring_set = set(ring)
        used = set()
        taken = []
        for point in ring:
            if len(taken) >= need or point in used:
                continue
            x, y = point
            rx, ry = x - center, y - center
            copies = set()
            for sx in (rx, -rx):
                for sy in (ry, -ry):
                    candidate = (round(center + sx), round(center + sy))
                    if candidate in ring_set and candidate not in selected_set:
                        copies.add(candidate)
            if copies and len(taken) + len(copies) <= need:
                taken.extend(sorted(copies))
                used.update(copies)
        for point in ring:
            if len(taken) >= need:
                break
            if point not in used and point not in selected_set:
                taken.append(point)
                used.add(point)
        selected.extend(taken[:need])
        break

    positions = selected[:n_cells]
    coordinate_to_index = {point: i for i, point in enumerate(positions)}
    adjacency = {i: [] for i in range(n_cells)}
    for i, (x, y) in enumerate(positions):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (x + dx, y + dy)
            if neighbour in coordinate_to_index:
                adjacency[i].append(coordinate_to_index[neighbour])

    peripheral = [
        i for i, (x, y) in enumerate(positions)
        if any((x + dx, y + dy) not in coordinate_to_index
               for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    ]
    layers = compute_layers(adjacency, n_cells)
    max_layer = max(layers)
    inner_threshold = max_layer - max_layer // 2
    inner = [i for i in range(n_cells) if layers[i] > inner_threshold]
    return peripheral, inner, adjacency, {i: point for i, point in enumerate(positions)}, layers


PERIPHERAL, INNER, ADJ, POS, LAYERS = generate_core_geometry(N_CELLS)
PERIPHERAL_SET = set(PERIPHERAL)
INNER_SET = set(INNER)
MIDDLE = [i for i in range(N_CELLS) if i not in PERIPHERAL_SET and i not in INNER_SET]
MIDDLE_SET = set(MIDDLE)
ADJ_PAIRS = [(i, j) for i in range(N_CELLS) for j in ADJ.get(i, []) if j > i]
NON_EXEMPT_PAIRS = [
    (i, j) for i, j in ADJ_PAIRS if i not in PERIPHERAL_SET and j not in PERIPHERAL_SET
]
ALL_FUEL_COORDS = {POS[i] for i in range(N_CELLS)}
WATER_POSITIONS = [
    (x, y) for x in range(GRID_SIDE) for y in range(GRID_SIDE)
    if (x, y) not in ALL_FUEL_COORDS
]


def pkey(row_or_pattern):
    if isinstance(row_or_pattern, dict):
        pattern = row_or_pattern.get('pattern', row_or_pattern.get('chromosome'))
    else:
        pattern = row_or_pattern
    if pattern is None:
        raise ValueError('Missing pattern/chromosome.')
    key = tuple(int(x) for x in pattern)
    if len(key) != N_CELLS:
        raise ValueError(f'Expected {N_CELLS} cells, got {len(key)}.')
    return key


def pattern_hash(pattern):
    arr = np.asarray(pkey(pattern), dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def chromosome_to_grid(chromosome, fill_value=0):
    grid = np.full((GRID_SIDE, GRID_SIDE), fill_value, dtype=int)
    for index, fuel_type in enumerate(np.asarray(chromosome, dtype=int)):
        col, row = POS[index]
        grid[row, col] = fuel_type
    return grid


def grid_to_chromosome(grid):
    grid = np.asarray(grid)
    return np.array([grid[POS[i][1], POS[i][0]] for i in range(N_CELLS)], dtype=int)


def d4_variants(chromosome):
    grid = chromosome_to_grid(chromosome)
    variants = []
    seen = set()
    for reflected in (False, True):
        base = np.fliplr(grid) if reflected else grid
        for k in range(4):
            candidate = grid_to_chromosome(np.rot90(base, k))
            key = tuple(candidate.tolist())
            if key not in seen:
                seen.add(key)
                variants.append(candidate)
    return variants


def canonical_d4_key(chromosome):
    return min(tuple(variant.tolist()) for variant in d4_variants(chromosome))


def d4_hash(chromosome):
    arr = np.asarray(canonical_d4_key(chromosome), dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build_d4_orbits():
    coordinate_to_index = {point: i for i, point in POS.items()}
    seen = set()
    orbits = []
    for index in range(N_CELLS):
        if index in seen:
            continue
        x, y = POS[index]
        points = set()
        for k in range(4):
            if k == 0:
                a, b = x, y
            elif k == 1:
                a, b = GRID_SIDE - 1 - y, x
            elif k == 2:
                a, b = GRID_SIDE - 1 - x, GRID_SIDE - 1 - y
            else:
                a, b = y, GRID_SIDE - 1 - x
            points.add((a, b))
            points.add((GRID_SIDE - 1 - a, b))
        orbit = tuple(sorted(coordinate_to_index[p] for p in points if p in coordinate_to_index))
        seen.update(orbit)
        orbits.append(orbit)
    return orbits


D4_ORBITS = build_d4_orbits()


def is_d4_symmetric(chromosome):
    pattern = np.asarray(chromosome, dtype=int)
    return all(len({int(pattern[i]) for i in orbit}) == 1 for orbit in D4_ORBITS)


def count_violations(chromosome):
    pattern = np.asarray(chromosome, dtype=int)
    if pattern.shape != (N_CELLS,):
        raise ValueError(f'Expected chromosome shape ({N_CELLS},), got {pattern.shape}')
    result = {
        'C1_invalid_gene': int(np.sum(~np.isin(pattern, (1, 2, 3)))),
        'C2_fresh': abs(int(np.sum(pattern == 1)) - N_FRESH),
        'C2_once': abs(int(np.sum(pattern == 2)) - N_ONCE),
        'C2_twice': abs(int(np.sum(pattern == 3)) - N_TWICE),
        'C3_twice_peripheral': int(np.sum(pattern[np.asarray(PERIPHERAL)] == 3)),
        'C3_fresh_inner': int(np.sum(pattern[np.asarray(INNER)] == 1)),
        'C4_fresh_fresh': int(sum(pattern[i] == 1 and pattern[j] == 1 for i, j in NON_EXEMPT_PAIRS)),
        'C4_twice_twice': int(sum(pattern[i] == 3 and pattern[j] == 3 for i, j in ADJ_PAIRS)),
    }
    result['total'] = int(sum(result.values()))
    return result


def is_hard_feasible(chromosome):
    return count_violations(chromosome)['total'] == 0


def true_physics_feasible(row):
    fa = float(row['peaking_assembly'])
    fqr = float(row.get('peaking_pin_radial', row.get('peaking_pin')))
    keff = float(row['keff'])
    return KEFF_LO <= keff <= KEFF_HI and fa <= FA_LIMIT and fqr <= FQR_LIMIT


def physics_violation_score(row):
    keff = float(row['keff'])
    fa = float(row['peaking_assembly'])
    fqr = float(row.get('peaking_pin_radial', row.get('peaking_pin')))
    kv = max(0.0, KEFF_LO - keff) / 0.02 + max(0.0, keff - KEFF_HI) / 0.02
    fav = max(0.0, fa - FA_LIMIT) / FA_LIMIT
    fqv = max(0.0, fqr - FQR_LIMIT) / FQR_LIMIT
    return kv + fav + fqv


def normalized_peaking_objective(row):
    fa = float(row['peaking_assembly'])
    fqr = float(row.get('peaking_pin_radial', row.get('peaking_pin')))
    return 0.5 * fa / FA_LIMIT + 0.5 * fqr / FQR_LIMIT


def patterns_to_onehot_grid(patterns):
    patterns = np.asarray(patterns, dtype=int)
    if patterns.ndim == 1:
        patterns = patterns[None, :]
    x = np.zeros((len(patterns), GRID_SIDE, GRID_SIDE, 3), dtype=np.float32)
    for fuel_index in range(N_CELLS):
        col, row = POS[fuel_index]
        fuel = patterns[:, fuel_index]
        for channel in range(3):
            x[fuel == channel + 1, row, col, channel] = 1.0
    return x
