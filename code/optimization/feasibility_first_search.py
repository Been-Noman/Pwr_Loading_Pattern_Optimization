#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import hashlib
import inspect
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pygad
import tensorflow as tf

from cnn_architecture import CNN_TRAINING_SEEDS, CUSTOM_OBJECTS, configure_tensorflow
from geometry import (
    ADJ_PAIRS,
    FA_LIMIT,
    FQR_LIMIT,
    INNER_SET,
    KEFF_HI,
    KEFF_LO,
    N_CELLS,
    N_FRESH,
    N_ONCE,
    N_TWICE,
    NON_EXEMPT_PAIRS,
    PERIPHERAL_SET,
    POS,
    canonical_d4_key,
    count_violations,
    is_hard_feasible,
    patterns_to_onehot_grid,
)

GA_BASE_SEED = 2026
GA_SEEDS = tuple(GA_BASE_SEED + 1009 * i for i in range(30))
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 300

# ------------------------------------------------------------
# V3 constrained-evolution architecture.
#
# Physics, objective, hard loading constraints, surrogate,
# residual-safety model, and conformal confidence remain frozen.
# These constants control only the evolutionary search mechanics.
# ------------------------------------------------------------

# Preserve the V2 parent-pool scale for a controlled comparison.
GA_PARENTS = 40

# Base local-search scale inherited from V2. V3 may escalate it
# only when explicit stagnation is detected.
GA_MUTATION_SWAPS = 8
GA_CROSSOVER_SWAPS = 12

# Persistent active archives.
V3_FEASIBLE_ARCHIVE_SIZE = 100
V3_NEAR_ARCHIVE_SIZE = 100

# Maximum archive members inserted directly into each generation.
V3_FEASIBLE_ELITES = 12
V3_NEAR_ELITES = 12

# Parent selection remains tournament-based, but ranking is V3
# feasibility-first constraint dominance rather than penalty fitness.
V3_TOURNAMENT_SIZE = 3

# Half of crossover offspring use the exploratory operator;
# the remainder use the conservative/exploitative operator.
V3_EXPLORATORY_CROSSOVER_FRACTION = 0.50

# Within the exploratory crossover channel, half of attempted swaps
# are parent-B-guided and half are globally random cross-type swaps.
# The random half is what allows genuine movement away from simple
# interpolation between the two parents.
V3_EXPLORATORY_GUIDED_FRACTION = 0.50

# Fresh hard-feasible immigrants preserve global exploration.
V3_BASE_IMMIGRANTS = 10
V3_STAGNANT_IMMIGRANTS = 20

# Stagnation is measured on minimum robust violation before
# feasibility and on best robust-safe J after feasibility.
V3_STAGNATION_WINDOW = 20
V3_PROGRESS_TOL = 1.0e-6

# Feasible-swap mutation escalates during prolonged stagnation.
V3_MUTATION_LEVELS = (8, 12, 16)

# Normalized 193-position Hamming-distance thresholds.
# Environmental selection starts strict and relaxes only as needed
# to fill the population.
V3_DIVERSITY_THRESHOLDS = (0.08, 0.06, 0.04, 0.02, 0.00)

# Retained temporarily so the copied V2 code stays syntactically
# valid while the V3 evolutionary loop is replaced incrementally.
HARD_CONSTRAINT_PENALTY = 1.0e9
PHYSICS_CONSTRAINT_PENALTY = 1.0e6

FA_WEIGHT = 0.5
FQR_WEIGHT = 0.5

CONSTRAINTS = {
    'N': N_CELLS,
    'N_fresh': N_FRESH,
    'N_once': N_ONCE,
    'N_twice': N_TWICE,
    'peripheral_set': PERIPHERAL_SET,
    'inner_set': INNER_SET,
    'adj_pairs': ADJ_PAIRS,
    'non_exempt_pairs': NON_EXEMPT_PAIRS,
    'pos': POS,
}


# ============================================================
# V3 constraint-dominance and diversity primitives.
# ============================================================

def v3_normalized_physics_violation(lower_k, upper_k, upper_fa, upper_fqr):
    """Return per-candidate normalized robust-physics violation.

    Zero means all learned-residual joint-95% physics bounds pass.
    Positive values measure distance outside the physical limits.
    No penalty constant is used.
    """
    lower_k = np.asarray(lower_k, dtype=float)
    upper_k = np.asarray(upper_k, dtype=float)
    upper_fa = np.asarray(upper_fa, dtype=float)
    upper_fqr = np.asarray(upper_fqr, dtype=float)

    return (
        np.maximum(0.0, KEFF_LO - lower_k) / KEFF_LO
        + np.maximum(0.0, upper_k - KEFF_HI) / KEFF_HI
        + np.maximum(0.0, upper_fa - FA_LIMIT) / FA_LIMIT
        + np.maximum(0.0, upper_fqr - FQR_LIMIT) / FQR_LIMIT
    )


def v3_canonical_array(pattern):
    """Canonical D4 representation used for diversity/deduplication."""
    return np.asarray(
        canonical_d4_key(np.asarray(pattern, dtype=int)),
        dtype=np.int8,
    )


def v3_hamming_distance(pattern_a, pattern_b):
    """Normalized Hamming distance in canonical D4 representation."""
    a = v3_canonical_array(pattern_a)
    b = v3_canonical_array(pattern_b)
    return float(np.mean(a != b))


def v3_population_diversity(population):
    """Mean pairwise normalized D4-canonical Hamming distance."""
    population = np.asarray(population, dtype=int)

    if len(population) < 2:
        return 0.0

    canonical = np.stack([
        v3_canonical_array(row)
        for row in population
    ])

    total = 0.0
    count = 0

    for i in range(len(canonical) - 1):
        distances = np.mean(
            canonical[i + 1:] != canonical[i],
            axis=1,
        )
        total += float(np.sum(distances))
        count += len(distances)

    return total / count if count else 0.0


def v3_constraint_order(safe, violation, objective):
    """Return best-to-worst indices under feasibility-first dominance.

    Ordering:
      1. Robust-feasible candidates before robust-infeasible candidates.
      2. Among feasible candidates: lower physical objective J is better.
      3. Among infeasible candidates: lower normalized violation is better,
         with J used only as a deterministic tie-breaker.
    """
    safe = np.asarray(safe, dtype=bool)
    violation = np.asarray(violation, dtype=float)
    objective = np.asarray(objective, dtype=float)

    if not (
        len(safe) == len(violation) == len(objective)
    ):
        raise ValueError("V3 ranking arrays must have equal length.")

    # np.lexsort uses the last key as the primary key.
    # safe candidates receive feasibility key 0, infeasible key 1.
    feasibility_key = (~safe).astype(np.int8)

    # For safe rows, violation is irrelevant and set to zero.
    # For infeasible rows, violation is the second ranking criterion.
    violation_key = np.where(safe, 0.0, violation)

    # For feasible rows J is the second criterion; for infeasible rows
    # it acts only as a tie-breaker after violation.
    objective_key = objective

    return np.lexsort(
        (
            objective_key,
            violation_key,
            feasibility_key,
        )
    ).astype(int)



# ============================================================
# V3 active archives and diversity-aware truncation.
# ============================================================

def v3_d4_key(pattern):
    """Hashable canonical D4 key."""
    return tuple(
        canonical_d4_key(np.asarray(pattern, dtype=int))
    )


def v3_diverse_truncation(
    population,
    ordered_indices,
    limit,
    thresholds=V3_DIVERSITY_THRESHOLDS,
):
    """Select up to ``limit`` D4-unique candidates with diversity pressure.

    Candidates are considered in the supplied quality order. The minimum
    normalized Hamming-distance requirement is progressively relaxed only
    when needed to fill the requested number of slots.
    """
    population = np.asarray(population, dtype=int)
    ordered_indices = [int(i) for i in ordered_indices]
    limit = int(limit)

    if limit <= 0 or len(ordered_indices) == 0:
        return np.empty(0, dtype=int)

    canonical = {
        i: v3_canonical_array(population[i])
        for i in ordered_indices
    }

    selected = []
    selected_keys = set()

    for threshold in thresholds:
        threshold = float(threshold)

        for idx in ordered_indices:
            if len(selected) >= limit:
                break

            key = tuple(canonical[idx].tolist())

            # D4 duplicates never receive a second slot, even when the
            # diversity threshold has been relaxed to zero.
            if key in selected_keys:
                continue

            if selected:
                min_distance = min(
                    float(np.mean(canonical[idx] != canonical[j]))
                    for j in selected
                )
                if min_distance + 1.0e-15 < threshold:
                    continue

            selected.append(idx)
            selected_keys.add(key)

        if len(selected) >= limit:
            break

    return np.asarray(selected, dtype=int)


def v3_archive_record(population, details, idx, generation):
    """Create one immutable-style archive record from an evaluated row."""
    idx = int(idx)
    pattern = np.asarray(population[idx], dtype=int)

    return {
        'pattern': pattern.copy(),
        'd4_key': v3_d4_key(pattern),
        'generation': int(generation),
        'safe': bool(details['safe'][idx]),
        'objective': float(details['secondary'][idx]),
        'normalized_violation': float(
            details['normalized_violation'][idx]
        ),
        'ensemble_mean_keff': float(details['mu'][idx, 0]),
        'ensemble_mean_Fa': float(details['mu'][idx, 1]),
        'ensemble_mean_Fq_r': float(details['mu'][idx, 2]),
        'calibrated_keff_lower': float(details['lower_k'][idx]),
        'calibrated_keff_upper': float(details['upper_k'][idx]),
        'calibrated_Fa_upper': float(details['upper_fa'][idx]),
        'calibrated_Fq_r_upper': float(details['upper_fqr'][idx]),
    }


def v3_archive_record_better(new, old):
    """Return True when ``new`` dominates ``old`` for the same D4 key."""
    if bool(new['safe']) != bool(old['safe']):
        return bool(new['safe'])

    if bool(new['safe']):
        return float(new['objective']) < float(old['objective'])

    nv = float(new['normalized_violation'])
    ov = float(old['normalized_violation'])

    if nv != ov:
        return nv < ov

    return float(new['objective']) < float(old['objective'])


def v3_prune_archive(archive, limit, feasible_archive):
    """Quality-order and diversity-truncate one persistent archive."""
    if not archive:
        return {}

    records = list(archive.values())
    population = np.stack([
        np.asarray(r['pattern'], dtype=int)
        for r in records
    ])

    if feasible_archive:
        order = np.argsort(
            np.asarray(
                [r['objective'] for r in records],
                dtype=float,
            ),
            kind='stable',
        )
    else:
        violation = np.asarray(
            [r['normalized_violation'] for r in records],
            dtype=float,
        )
        objective = np.asarray(
            [r['objective'] for r in records],
            dtype=float,
        )
        # Primary: violation. Secondary: J.
        order = np.lexsort((objective, violation))

    selected = v3_diverse_truncation(
        population,
        order,
        limit=int(limit),
    )

    return {
        records[int(i)]['d4_key']: records[int(i)]
        for i in selected
    }


def v3_update_archives(
    feasible_archive,
    near_archive,
    population,
    details,
    generation,
):
    """Update persistent D4-unique feasible and near-feasible archives."""
    feasible_archive = dict(feasible_archive)
    near_archive = dict(near_archive)

    population = np.asarray(population, dtype=int)

    hard = np.asarray(details['hard_violation'], dtype=float)
    novel = np.asarray(details['d4_novel'], dtype=bool)
    safe = np.asarray(details['safe'], dtype=bool)

    for idx in range(len(population)):
        # Archives contain only structurally valid, externally novel rows.
        if hard[idx] != 0.0 or not novel[idx]:
            continue

        record = v3_archive_record(
            population,
            details,
            idx,
            generation,
        )
        key = record['d4_key']

        if safe[idx]:
            old = feasible_archive.get(key)
            if old is None or v3_archive_record_better(record, old):
                feasible_archive[key] = record

            # Once a D4 class becomes robust-feasible it should no longer
            # occupy the near-feasible archive.
            near_archive.pop(key, None)

        else:
            # Never downgrade a D4 class that has already reached
            # robust feasibility.
            if key in feasible_archive:
                continue

            old = near_archive.get(key)
            if old is None or v3_archive_record_better(record, old):
                near_archive[key] = record

    feasible_archive = v3_prune_archive(
        feasible_archive,
        V3_FEASIBLE_ARCHIVE_SIZE,
        feasible_archive=True,
    )
    near_archive = v3_prune_archive(
        near_archive,
        V3_NEAR_ARCHIVE_SIZE,
        feasible_archive=False,
    )

    return feasible_archive, near_archive


def v3_archive_patterns(archive, limit=None):
    """Return archive chromosomes in stored quality/diversity order."""
    records = list(archive.values())

    if limit is not None:
        records = records[:int(limit)]

    if not records:
        return np.empty((0, N_CELLS), dtype=int)

    return np.stack([
        np.asarray(r['pattern'], dtype=int)
        for r in records
    ]).astype(int)


def construct_feasible_chromosome(constraints=CONSTRAINTS, rng=None, max_attempts=200):
    rng = np.random.default_rng() if rng is None else rng
    parity = np.array([
        (constraints['pos'][i][0] + constraints['pos'][i][1]) % 2
        for i in range(constraints['N'])
    ])

    for _ in range(max_attempts):
        pattern = np.full(constraints['N'], 2, dtype=int)
        twice_allowed = np.array([
            i for i in range(constraints['N'])
            if i not in constraints['peripheral_set']
        ], dtype=int)
        twice_parities = [
            value for value in (0, 1)
            if int(np.sum(parity[twice_allowed] == value)) >= constraints['N_twice']
        ]
        if not twice_parities:
            break
        twice_parity = int(rng.choice(twice_parities))
        twice_candidates = twice_allowed[parity[twice_allowed] == twice_parity]
        twice_positions = rng.choice(twice_candidates, constraints['N_twice'], replace=False)
        pattern[twice_positions] = 3

        peripheral = np.array(sorted(constraints['peripheral_set']), dtype=int)
        middle_available = np.array([
            i for i in range(constraints['N'])
            if i not in constraints['peripheral_set']
            and i not in constraints['inner_set']
            and pattern[i] != 3
        ], dtype=int)

        fresh_options = []
        for fresh_parity in (0, 1):
            middle_candidates = middle_available[parity[middle_available] == fresh_parity]
            min_peripheral = max(0, constraints['N_fresh'] - len(middle_candidates))
            max_peripheral = min(len(peripheral), constraints['N_fresh'])
            if min_peripheral <= max_peripheral:
                fresh_options.append((middle_candidates, min_peripheral, max_peripheral))
        if not fresh_options:
            continue

        middle_candidates, min_peripheral, max_peripheral = fresh_options[
            int(rng.integers(len(fresh_options)))
        ]
        n_peripheral = int(rng.integers(min_peripheral, max_peripheral + 1))
        n_middle = constraints['N_fresh'] - n_peripheral
        if n_peripheral:
            pattern[rng.choice(peripheral, n_peripheral, replace=False)] = 1
        if n_middle:
            pattern[rng.choice(middle_candidates, n_middle, replace=False)] = 1

        if is_hard_feasible(pattern):
            return pattern
    raise RuntimeError('Unable to construct a hard-feasible chromosome.')


def perturb_feasible_chromosome(chromosome, n_swaps, constraints=CONSTRAINTS, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    pattern = np.asarray(chromosome, dtype=int).copy()
    if not is_hard_feasible(pattern):
        raise RuntimeError('Mutation received a hard-infeasible chromosome.')

    accepted = 0
    attempts = 0
    max_attempts = max(100, 40 * n_swaps)
    while accepted < n_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(constraints['N'], size=2, replace=False)
        if pattern[i] == pattern[j]:
            continue
        pattern[i], pattern[j] = pattern[j], pattern[i]
        if is_hard_feasible(pattern):
            accepted += 1
        else:
            pattern[i], pattern[j] = pattern[j], pattern[i]
    return pattern


def build_feasible_population(
    population_size,
    constraints=CONSTRAINTS,
    rng=None,
    base_chromosomes=None,
    forbidden_d4=None,
):
    """Build a unique, hard-feasible, D4-novel population around supplied anchors."""
    rng = np.random.default_rng() if rng is None else rng
    bases = [] if base_chromosomes is None else [
        np.asarray(row, dtype=int).copy() for row in base_chromosomes
        if is_hard_feasible(np.asarray(row, dtype=int))
    ]
    forbidden_d4 = set() if forbidden_d4 is None else set(forbidden_d4)

    rows = []
    seen_exact = set()
    seen_d4 = set()
    attempts = 0
    max_attempts = population_size * 500

    while len(rows) < population_size and attempts < max_attempts:
        attempts += 1

        if bases and rng.random() < 0.90:
            base = bases[int(rng.integers(len(bases)))]
            candidate = perturb_feasible_chromosome(
                base,
                n_swaps=int(rng.integers(1, 9)),
                constraints=constraints,
                rng=rng,
            )
        else:
            candidate = construct_feasible_chromosome(constraints, rng)

        if not is_hard_feasible(candidate):
            continue

        exact_key = tuple(candidate.tolist())
        d4_key = tuple(canonical_d4_key(candidate))

        if exact_key in seen_exact or d4_key in seen_d4 or d4_key in forbidden_d4:
            continue

        rows.append(candidate)
        seen_exact.add(exact_key)
        seen_d4.add(d4_key)

    if len(rows) != population_size:
        raise RuntimeError(
            f"Generated {len(rows)} unique D4-novel feasible chromosomes; "
            f"expected {population_size}."
        )

    population = np.stack(rows).astype(int)
    if not all(is_hard_feasible(row) for row in population):
        raise RuntimeError("Initial population hard-constraint audit failed.")
    return population


def _ga_rng(ga_instance):
    if not hasattr(ga_instance, '_constraint_rng'):
        seed = int(getattr(ga_instance, 'random_seed', GA_BASE_SEED) or GA_BASE_SEED)
        ga_instance._constraint_rng = np.random.default_rng(seed)
    return ga_instance._constraint_rng


def count_preserving_crossover(parents, offspring_size, ga_instance):
    rng = _ga_rng(ga_instance)
    constraints = ga_instance._constraint_tables
    n_swaps = int(getattr(ga_instance, '_crossover_swaps', GA_CROSSOVER_SWAPS))
    offspring = np.empty(offspring_size, dtype=int)

    for child_index in range(offspring_size[0]):
        parent_a = np.asarray(parents[child_index % len(parents)], dtype=int)
        parent_b = np.asarray(parents[(child_index + 1) % len(parents)], dtype=int)
        child = parent_a.copy()
        accepted = 0
        attempts = 0
        max_attempts = max(50, 20 * n_swaps)

        while accepted < n_swaps and attempts < max_attempts:
            attempts += 1
            mismatched = np.where(child != parent_b)[0]
            if len(mismatched) < 2:
                break
            first = int(rng.choice(mismatched))
            desired_type = int(parent_b[first])
            reciprocal = np.where((child == desired_type) & (parent_b == child[first]))[0]
            candidates = reciprocal[reciprocal != first]
            if not len(candidates):
                candidates = np.where(
                    (child == desired_type) & (np.arange(constraints['N']) != first)
                )[0]
            if not len(candidates):
                continue
            second = int(rng.choice(candidates))
            mismatch_before = int(child[first] != parent_b[first]) + int(child[second] != parent_b[second])
            child[first], child[second] = child[second], child[first]
            mismatch_after = int(child[first] != parent_b[first]) + int(child[second] != parent_b[second])
            if mismatch_after <= mismatch_before and is_hard_feasible(child):
                accepted += 1
            else:
                child[first], child[second] = child[second], child[first]

        if accepted == 0:
            child = perturb_feasible_chromosome(child, 1, constraints, rng)
        offspring[child_index] = child
    return offspring


def feasibility_preserving_swap_mutation(offspring, ga_instance):
    rng = _ga_rng(ga_instance)
    constraints = ga_instance._constraint_tables
    n_swaps = int(getattr(ga_instance, '_mutation_swaps', GA_MUTATION_SWAPS))
    mutated = np.asarray(offspring, dtype=int).copy()
    for i in range(len(mutated)):
        mutated[i] = perturb_feasible_chromosome(mutated[i], n_swaps, constraints, rng)
    return mutated



# ============================================================
# V3 reproduction primitives.
# ============================================================

def v3_rank_population(details):
    """Return best-to-worst indices under V3 constraint dominance.

    V3 generations are required to contain only hard-feasible,
    externally D4-novel chromosomes. Violations here indicate a bug
    in generation construction rather than something selection should
    repair with a numerical penalty.
    """
    hard = np.asarray(details['hard_violation'], dtype=float)
    novel = np.asarray(details['d4_novel'], dtype=bool)

    if np.any(hard != 0.0):
        raise RuntimeError(
            "V3 population ranking received a hard-infeasible chromosome."
        )

    if not np.all(novel):
        raise RuntimeError(
            "V3 population ranking received a known-OpenMC D4 chromosome."
        )

    return v3_constraint_order(
        details['safe'],
        details['normalized_violation'],
        details['secondary'],
    )


def v3_tournament_parent_indices(
    details,
    n_parents,
    rng,
    tournament_size=V3_TOURNAMENT_SIZE,
):
    """Select parent indices using V3 feasibility-first tournament ranking."""
    n = len(np.asarray(details['safe']))
    n_parents = int(n_parents)
    tournament_size = min(int(tournament_size), n)

    if n <= 0:
        raise ValueError("Cannot select parents from an empty population.")

    order = v3_rank_population(details)

    # rank_position[i] = 0 for the best chromosome, larger is worse.
    rank_position = np.empty(n, dtype=int)
    rank_position[order] = np.arange(n, dtype=int)

    selected = np.empty(n_parents, dtype=int)

    for k in range(n_parents):
        competitors = rng.choice(
            n,
            size=tournament_size,
            replace=False,
        )
        selected[k] = int(
            competitors[
                np.argmin(rank_position[competitors])
            ]
        )

    return selected



def v3_select_mating_pool_indices(
    details,
    n_parents,
    rng,
):
    """Select a D4-distinct mating pool by V3 tournaments.

    This preserves the frozen V2 mating-parent scale (40) while using
    V3 feasibility-first ranking. Tournament winners are accumulated
    without replacement at the candidate-index level.
    """
    n = len(np.asarray(details['safe']))
    target = min(int(n_parents), n)

    if target < 2:
        raise RuntimeError(
            "V3 mating pool requires at least 2 candidates."
        )

    chosen = []
    chosen_set = set()

    attempts = 0
    max_attempts = max(1000, target * 200)

    while len(chosen) < target and attempts < max_attempts:
        attempts += 1

        idx = int(
            v3_tournament_parent_indices(
                details,
                n_parents=1,
                rng=rng,
            )[0]
        )

        if idx in chosen_set:
            continue

        chosen.append(idx)
        chosen_set.add(idx)

    # Extremely defensive fallback: if stochastic tournaments have not
    # populated all unique slots, fill the remainder in strict V3 rank
    # order. This preserves feasibility-first quality semantics.
    if len(chosen) < target:
        order = v3_rank_population(details)

        for idx in order:
            idx = int(idx)

            if idx in chosen_set:
                continue

            chosen.append(idx)
            chosen_set.add(idx)

            if len(chosen) >= target:
                break

    if len(chosen) != target:
        raise RuntimeError(
            f"V3 mating-pool selection produced "
            f"{len(chosen)}/{target} unique parents."
        )

    return np.asarray(chosen, dtype=int)


def v3_exploitative_crossover(
    parent_a,
    parent_b,
    rng,
    n_swaps=GA_CROSSOVER_SWAPS,
    constraints=CONSTRAINTS,
):
    """Conservative V2-like count-preserving crossover.

    Accepted swaps must not increase mismatch against parent B.
    This channel exploits an existing basin.
    """
    child = np.asarray(parent_a, dtype=int).copy()
    parent_b = np.asarray(parent_b, dtype=int)

    if not is_hard_feasible(child) or not is_hard_feasible(parent_b):
        raise RuntimeError(
            "V3 exploitative crossover received hard-infeasible parent."
        )

    accepted = 0
    attempts = 0
    max_attempts = max(50, 20 * int(n_swaps))

    while accepted < int(n_swaps) and attempts < max_attempts:
        attempts += 1

        mismatched = np.where(child != parent_b)[0]
        if len(mismatched) < 2:
            break

        first = int(rng.choice(mismatched))
        desired_type = int(parent_b[first])

        reciprocal = np.where(
            (child == desired_type)
            & (parent_b == child[first])
        )[0]

        candidates = reciprocal[reciprocal != first]

        if not len(candidates):
            candidates = np.where(
                (child == desired_type)
                & (np.arange(constraints['N']) != first)
            )[0]

        if not len(candidates):
            continue

        second = int(rng.choice(candidates))

        mismatch_before = (
            int(child[first] != parent_b[first])
            + int(child[second] != parent_b[second])
        )

        child[first], child[second] = (
            child[second],
            child[first],
        )

        mismatch_after = (
            int(child[first] != parent_b[first])
            + int(child[second] != parent_b[second])
        )

        if (
            mismatch_after <= mismatch_before
            and is_hard_feasible(child)
        ):
            accepted += 1
        else:
            child[first], child[second] = (
                child[second],
                child[first],
            )

    if accepted == 0:
        child = perturb_feasible_chromosome(
            child,
            1,
            constraints,
            rng,
        )

    return child


def v3_exploratory_crossover(
    parent_a,
    parent_b,
    rng,
    n_swaps=GA_CROSSOVER_SWAPS,
    constraints=CONSTRAINTS,
):
    """Exploratory count-preserving crossover.

    Parent B suggests useful assembly types, but unlike the exploitative
    operator, a swap does NOT have to reduce Hamming mismatch to B.
    Acceptance requires only exact-count preservation (inherent in swaps)
    and hard loading feasibility. This allows movement away from simple
    interpolation between the two parents.
    """
    child = np.asarray(parent_a, dtype=int).copy()
    parent_b = np.asarray(parent_b, dtype=int)

    if not is_hard_feasible(child) or not is_hard_feasible(parent_b):
        raise RuntimeError(
            "V3 exploratory crossover received hard-infeasible parent."
        )

    accepted = 0
    attempts = 0
    max_attempts = max(100, 40 * int(n_swaps))

    while accepted < int(n_swaps) and attempts < max_attempts:
        attempts += 1

        guided = (
            rng.random()
            < V3_EXPLORATORY_GUIDED_FRACTION
        )

        mismatched = np.where(child != parent_b)[0]

        if guided and len(mismatched):
            # Parent-guided exploratory move.
            first = int(rng.choice(mismatched))
            desired_type = int(parent_b[first])

            candidates = np.where(
                (child == desired_type)
                & (np.arange(constraints['N']) != first)
            )[0]
        else:
            # Globally exploratory move: choose any locus and exchange
            # it with a different fuel type. This preserves exact counts
            # but is not directed toward either parent.
            first = int(rng.integers(constraints['N']))

            candidates = np.where(
                (child != child[first])
                & (np.arange(constraints['N']) != first)
            )[0]

        if not len(candidates):
            continue

        second = int(rng.choice(candidates))

        if child[first] == child[second]:
            continue

        child[first], child[second] = (
            child[second],
            child[first],
        )

        if is_hard_feasible(child):
            accepted += 1
        else:
            child[first], child[second] = (
                child[second],
                child[first],
            )

    if accepted == 0:
        child = perturb_feasible_chromosome(
            child,
            1,
            constraints,
            rng,
        )

    return child


def v3_reproduce_child(
    parent_a,
    parent_b,
    rng,
    mutation_swaps,
    exploratory,
):
    """Create one hard-feasible child and apply adaptive local mutation."""
    if exploratory:
        child = v3_exploratory_crossover(
            parent_a,
            parent_b,
            rng,
        )
    else:
        child = v3_exploitative_crossover(
            parent_a,
            parent_b,
            rng,
        )

    child = perturb_feasible_chromosome(
        child,
        int(mutation_swaps),
        CONSTRAINTS,
        rng,
    )

    if not is_hard_feasible(child):
        raise RuntimeError(
            "V3 reproduction produced a hard-infeasible child."
        )

    return child


def v3_generate_immigrants(
    count,
    rng,
    forbidden_d4,
    blocked_d4=None,
):
    """Generate globally fresh hard-feasible D4-unique chromosomes."""
    count = int(count)

    if count <= 0:
        return np.empty((0, N_CELLS), dtype=int)

    forbidden_d4 = set(forbidden_d4)
    blocked = set() if blocked_d4 is None else set(blocked_d4)

    rows = []
    seen = set(blocked)
    attempts = 0
    max_attempts = max(1000, count * 500)

    while len(rows) < count and attempts < max_attempts:
        attempts += 1

        candidate = construct_feasible_chromosome(
            CONSTRAINTS,
            rng,
        )

        key = v3_d4_key(candidate)

        if key in forbidden_d4 or key in seen:
            continue

        rows.append(candidate)
        seen.add(key)

    if len(rows) != count:
        raise RuntimeError(
            f"V3 immigrant generation produced {len(rows)}/{count} "
            "requested D4-unique chromosomes."
        )

    return np.stack(rows).astype(int)



# ============================================================
# V3 environmental survival and adaptive-search control.
# ============================================================

def v3_environmental_select(
    population,
    details,
    population_size=GA_POPULATION_SIZE,
):
    """Feasibility-first, diversity-aware survivor selection.

    Quality is determined only by V3 constraint dominance. Diversity is
    then used during truncation so that one local loading basin cannot
    monopolize the complete next generation.
    """
    population = np.asarray(population, dtype=int)
    population_size = int(population_size)

    if len(population) < population_size:
        raise RuntimeError(
            f"V3 environmental selection received only {len(population)} "
            f"candidates for {population_size} survivor slots."
        )

    hard = np.asarray(details['hard_violation'], dtype=float)
    novel = np.asarray(details['d4_novel'], dtype=bool)

    if np.any(hard != 0.0):
        raise RuntimeError(
            "V3 environmental selection received hard-infeasible rows."
        )

    if not np.all(novel):
        raise RuntimeError(
            "V3 environmental selection received known-OpenMC D4 rows."
        )

    safe = np.asarray(details['safe'], dtype=bool)
    objective = np.asarray(details['secondary'], dtype=float)
    violation = np.asarray(
        details['normalized_violation'],
        dtype=float,
    )

    safe_indices = np.flatnonzero(safe)
    infeasible_indices = np.flatnonzero(~safe)

    selected_parts = []

    # --------------------------------------------------------
    # Tier 1: robust-feasible candidates.
    #
    # If there are <= population_size feasible rows, retain ALL
    # D4-unique feasible rows before considering any infeasible row.
    # This enforces strict feasibility dominance.
    #
    # If feasible rows alone exceed population capacity, truncate
    # within the feasible tier using J quality + diversity.
    # --------------------------------------------------------
    if len(safe_indices):
        safe_order = safe_indices[
            np.argsort(
                objective[safe_indices],
                kind='stable',
            )
        ]

        if len(safe_order) <= population_size:
            safe_selected = np.asarray(
                safe_order,
                dtype=int,
            )
        else:
            safe_selected = v3_diverse_truncation(
                population,
                safe_order,
                limit=population_size,
            )

        selected_parts.append(safe_selected)

    n_selected = sum(len(x) for x in selected_parts)
    remaining = population_size - n_selected

    # --------------------------------------------------------
    # Tier 2: robust-infeasible candidates.
    #
    # Only slots not occupied by feasible rows may be filled by
    # infeasible candidates. Within this tier, normalized robust
    # violation is primary and J is only a tie-breaker.
    # --------------------------------------------------------
    if remaining > 0:
        if len(infeasible_indices) < remaining:
            raise RuntimeError(
                "V3 environmental selection does not have enough "
                "infeasible candidates to fill remaining population slots."
            )

        infeasible_order_local = np.lexsort(
            (
                objective[infeasible_indices],
                violation[infeasible_indices],
            )
        )

        infeasible_order = infeasible_indices[
            infeasible_order_local
        ]

        infeasible_selected = v3_diverse_truncation(
            population,
            infeasible_order,
            limit=remaining,
        )

        selected_parts.append(infeasible_selected)

    selected = np.concatenate(selected_parts).astype(int)

    if len(selected) != population_size:
        raise RuntimeError(
            f"V3 environmental selection retained {len(selected)}/"
            f"{population_size} required D4-unique survivors."
        )

    # Strict invariant: if fewer than population_size feasible rows
    # existed, every one of them must survive.
    if len(safe_indices) <= population_size:
        if not set(safe_indices.tolist()).issubset(
            set(selected.tolist())
        ):
            raise RuntimeError(
                "V3 feasibility-dominance invariant violated: "
                "a feasible candidate was discarded for an infeasible one."
            )

    return selected


def v3_search_progress(details):
    """Return the active V3 search phase and its minimization metric.

    Before robust feasibility:
        minimize the best normalized robust violation.

    After at least one robust-feasible chromosome exists:
        minimize the best robust-safe physical objective J.
    """
    safe = np.asarray(details['safe'], dtype=bool)

    if np.any(safe):
        metric = float(
            np.min(
                np.asarray(details['secondary'], dtype=float)[safe]
            )
        )
        return 'feasible_objective', metric

    metric = float(
        np.min(
            np.asarray(
                details['normalized_violation'],
                dtype=float,
            )
        )
    )
    return 'constraint_violation', metric


def v3_new_search_state():
    """Create deterministic phase-aware stagnation state."""
    return {
        'phase': None,
        'best_metric': None,
        'stagnation': 0,
        'mutation_level_index': 0,
        'mutation_swaps': int(V3_MUTATION_LEVELS[0]),
        'immigrants': int(V3_BASE_IMMIGRANTS),
    }


def v3_update_search_state(state, details):
    """Update stagnation, mutation strength, and immigrant pressure.

    A phase transition from infeasible search to feasible-objective search
    resets stagnation because the optimizer has reached a fundamentally
    different part of the constrained problem.
    """
    state = dict(state)

    phase, metric = v3_search_progress(details)

    phase_changed = state['phase'] != phase
    improved = False

    if phase_changed:
        state['phase'] = phase
        state['best_metric'] = metric
        state['stagnation'] = 0
        improved = True

    elif (
        state['best_metric'] is None
        or metric < float(state['best_metric']) - V3_PROGRESS_TOL
    ):
        state['best_metric'] = metric
        state['stagnation'] = 0
        improved = True

    else:
        state['stagnation'] = int(state['stagnation']) + 1

    level_index = min(
        int(state['stagnation']) // int(V3_STAGNATION_WINDOW),
        len(V3_MUTATION_LEVELS) - 1,
    )

    state['mutation_level_index'] = int(level_index)
    state['mutation_swaps'] = int(
        V3_MUTATION_LEVELS[level_index]
    )

    state['immigrants'] = int(
        V3_STAGNANT_IMMIGRANTS
        if state['stagnation'] >= V3_STAGNATION_WINDOW
        else V3_BASE_IMMIGRANTS
    )

    return state, {
        'phase': phase,
        'metric': metric,
        'improved': bool(improved),
        'stagnation': int(state['stagnation']),
        'mutation_swaps': int(state['mutation_swaps']),
        'immigrants': int(state['immigrants']),
    }


def v3_combine_unique_candidates(
    groups,
    forbidden_d4,
):
    """Combine candidate groups while rejecting forbidden/D4 duplicates."""
    forbidden_d4 = set(forbidden_d4)

    rows = []
    seen = set()

    for group in groups:
        arr = np.asarray(group, dtype=int)

        if arr.size == 0:
            continue

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        for row in arr:
            if not is_hard_feasible(row):
                raise RuntimeError(
                    "V3 candidate combination received a hard-infeasible row."
                )

            key = v3_d4_key(row)

            if key in forbidden_d4 or key in seen:
                continue

            rows.append(np.asarray(row, dtype=int).copy())
            seen.add(key)

    if not rows:
        return np.empty((0, N_CELLS), dtype=int)

    return np.stack(rows).astype(int)


def inverse_outputs(model, x, scalers):
    outputs = model(tf.constant(x), training=False)
    return np.column_stack([
        scalers['keff'].inverse_transform(outputs[0].numpy().reshape(-1, 1)).ravel(),
        scalers['fa'].inverse_transform(outputs[1].numpy().reshape(-1, 1)).ravel(),
        scalers['fqr'].inverse_transform(outputs[2].numpy().reshape(-1, 1)).ravel(),
    ]).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round-dir', required=True)
    ap.add_argument(
        '--residual-model-dir',
        required=True,
        help='Frozen learned-residual safety models and joint95 metadata.'
    )
    ap.add_argument('--output-dir', required=True)
    ap.add_argument(
        '--strategy',
        required=True,
        choices=('random', 'sa', 'sqa'),
        help='Initialization strategy; all downstream GA mechanics remain identical.'
    )
    ap.add_argument(
        '--anchor-json',
        default=None,
        help='Fresh independent SA/SQA anchor bank. Required for sa/sqa; forbidden for random.'
    )
    ap.add_argument('--labelled-json', required=True)
    ap.add_argument('--seed-index', type=int, required=True)
    ap.add_argument('--threads', type=int, default=48)
    args = ap.parse_args()

    if not 0 <= args.seed_index < len(GA_SEEDS):
        raise SystemExit(f'--seed-index must be 0..{len(GA_SEEDS)-1}')
    seed = int(GA_SEEDS[args.seed_index])
    configure_tensorflow(args.threads)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    round_dir = Path(args.round_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labelled_path = Path(args.labelled_json)
    labelled_sha256 = hashlib.sha256(labelled_path.read_bytes()).hexdigest()
    labelled_rows = json.loads(labelled_path.read_text())

    if args.strategy == 'random':
        if args.anchor_json is not None:
            raise SystemExit('--anchor-json must not be supplied for strategy=random')
        anchor_path = None
        anchor_sha256 = None
        anchor_rows = []
        base_chromosomes = []
    else:
        if args.anchor_json is None:
            raise SystemExit(f'--anchor-json is required for strategy={args.strategy}')
        anchor_path = Path(args.anchor_json)
        anchor_sha256 = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
        anchor_rows = json.loads(anchor_path.read_text())
        base_chromosomes = [
            np.asarray(row['pattern'], dtype=int)
            for row in anchor_rows
        ]
        if not base_chromosomes:
            raise RuntimeError(f'No anchors found for strategy={args.strategy}')

    forbidden_d4 = {
        tuple(canonical_d4_key(row['pattern']))
        for row in labelled_rows
    }

    metadata = json.loads((round_dir / 'round_metadata.json').read_text())

    residual_model_dir = Path(args.residual_model_dir)
    residual_metadata_path = (
        residual_model_dir / 'residual_safety_metadata.json'
    )
    residual_metadata = json.loads(
        residual_metadata_path.read_text()
    )
    q = residual_metadata['joint95_conformal_q']

    residual_metadata_sha256 = hashlib.sha256(
        residual_metadata_path.read_bytes()
    ).hexdigest()

    round_arrays_path = round_dir / 'round_arrays.npz'
    round_metadata_path = round_dir / 'round_metadata.json'

    round_arrays_sha256 = hashlib.sha256(
        round_arrays_path.read_bytes()
    ).hexdigest()

    round_metadata_sha256 = hashlib.sha256(
        round_metadata_path.read_bytes()
    ).hexdigest()

    runner_script_sha256 = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()

    context = {
        # --------------------------------------------------------
        # Frozen surrogate / uncertainty / physics provenance.
        # --------------------------------------------------------
        'strategy': args.strategy,
        'round_version': metadata['version'],
        'round_arrays_sha256': round_arrays_sha256,
        'round_metadata_sha256': round_metadata_sha256,
        'round2_labels_sha256': metadata.get('round2_labels_sha256'),
        'training_unique_total': int(metadata['training_unique_total']),
        'cnn_seeds': list(CNN_TRAINING_SEEDS),
        'conformal_mode': 'learned_residual_joint95_bonferroni',
        'uncertainty_scale': 'crossfit_learned_absolute_error',
        'residual_model_metadata_sha256': residual_metadata_sha256,
        'residual_training_rows': int(
            residual_metadata['training_rows']
        ),
        'q': q,
        'anchor_sha256': anchor_sha256,
        'anchor_count': len(base_chromosomes),
        'labelled_exclusion_sha256': labelled_sha256,
        'labelled_d4_exclusion_count': len(forbidden_d4),
        'require_d4_novel_vs_known_openmc': True,

        # --------------------------------------------------------
        # V3 runner identity.
        # --------------------------------------------------------
        'runner_mode': (
            'v3_feasibility_first_diversity_preserving_'
            'adaptive_constrained_ga'
        ),
        'runner_script_sha256': runner_script_sha256,
        'selection_mode': (
            'feasibility_first_constraint_dominance'
        ),
        'initialization_source': (
            'fresh_random_hard_feasible'
            if args.strategy == 'random'
            else f'fresh_independent_{args.strategy}_anchors'
        ),

        # --------------------------------------------------------
        # Frozen paired-experiment controls.
        # --------------------------------------------------------
        'ga_seed': seed,
        'population_size': GA_POPULATION_SIZE,
        'generations': GA_GENERATIONS,
        'parent_pool_target': GA_PARENTS,
        'base_mutation_swaps': GA_MUTATION_SWAPS,
        'crossover_swaps': GA_CROSSOVER_SWAPS,
        'fa_weight': FA_WEIGHT,
        'fqr_weight': FQR_WEIGHT,

        # --------------------------------------------------------
        # V3 architecture controls.
        # --------------------------------------------------------
        'feasible_archive_size': V3_FEASIBLE_ARCHIVE_SIZE,
        'near_archive_size': V3_NEAR_ARCHIVE_SIZE,
        'feasible_elites': V3_FEASIBLE_ELITES,
        'near_elites': V3_NEAR_ELITES,
        'tournament_size': V3_TOURNAMENT_SIZE,
        'exploratory_crossover_fraction': (
            V3_EXPLORATORY_CROSSOVER_FRACTION
        ),
        'exploratory_guided_fraction': (
            V3_EXPLORATORY_GUIDED_FRACTION
        ),
        'base_immigrants': V3_BASE_IMMIGRANTS,
        'stagnant_immigrants': V3_STAGNANT_IMMIGRANTS,
        'stagnation_window': V3_STAGNATION_WINDOW,
        'progress_tolerance': V3_PROGRESS_TOL,
        'mutation_levels': list(V3_MUTATION_LEVELS),
        'diversity_thresholds': list(
            V3_DIVERSITY_THRESHOLDS
        ),
        'diversity_metric': (
            'normalized_D4_canonical_Hamming_193'
        ),
        'tail_capture_generations': [
            0, 1, 5, 10, 25, 50, 75, 100, 125, 150,
            175, 200, 225, 250, 275, 300,
        ],
    }
    context_sha = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    summary_path = out_dir / f'ga_seed_{seed}_summary.json'
    archive_path = out_dir / f'ga_seed_{seed}_archive.json'
    if summary_path.exists() and archive_path.exists():
        try:
            old = json.loads(summary_path.read_text())
            if old.get('status') == 'COMPLETED' and old.get('context_sha256') == context_sha:
                print(json.dumps({'status': 'REUSED', 'seed': seed, 'summary': str(summary_path)}, indent=2))
                return
        except Exception:
            pass

    scalers = {
        name: joblib.load(round_dir / f'{name}_scaler.pkl')
        for name in ('keff', 'fa', 'fqr')
    }
    models = []
    for cnn_seed in CNN_TRAINING_SEEDS:
        path = round_dir / 'models' / f'cnn_seed_{cnn_seed}.keras'
        models.append(
            tf.keras.models.load_model(
                path,
                custom_objects=CUSTOM_OBJECTS,
                compile=False,
            )
        )

    residual_models = {
        'keff': joblib.load(
            residual_model_dir / 'residual_keff.joblib'
        ),
        'Fa': joblib.load(
            residual_model_dir / 'residual_Fa.joblib'
        ),
        'Fq_r': joblib.load(
            residual_model_dir / 'residual_Fq_r.joblib'
        ),
    }

    def make_residual_features(population, mu, sd):
        population = np.asarray(population, dtype=np.float32)

        predicted_j = (
            FA_WEIGHT * mu[:, 1] / FA_LIMIT
            + FQR_WEIGHT * mu[:, 2] / FQR_LIMIT
        )[:, None]

        fa_margin = (FA_LIMIT - mu[:, 1])[:, None]
        fqr_margin = (FQR_LIMIT - mu[:, 2])[:, None]

        return np.concatenate(
            [
                population,
                mu.astype(np.float32),
                sd.astype(np.float32),
                predicted_j.astype(np.float32),
                fa_margin.astype(np.float32),
                fqr_margin.astype(np.float32),
            ],
            axis=1,
        )

    def predict_population(population):
        population = np.asarray(population, dtype=int)
        x = patterns_to_onehot_grid(population)

        per_seed = np.stack(
            [
                inverse_outputs(model, x, scalers)
                for model in models
            ],
            axis=0,
        )

        mu = per_seed.mean(axis=0)
        sd = per_seed.std(axis=0, ddof=1)

        residual_x = make_residual_features(
            population,
            mu,
            sd,
        )

        learned_scale = np.column_stack([
            np.maximum(
                residual_models['keff'].predict(residual_x),
                1e-12,
            ),
            np.maximum(
                residual_models['Fa'].predict(residual_x),
                1e-12,
            ),
            np.maximum(
                residual_models['Fq_r'].predict(residual_x),
                1e-12,
            ),
        ]).astype(float)

        lower_k = (
            mu[:, 0]
            - float(q['keff_lower']) * learned_scale[:, 0]
        )
        upper_k = (
            mu[:, 0]
            + float(q['keff_upper']) * learned_scale[:, 0]
        )
        upper_fa = (
            mu[:, 1]
            + float(q['Fa_upper']) * learned_scale[:, 1]
        )
        upper_fqr = (
            mu[:, 2]
            + float(q['Fq_r_upper']) * learned_scale[:, 2]
        )

        return (
            mu,
            sd,
            learned_scale,
            lower_k,
            upper_k,
            upper_fa,
            upper_fqr,
        )

    def compute_ga_fitness(population):
        population = np.asarray(population, dtype=int)
        (
            mu,
            sd,
            learned_scale,
            lower_k,
            upper_k,
            upper_fa,
            upper_fqr,
        ) = predict_population(population)
        hard_loading = np.array(
            [count_violations(row)['total'] for row in population],
            dtype=float,
        )
        d4_novel = np.array(
            [tuple(canonical_d4_key(row)) not in forbidden_d4 for row in population],
            dtype=bool,
        )
        hard = hard_loading + (~d4_novel).astype(float)

        physics_violation = (
            np.maximum(0.0, KEFF_LO - lower_k) ** 2
            + np.maximum(0.0, upper_k - KEFF_HI) ** 2
            + np.maximum(0.0, upper_fa - FA_LIMIT) ** 2
            + np.maximum(0.0, upper_fqr - FQR_LIMIT) ** 2
        )

        # V3 normalized robust-constraint violation.
        # This is the quantity used by feasibility-first ranking.
        # The copied V2 penalty fitness remains temporarily below only
        # until the explicit V3 evolutionary loop replaces PyGAD.
        normalized_violation = v3_normalized_physics_violation(
            lower_k,
            upper_k,
            upper_fa,
            upper_fqr,
        )

        secondary = (
            FA_WEIGHT * mu[:, 1] / FA_LIMIT
            + FQR_WEIGHT * mu[:, 2] / FQR_LIMIT
        )

        safe = (
            (hard == 0)
            & (normalized_violation <= 0.0)
        )
        fitness = np.where(
            safe,
            -secondary,
            -PHYSICS_CONSTRAINT_PENALTY
            - HARD_CONSTRAINT_PENALTY * hard
            - PHYSICS_CONSTRAINT_PENALTY * physics_violation
            - secondary,
        )
        details = {
            'mu': mu,
            'sd': sd,
            'learned_scale': learned_scale,
            'lower_k': lower_k,
            'upper_k': upper_k,
            'upper_fa': upper_fa, 'upper_fqr': upper_fqr,
            'safe': safe,
            'secondary': secondary,
            'normalized_violation': normalized_violation,
            'hard_violation': hard,
            'd4_novel': d4_novel,
        }
        return fitness.astype(float), details

    rng = np.random.default_rng(seed)
    initial_population = build_feasible_population(
        GA_POPULATION_SIZE,
        rng=rng,
        base_chromosomes=base_chromosomes,
        forbidden_d4=forbidden_d4,
    )
    run_archive = {}

    # ============================================================
    # V3 EXPLICIT CONSTRAINED-EVOLUTION LOOP
    #
    # IMPORTANT:
    # - The legacy million-scale ``fitness`` returned above is retained
    #   only for backwards-compatible diagnostic artifacts.
    # - It is NOT used for V3 selection, survival, archives, or progress.
    # - V3 evolutionary decisions use:
    #       feasibility first
    #       -> safe J
    #       -> normalized robust violation
    #       -> diversity-aware survival.
    # ============================================================

    # ------------------------------------------------------------
    # Explicit generation-0 evaluation.
    # ------------------------------------------------------------
    initial_fit, initial_d = compute_ga_fitness(
        initial_population
    )

    initial_order = v3_rank_population(initial_d)
    initial_best_index = int(initial_order[0])
    initial_safe_mask = np.asarray(
        initial_d['safe'],
        dtype=bool,
    )

    initial_best_fitness = float(
        initial_fit[initial_best_index]
    )
    initial_best_objective = float(
        initial_d['secondary'][initial_best_index]
    )
    initial_best_calibrated_safe = bool(
        initial_safe_mask[initial_best_index]
    )
    initial_calibrated_safe_fraction = float(
        np.mean(initial_safe_mask)
    )

    if bool(np.any(initial_safe_mask)):
        initial_safe_indices = np.flatnonzero(
            initial_safe_mask
        )
        initial_best_safe_index = int(
            initial_safe_indices[
                np.argmin(
                    initial_d['secondary'][
                        initial_safe_indices
                    ]
                )
            ]
        )
        initial_best_safe_objective = float(
            initial_d['secondary'][
                initial_best_safe_index
            ]
        )
    else:
        initial_best_safe_objective = None

    # Legacy compatibility history: stores the old scalar diagnostic
    # fitness evaluated at the V3-best chromosome of each generation.
    # It does not drive V3 evolution.
    convergence = [
        initial_best_fitness
    ]

    safe_fraction_history = [
        initial_calibrated_safe_fraction
    ]

    safe_objective_history = [
        initial_best_safe_objective
    ]

    first_safe_generation = (
        0 if bool(np.any(initial_safe_mask)) else None
    )

    # ------------------------------------------------------------
    # Active V3 archives.
    # ------------------------------------------------------------
    feasible_archive = {}
    near_archive = {}

    feasible_archive, near_archive = v3_update_archives(
        feasible_archive,
        near_archive,
        initial_population,
        initial_d,
        generation=0,
    )

    # ------------------------------------------------------------
    # Phase-aware adaptive-search state.
    # ------------------------------------------------------------
    search_state = v3_new_search_state()
    search_state, initial_search_info = (
        v3_update_search_state(
            search_state,
            initial_d,
        )
    )

    # ============================================================
    # V3 optimizer-tail capture.
    #
    # Generation 150 remains explicit for direct V2-vs-V3
    # comparison at equal search depth. V3 then continues to 300.
    # ============================================================
    TAIL_CAPTURE_GENERATIONS = (
        0, 1, 5, 10, 25, 50, 75, 100, 125, 150,
        175, 200, 225, 250, 275, 300,
    )

    tail_snapshots = []

    def capture_tail_snapshot(
        generation,
        population,
        fit,
        d,
    ):
        generation = int(generation)

        if generation not in TAIL_CAPTURE_GENERATIONS:
            return

        population = np.asarray(
            population,
            dtype=np.int8,
        ).copy()

        fit = np.asarray(
            fit,
            dtype=np.float32,
        ).copy()

        tail_snapshots.append({
            'generation': generation,
            'population': population,
            'fitness': fit,
            'objective': np.asarray(
                d['secondary'],
                dtype=np.float32,
            ).copy(),
            'normalized_violation': np.asarray(
                d['normalized_violation'],
                dtype=np.float32,
            ).copy(),
            'ensemble_mean': np.asarray(
                d['mu'],
                dtype=np.float32,
            ).copy(),
            'ensemble_std': np.asarray(
                d['sd'],
                dtype=np.float32,
            ).copy(),
            'calibrated_keff_lower': np.asarray(
                d['lower_k'],
                dtype=np.float32,
            ).copy(),
            'calibrated_keff_upper': np.asarray(
                d['upper_k'],
                dtype=np.float32,
            ).copy(),
            'calibrated_Fa_upper': np.asarray(
                d['upper_fa'],
                dtype=np.float32,
            ).copy(),
            'calibrated_Fq_r_upper': np.asarray(
                d['upper_fqr'],
                dtype=np.float32,
            ).copy(),
            'calibrated_safe': np.asarray(
                d['safe'],
                dtype=np.uint8,
            ).copy(),
            'd4_novel': np.asarray(
                d['d4_novel'],
                dtype=np.uint8,
            ).copy(),
            'population_diversity': float(
                v3_population_diversity(
                    population
                )
            ),
        })

    # ------------------------------------------------------------
    # Backwards-compatible observational archive.
    #
    # Unlike V2, candidate ordering here follows V3 constraint
    # dominance. This archive still does not itself drive evolution;
    # active feasible/near archives above do.
    # ------------------------------------------------------------
    def record_observational_archive(
        generation,
        population,
        fit,
        d,
        limit=10,
    ):
        order = v3_rank_population(d)

        for idx in order[:int(limit)]:
            idx = int(idx)
            chrom = np.asarray(
                population[idx],
                dtype=int,
            )

            if not is_hard_feasible(chrom):
                continue

            if not bool(d['d4_novel'][idx]):
                continue

            key = tuple(chrom.tolist())

            row = {
                'pattern': chrom.tolist(),
                'fitness': float(fit[idx]),
                'objective': float(
                    d['secondary'][idx]
                ),
                'normalized_violation': float(
                    d['normalized_violation'][idx]
                ),
                'ensemble_mean_keff': float(
                    d['mu'][idx, 0]
                ),
                'ensemble_mean_Fa': float(
                    d['mu'][idx, 1]
                ),
                'ensemble_mean_Fq_r': float(
                    d['mu'][idx, 2]
                ),
                'ensemble_std_keff': float(
                    d['sd'][idx, 0]
                ),
                'ensemble_std_Fa': float(
                    d['sd'][idx, 1]
                ),
                'ensemble_std_Fq_r': float(
                    d['sd'][idx, 2]
                ),
                'calibrated_keff_lower': float(
                    d['lower_k'][idx]
                ),
                'calibrated_keff_upper': float(
                    d['upper_k'][idx]
                ),
                'calibrated_Fa_upper': float(
                    d['upper_fa'][idx]
                ),
                'calibrated_Fq_r_upper': float(
                    d['upper_fqr'][idx]
                ),
                'calibrated_safe': bool(
                    d['safe'][idx]
                ),
                'd4_novel_vs_labelled_10000': bool(
                    d['d4_novel'][idx]
                ),
                'generation': int(generation),
                'ga_seed': seed,
            }

            old = run_archive.get(key)

            if old is None:
                run_archive[key] = row
                continue

            new_safe = bool(
                row['calibrated_safe']
            )
            old_safe = bool(
                old['calibrated_safe']
            )

            better = False

            if new_safe != old_safe:
                better = new_safe

            elif new_safe:
                better = (
                    row['objective']
                    < old['objective']
                )

            else:
                if (
                    row['normalized_violation']
                    != old.get(
                        'normalized_violation',
                        float('inf'),
                    )
                ):
                    better = (
                        row['normalized_violation']
                        < old.get(
                            'normalized_violation',
                            float('inf'),
                        )
                    )
                else:
                    better = (
                        row['objective']
                        < old['objective']
                    )

            if better:
                run_archive[key] = row

    # ------------------------------------------------------------
    # Generation-0 capture and diagnostics.
    # ------------------------------------------------------------
    capture_tail_snapshot(
        0,
        initial_population,
        initial_fit,
        initial_d,
    )

    record_observational_archive(
        0,
        initial_population,
        initial_fit,
        initial_d,
    )

    initial_best_violation = float(
        np.min(
            initial_d['normalized_violation']
        )
    )

    v3_diagnostics = [{
        'generation': 0,
        'phase': initial_search_info['phase'],
        'progress_metric': float(
            initial_search_info['metric']
        ),
        'improved': bool(
            initial_search_info['improved']
        ),
        'stagnation': int(
            initial_search_info['stagnation']
        ),
        'mutation_swaps': int(
            initial_search_info['mutation_swaps']
        ),
        'immigrant_target': int(
            initial_search_info['immigrants']
        ),
        'immigrants_generated': 0,
        'offspring_generated': 0,
        'exploratory_offspring': 0,
        'offspring_fallback_immigrants': 0,
        'candidate_pool_size': int(
            len(initial_population)
        ),
        'safe_fraction': float(
            np.mean(initial_d['safe'])
        ),
        'safe_count': int(
            np.sum(initial_d['safe'])
        ),
        'best_normalized_violation': (
            initial_best_violation
        ),
        'best_safe_objective': (
            initial_best_safe_objective
        ),
        'population_diversity': float(
            v3_population_diversity(
                initial_population
            )
        ),
        'feasible_archive_size': int(
            len(feasible_archive)
        ),
        'near_archive_size': int(
            len(near_archive)
        ),
        'min_Fa_upper': float(
            np.min(initial_d['upper_fa'])
        ),
        'median_Fa_upper': float(
            np.median(initial_d['upper_fa'])
        ),
    }]

    # ------------------------------------------------------------
    # Current generation state.
    # ------------------------------------------------------------
    current_population = np.asarray(
        initial_population,
        dtype=int,
    ).copy()

    current_fit = np.asarray(
        initial_fit,
        dtype=float,
    ).copy()

    current_d = {
        k: np.asarray(v).copy()
        for k, v in initial_d.items()
    }

    start = time.perf_counter()

    # ============================================================
    # V3 generations 1..300.
    # ============================================================
    for generation in range(
        1,
        GA_GENERATIONS + 1,
    ):
        # --------------------------------------------------------
        # Active archive elites.
        # --------------------------------------------------------
        feasible_elites = v3_archive_patterns(
            feasible_archive,
            limit=V3_FEASIBLE_ELITES,
        )

        near_elites = v3_archive_patterns(
            near_archive,
            limit=V3_NEAR_ELITES,
        )

        # --------------------------------------------------------
        # Parent pool:
        # current population + persistent archive representatives.
        #
        # D4 duplicates are collapsed here, so archives contribute
        # only when they carry information absent from the current
        # generation.
        # --------------------------------------------------------
        parent_pool = v3_combine_unique_candidates(
            [
                current_population,
                feasible_elites,
                near_elites,
            ],
            forbidden_d4=forbidden_d4,
        )

        if len(parent_pool) < 2:
            raise RuntimeError(
                "V3 parent pool contains fewer than 2 "
                "D4-unique candidates."
            )

        parent_fit, parent_d = compute_ga_fitness(
            parent_pool
        )

        # --------------------------------------------------------
        # Preserve the frozen 40-parent mating scale.
        #
        # Parent candidates may come from the current population or
        # persistent active archives, but exactly GA_PARENTS unique
        # mating parents are chosen using V3 feasibility-first
        # tournaments before offspring production.
        # --------------------------------------------------------
        mating_indices = v3_select_mating_pool_indices(
            parent_d,
            n_parents=GA_PARENTS,
            rng=rng,
        )

        mating_pool = np.asarray(
            parent_pool[mating_indices],
            dtype=int,
        )

        mating_d = {
            k: np.asarray(v)[mating_indices]
            for k, v in parent_d.items()
        }

        # --------------------------------------------------------
        # Generate one population-sized offspring cohort.
        #
        # Accepted offspring must be:
        # - exact-inventory
        # - hard-feasible
        # - outside known OpenMC D4 set
        # - D4-distinct from current/archive elites
        # - D4-distinct from each other.
        # --------------------------------------------------------
        blocked_d4 = {
            v3_d4_key(row)
            for row in parent_pool
        }

        offspring = []
        offspring_keys = set()
        exploratory_accepted = 0

        attempts = 0
        max_attempts = (
            GA_POPULATION_SIZE * 250
        )

        mutation_swaps = int(
            search_state['mutation_swaps']
        )

        while (
            len(offspring) < GA_POPULATION_SIZE
            and attempts < max_attempts
        ):
            attempts += 1

            pair = v3_tournament_parent_indices(
                mating_d,
                n_parents=2,
                rng=rng,
            )

            ia = int(pair[0])
            ib = int(pair[1])

            if ia == ib:
                alternatives = np.delete(
                    np.arange(len(mating_pool)),
                    ia,
                )
                ib = int(rng.choice(alternatives))

            exploratory = bool(
                rng.random()
                < V3_EXPLORATORY_CROSSOVER_FRACTION
            )

            child = v3_reproduce_child(
                mating_pool[ia],
                mating_pool[ib],
                rng,
                mutation_swaps=mutation_swaps,
                exploratory=exploratory,
            )

            key = v3_d4_key(child)

            if (
                key in forbidden_d4
                or key in blocked_d4
                or key in offspring_keys
            ):
                continue

            offspring.append(child)
            offspring_keys.add(key)

            if exploratory:
                exploratory_accepted += 1

        # If crossover/mutation cannot produce enough distinct D4
        # offspring, fill only the shortfall using globally fresh
        # hard-feasible constructors rather than duplicating rows.
        fallback_count = (
            GA_POPULATION_SIZE
            - len(offspring)
        )

        if fallback_count > 0:
            fallback_blocked = (
                blocked_d4
                | offspring_keys
            )

            fallback = v3_generate_immigrants(
                fallback_count,
                rng,
                forbidden_d4=forbidden_d4,
                blocked_d4=fallback_blocked,
            )

            for row in fallback:
                key = v3_d4_key(row)
                offspring.append(row)
                offspring_keys.add(key)

        offspring = np.stack(
            offspring
        ).astype(int)

        # --------------------------------------------------------
        # Explicit global immigrants.
        # --------------------------------------------------------
        immigrant_target = int(
            search_state['immigrants']
        )

        immigrant_blocked = (
            blocked_d4
            | offspring_keys
        )

        immigrants = v3_generate_immigrants(
            immigrant_target,
            rng,
            forbidden_d4=forbidden_d4,
            blocked_d4=immigrant_blocked,
        )

        # --------------------------------------------------------
        # Candidate pool = parents/survivors + archive elites
        #                  + offspring + fresh immigrants.
        # --------------------------------------------------------
        candidate_pool = v3_combine_unique_candidates(
            [
                current_population,
                feasible_elites,
                near_elites,
                offspring,
                immigrants,
            ],
            forbidden_d4=forbidden_d4,
        )

        if len(candidate_pool) < GA_POPULATION_SIZE:
            raise RuntimeError(
                f"V3 generation {generation}: only "
                f"{len(candidate_pool)} unique candidates "
                "available for environmental selection."
            )

        candidate_fit, candidate_d = (
            compute_ga_fitness(
                candidate_pool
            )
        )

        # --------------------------------------------------------
        # Update active archives BEFORE survivor truncation.
        #
        # Therefore a valuable candidate is not forgotten merely
        # because environmental diversity removes it this generation.
        # --------------------------------------------------------
        feasible_archive, near_archive = (
            v3_update_archives(
                feasible_archive,
                near_archive,
                candidate_pool,
                candidate_d,
                generation=generation,
            )
        )

        # --------------------------------------------------------
        # Feasibility-first + diversity-aware survivor selection.
        # --------------------------------------------------------
        survivor_indices = (
            v3_environmental_select(
                candidate_pool,
                candidate_d,
                population_size=GA_POPULATION_SIZE,
            )
        )

        current_population = np.asarray(
            candidate_pool[survivor_indices],
            dtype=int,
        ).copy()

        current_fit = np.asarray(
            candidate_fit[survivor_indices],
            dtype=float,
        ).copy()

        current_d = {
            k: np.asarray(v)[
                survivor_indices
            ].copy()
            for k, v in candidate_d.items()
        }

        # Population construction invariants.
        if not all(
            is_hard_feasible(row)
            for row in current_population
        ):
            raise RuntimeError(
                f"V3 generation {generation}: "
                "hard-feasibility invariant failed."
            )

        population_d4 = [
            v3_d4_key(row)
            for row in current_population
        ]

        if (
            len(population_d4)
            != len(set(population_d4))
        ):
            raise RuntimeError(
                f"V3 generation {generation}: "
                "population contains D4 duplicates."
            )

        if any(
            key in forbidden_d4
            for key in population_d4
        ):
            raise RuntimeError(
                f"V3 generation {generation}: "
                "known-OpenMC D4 pattern entered population."
            )

        # --------------------------------------------------------
        # Update adaptive-search controller from the selected
        # population. The best candidate under V3 dominance is
        # guaranteed to survive environmental selection.
        # --------------------------------------------------------
        search_state, search_info = (
            v3_update_search_state(
                search_state,
                current_d,
            )
        )

        current_order = v3_rank_population(
            current_d
        )
        best_index = int(current_order[0])

        convergence.append(
            float(current_fit[best_index])
        )

        safe_mask = np.asarray(
            current_d['safe'],
            dtype=bool,
        )

        safe_fraction = float(
            np.mean(safe_mask)
        )

        safe_fraction_history.append(
            safe_fraction
        )

        if bool(np.any(safe_mask)):
            safe_indices = np.flatnonzero(
                safe_mask
            )

            best_safe_objective = float(
                np.min(
                    current_d['secondary'][
                        safe_indices
                    ]
                )
            )

            safe_objective_history.append(
                best_safe_objective
            )
        else:
            best_safe_objective = None
            safe_objective_history.append(
                None
            )

        if (
            first_safe_generation is None
            and bool(np.any(safe_mask))
        ):
            first_safe_generation = int(
                generation
            )

        capture_tail_snapshot(
            generation,
            current_population,
            current_fit,
            current_d,
        )

        record_observational_archive(
            generation,
            current_population,
            current_fit,
            current_d,
        )

        best_violation = float(
            np.min(
                current_d[
                    'normalized_violation'
                ]
            )
        )

        population_diversity = float(
            v3_population_diversity(
                current_population
            )
        )

        v3_diagnostics.append({
            'generation': int(generation),
            'phase': search_info['phase'],
            'progress_metric': float(
                search_info['metric']
            ),
            'improved': bool(
                search_info['improved']
            ),
            'stagnation': int(
                search_info['stagnation']
            ),
            'mutation_swaps': int(
                search_info['mutation_swaps']
            ),
            'immigrant_target': int(
                search_info['immigrants']
            ),
            'immigrants_generated': int(
                len(immigrants)
            ),
            'offspring_generated': int(
                len(offspring)
            ),
            'exploratory_offspring': int(
                exploratory_accepted
            ),
            'offspring_fallback_immigrants': int(
                fallback_count
            ),
            'candidate_pool_size': int(
                len(candidate_pool)
            ),
            'mating_pool_size': int(
                len(mating_pool)
            ),
            'safe_fraction': safe_fraction,
            'safe_count': int(
                np.sum(safe_mask)
            ),
            'best_normalized_violation': (
                best_violation
            ),
            'best_safe_objective': (
                best_safe_objective
            ),
            'population_diversity': (
                population_diversity
            ),
            'feasible_archive_size': int(
                len(feasible_archive)
            ),
            'near_archive_size': int(
                len(near_archive)
            ),
            'min_Fa_upper': float(
                np.min(current_d['upper_fa'])
            ),
            'median_Fa_upper': float(
                np.median(current_d['upper_fa'])
            ),
        })

        if (
            generation == 1
            or generation % 25 == 0
        ):
            safe_j_text = (
                "None"
                if best_safe_objective is None
                else f"{best_safe_objective:.6f}"
            )

            print(
                f"Seed {seed}; "
                f"generation {generation:3d}: "
                f"phase={search_info['phase']}, "
                f"best_violation={best_violation:.8f}, "
                f"safe={safe_fraction:.1%}, "
                f"best_safe_J={safe_j_text}, "
                f"diversity={population_diversity:.4f}, "
                f"mutation={search_info['mutation_swaps']}, "
                f"immigrants={search_info['immigrants']}, "
                f"AF={len(feasible_archive)}, "
                f"AN={len(near_archive)}",
                flush=True,
            )

    elapsed = time.perf_counter() - start

    # Explicit final state for the downstream artifact writer.
    final_population = np.asarray(
        current_population,
        dtype=int,
    ).copy()

    final_fit = np.asarray(
        current_fit,
        dtype=float,
    ).copy()

    final_d = {
        k: np.asarray(v).copy()
        for k, v in current_d.items()
    }

    final_order = v3_rank_population(
        final_d
    )

    final_best_index = int(
        final_order[0]
    )

    final_best_chromosome = np.asarray(
        final_population[
            final_best_index
        ],
        dtype=int,
    ).copy()

    # ============================================================
    # Freeze optimizer-tail trajectory for this run.
    # Shape conventions:
    #   population    : [checkpoint, population, 193]
    #   fitness       : [checkpoint, population]
    #   objective     : [checkpoint, population]
    #   ensemble_mean : [checkpoint, population, 3]
    #   ensemble_std  : [checkpoint, population, 3]
    # ============================================================
    captured_generations = [
        int(s['generation']) for s in tail_snapshots
    ]

    expected_generations = list(TAIL_CAPTURE_GENERATIONS)

    if captured_generations != expected_generations:
        raise RuntimeError(
            'Tail capture generation mismatch: '
            f'got={captured_generations}, '
            f'expected={expected_generations}'
        )

    tail_path = (
        out_dir /
        f'ga_seed_{seed}_tail_capture.npz'
    )

    np.savez_compressed(
        tail_path,
        generations=np.asarray(
            captured_generations,
            dtype=np.int16
        ),
        population=np.stack([
            s['population'] for s in tail_snapshots
        ]),
        fitness=np.stack([
            s['fitness'] for s in tail_snapshots
        ]),
        objective=np.stack([
            s['objective'] for s in tail_snapshots
        ]),
        normalized_violation=np.stack([
            s['normalized_violation']
            for s in tail_snapshots
        ]),
        ensemble_mean=np.stack([
            s['ensemble_mean'] for s in tail_snapshots
        ]),
        ensemble_std=np.stack([
            s['ensemble_std'] for s in tail_snapshots
        ]),
        calibrated_keff_lower=np.stack([
            s['calibrated_keff_lower']
            for s in tail_snapshots
        ]),
        calibrated_keff_upper=np.stack([
            s['calibrated_keff_upper']
            for s in tail_snapshots
        ]),
        calibrated_Fa_upper=np.stack([
            s['calibrated_Fa_upper']
            for s in tail_snapshots
        ]),
        calibrated_Fq_r_upper=np.stack([
            s['calibrated_Fq_r_upper']
            for s in tail_snapshots
        ]),
        calibrated_safe=np.stack([
            s['calibrated_safe']
            for s in tail_snapshots
        ]),
        d4_novel=np.stack([
            s['d4_novel']
            for s in tail_snapshots
        ]),
        population_diversity=np.asarray([
            s['population_diversity']
            for s in tail_snapshots
        ], dtype=np.float32),
        ga_seed=np.asarray(seed, dtype=np.int64),
        seed_index=np.asarray(
            args.seed_index,
            dtype=np.int16
        ),
        strategy=np.asarray(args.strategy),
    )

    # ============================================================
    # V3 final winner and archive outputs.
    #
    # ``final_best_chromosome`` was selected by V3 constraint
    # dominance, not by the legacy million-scale penalty fitness.
    # ============================================================
    best_chromosome = np.asarray(
        final_best_chromosome,
        dtype=int,
    )

    hard_audit = count_violations(
        best_chromosome
    )

    if hard_audit['total'] != 0:
        raise RuntimeError(
            f'Final V3 chromosome failed hard constraints: '
            f'{hard_audit}'
        )

    best_fit, best_d = compute_ga_fitness(
        best_chromosome.reshape(1, -1)
    )

    best_d4_novel = bool(
        best_d['d4_novel'][0]
    )

    if not best_d4_novel:
        raise RuntimeError(
            'Final V3 chromosome is D4-equivalent '
            'to a known OpenMC pattern.'
        )

    best_key = tuple(
        best_chromosome.tolist()
    )

    best_row = {
        'pattern': best_chromosome.tolist(),
        'fitness': float(best_fit[0]),
        'objective': float(
            best_d['secondary'][0]
        ),
        'normalized_violation': float(
            best_d['normalized_violation'][0]
        ),
        'ensemble_mean_keff': float(
            best_d['mu'][0, 0]
        ),
        'ensemble_mean_Fa': float(
            best_d['mu'][0, 1]
        ),
        'ensemble_mean_Fq_r': float(
            best_d['mu'][0, 2]
        ),
        'ensemble_std_keff': float(
            best_d['sd'][0, 0]
        ),
        'ensemble_std_Fa': float(
            best_d['sd'][0, 1]
        ),
        'ensemble_std_Fq_r': float(
            best_d['sd'][0, 2]
        ),
        'calibrated_keff_lower': float(
            best_d['lower_k'][0]
        ),
        'calibrated_keff_upper': float(
            best_d['upper_k'][0]
        ),
        'calibrated_Fa_upper': float(
            best_d['upper_fa'][0]
        ),
        'calibrated_Fq_r_upper': float(
            best_d['upper_fqr'][0]
        ),
        'calibrated_safe': bool(
            best_d['safe'][0]
        ),
        'd4_novel_vs_labelled_10000': (
            best_d4_novel
        ),
        'generation': GA_GENERATIONS,
        'ga_seed': seed,
    }

    old = run_archive.get(best_key)

    if old is None:
        run_archive[best_key] = best_row
    else:
        new_safe = bool(
            best_row['calibrated_safe']
        )
        old_safe = bool(
            old['calibrated_safe']
        )

        if (
            (new_safe and not old_safe)
            or (
                new_safe
                and old_safe
                and best_row['objective']
                < old['objective']
            )
            or (
                not new_safe
                and not old_safe
                and (
                    best_row['normalized_violation']
                    < old.get(
                        'normalized_violation',
                        float('inf'),
                    )
                    or (
                        best_row['normalized_violation']
                        == old.get(
                            'normalized_violation',
                            float('inf'),
                        )
                        and best_row['objective']
                        < old['objective']
                    )
                )
            )
        ):
            run_archive[best_key] = best_row

    # Observational archive ordering now follows the same V3
    # feasibility-first semantics used by the optimizer.
    archive = sorted(
        run_archive.values(),
        key=lambda r: (
            0
            if bool(r['calibrated_safe'])
            else 1,
            (
                float(r['objective'])
                if bool(r['calibrated_safe'])
                else float(
                    r.get(
                        'normalized_violation',
                        float('inf'),
                    )
                )
            ),
            float(r['objective']),
        ),
    )

    for r in archive:
        r['d4_key_sha256'] = hashlib.sha256(
            np.asarray(
                canonical_d4_key(r['pattern']),
                dtype=np.uint8,
            ).tobytes()
        ).hexdigest()

    def serialize_active_archive(active_archive):
        rows = []

        for r in active_archive.values():
            out = {
                k: v
                for k, v in r.items()
                if k not in ('pattern', 'd4_key')
            }

            out['pattern'] = np.asarray(
                r['pattern'],
                dtype=int,
            ).tolist()

            out['d4_key_sha256'] = hashlib.sha256(
                np.asarray(
                    r['d4_key'],
                    dtype=np.uint8,
                ).tobytes()
            ).hexdigest()

            rows.append(out)

        return rows

    feasible_archive_rows = (
        serialize_active_archive(
            feasible_archive
        )
    )

    near_archive_rows = (
        serialize_active_archive(
            near_archive
        )
    )

    summary = {
        'status': 'COMPLETED',
        'context_sha256': context_sha,
        'context': context,
        'ga_seed': seed,
        # Legacy scalar fitness is retained only for compatibility.
        # V3 did not use this value for evolutionary selection.
        'best_fitness': float(best_fit[0]),
        'best_selection_mode': 'v3_feasibility_first_constraint_dominance',
        'best_objective': float(best_d['secondary'][0]),
        'best_normalized_violation': float(
            best_d['normalized_violation'][0]
        ),
        'best_ensemble_mean_keff': float(best_d['mu'][0, 0]),
        'best_ensemble_mean_Fa': float(best_d['mu'][0, 1]),
        'best_ensemble_mean_Fq_r': float(best_d['mu'][0, 2]),
        'best_calibrated_keff_lower': float(best_d['lower_k'][0]),
        'best_calibrated_keff_upper': float(best_d['upper_k'][0]),
        'best_calibrated_Fa_upper': float(best_d['upper_fa'][0]),
        'best_calibrated_Fq_r_upper': float(best_d['upper_fqr'][0]),
        'best_calibrated_safe': bool(best_d['safe'][0]),
        'best_d4_novel_vs_labelled_10000': best_d4_novel,

        'initial_best_fitness': initial_best_fitness,
        'initial_best_objective': initial_best_objective,
        'initial_best_calibrated_safe': initial_best_calibrated_safe,
        'initial_best_safe_objective': initial_best_safe_objective,
        'initial_calibrated_safe_fraction': initial_calibrated_safe_fraction,
        'objective_improvement_from_initial_safe_best': (
            float(initial_best_safe_objective - best_d['secondary'][0])
            if initial_best_safe_objective is not None
            and bool(best_d['safe'][0])
            else None
        ),

        'first_calibrated_safe_generation': first_safe_generation,
        'final_calibrated_safe_fraction': safe_fraction_history[-1] if safe_fraction_history else None,
        'hard_constraint_audit': hard_audit,
        'unique_archived_candidates': len(archive),

        # V3-native final diagnostics.
        'v3_generations': int(GA_GENERATIONS),
        'v3_final_population_diversity': float(
            v3_population_diversity(
                final_population
            )
        ),
        'v3_final_feasible_archive_size': int(
            len(feasible_archive)
        ),
        'v3_final_near_archive_size': int(
            len(near_archive)
        ),
        'v3_final_search_phase': str(
            search_state['phase']
        ),
        'v3_final_stagnation': int(
            search_state['stagnation']
        ),
        'v3_final_mutation_swaps': int(
            search_state['mutation_swaps']
        ),
        'v3_final_immigrant_target': int(
            search_state['immigrants']
        ),
        'v3_diagnostic_generations': int(
            len(v3_diagnostics)
        ),

        'ga_wall_seconds': float(elapsed),
        'threads': int(args.threads),
    }

    archive_path.write_text(
        json.dumps(archive, indent=2),
        encoding='utf-8',
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8',
    )

    (
        out_dir
        / f'ga_seed_{seed}_v3_feasible_archive.json'
    ).write_text(
        json.dumps(
            feasible_archive_rows,
            indent=2,
        ),
        encoding='utf-8',
    )

    (
        out_dir
        / f'ga_seed_{seed}_v3_near_archive.json'
    ).write_text(
        json.dumps(
            near_archive_rows,
            indent=2,
        ),
        encoding='utf-8',
    )

    (
        out_dir
        / f'ga_seed_{seed}_v3_diagnostics.json'
    ).write_text(
        json.dumps(
            v3_diagnostics,
            indent=2,
        ),
        encoding='utf-8',
    )

    (out_dir / f'ga_seed_{seed}_convergence.json').write_text(json.dumps(convergence), encoding='utf-8')
    (out_dir / f'ga_seed_{seed}_safe_fraction.json').write_text(json.dumps(safe_fraction_history), encoding='utf-8')
    (out_dir / f'ga_seed_{seed}_safe_objective.json').write_text(json.dumps(safe_objective_history), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
