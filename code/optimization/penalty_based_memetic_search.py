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

from local_refinement import (
    MEMETIC_CHECKPOINTS,
    MEMETIC_ESCAPE_SAMPLES,
    MEMETIC_MAX_DESCENT_MOVES,
    MEMETIC_MAX_EVALS_PER_SESSION,
    candidate_better,
    candidate_record,
    run_memetic_session,
    select_memetic_seed_index,
)

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
GA_PARENTS = 40
GA_MUTATION_SWAPS = 8
GA_CROSSOVER_SWAPS = 12
GA_ELITISM = 12
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
        'require_d4_novel_vs_labelled_10000': True,
        'runner_mode': 'v2m_300_penalty_ga_common_bounded_memetic',
        'runner_script_sha256': runner_script_sha256,
        'global_selection_mode': 'v2_penalty_fitness_pygad',
        'memetic_selection_mode': 'common_feasibility_first_constraint_dominance',
        'initialization_source': (
            'fresh_random_hard_feasible'
            if args.strategy == 'random'
            else f'fresh_independent_{args.strategy}_anchors'
        ),
        'ga_seed': seed,
        'population_size': GA_POPULATION_SIZE,
        'generations': GA_GENERATIONS,
        'parents': GA_PARENTS,
        'mutation_swaps': GA_MUTATION_SWAPS,
        'crossover_swaps': GA_CROSSOVER_SWAPS,
        'elitism': GA_ELITISM,
        'fa_weight': FA_WEIGHT,
        'fqr_weight': FQR_WEIGHT,

        # Common V2M/V3M memetic controls.
        'memetic_checkpoints': list(MEMETIC_CHECKPOINTS),
        'memetic_max_descent_moves': int(
            MEMETIC_MAX_DESCENT_MOVES
        ),
        'memetic_escape_operator': 'exact_two_accepted_feasible_swaps',
        'memetic_escape_samples': int(
            MEMETIC_ESCAPE_SAMPLES
        ),
        'memetic_max_evals_per_session': int(
            MEMETIC_MAX_EVALS_PER_SESSION
        ),
        'memetic_local_rule': (
            'feasible_over_infeasible__'
            'infeasible_min_normalized_violation_then_J__'
            'feasible_min_J'
        ),
        'memetic_d4_live_population_guard': True,
        'comparison_population_size': 100,
        'comparison_generations': 300,
        'comparison_parent_count': 40,
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

    surrogate_eval_counter = {
        'total': 0,
        'memetic': 0,
    }

    def compute_ga_fitness(population):
        population = np.asarray(population, dtype=int)

        if population.ndim == 1:
            population = population.reshape(1, -1)

        surrogate_eval_counter['total'] += int(len(population))
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
        secondary = FA_WEIGHT * mu[:, 1] / FA_LIMIT + FQR_WEIGHT * mu[:, 2] / FQR_LIMIT

        normalized_violation = (
            np.maximum(0.0, KEFF_LO - lower_k) / KEFF_LO
            + np.maximum(0.0, upper_k - KEFF_HI) / KEFF_HI
            + np.maximum(0.0, upper_fa - FA_LIMIT) / FA_LIMIT
            + np.maximum(0.0, upper_fqr - FQR_LIMIT) / FQR_LIMIT
        )

        safe = (hard == 0) & (physics_violation == 0)
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
            'd4_novel': d4_novel,
        }
        return fitness.astype(float), details

    def score_memetic_population(population):
        arr = np.asarray(population, dtype=int)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        _, d = compute_ga_fitness(arr)

        surrogate_eval_counter['memetic'] += int(len(arr))

        return {
            'safe': np.asarray(d['safe'], dtype=bool),
            'normalized_violation': np.asarray(
                d['normalized_violation'],
                dtype=float,
            ),
            'objective': np.asarray(
                d['secondary'],
                dtype=float,
            ),
        }

    rng = np.random.default_rng(seed)

    # Dedicated memetic RNG. It never consumes the PyGAD/global-search
    # RNG stream, so V2M differs from V2 only through candidate injection.
    memetic_rng = np.random.default_rng(seed + 3_000_001)

    memetic_sessions = []
    initial_population = build_feasible_population(
        GA_POPULATION_SIZE,
        rng=rng,
        base_chromosomes=base_chromosomes,
        forbidden_d4=forbidden_d4,
    )
    run_archive = {}

    # ------------------------------------------------------------
    # Explicit generation-0 evaluation.
    # History index now corresponds exactly to GA generation:
    # index 0 = initialized population, indices 1..150 = evolved.
    # ------------------------------------------------------------
    initial_fit, initial_d = compute_ga_fitness(initial_population)
    initial_best_index = int(np.argmax(initial_fit))
    initial_safe_mask = np.asarray(initial_d['safe'], dtype=bool)

    initial_best_fitness = float(initial_fit[initial_best_index])
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
        initial_safe_indices = np.flatnonzero(initial_safe_mask)
        initial_best_safe_index = int(
            initial_safe_indices[
                np.argmin(
                    initial_d['secondary'][initial_safe_indices]
                )
            ]
        )
        initial_best_safe_objective = float(
            initial_d['secondary'][initial_best_safe_index]
        )
    else:
        initial_best_safe_objective = None

    convergence = [initial_best_fitness]
    safe_fraction_history = [initial_calibrated_safe_fraction]
    safe_objective_history = [initial_best_safe_objective]

    first_safe_generation = (
        0 if bool(np.any(initial_safe_mask)) else None
    )

    # ============================================================
    # V2 OPTIMIZER-TAIL CAPTURE
    #
    # Pure observation only:
    # - no RNG calls
    # - no population modification
    # - no selection/mutation/crossover changes
    #
    # Full 100-member populations are frozen at fixed generations.
    # ============================================================
    TAIL_CAPTURE_GENERATIONS = (
        0, 1, 5, 10, 25, 50, 75, 100, 125, 150,
        175, 200, 225, 250, 275, 300,
    )

    tail_snapshots = []

    def capture_tail_snapshot(generation, population, fit, d):
        generation = int(generation)

        if generation not in TAIL_CAPTURE_GENERATIONS:
            return

        population = np.asarray(population, dtype=np.int8).copy()
        fit = np.asarray(fit, dtype=np.float32).copy()

        tail_snapshots.append({
            'generation': generation,
            'population': population,
            'fitness': fit,
            'objective': np.asarray(
                d['secondary'], dtype=np.float32
            ).copy(),
            'ensemble_mean': np.asarray(
                d['mu'], dtype=np.float32
            ).copy(),
            'ensemble_std': np.asarray(
                d['sd'], dtype=np.float32
            ).copy(),
            'calibrated_keff_lower': np.asarray(
                d['lower_k'], dtype=np.float32
            ).copy(),
            'calibrated_keff_upper': np.asarray(
                d['upper_k'], dtype=np.float32
            ).copy(),
            'calibrated_Fa_upper': np.asarray(
                d['upper_fa'], dtype=np.float32
            ).copy(),
            'calibrated_Fq_r_upper': np.asarray(
                d['upper_fqr'], dtype=np.float32
            ).copy(),
            'calibrated_safe': np.asarray(
                d['safe'], dtype=np.uint8
            ).copy(),
            'd4_novel': np.asarray(
                d['d4_novel'], dtype=np.uint8
            ).copy(),
        })

    # Explicitly capture generation 0.
    capture_tail_snapshot(
        0,
        initial_population,
        initial_fit,
        initial_d,
    )

    def fitness_function(ga_instance, solutions, solution_indices):
        arr = np.asarray(solutions, dtype=int)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)
        fit, _ = compute_ga_fitness(arr)
        return float(fit[0]) if single else fit.tolist()

    def on_generation(ga_instance):
        nonlocal first_safe_generation

        generation = int(
            ga_instance.generations_completed
        )

        population = np.asarray(
            ga_instance.population,
            dtype=int,
        ).copy()

        fit, d = compute_ga_fitness(population)

        # --------------------------------------------------------
        # V2M common memetic checkpoint.
        #
        # Global V2 selection remains penalty-based PyGAD.
        # Only this bounded local refinement uses the common
        # V2M/V3M feasibility-first memetic rule.
        # --------------------------------------------------------
        if generation in MEMETIC_CHECKPOINTS:
            local_score = {
                'safe': np.asarray(
                    d['safe'],
                    dtype=bool,
                ),
                'normalized_violation': np.asarray(
                    d['normalized_violation'],
                    dtype=float,
                ),
                'objective': np.asarray(
                    d['secondary'],
                    dtype=float,
                ),
            }

            seed_index = select_memetic_seed_index(
                local_score
            )

            start_local = candidate_record(
                local_score,
                seed_index,
            )

            session_forbidden_d4 = set(
                forbidden_d4
            )

            session_forbidden_d4.update(
                tuple(canonical_d4_key(row))
                for i, row in enumerate(population)
                if int(i) != int(seed_index)
            )

            session = run_memetic_session(
                population[seed_index],
                score_population=score_memetic_population,
                rng=memetic_rng,
                n_cells=N_CELLS,
                is_hard_feasible=is_hard_feasible,
                d4_key=lambda row: tuple(
                    canonical_d4_key(row)
                ),
                forbidden_d4=session_forbidden_d4,
                max_descent_moves=MEMETIC_MAX_DESCENT_MOVES,
                escape_samples=MEMETIC_ESCAPE_SAMPLES,
                max_evaluations=MEMETIC_MAX_EVALS_PER_SESSION,
            )

            final_local = session['final']

            injected = candidate_better(
                start_local,
                final_local,
            )

            if injected:
                population[seed_index] = np.asarray(
                    session['pattern'],
                    dtype=int,
                )

                ga_instance.population = (
                    population.copy()
                )

                fit, d = compute_ga_fitness(
                    population
                )

            session_record = {
                'generation': generation,
                'seed_index': int(seed_index),
                'injected': bool(injected),
                **{
                    k: v
                    for k, v in session.items()
                    if k != 'pattern'
                },
            }

            memetic_sessions.append(
                session_record
            )

            print(
                f'V2M seed {seed}; generation {generation:3d}; '
                f'memetic_evals={session["evaluation_count"]}; '
                f'moves={session["accepted_descent_moves"]}; '
                f'escape={session["escape_accepted"]}; '
                f'injected={injected}; '
                f'V={final_local["normalized_violation"]:.8f}; '
                f'J={final_local["objective"]:.8f}; '
                f'safe={final_local["safe"]}',
                flush=True,
            )

        ga_instance.last_generation_fitness = fit
        best_index = int(np.argmax(fit))
        convergence.append(float(fit[best_index]))
        safe_fraction_history.append(float(np.mean(d['safe'])))

        safe_mask = np.asarray(d['safe'], dtype=bool)
        if bool(np.any(safe_mask)):
            safe_indices = np.flatnonzero(safe_mask)
            safe_objective_history.append(
                float(np.min(d['secondary'][safe_indices]))
            )
        else:
            safe_objective_history.append(None)

        if first_safe_generation is None and bool(np.any(d['safe'])):
            first_safe_generation = int(ga_instance.generations_completed)

        g = generation

        capture_tail_snapshot(
            g,
            population,
            fit,
            d,
        )

        for idx in np.argsort(fit)[-10:][::-1]:
            chrom = population[idx]
            if not is_hard_feasible(chrom):
                continue
            if not bool(d['d4_novel'][idx]):
                continue
            key = tuple(chrom.tolist())
            row = {
                'pattern': chrom.tolist(),
                'fitness': float(fit[idx]),
                'objective': float(d['secondary'][idx]),
                'ensemble_mean_keff': float(d['mu'][idx, 0]),
                'ensemble_mean_Fa': float(d['mu'][idx, 1]),
                'ensemble_mean_Fq_r': float(d['mu'][idx, 2]),
                'ensemble_std_keff': float(d['sd'][idx, 0]),
                'ensemble_std_Fa': float(d['sd'][idx, 1]),
                'ensemble_std_Fq_r': float(d['sd'][idx, 2]),
                'calibrated_keff_lower': float(d['lower_k'][idx]),
                'calibrated_keff_upper': float(d['upper_k'][idx]),
                'calibrated_Fa_upper': float(d['upper_fa'][idx]),
                'calibrated_Fq_r_upper': float(d['upper_fqr'][idx]),
                'calibrated_safe': bool(d['safe'][idx]),
                'd4_novel_vs_labelled_10000': bool(d['d4_novel'][idx]),
                'generation': int(ga_instance.generations_completed),
                'ga_seed': seed,
            }
            if key not in run_archive or row['fitness'] > run_archive[key]['fitness']:
                run_archive[key] = row

        g = generation
        if g == 1 or g % 25 == 0:
            print(
                f'Seed {seed}; generation {g:3d}: best fitness={fit[best_index]:.8f}, '
                f'calibrated-safe={np.mean(d["safe"]):.1%}',
                flush=True,
            )

    ga_args = dict(
        num_generations=GA_GENERATIONS,
        num_parents_mating=GA_PARENTS,
        fitness_func=fitness_function,
        initial_population=initial_population,
        sol_per_pop=GA_POPULATION_SIZE,
        num_genes=N_CELLS,
        gene_space=[1, 2, 3],
        gene_type=int,
        parent_selection_type='tournament',
        crossover_type=count_preserving_crossover,
        mutation_type=feasibility_preserving_swap_mutation,
        on_generation=on_generation,
    )
    signature = inspect.signature(pygad.GA.__init__).parameters
    if 'random_seed' in signature:
        ga_args['random_seed'] = seed
    if 'suppress_warnings' in signature:
        ga_args['suppress_warnings'] = True
    if 'fitness_batch_size' in signature:
        ga_args['fitness_batch_size'] = GA_POPULATION_SIZE
    if 'keep_elitism' in signature:
        ga_args['keep_elitism'] = GA_ELITISM
    else:
        ga_args['keep_parents'] = min(4, GA_PARENTS)

    ga = pygad.GA(**ga_args)
    ga._constraint_tables = CONSTRAINTS
    ga._constraint_rng = np.random.default_rng(seed + 1)
    ga._mutation_swaps = GA_MUTATION_SWAPS
    ga._crossover_swaps = GA_CROSSOVER_SWAPS

    start = time.perf_counter()
    ga.run()
    elapsed = time.perf_counter() - start

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
        ga_seed=np.asarray(seed, dtype=np.int64),
        seed_index=np.asarray(
            args.seed_index,
            dtype=np.int16
        ),
        strategy=np.asarray(args.strategy),
    )

    best_chromosome, _, _ = ga.best_solution()
    best_chromosome = np.asarray(best_chromosome, dtype=int)
    hard_audit = count_violations(best_chromosome)
    if hard_audit['total'] != 0:
        raise RuntimeError(f'Final GA chromosome failed hard constraints: {hard_audit}')
    best_fit, best_d = compute_ga_fitness(best_chromosome.reshape(1, -1))
    best_d4_novel = bool(best_d['d4_novel'][0])
    if not best_d4_novel:
        raise RuntimeError('Final GA chromosome is D4-equivalent to a labelled pattern.')

    best_key = tuple(best_chromosome.tolist())
    if best_key not in run_archive:
        run_archive[best_key] = {
            'pattern': best_chromosome.tolist(),
            'fitness': float(best_fit[0]),
            'objective': float(best_d['secondary'][0]),
            'ensemble_mean_keff': float(best_d['mu'][0, 0]),
            'ensemble_mean_Fa': float(best_d['mu'][0, 1]),
            'ensemble_mean_Fq_r': float(best_d['mu'][0, 2]),
            'ensemble_std_keff': float(best_d['sd'][0, 0]),
            'ensemble_std_Fa': float(best_d['sd'][0, 1]),
            'ensemble_std_Fq_r': float(best_d['sd'][0, 2]),
            'calibrated_keff_lower': float(best_d['lower_k'][0]),
            'calibrated_keff_upper': float(best_d['upper_k'][0]),
            'calibrated_Fa_upper': float(best_d['upper_fa'][0]),
            'calibrated_Fq_r_upper': float(best_d['upper_fqr'][0]),
            'calibrated_safe': bool(best_d['safe'][0]),
            'd4_novel_vs_labelled_10000': best_d4_novel,
            'generation': GA_GENERATIONS,
            'ga_seed': seed,
        }

    archive = sorted(run_archive.values(), key=lambda r: r['fitness'], reverse=True)
    for r in archive:
        r['d4_key_sha256'] = hashlib.sha256(np.asarray(canonical_d4_key(r['pattern']), dtype=np.uint8).tobytes()).hexdigest()

    summary = {
        'status': 'COMPLETED',
        'context_sha256': context_sha,
        'context': context,
        'ga_seed': seed,
        'best_fitness': float(best_fit[0]),
        'best_objective': float(best_d['secondary'][0]),
        'best_normalized_violation': float(
            best_d['normalized_violation'][0]
        ),
        'best_selection_mode': 'v2_penalty_global_plus_common_memetic',
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

        # V2M/common-memetic diagnostics.
        'solver_family': 'V2M',
        'comparison_population_size': int(GA_POPULATION_SIZE),
        'comparison_generations': int(GA_GENERATIONS),
        'comparison_parent_count': int(GA_PARENTS),
        'memetic_session_count': int(len(memetic_sessions)),
        'memetic_checkpoints': list(MEMETIC_CHECKPOINTS),
        'memetic_total_scored_candidates': int(
            surrogate_eval_counter['memetic']
        ),
        'surrogate_total_scored_candidates': int(
            surrogate_eval_counter['total']
        ),
        'surrogate_nonmemetic_scored_candidates': int(
            surrogate_eval_counter['total']
            - surrogate_eval_counter['memetic']
        ),
        'memetic_total_accepted_descent_moves': int(
            sum(
                int(x['accepted_descent_moves'])
                for x in memetic_sessions
            )
        ),
        'memetic_escape_attempts': int(
            sum(
                bool(x['escape_attempted'])
                for x in memetic_sessions
            )
        ),
        'memetic_escape_accepts': int(
            sum(
                bool(x['escape_accepted'])
                for x in memetic_sessions
            )
        ),
        'memetic_injections': int(
            sum(
                bool(x['injected'])
                for x in memetic_sessions
            )
        ),

        'ga_wall_seconds': float(elapsed),
        'threads': int(args.threads),
    }

    archive_path.write_text(json.dumps(archive, indent=2), encoding='utf-8')
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    (
        out_dir / f'ga_seed_{seed}_memetic_sessions.json'
    ).write_text(
        json.dumps(memetic_sessions, indent=2),
        encoding='utf-8',
    )
    (out_dir / f'ga_seed_{seed}_convergence.json').write_text(json.dumps(convergence), encoding='utf-8')
    (out_dir / f'ga_seed_{seed}_safe_fraction.json').write_text(json.dumps(safe_fraction_history), encoding='utf-8')
    (out_dir / f'ga_seed_{seed}_safe_objective.json').write_text(json.dumps(safe_objective_history), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
