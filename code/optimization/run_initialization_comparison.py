#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
GA_GENERATIONS = 150
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
    ap.add_argument('--audit-json', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--warm-start-json', required=True)
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

    warm_start_path = Path(args.warm_start_json)
    labelled_path = Path(args.labelled_json)

    warm_start_sha256 = hashlib.sha256(warm_start_path.read_bytes()).hexdigest()
    labelled_sha256 = hashlib.sha256(labelled_path.read_bytes()).hexdigest()

    warm_start_rows = json.loads(warm_start_path.read_text())
    labelled_rows = json.loads(labelled_path.read_text())

    base_chromosomes = [
        np.asarray(row['pattern'], dtype=int)
        for row in warm_start_rows
    ]

    forbidden_d4 = {
        tuple(canonical_d4_key(row['pattern']))
        for row in labelled_rows
    }

    metadata = json.loads((round_dir / 'round_metadata.json').read_text())
    audit = json.loads(Path(args.audit_json).read_text())
    q = audit['adaptive_conformal_joint_95_bonferroni']['q']

    context = {
        'dataset_sha256': metadata['dataset_sha256'],
        'split_sha256': metadata['split_sha256'],
        'cnn_seeds': list(CNN_TRAINING_SEEDS),
        'conformal_mode': 'adaptive_conformal_joint_95_bonferroni',
        'q': q,
        'warm_start_sha256': warm_start_sha256,
        'warm_start_count': len(base_chromosomes),
        'labelled_exclusion_sha256': labelled_sha256,
        'labelled_d4_exclusion_count': len(forbidden_d4),
        'require_d4_novel_vs_labelled_10000': True,
        'warm_start_source': 'training_split_only_surrogate_ranked',
        'ga_seed': seed,
        'population_size': GA_POPULATION_SIZE,
        'generations': GA_GENERATIONS,
        'parents': GA_PARENTS,
        'mutation_swaps': GA_MUTATION_SWAPS,
        'crossover_swaps': GA_CROSSOVER_SWAPS,
        'elitism': GA_ELITISM,
        'fa_weight': FA_WEIGHT,
        'fqr_weight': FQR_WEIGHT,
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
        models.append(tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False))

    def predict_population(population):
        x = patterns_to_onehot_grid(np.asarray(population, dtype=int))
        per_seed = np.stack([inverse_outputs(model, x, scalers) for model in models], axis=0)
        mu = per_seed.mean(axis=0)
        sd = per_seed.std(axis=0, ddof=1)
        lower_k = mu[:, 0] - float(q['keff_lower']) * sd[:, 0]
        upper_k = mu[:, 0] + float(q['keff_upper']) * sd[:, 0]
        upper_fa = mu[:, 1] + float(q['Fa_upper']) * sd[:, 1]
        upper_fqr = mu[:, 2] + float(q['Fq_r_upper']) * sd[:, 2]
        return mu, sd, lower_k, upper_k, upper_fa, upper_fqr

    def compute_ga_fitness(population):
        population = np.asarray(population, dtype=int)
        mu, sd, lower_k, upper_k, upper_fa, upper_fqr = predict_population(population)
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
            'mu': mu, 'sd': sd,
            'lower_k': lower_k, 'upper_k': upper_k,
            'upper_fa': upper_fa, 'upper_fqr': upper_fqr,
            'safe': safe, 'secondary': secondary,
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
    convergence = []
    safe_fraction_history = []
    first_safe_generation = None

    def fitness_function(ga_instance, solutions, solution_indices):
        arr = np.asarray(solutions, dtype=int)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)
        fit, _ = compute_ga_fitness(arr)
        return float(fit[0]) if single else fit.tolist()

    def on_generation(ga_instance):
        nonlocal first_safe_generation
        population = np.asarray(ga_instance.population, dtype=int)
        fit, d = compute_ga_fitness(population)
        ga_instance.last_generation_fitness = fit
        best_index = int(np.argmax(fit))
        convergence.append(float(fit[best_index]))
        safe_fraction_history.append(float(np.mean(d['safe'])))
        if first_safe_generation is None and bool(np.any(d['safe'])):
            first_safe_generation = int(ga_instance.generations_completed)

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

        g = int(ga_instance.generations_completed)
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
        'best_ensemble_mean_keff': float(best_d['mu'][0, 0]),
        'best_ensemble_mean_Fa': float(best_d['mu'][0, 1]),
        'best_ensemble_mean_Fq_r': float(best_d['mu'][0, 2]),
        'best_calibrated_keff_lower': float(best_d['lower_k'][0]),
        'best_calibrated_keff_upper': float(best_d['upper_k'][0]),
        'best_calibrated_Fa_upper': float(best_d['upper_fa'][0]),
        'best_calibrated_Fq_r_upper': float(best_d['upper_fqr'][0]),
        'best_calibrated_safe': bool(best_d['safe'][0]),
        'best_d4_novel_vs_labelled_10000': best_d4_novel,
        'first_calibrated_safe_generation': first_safe_generation,
        'final_calibrated_safe_fraction': safe_fraction_history[-1] if safe_fraction_history else None,
        'hard_constraint_audit': hard_audit,
        'unique_archived_candidates': len(archive),
        'ga_wall_seconds': float(elapsed),
        'threads': int(args.threads),
    }

    archive_path.write_text(json.dumps(archive, indent=2), encoding='utf-8')
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out_dir / f'ga_seed_{seed}_convergence.json').write_text(json.dumps(convergence), encoding='utf-8')
    (out_dir / f'ga_seed_{seed}_safe_fraction.json').write_text(json.dumps(safe_fraction_history), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
