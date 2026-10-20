#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


MEMETIC_CHECKPOINTS = (75, 150, 225, 300)
MEMETIC_MAX_DESCENT_MOVES = 6
MEMETIC_ESCAPE_SAMPLES = 25_000
MEMETIC_MAX_EVALS_PER_SESSION = 50_000
MEMETIC_IMPROVEMENT_TOL = 1e-12


def common_rank_order(score):
    """
    Common V2M/V3M local ranking.

    Feasible always beats infeasible.
    Among infeasible candidates:
        lower normalized robust violation wins,
        then lower objective J.
    Among feasible candidates:
        lower objective J wins.
    """
    safe = np.asarray(score["safe"], dtype=bool)
    violation = np.asarray(
        score["normalized_violation"],
        dtype=float,
    )
    objective = np.asarray(
        score["objective"],
        dtype=float,
    )

    return np.lexsort(
        (
            objective,
            violation,
            (~safe).astype(int),
        )
    )


def candidate_record(score, index):
    i = int(index)

    return {
        "safe": bool(score["safe"][i]),
        "normalized_violation": float(
            score["normalized_violation"][i]
        ),
        "objective": float(
            score["objective"][i]
        ),
    }


def candidate_better(
    current,
    candidate,
    tol=MEMETIC_IMPROVEMENT_TOL,
):
    """
    Return True iff candidate strictly dominates current under the
    common memetic decision rule.
    """
    csafe = bool(current["safe"])
    nsafe = bool(candidate["safe"])

    if nsafe and not csafe:
        return True

    if csafe and not nsafe:
        return False

    if csafe and nsafe:
        return (
            float(candidate["objective"])
            <
            float(current["objective"]) - float(tol)
        )

    cv = float(current["normalized_violation"])
    nv = float(candidate["normalized_violation"])

    if nv < cv - float(tol):
        return True

    if abs(nv - cv) <= float(tol):
        return (
            float(candidate["objective"])
            <
            float(current["objective"]) - float(tol)
        )

    return False


def select_memetic_seed_index(score):
    """
    Select the current population member from which memetic
    refinement starts, using exactly the common local ranking.
    """
    order = common_rank_order(score)

    if len(order) == 0:
        raise RuntimeError(
            "Cannot select memetic seed from empty population."
        )

    return int(order[0])


def exhaustive_one_swap_neighbors(
    base,
    *,
    n_cells,
    is_hard_feasible,
    d4_key,
    forbidden_d4,
):
    """
    Enumerate every hard-feasible, D4-novel, D4-unique,
    one cross-type swap neighbor.
    """
    base = np.asarray(base, dtype=int)

    rows = []
    seen = set()

    for i in range(int(n_cells) - 1):
        for j in range(i + 1, int(n_cells)):
            if base[i] == base[j]:
                continue

            x = base.copy()
            x[i], x[j] = x[j], x[i]

            if not is_hard_feasible(x):
                continue

            key = d4_key(x)

            if key in forbidden_d4 or key in seen:
                continue

            rows.append(x)
            seen.add(key)

    if not rows:
        return np.empty(
            (0, int(n_cells)),
            dtype=int,
        )

    return np.stack(rows).astype(int)


def exact_k_feasible_swaps(
    base,
    *,
    k,
    rng,
    n_cells,
    is_hard_feasible,
    max_attempts=None,
):
    """
    Apply exactly k accepted cross-type swaps.

    A position may participate in more than one accepted swap, so the
    final Hamming distance need not equal exactly 2*k.
    """
    x = np.asarray(base, dtype=int).copy()

    if not is_hard_feasible(x):
        raise RuntimeError(
            "Memetic escape received hard-infeasible base."
        )

    k = int(k)

    if max_attempts is None:
        max_attempts = max(
            1000,
            200 * k,
        )

    accepted = 0
    attempts = 0

    while (
        accepted < k
        and attempts < int(max_attempts)
    ):
        attempts += 1

        i, j = rng.choice(
            int(n_cells),
            size=2,
            replace=False,
        )

        i = int(i)
        j = int(j)

        if x[i] == x[j]:
            continue

        x[i], x[j] = x[j], x[i]

        if is_hard_feasible(x):
            accepted += 1
        else:
            x[i], x[j] = x[j], x[i]

    if accepted != k:
        return None

    return x


def sample_two_swap_escape_neighbors(
    base,
    *,
    target,
    rng,
    n_cells,
    is_hard_feasible,
    d4_key,
    forbidden_d4,
):
    """
    Generate a bounded set of D4-unique exact-two-swap
    hard-feasible escape candidates.
    """
    target = int(target)

    if target <= 0:
        return np.empty(
            (0, int(n_cells)),
            dtype=int,
        )

    base = np.asarray(base, dtype=int)
    base_key = d4_key(base)

    rows = []
    seen = {base_key}

    attempts = 0
    max_attempts = max(
        target * 200,
        10_000,
    )

    while (
        len(rows) < target
        and attempts < max_attempts
    ):
        attempts += 1

        x = exact_k_feasible_swaps(
            base,
            k=2,
            rng=rng,
            n_cells=n_cells,
            is_hard_feasible=is_hard_feasible,
        )

        if x is None:
            continue

        key = d4_key(x)

        if (
            key in forbidden_d4
            or key in seen
        ):
            continue

        rows.append(x)
        seen.add(key)

    if not rows:
        return np.empty(
            (0, int(n_cells)),
            dtype=int,
        )

    return np.stack(rows).astype(int)


def run_memetic_session(
    start_pattern,
    *,
    score_population,
    rng,
    n_cells,
    is_hard_feasible,
    d4_key,
    forbidden_d4,
    max_descent_moves=MEMETIC_MAX_DESCENT_MOVES,
    escape_samples=MEMETIC_ESCAPE_SAMPLES,
    max_evaluations=MEMETIC_MAX_EVALS_PER_SESSION,
    improvement_tol=MEMETIC_IMPROVEMENT_TOL,
):
    """
    Common V2M/V3M bounded memetic refinement.

    Flow:
        start
          -> exhaustive best-improvement 1-swap descent
          -> if infeasible 1-swap local optimum:
               bounded exact-2-swap escape search
          -> if improving escape exists:
               accept best escape and resume 1-swap descent

    Only one escape search is allowed per checkpoint/session.

    score_population(population) must return arrays:
        safe
        normalized_violation
        objective

    Every surrogate-scored pattern is counted in evaluation_count.
    """
    current_pattern = np.asarray(
        start_pattern,
        dtype=int,
    ).copy()

    if not is_hard_feasible(current_pattern):
        raise RuntimeError(
            "Memetic session received hard-infeasible start."
        )

    current_key = d4_key(current_pattern)

    if current_key in forbidden_d4:
        raise RuntimeError(
            "Memetic session received known-OpenMC/D4-forbidden start."
        )

    evaluation_count = 0
    accepted_descent_moves = 0
    escape_attempted = False
    escape_accepted = False
    trace = []

    # ------------------------------------------------------------
    # Score starting state.
    # ------------------------------------------------------------
    start_score = score_population(
        current_pattern.reshape(1, -1)
    )
    evaluation_count += 1

    current = candidate_record(
        start_score,
        0,
    )

    initial = dict(current)

    stop_reason = None
    iteration = 0

    while True:
        iteration += 1

        if (
            accepted_descent_moves
            >= int(max_descent_moves)
        ):
            stop_reason = "DESCENT_MOVE_LIMIT"
            break

        # --------------------------------------------------------
        # Exhaustive one-swap local neighborhood.
        # --------------------------------------------------------
        neighbors = exhaustive_one_swap_neighbors(
            current_pattern,
            n_cells=n_cells,
            is_hard_feasible=is_hard_feasible,
            d4_key=d4_key,
            forbidden_d4=forbidden_d4,
        )

        n_neighbors = int(len(neighbors))

        if n_neighbors == 0:
            stop_reason = "NO_VALID_ONE_SWAP_NEIGHBORS"
            break

        # Preserve exhaustive semantics. Never evaluate a partial
        # one-swap neighborhood merely to fill the budget.
        if (
            evaluation_count + n_neighbors
            > int(max_evaluations)
        ):
            stop_reason = "MEMETIC_EVALUATION_BUDGET"
            break

        score = score_population(neighbors)
        evaluation_count += n_neighbors

        order = common_rank_order(score)
        best_index = int(order[0])

        best = candidate_record(
            score,
            best_index,
        )

        improving_count = 0

        for i in range(n_neighbors):
            candidate = candidate_record(
                score,
                i,
            )

            if candidate_better(
                current,
                candidate,
                tol=improvement_tol,
            ):
                improving_count += 1

        trace.append(
            {
                "iteration": int(iteration),
                "action": "one_swap_neighborhood",
                "neighbors_evaluated": n_neighbors,
                "improving_neighbors": int(
                    improving_count
                ),
                "current_safe": bool(
                    current["safe"]
                ),
                "current_violation": float(
                    current[
                        "normalized_violation"
                    ]
                ),
                "current_objective": float(
                    current["objective"]
                ),
                "best_safe": bool(
                    best["safe"]
                ),
                "best_violation": float(
                    best[
                        "normalized_violation"
                    ]
                ),
                "best_objective": float(
                    best["objective"]
                ),
                "evaluation_count": int(
                    evaluation_count
                ),
            }
        )

        if candidate_better(
            current,
            best,
            tol=improvement_tol,
        ):
            current_pattern = np.asarray(
                neighbors[best_index],
                dtype=int,
            ).copy()

            current = best

            accepted_descent_moves += 1
            continue

        # --------------------------------------------------------
        # Genuine one-swap local optimum.
        # --------------------------------------------------------
        if bool(current["safe"]):
            stop_reason = (
                "SAFE_ONE_SWAP_LOCAL_OPTIMUM"
            )
            break

        if escape_attempted:
            stop_reason = (
                "POST_ESCAPE_ONE_SWAP_LOCAL_OPTIMUM"
            )
            break

        # --------------------------------------------------------
        # One bounded exact-2-swap escape search.
        # --------------------------------------------------------
        escape_attempted = True

        remaining_budget = (
            int(max_evaluations)
            - int(evaluation_count)
        )

        target = min(
            int(escape_samples),
            int(remaining_budget),
        )

        if target <= 0:
            stop_reason = (
                "MEMETIC_EVALUATION_BUDGET"
            )
            break

        escapes = sample_two_swap_escape_neighbors(
            current_pattern,
            target=target,
            rng=rng,
            n_cells=n_cells,
            is_hard_feasible=is_hard_feasible,
            d4_key=d4_key,
            forbidden_d4=forbidden_d4,
        )

        if len(escapes) == 0:
            stop_reason = (
                "NO_VALID_TWO_SWAP_ESCAPES"
            )
            break

        # Generation can theoretically return fewer than target if
        # the local space is exhausted. Count only actual scoring.
        if (
            evaluation_count + len(escapes)
            > int(max_evaluations)
        ):
            escapes = escapes[
                : int(max_evaluations)
                - int(evaluation_count)
            ]

        if len(escapes) == 0:
            stop_reason = (
                "MEMETIC_EVALUATION_BUDGET"
            )
            break

        escape_score = score_population(
            escapes
        )
        evaluation_count += int(
            len(escapes)
        )

        escape_order = common_rank_order(
            escape_score
        )
        escape_index = int(
            escape_order[0]
        )

        best_escape = candidate_record(
            escape_score,
            escape_index,
        )

        improving_escape_count = 0

        for i in range(len(escapes)):
            candidate = candidate_record(
                escape_score,
                i,
            )

            if candidate_better(
                current,
                candidate,
                tol=improvement_tol,
            ):
                improving_escape_count += 1

        trace.append(
            {
                "iteration": int(iteration),
                "action": "two_swap_escape",
                "escapes_evaluated": int(
                    len(escapes)
                ),
                "improving_escapes": int(
                    improving_escape_count
                ),
                "current_safe": bool(
                    current["safe"]
                ),
                "current_violation": float(
                    current[
                        "normalized_violation"
                    ]
                ),
                "current_objective": float(
                    current["objective"]
                ),
                "best_safe": bool(
                    best_escape["safe"]
                ),
                "best_violation": float(
                    best_escape[
                        "normalized_violation"
                    ]
                ),
                "best_objective": float(
                    best_escape["objective"]
                ),
                "evaluation_count": int(
                    evaluation_count
                ),
            }
        )

        if not candidate_better(
            current,
            best_escape,
            tol=improvement_tol,
        ):
            stop_reason = (
                "NO_IMPROVING_TWO_SWAP_ESCAPE"
            )
            break

        current_pattern = np.asarray(
            escapes[escape_index],
            dtype=int,
        ).copy()

        current = best_escape
        escape_accepted = True

        # Resume exhaustive 1-swap descent from escaped basin.

    if stop_reason is None:
        stop_reason = "UNKNOWN"

    return {
        "pattern": current_pattern,
        "initial": initial,
        "final": dict(current),
        "evaluation_count": int(
            evaluation_count
        ),
        "accepted_descent_moves": int(
            accepted_descent_moves
        ),
        "escape_attempted": bool(
            escape_attempted
        ),
        "escape_accepted": bool(
            escape_accepted
        ),
        "stop_reason": str(
            stop_reason
        ),
        "trace": trace,
    }
