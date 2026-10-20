#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
import sys

import numpy as np
from scipy.stats import (
    friedmanchisquare,
    rankdata,
    wilcoxon,
)


PROJECT = Path(
    __import__("os").environ["PWR_PROJECT"]
)

PROS = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360"
)

PROTOCOL = PROS / "protocol"

MERGED = (
    PROTOCOL
    / "cycle5_q5_prospective360_openmc_merged_360_v1.json"
)

EVALUATION = (
    PROTOCOL
    / "cycle5_q5_prospective360_evaluation_v1.json"
)

WINNERS = (
    PROTOCOL
    / "cycle5_q5_prospective360_winner_manifest_360_v1.json"
)

FINAL_FREEZE = (
    PROTOCOL
    / "cycle5_q5_final_deployment_freeze_v1.json"
)

FINAL_CERTIFICATE = (
    PROJECT
    / "run/optimizer_tail_v2/models/"
      "learned_residual_safety_round5_q5_prospective_search/"
      "deployment_certificate_final_q5.json"
)

OUT_JSON = (
    PROTOCOL
    / "cycle5_q5_final_paired_optimizer_benchmark_v1.json"
)

OUT_TSV = (
    PROTOCOL
    / "cycle5_q5_final_paired_optimizer_benchmark_v1.tsv"
)

PAIRWISE_TSV = (
    PROTOCOL
    / "cycle5_q5_final_paired_optimizer_pairwise_v1.tsv"
)


EXPECTED = {
    "merged":
        "575c90a18e1ec20c00099c91af3e066d5d6f6742ff4f2c87c0040aa4084e640c",

    "evaluation":
        "7a0edf92848ff57df26bbf027f29f679f830a1adad54d8a1a26c0d1986fa894c",

    "winners":
        "73e4049c0c1c89f0306537a07ece92603ae6b25f6afc3af46f4e7c87c6321ac0",

    "final_freeze":
        "4287f2a8afdd1d8be9b237267177c9f17a6a7d3c07387adacb358cc39c6e3174",

    "final_certificate":
        "4469a191188f940ad79b64f0bd1eff12d7070eb0b665dcd66ff65ad08d45a6f4",
}


SOLVERS = [
    "v2",
    "v2m",
    "v3",
    "v3m",
]

INITIALIZERS = [
    "random",
    "sa",
    "sqa",
]

ARMS = [
    f"{solver}_{initializer}"
    for solver in SOLVERS
    for initializer in INITIALIZERS
]

N_REPLICATES = 30
N_ARMS = 12
ALPHA = 0.05


# ======================================================================
# HELPERS
# ======================================================================

def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def rel(path):
    return str(
        path.relative_to(PROJECT)
    )


def frozen_write_json(path, obj):

    data = (
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():

        if path.read_bytes() != data:
            raise RuntimeError(
                "Frozen artifact already exists "
                f"with different bytes:\n{path}"
            )

        return

    path.write_bytes(data)


def describe(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    return {
        "n":
            int(len(x)),

        "mean":
            float(np.mean(x)),

        "std":
            float(np.std(
                x,
                ddof=1,
            )),

        "min":
            float(np.min(x)),

        "q25":
            float(np.quantile(
                x,
                0.25,
            )),

        "median":
            float(np.median(x)),

        "q75":
            float(np.quantile(
                x,
                0.75,
            )),

        "max":
            float(np.max(x)),
    }


def paired_wilcoxon(a, b):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if a.shape != b.shape:
        raise RuntimeError(
            "Paired vectors have different shapes"
        )

    d = a - b

    nonzero = d[
        np.abs(d) > 1e-15
    ]

    if len(nonzero) == 0:

        return {
            "statistic":
                0.0,

            "p_raw":
                1.0,

            "median_delta_a_minus_b":
                0.0,

            "mean_delta_a_minus_b":
                0.0,

            "a_better":
                0,

            "b_better":
                0,

            "ties":
                int(len(d)),
        }

    try:

        result = wilcoxon(
            a,
            b,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )

        statistic = float(
            result.statistic
        )

        p = float(
            result.pvalue
        )

    except ValueError:

        statistic = 0.0
        p = 1.0

    # Lower J is better.
    a_better = int(
        np.sum(d < -1e-15)
    )

    b_better = int(
        np.sum(d > 1e-15)
    )

    ties = int(
        np.sum(
            np.abs(d) <= 1e-15
        )
    )

    return {
        "statistic":
            statistic,

        "p_raw":
            p,

        "median_delta_a_minus_b":
            float(np.median(d)),

        "mean_delta_a_minus_b":
            float(np.mean(d)),

        "a_better":
            a_better,

        "b_better":
            b_better,

        "ties":
            ties,
    }


def holm_adjust(records):

    if not records:
        return records

    order = sorted(
        range(len(records)),
        key=lambda i: records[i]["p_raw"],
    )

    m = len(records)

    running = 0.0

    adjusted = [None] * m

    for rank_position, idx in enumerate(order):

        multiplier = (
            m - rank_position
        )

        value = min(
            1.0,
            multiplier
            * records[idx]["p_raw"],
        )

        running = max(
            running,
            value,
        )

        adjusted[idx] = min(
            1.0,
            running,
        )

    for i, record in enumerate(records):

        record["p_holm"] = float(
            adjusted[i]
        )

        record["significant_holm_0.05"] = bool(
            adjusted[i] < ALPHA
        )

    return records


def write_tsv(path, rows, fields):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    from io import StringIO

    buf = StringIO()

    writer = csv.DictWriter(
        buf,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(rows)

    data = buf.getvalue().encode()

    if path.exists():

        if path.read_bytes() != data:
            raise RuntimeError(
                "Frozen TSV already exists "
                f"with different bytes:\n{path}"
            )

        return

    path.write_bytes(data)


# ======================================================================
# 1. VERIFY FINAL FROZEN SCIENTIFIC CHAIN
# ======================================================================

dependencies = [
    (
        "prospective OpenMC360",
        MERGED,
        EXPECTED["merged"],
    ),
    (
        "prospective evaluation",
        EVALUATION,
        EXPECTED["evaluation"],
    ),
    (
        "winner manifest",
        WINNERS,
        EXPECTED["winners"],
    ),
    (
        "final q5 freeze",
        FINAL_FREEZE,
        EXPECTED["final_freeze"],
    ),
    (
        "final q5 certificate",
        FINAL_CERTIFICATE,
        EXPECTED["final_certificate"],
    ),
]

for name, path, expected in dependencies:

    if not path.is_file():

        raise RuntimeError(
            f"Missing frozen {name}:\n{path}"
        )

    actual = sha(path)

    if actual != expected:

        raise RuntimeError(
            f"{name} SHA changed:\n"
            f"expected={expected}\n"
            f"actual  ={actual}"
        )


evaluation = json.loads(
    EVALUATION.read_text()
)

if (
    evaluation["status"]
    != "PASS_FROZEN_Q5_PROSPECTIVE_VALIDATION"
):

    raise RuntimeError(
        "Cannot benchmark: prospective q5 "
        "validation was not PASS"
    )


certificate = json.loads(
    FINAL_CERTIFICATE.read_text()
)

if (
    certificate["status"]
    != "FINAL_Q5_DEPLOYMENT_CERTIFICATE"
):

    raise RuntimeError(
        "Final q5 deployment certificate "
        "is not frozen"
    )


# ======================================================================
# 2. LOAD 360 OPENMC RESULTS + WINNER SUMMARIES
# ======================================================================

merged_obj = json.loads(
    MERGED.read_text()
)

rows = merged_obj["rows"]

winner_obj = json.loads(
    WINNERS.read_text()
)

winners = winner_obj["winners"]


if len(rows) != 360:
    raise RuntimeError(
        f"Expected 360 merged rows, got {len(rows)}"
    )

if len(winners) != 360:
    raise RuntimeError(
        f"Expected 360 winners, got {len(winners)}"
    )


winner_by_task = {
    int(w["task_id"]): w
    for w in winners
}

if sorted(winner_by_task) != list(
    range(360)
):
    raise RuntimeError(
        "Winner task IDs are not exactly 0..359"
    )


# ======================================================================
# 3. BUILD STRICT 30 x 12 PAIRED MATRIX
# ======================================================================

records = []

for row in rows:

    task = int(
        row["task_id"]
    )

    rep = int(
        row["replicate"]
    )

    solver = row["solver"]
    initializer = row["initializer"]

    arm = (
        f"{solver}_{initializer}"
    )

    if arm != row["arm"]:
        raise RuntimeError(
            f"Arm mismatch at task {task}"
        )

    if solver not in SOLVERS:
        raise RuntimeError(
            f"Unexpected solver {solver}"
        )

    if initializer not in INITIALIZERS:
        raise RuntimeError(
            f"Unexpected initializer {initializer}"
        )

    w = winner_by_task[task]

    summary_path = (
        PROJECT
        / w["summary_path"]
    )

    if not summary_path.is_file():

        raise RuntimeError(
            f"Missing GA summary:\n{summary_path}"
        )

    actual_summary_sha = sha(
        summary_path
    )

    if (
        actual_summary_sha
        != w["summary_sha256"]
    ):

        raise RuntimeError(
            f"GA summary SHA changed "
            f"for task {task}"
        )

    summary = json.loads(
        summary_path.read_text()
    )

    if (
        summary.get("status")
        != "COMPLETED"
    ):

        raise RuntimeError(
            f"Task {task}: GA summary "
            "not COMPLETED"
        )

    if "ga_wall_seconds" not in summary:

        raise RuntimeError(
            f"Task {task}: summary missing "
            "ga_wall_seconds"
        )

    wall = float(
        summary["ga_wall_seconds"]
    )

    if (
        not math.isfinite(wall)
        or wall <= 0
    ):

        raise RuntimeError(
            f"Task {task}: invalid GA runtime"
        )

    k = float(
        row["openmc"]["keff"]
    )

    fa = float(
        row["openmc"]["Fa"]
    )

    fq = float(
        row["openmc"]["Fq_r"]
    )

    J = float(
        row["openmc"]["J"]
    )

    pred_J = float(
        row["prediction"]["J"]
    )

    # Verify objective definition independently.
    J_recalc = (
        0.5 * (fa / 2.0)
        + 0.5 * (fq / 2.5)
    )

    if abs(
        J - J_recalc
    ) > 1e-12:

        raise RuntimeError(
            f"Task {task}: OpenMC J mismatch"
        )

    records.append({
        "task_id":
            task,

        "replicate":
            rep,

        "solver":
            solver,

        "initializer":
            initializer,

        "arm":
            arm,

        "paired_ga_seed":
            int(
                row["paired_ga_seed"]
            ),

        "openmc_keff":
            k,

        "openmc_Fa":
            fa,

        "openmc_Fq_r":
            fq,

        "openmc_J":
            J,

        "surrogate_J":
            pred_J,

        "ga_wall_seconds":
            wall,

        "actual_safe":
            bool(
                row[
                    "actual_constraints_pass"
                ]
            ),

        "q5_joint_covered":
            bool(
                row["coverage"][
                    "joint_all_four"
                ]
            ),
    })


# Check exact paired design.
by_rep = defaultdict(list)

for r in records:
    by_rep[r["replicate"]].append(r)


if sorted(by_rep) != list(
    range(N_REPLICATES)
):

    raise RuntimeError(
        "Replicates are not exactly 0..29"
    )


for rep in range(N_REPLICATES):

    rr = by_rep[rep]

    if len(rr) != N_ARMS:

        raise RuntimeError(
            f"Replicate {rep}: expected "
            f"12 arms, got {len(rr)}"
        )

    rep_arms = {
        r["arm"]
        for r in rr
    }

    if rep_arms != set(ARMS):

        raise RuntimeError(
            f"Replicate {rep}: arm set mismatch"
        )

    seeds = {
        r["paired_ga_seed"]
        for r in rr
    }

    if len(seeds) != 1:

        raise RuntimeError(
            f"Replicate {rep}: paired GA "
            "seed property broken"
        )


# Matrix:
# rows = replicate
# columns = arm
record_map = {
    (
        r["replicate"],
        r["arm"],
    ): r
    for r in records
}

J_matrix = np.asarray([
    [
        record_map[
            (rep, arm)
        ]["openmc_J"]
        for arm in ARMS
    ]
    for rep in range(N_REPLICATES)
])

runtime_matrix = np.asarray([
    [
        record_map[
            (rep, arm)
        ]["ga_wall_seconds"]
        for arm in ARMS
    ]
    for rep in range(N_REPLICATES)
])


if J_matrix.shape != (30, 12):
    raise RuntimeError(
        f"Unexpected J matrix shape "
        f"{J_matrix.shape}"
    )


# ======================================================================
# 4. WITHIN-REPLICATE RANKS
#
# Lower OpenMC J = better.
# ======================================================================

rank_matrix = np.asarray([
    rankdata(
        J_matrix[rep],
        method="average",
    )
    for rep in range(N_REPLICATES)
])


arm_rank_stats = {}

for j, arm in enumerate(ARMS):

    ranks = rank_matrix[:, j]

    arm_rank_stats[arm] = {
        "mean_rank":
            float(np.mean(ranks)),

        "median_rank":
            float(np.median(ranks)),

        "best_in_replicate_count":
            int(np.sum(
                ranks
                == np.min(
                    rank_matrix,
                    axis=1,
                )
            )),
    }


# ======================================================================
# 5. PER-ARM OPENMC QUALITY + RUNTIME SUMMARY
# ======================================================================

arm_stats = {}

for j, arm in enumerate(ARMS):

    rr = [
        r
        for r in records
        if r["arm"] == arm
    ]

    J_values = np.asarray([
        r["openmc_J"]
        for r in rr
    ])

    Fa_values = np.asarray([
        r["openmc_Fa"]
        for r in rr
    ])

    Fq_values = np.asarray([
        r["openmc_Fq_r"]
        for r in rr
    ])

    k_values = np.asarray([
        r["openmc_keff"]
        for r in rr
    ])

    pred_J_values = np.asarray([
        r["surrogate_J"]
        for r in rr
    ])

    runtime_values = np.asarray([
        r["ga_wall_seconds"]
        for r in rr
    ])

    q5_cov = sum(
        r["q5_joint_covered"]
        for r in rr
    )

    actual_safe = sum(
        r["actual_safe"]
        for r in rr
    )

    bias = (
        J_values
        - pred_J_values
    )

    arm_stats[arm] = {
        "solver":
            rr[0]["solver"],

        "initializer":
            rr[0]["initializer"],

        "n":
            30,

        "openmc_J":
            describe(J_values),

        "openmc_Fa":
            describe(Fa_values),

        "openmc_Fq_r":
            describe(Fq_values),

        "openmc_keff":
            describe(k_values),

        "ga_wall_seconds":
            describe(runtime_values),

        "ga_wall_minutes_mean":
            float(
                np.mean(runtime_values)
                / 60.0
            ),

        "q5_joint_coverage":
            {
                "covered":
                    int(q5_cov),

                "total":
                    30,

                "rate":
                    float(
                        q5_cov / 30.0
                    ),
            },

        "actual_safe":
            {
                "passed":
                    int(actual_safe),

                "total":
                    30,

                "rate":
                    float(
                        actual_safe / 30.0
                    ),
            },

        "surrogate_minus_openmc_J": {
            # positive means surrogate predicted
            # larger/worse J than OpenMC.
            "mean":
                float(
                    np.mean(
                        pred_J_values
                        - J_values
                    )
                ),

            "median":
                float(
                    np.median(
                        pred_J_values
                        - J_values
                    )
                ),

            "mae":
                float(
                    np.mean(
                        np.abs(
                            pred_J_values
                            - J_values
                        )
                    )
                ),
        },

        **arm_rank_stats[arm],
    }


# ======================================================================
# 6. 12-ARM FRIEDMAN OMNIBUS
# ======================================================================

friedman = friedmanchisquare(
    *[
        J_matrix[:, j]
        for j in range(N_ARMS)
    ]
)

friedman_12 = {
    "test":
        "Friedman repeated-measures test",

    "n_replicates":
        30,

    "n_arms":
        12,

    "statistic":
        float(
            friedman.statistic
        ),

    "p_value":
        float(
            friedman.pvalue
        ),

    "significant_0.05":
        bool(
            friedman.pvalue
            < ALPHA
        ),
}


# ======================================================================
# 7. ALL 66 PAIRED ARM COMPARISONS
#    Wilcoxon + Holm correction.
# ======================================================================

pairwise = []

for a_idx, b_idx in itertools.combinations(
    range(N_ARMS),
    2,
):

    arm_a = ARMS[a_idx]
    arm_b = ARMS[b_idx]

    result = paired_wilcoxon(
        J_matrix[:, a_idx],
        J_matrix[:, b_idx],
    )

    pairwise.append({
        "arm_a":
            arm_a,

        "arm_b":
            arm_b,

        **result,
    })


pairwise = holm_adjust(
    pairwise
)


# ======================================================================
# 8. SOLVER MAIN-EFFECT PAIRED SUMMARY
#
# For each replicate:
# average the three initializer-specific OpenMC J values
# within each solver.
#
# This is a descriptive repeated-measures main-effect analysis;
# the full 12-arm result remains primary.
# ======================================================================

solver_matrix = np.asarray([
    [
        np.mean([
            record_map[
                (
                    rep,
                    f"{solver}_{init}",
                )
            ]["openmc_J"]
            for init in INITIALIZERS
        ])
        for solver in SOLVERS
    ]
    for rep in range(N_REPLICATES)
])


solver_friedman_result = friedmanchisquare(
    *[
        solver_matrix[:, j]
        for j in range(
            len(SOLVERS)
        )
    ]
)

solver_main = {
    "method":
        "within-replicate mean across initializers",

    "friedman": {
        "statistic":
            float(
                solver_friedman_result.statistic
            ),

        "p_value":
            float(
                solver_friedman_result.pvalue
            ),

        "significant_0.05":
            bool(
                solver_friedman_result.pvalue
                < ALPHA
            ),
    },

    "summary": {
        solver: describe(
            solver_matrix[:, j]
        )
        for j, solver
        in enumerate(SOLVERS)
    },
}


solver_pairs = []

for a, b in itertools.combinations(
    range(len(SOLVERS)),
    2,
):

    result = paired_wilcoxon(
        solver_matrix[:, a],
        solver_matrix[:, b],
    )

    solver_pairs.append({
        "solver_a":
            SOLVERS[a],

        "solver_b":
            SOLVERS[b],

        **result,
    })

solver_main["pairwise_wilcoxon_holm"] = (
    holm_adjust(
        solver_pairs
    )
)


# ======================================================================
# 9. INITIALIZER MAIN-EFFECT PAIRED SUMMARY
#
# For each replicate:
# average the four solver-specific OpenMC J values
# within each initializer.
# ======================================================================

initializer_matrix = np.asarray([
    [
        np.mean([
            record_map[
                (
                    rep,
                    f"{solver}_{initializer}",
                )
            ]["openmc_J"]
            for solver in SOLVERS
        ])
        for initializer in INITIALIZERS
    ]
    for rep in range(N_REPLICATES)
])


init_friedman_result = friedmanchisquare(
    *[
        initializer_matrix[:, j]
        for j in range(
            len(INITIALIZERS)
        )
    ]
)

initializer_main = {
    "method":
        "within-replicate mean across solvers",

    "friedman": {
        "statistic":
            float(
                init_friedman_result.statistic
            ),

        "p_value":
            float(
                init_friedman_result.pvalue
            ),

        "significant_0.05":
            bool(
                init_friedman_result.pvalue
                < ALPHA
            ),
    },

    "summary": {
        initializer: describe(
            initializer_matrix[:, j]
        )
        for j, initializer
        in enumerate(
            INITIALIZERS
        )
    },
}


initializer_pairs = []

for a, b in itertools.combinations(
    range(len(INITIALIZERS)),
    2,
):

    result = paired_wilcoxon(
        initializer_matrix[:, a],
        initializer_matrix[:, b],
    )

    initializer_pairs.append({
        "initializer_a":
            INITIALIZERS[a],

        "initializer_b":
            INITIALIZERS[b],

        **result,
    })

initializer_main[
    "pairwise_wilcoxon_holm"
] = holm_adjust(
    initializer_pairs
)


# ======================================================================
# 10. RANKINGS
# ======================================================================

ranking_by_mean_J = sorted(
    ARMS,
    key=lambda arm:
        arm_stats[arm][
            "openmc_J"
        ]["mean"],
)

ranking_by_median_J = sorted(
    ARMS,
    key=lambda arm:
        arm_stats[arm][
            "openmc_J"
        ]["median"],
)

ranking_by_mean_rank = sorted(
    ARMS,
    key=lambda arm:
        arm_stats[arm][
            "mean_rank"
        ],
)

ranking_by_runtime = sorted(
    ARMS,
    key=lambda arm:
        arm_stats[arm][
            "ga_wall_seconds"
        ]["mean"],
)


# Primary benchmark winner:
# lowest mean within-replicate rank.
primary_best_arm = (
    ranking_by_mean_rank[0]
)


# ======================================================================
# 11. PAIRWISE RESULT VS PRIMARY BEST ARM
# ======================================================================

best_vs_others = []

for record in pairwise:

    if (
        record["arm_a"]
        == primary_best_arm
    ):

        other = record["arm_b"]

        delta_best_minus_other = (
            record[
                "mean_delta_a_minus_b"
            ]
        )

        best_wins = (
            record["a_better"]
        )

        other_wins = (
            record["b_better"]
        )

    elif (
        record["arm_b"]
        == primary_best_arm
    ):

        other = record["arm_a"]

        delta_best_minus_other = (
            -record[
                "mean_delta_a_minus_b"
            ]
        )

        best_wins = (
            record["b_better"]
        )

        other_wins = (
            record["a_better"]
        )

    else:
        continue

    best_vs_others.append({
        "other_arm":
            other,

        "mean_delta_best_minus_other":
            float(
                delta_best_minus_other
            ),

        "best_better_replicates":
            int(best_wins),

        "other_better_replicates":
            int(other_wins),

        "ties":
            int(record["ties"]),

        "p_raw":
            float(record["p_raw"]),

        "p_holm":
            float(record["p_holm"]),

        "significant_holm_0.05":
            bool(
                record[
                    "significant_holm_0.05"
                ]
            ),
    })


best_vs_others.sort(
    key=lambda x:
        x["other_arm"]
)


# ======================================================================
# 12. FREEZE BENCHMARK JSON
# ======================================================================

benchmark = {
    "stage":
        "CYCLE5_FINAL_PAIRED_OPTIMIZER_BENCHMARK",

    "status":
        "FROZEN_COMPLETE",

    "primary_endpoint":
        "OpenMC objective J = "
        "0.5*(Fa/2.0) + "
        "0.5*(Fq_r/2.5); lower is better",

    "statistical_unit":
        "one frozen final winner from one "
        "solver x initializer run",

    "design": {
        "replicates":
            30,

        "arms":
            12,

        "total_runs":
            360,

        "paired_seed_within_replicate":
            True,

        "solvers":
            SOLVERS,

        "initializers":
            INITIALIZERS,
    },

    "scientific_status": {
        "q5_final_certificate":
            True,

        "q5_refit":
            False,

        "Residual5_refit":
            False,

        "retrospective_winner_reselection":
            False,

        "all_OpenMC_actual_safe":
            True,

        "prospective_q5_validation_pass":
            True,
    },

    "arm_statistics":
        arm_stats,

    "primary_12_arm_friedman":
        friedman_12,

    "pairwise_12_arm_wilcoxon_holm":
        pairwise,

    "solver_main_effect":
        solver_main,

    "initializer_main_effect":
        initializer_main,

    "rankings": {
        "by_mean_openmc_J":
            ranking_by_mean_J,

        "by_median_openmc_J":
            ranking_by_median_J,

        "by_mean_within_replicate_rank":
            ranking_by_mean_rank,

        "by_mean_ga_runtime":
            ranking_by_runtime,
    },

    "primary_best_arm_by_mean_rank":
        primary_best_arm,

    "primary_best_vs_other_arms":
        best_vs_others,

    "source_artifacts": {
        "prospective_openmc360": {
            "file":
                rel(MERGED),

            "sha256":
                sha(MERGED),
        },

        "prospective_evaluation": {
            "file":
                rel(EVALUATION),

            "sha256":
                sha(EVALUATION),
        },

        "winner_manifest": {
            "file":
                rel(WINNERS),

            "sha256":
                sha(WINNERS),
        },

        "final_q5_certificate": {
            "file":
                rel(FINAL_CERTIFICATE),

            "sha256":
                sha(FINAL_CERTIFICATE),
        },

        "final_q5_freeze": {
            "file":
                rel(FINAL_FREEZE),

            "sha256":
                sha(FINAL_FREEZE),
        },
    },

    "interpretation_rule":
        "The 12-arm paired OpenMC comparison is primary. "
        "Solver and initializer main-effect summaries are "
        "secondary descriptive repeated-measures analyses. "
        "No method is declared statistically superior solely "
        "from raw mean J; Holm-adjusted paired comparisons and "
        "within-replicate ranks are retained.",
}

frozen_write_json(
    OUT_JSON,
    benchmark,
)


# ======================================================================
# 13. FREEZE COMPACT ARM TABLE
# ======================================================================

arm_table = []

for rank_no, arm in enumerate(
    ranking_by_mean_rank,
    start=1,
):

    s = arm_stats[arm]

    arm_table.append({
        "rank":
            rank_no,

        "arm":
            arm,

        "solver":
            s["solver"],

        "initializer":
            s["initializer"],

        "mean_rank":
            f"{s['mean_rank']:.6f}",

        "mean_OpenMC_J":
            f"{s['openmc_J']['mean']:.9f}",

        "median_OpenMC_J":
            f"{s['openmc_J']['median']:.9f}",

        "std_OpenMC_J":
            f"{s['openmc_J']['std']:.9f}",

        "mean_Fa":
            f"{s['openmc_Fa']['mean']:.9f}",

        "mean_Fq_r":
            f"{s['openmc_Fq_r']['mean']:.9f}",

        "mean_keff":
            f"{s['openmc_keff']['mean']:.9f}",

        "joint_q5_coverage":
            (
                f"{s['q5_joint_coverage']['covered']}/30"
            ),

        "actual_safe":
            (
                f"{s['actual_safe']['passed']}/30"
            ),

        "mean_GA_minutes":
            f"{s['ga_wall_minutes_mean']:.4f}",

        "surrogate_J_MAE":
            (
                f"{s['surrogate_minus_openmc_J']['mae']:.9f}"
            ),

        "best_in_replicate_count":
            s["best_in_replicate_count"],
    })


write_tsv(
    OUT_TSV,
    arm_table,
    [
        "rank",
        "arm",
        "solver",
        "initializer",
        "mean_rank",
        "mean_OpenMC_J",
        "median_OpenMC_J",
        "std_OpenMC_J",
        "mean_Fa",
        "mean_Fq_r",
        "mean_keff",
        "joint_q5_coverage",
        "actual_safe",
        "mean_GA_minutes",
        "surrogate_J_MAE",
        "best_in_replicate_count",
    ],
)


# ======================================================================
# 14. FREEZE PAIRWISE TABLE
# ======================================================================

pairwise_rows = []

for p in pairwise:

    pairwise_rows.append({
        "arm_a":
            p["arm_a"],

        "arm_b":
            p["arm_b"],

        "mean_delta_a_minus_b":
            f"{p['mean_delta_a_minus_b']:.10f}",

        "median_delta_a_minus_b":
            f"{p['median_delta_a_minus_b']:.10f}",

        "a_better":
            p["a_better"],

        "b_better":
            p["b_better"],

        "ties":
            p["ties"],

        "p_raw":
            f"{p['p_raw']:.12g}",

        "p_holm":
            f"{p['p_holm']:.12g}",

        "significant_holm_0.05":
            p[
                "significant_holm_0.05"
            ],
    })


write_tsv(
    PAIRWISE_TSV,
    pairwise_rows,
    [
        "arm_a",
        "arm_b",
        "mean_delta_a_minus_b",
        "median_delta_a_minus_b",
        "a_better",
        "b_better",
        "ties",
        "p_raw",
        "p_holm",
        "significant_holm_0.05",
    ],
)


# ======================================================================
# 15. PRINT PAPER-READY RESULT
# ======================================================================

print("=" * 116)
print("CYCLE 5 STEP 13A — FINAL PAIRED OPTIMIZER BENCHMARK")
print("=" * 116)

print()
print("DESIGN")
print("  paired replicates              = 30")
print("  arms                           = 12")
print("  total OpenMC-validated runs    = 360")
print("  primary endpoint               = OpenMC J (lower is better)")
print("  actual physical feasibility    = 360/360")
print("  frozen q5 deployment cert      = YES")

print()
print("12-ARM FRIEDMAN TEST")
print(
    "  statistic                      = "
    f"{friedman_12['statistic']:.6f}"
)
print(
    "  p-value                        = "
    f"{friedman_12['p_value']:.12g}"
)
print(
    "  significant @ 0.05             = "
    f"{friedman_12['significant_0.05']}"
)

print()
print("FINAL ARM RANKING — OPENMC TRUTH")
print("-" * 116)
print(
    f"{'RK':>2}  "
    f"{'ARM':12s} "
    f"{'MEAN RANK':>10s} "
    f"{'MEAN J':>12s} "
    f"{'MEDIAN J':>12s} "
    f"{'BEST/30':>8s} "
    f"{'Q5 COV':>8s} "
    f"{'GA MIN':>10s}"
)
print("-" * 116)

for i, arm in enumerate(
    ranking_by_mean_rank,
    start=1,
):

    s = arm_stats[arm]

    print(
        f"{i:2d}  "
        f"{arm:12s} "
        f"{s['mean_rank']:10.4f} "
        f"{s['openmc_J']['mean']:12.8f} "
        f"{s['openmc_J']['median']:12.8f} "
        f"{s['best_in_replicate_count']:8d} "
        f"{s['q5_joint_coverage']['covered']:2d}/30   "
        f"{s['ga_wall_minutes_mean']:10.3f}"
    )

print("-" * 116)

print()
print("PRIMARY BEST ARM")
print("  arm                            =", primary_best_arm)
print(
    "  mean within-replicate rank     =",
    f"{arm_stats[primary_best_arm]['mean_rank']:.6f}",
)
print(
    "  mean OpenMC J                  =",
    f"{arm_stats[primary_best_arm]['openmc_J']['mean']:.9f}",
)
print(
    "  median OpenMC J                =",
    f"{arm_stats[primary_best_arm]['openmc_J']['median']:.9f}",
)
print(
    "  best replicate count           =",
    arm_stats[
        primary_best_arm
    ]["best_in_replicate_count"],
    "/30",
)

print()
print("SOLVER MAIN EFFECT")
print(
    "  Friedman p                     =",
    f"{solver_main['friedman']['p_value']:.12g}",
)

for solver in SOLVERS:
    print(
        f"  {solver:4s} mean J                  = "
        f"{solver_main['summary'][solver]['mean']:.9f}"
    )

print()
print("INITIALIZER MAIN EFFECT")
print(
    "  Friedman p                     =",
    f"{initializer_main['friedman']['p_value']:.12g}",
)

for initializer in INITIALIZERS:
    print(
        f"  {initializer:6s} mean J                = "
        f"{initializer_main['summary'][initializer]['mean']:.9f}"
    )

print()
print("FROZEN ARTIFACTS")
print("  benchmark JSON                 =", OUT_JSON)
print("  benchmark JSON SHA256          =", sha(OUT_JSON))
print()
print("  arm ranking TSV                =", OUT_TSV)
print("  arm ranking TSV SHA256         =", sha(OUT_TSV))
print()
print("  pairwise TSV                   =", PAIRWISE_TSV)
print("  pairwise TSV SHA256            =", sha(PAIRWISE_TSV))

print()
print("Q5 REFIT                        = NO")
print("RESIDUAL5 REFIT                 = NO")
print("NEW OPENMC RUNS                 = NO")
print("RETROSPECTIVE WINNER RESELECTION= NO")

print()
print("CYCLE5_STEP13A_FINAL_PAIRED_BENCHMARK_FROZEN")
