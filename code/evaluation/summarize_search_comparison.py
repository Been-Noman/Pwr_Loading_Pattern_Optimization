#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json


PROJECT = Path(
    __import__("os").environ["PWR_PROJECT"]
)

PROTOCOL = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/"
      "q5_prospective360/protocol"
)

BENCHMARK = (
    PROTOCOL
    / "cycle5_q5_final_paired_optimizer_benchmark_v1.json"
)

PAIRWISE_TSV = (
    PROTOCOL
    / "cycle5_q5_final_paired_optimizer_pairwise_v1.tsv"
)

FINAL_CERTIFICATE = (
    PROJECT
    / "run/optimizer_tail_v2/models/"
      "learned_residual_safety_round5_q5_prospective_search/"
      "deployment_certificate_final_q5.json"
)

OUT = (
    PROTOCOL
    / "cycle5_q5_final_deployment_method_decision_v1.json"
)


EXPECTED_BENCHMARK_SHA = (
    "8f5b3612131883eca0fc84e0a74cfea173f9622d2435753a3b77a150970bb2b9"
)

EXPECTED_PAIRWISE_SHA = (
    "b3b2e34170abd673a9827386a8e9e0d2f24ce2f33cc2b9bcc7aab3887c1e28de"
)

EXPECTED_CERT_SHA = (
    "4469a191188f940ad79b64f0bd1eff12d7070eb0b665dcd66ff65ad08d45a6f4"
)

ALPHA = 0.05


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def rel(path):
    return str(
        path.relative_to(PROJECT)
    )


def frozen_write(path, obj):

    data = (
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()

    if path.exists():

        if path.read_bytes() != data:
            raise RuntimeError(
                f"Frozen artifact exists with different bytes:\n{path}"
            )

        return

    path.write_bytes(data)


# ================================================================
# 1. VERIFY FROZEN INPUTS
# ================================================================

for name, path, expected in [
    (
        "paired benchmark",
        BENCHMARK,
        EXPECTED_BENCHMARK_SHA,
    ),
    (
        "pairwise table",
        PAIRWISE_TSV,
        EXPECTED_PAIRWISE_SHA,
    ),
    (
        "q5 deployment certificate",
        FINAL_CERTIFICATE,
        EXPECTED_CERT_SHA,
    ),
]:

    if not path.is_file():
        raise RuntimeError(
            f"Missing {name}:\n{path}"
        )

    actual = sha(path)

    if actual != expected:
        raise RuntimeError(
            f"{name} SHA changed:\n"
            f"expected={expected}\n"
            f"actual  ={actual}"
        )


b = json.loads(
    BENCHMARK.read_text()
)

if b["status"] != "FROZEN_COMPLETE":
    raise RuntimeError(
        "Final paired benchmark is not frozen"
    )

if not b["scientific_status"][
    "q5_final_certificate"
]:
    raise RuntimeError(
        "q5 final certificate flag is false"
    )

if not b["scientific_status"][
    "prospective_q5_validation_pass"
]:
    raise RuntimeError(
        "Prospective q5 validation did not pass"
    )


# ================================================================
# 2. PRIMARY WINNER
# ================================================================

primary = b[
    "primary_best_arm_by_mean_rank"
]

if primary != "v3m_sa":
    raise RuntimeError(
        f"Expected primary arm v3m_sa, got {primary}"
    )

stats = b["arm_statistics"]

primary_stats = stats[primary]


# ================================================================
# 3. HOLM-ADJUSTED PRIMARY-vs-ALL COMPARISONS
#
# Negative delta = primary has LOWER / BETTER OpenMC J.
#
# Important:
# p >= .05 means "not statistically separated";
# it does NOT prove equivalence.
# ================================================================

comparisons = []

for x in b["primary_best_vs_other_arms"]:

    other = x["other_arm"]

    comparisons.append({
        "other_arm":
            other,

        "mean_delta_primary_minus_other":
            float(
                x[
                    "mean_delta_best_minus_other"
                ]
            ),

        "primary_better_replicates":
            int(
                x[
                    "best_better_replicates"
                ]
            ),

        "other_better_replicates":
            int(
                x[
                    "other_better_replicates"
                ]
            ),

        "ties":
            int(x["ties"]),

        "p_raw":
            float(x["p_raw"]),

        "p_holm":
            float(x["p_holm"]),

        "significant_holm":
            bool(
                x[
                    "significant_holm_0.05"
                ]
            ),

        "primary_mean_J":
            float(
                primary_stats[
                    "openmc_J"
                ]["mean"]
            ),

        "other_mean_J":
            float(
                stats[other][
                    "openmc_J"
                ]["mean"]
            ),

        "primary_mean_runtime_minutes":
            float(
                primary_stats[
                    "ga_wall_minutes_mean"
                ]
            ),

        "other_mean_runtime_minutes":
            float(
                stats[other][
                    "ga_wall_minutes_mean"
                ]
            ),
    })


comparisons.sort(
    key=lambda x:
        x["other_mean_J"]
)


not_separated = [
    primary
]

significantly_outperformed = []

unexpected_primary_losses = []

for x in comparisons:

    delta = x[
        "mean_delta_primary_minus_other"
    ]

    sig = x[
        "significant_holm"
    ]

    other = x[
        "other_arm"
    ]

    if not sig:
        not_separated.append(
            other
        )

    elif delta < 0:
        # Primary significantly lower J.
        significantly_outperformed.append(
            other
        )

    elif delta > 0:
        # Another arm significantly lower J than primary.
        unexpected_primary_losses.append(
            other
        )


if unexpected_primary_losses:
    raise RuntimeError(
        "Primary ranking winner is significantly "
        "worse than another arm: "
        f"{unexpected_primary_losses}"
    )


# ================================================================
# 4. SOLVER MAIN-EFFECT RESULTS
# ================================================================

solver_main = b[
    "solver_main_effect"
]

solver_pairs = (
    solver_main[
        "pairwise_wilcoxon_holm"
    ]
)

solver_summary = (
    solver_main[
        "summary"
    ]
)


# ================================================================
# 5. INITIALIZER MAIN-EFFECT RESULTS
# ================================================================

initializer_main = b[
    "initializer_main_effect"
]

initializer_pairs = (
    initializer_main[
        "pairwise_wilcoxon_holm"
    ]
)

initializer_summary = (
    initializer_main[
        "summary"
    ]
)


# ================================================================
# 6. RUNTIME-SENSITIVE SECONDARY REFERENCE
#
# This does NOT replace the primary deployment method.
#
# Among methods not statistically separated from v3m_sa,
# identify the fastest arm only as an engineering reference.
# ================================================================

fastest_not_separated = min(
    not_separated,
    key=lambda arm:
        stats[arm][
            "ga_wall_minutes_mean"
        ],
)

runtime_primary = float(
    stats[primary][
        "ga_wall_minutes_mean"
    ]
)

runtime_fast = float(
    stats[
        fastest_not_separated
    ][
        "ga_wall_minutes_mean"
    ]
)

runtime_ratio = (
    runtime_primary
    / runtime_fast
)


# ================================================================
# 7. FINAL METHOD DECISION
#
# We retain the pre-specified primary endpoint/ranking rule.
# Runtime does not retrospectively change the winner.
# ================================================================

decision = {
    "stage":
        "CYCLE5_FINAL_DEPLOYMENT_METHOD_DECISION",

    "status":
        "FROZEN",

    "primary_endpoint":
        "paired OpenMC objective J; lower is better",

    "primary_ranking_rule":
        "lowest mean within-replicate rank across 30 paired replicates",

    "primary_deployment_arm":
        primary,

    "primary_mean_rank":
        float(
            primary_stats[
                "mean_rank"
            ]
        ),

    "primary_mean_OpenMC_J":
        float(
            primary_stats[
                "openmc_J"
            ]["mean"]
        ),

    "primary_median_OpenMC_J":
        float(
            primary_stats[
                "openmc_J"
            ]["median"]
        ),

    "primary_best_replicates":
        int(
            primary_stats[
                "best_in_replicate_count"
            ]
        ),

    "primary_mean_GA_minutes":
        runtime_primary,

    "global_12_arm_test": {
        "friedman_statistic":
            float(
                b[
                    "primary_12_arm_friedman"
                ]["statistic"]
            ),

        "p_value":
            float(
                b[
                    "primary_12_arm_friedman"
                ]["p_value"]
            ),

        "significant":
            bool(
                b[
                    "primary_12_arm_friedman"
                ][
                    "significant_0.05"
                ]
            ),
    },

    "primary_vs_all_Holm":
        comparisons,

    "arms_not_statistically_separated_from_primary":
        not_separated,

    "interpretation_of_not_separated":
        (
            "Failure to reject a paired difference after Holm correction "
            "is not evidence of equivalence. These arms are reported only "
            "as not statistically separated from the primary method."
        ),

    "arms_significantly_outperformed_by_primary":
        significantly_outperformed,

    "arms_significantly_better_than_primary":
        unexpected_primary_losses,

    "solver_main_effect": {
        "friedman_p":
            float(
                solver_main[
                    "friedman"
                ]["p_value"]
            ),

        "mean_OpenMC_J":
            {
                solver:
                    float(
                        solver_summary[
                            solver
                        ]["mean"]
                    )
                for solver
                in (
                    "v2",
                    "v2m",
                    "v3",
                    "v3m",
                )
            },

        "pairwise_wilcoxon_holm":
            solver_pairs,
    },

    "initializer_main_effect": {
        "friedman_p":
            float(
                initializer_main[
                    "friedman"
                ]["p_value"]
            ),

        "mean_OpenMC_J":
            {
                initializer:
                    float(
                        initializer_summary[
                            initializer
                        ]["mean"]
                    )
                for initializer
                in (
                    "random",
                    "sa",
                    "sqa",
                )
            },

        "pairwise_wilcoxon_holm":
            initializer_pairs,
    },

    "secondary_runtime_reference": {
        "fastest_arm_not_statistically_separated_from_primary":
            fastest_not_separated,

        "fastest_arm_mean_GA_minutes":
            runtime_fast,

        "primary_to_fastest_runtime_ratio":
            runtime_ratio,

        "role":
            (
                "engineering/runtime reference only; "
                "does not replace the primary deployment arm"
            ),
    },

    "scientific_decision": {
        "deployment_method":
            primary,

        "q5":
            "frozen final deployment certificate",

        "q5_refit":
            False,

        "Residual5_refit":
            False,

        "new_OpenMC_used":
            False,

        "winner_reselection":
            False,

        "reason":
            (
                "v3m_sa is retained because it is the winner under "
                "the frozen primary paired ranking rule. Statistical "
                "pairwise results are reported as supporting inference "
                "and do not retrospectively redefine the endpoint."
            ),
    },

    "sources": {
        "benchmark":
            rel(BENCHMARK),

        "benchmark_sha256":
            sha(BENCHMARK),

        "pairwise_tsv":
            rel(PAIRWISE_TSV),

        "pairwise_tsv_sha256":
            sha(PAIRWISE_TSV),

        "q5_certificate":
            rel(FINAL_CERTIFICATE),

        "q5_certificate_sha256":
            sha(FINAL_CERTIFICATE),
    },
}

frozen_write(
    OUT,
    decision,
)


# ================================================================
# 8. OUTPUT
# ================================================================

print("=" * 112)
print("CYCLE 5 STEP 13B — FINAL DEPLOYMENT METHOD DECISION")
print("=" * 112)

print()
print("PRIMARY METHOD")
print("  deployment arm                =", primary)
print(
    "  mean paired rank              =",
    f"{primary_stats['mean_rank']:.4f}",
)
print(
    "  mean OpenMC J                 =",
    f"{primary_stats['openmc_J']['mean']:.9f}",
)
print(
    "  median OpenMC J               =",
    f"{primary_stats['openmc_J']['median']:.9f}",
)
print(
    "  best replicate count          =",
    f"{primary_stats['best_in_replicate_count']}/30",
)
print(
    "  mean GA runtime               =",
    f"{runtime_primary:.3f} min",
)

print()
print("v3m_sa vs OTHER ARMS — HOLM-ADJUSTED PAIRED TESTS")
print("-" * 112)

print(
    f"{'OTHER ARM':12s} "
    f"{'ΔJ PRIMARY-OTHER':>18s} "
    f"{'W-L-T':>10s} "
    f"{'p(Holm)':>14s} "
    f"{'SIG':>6s}"
)

print("-" * 112)

for x in comparisons:

    print(
        f"{x['other_arm']:12s} "
        f"{x['mean_delta_primary_minus_other']:+18.9f} "
        f"{x['primary_better_replicates']:2d}-"
        f"{x['other_better_replicates']:2d}-"
        f"{x['ties']:2d} "
        f"{x['p_holm']:14.8g} "
        f"{str(x['significant_holm']):>6s}"
    )

print("-" * 112)

print()
print("NOT STATISTICALLY SEPARATED FROM v3m_sa")
for arm in not_separated:
    print(
        " ",
        arm,
        f"meanJ={stats[arm]['openmc_J']['mean']:.9f}",
        f"runtime={stats[arm]['ga_wall_minutes_mean']:.3f} min",
    )

print()
print("SIGNIFICANTLY OUTPERFORMED BY v3m_sa")
if significantly_outperformed:
    for arm in significantly_outperformed:
        print(" ", arm)
else:
    print("  NONE")

print()
print("SOLVER EFFECT")
print(
    "  Friedman p                    =",
    f"{solver_main['friedman']['p_value']:.12g}",
)
for solver in ("v2", "v2m", "v3", "v3m"):
    print(
        f"  {solver:4s} mean OpenMC J             = "
        f"{solver_summary[solver]['mean']:.9f}"
    )

print()
print("INITIALIZER EFFECT")
print(
    "  Friedman p                    =",
    f"{initializer_main['friedman']['p_value']:.12g}",
)
for initializer in ("random", "sa", "sqa"):
    print(
        f"  {initializer:6s} mean OpenMC J           = "
        f"{initializer_summary[initializer]['mean']:.9f}"
    )

print()
print("SECONDARY RUNTIME REFERENCE")
print(
    "  fastest non-separated arm     =",
    fastest_not_separated,
)
print(
    "  its mean GA runtime           =",
    f"{runtime_fast:.3f} min",
)
print(
    "  v3m_sa runtime / reference    =",
    f"{runtime_ratio:.3f}x",
)

print()
print("DEPLOYMENT DECISION")
print("  PRIMARY METHOD                 = v3m_sa")
print("  q5                             = FROZEN FINAL")
print("  q5 refit                       = NO")
print("  Residual5 refit                = NO")
print("  new OpenMC                     = NO")
print("  retrospective endpoint change = NO")

print()
print("FROZEN")
print("  decision                       =", OUT)
print("  decision SHA256                =", sha(OUT))

print()
print("CYCLE5_STEP13B_DEPLOYMENT_METHOD_DECISION_FROZEN")
