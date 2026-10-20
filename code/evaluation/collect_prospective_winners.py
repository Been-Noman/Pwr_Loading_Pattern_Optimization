#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import sys

PROJECT = Path(__import__("os").environ["PWR_PROJECT"])
sys.path.insert(0, str(PROJECT / "scripts"))

from geometry import canonical_d4_key, is_hard_feasible


# ======================================================================
# PATHS / FROZEN IDENTITIES
# ======================================================================

PROS = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360"
)

PROTOCOL = PROS / "protocol"

EXEC_MANIFEST = (
    PROTOCOL
    / "cycle5_q5_prospective360_execution_manifest_360_v1.json"
)

SCIENTIFIC_FIREWALL = (
    PROTOCOL
    / "cycle5_q5_prospective360_known_openmc_plus_calibration_d4_exclusion_v1.json"
)

RESIDUAL_METADATA = (
    PROJECT
    / "run/optimizer_tail_v2/models/"
      "learned_residual_safety_round5_q5_prospective_search/"
      "residual_safety_metadata.json"
)

WINNER_MANIFEST = (
    PROTOCOL
    / "cycle5_q5_prospective360_winner_manifest_360_v1.json"
)

WINNER_TSV = (
    PROTOCOL
    / "cycle5_q5_prospective360_winner_manifest_360_v1.tsv"
)

WINNER_AUDIT = (
    PROTOCOL
    / "cycle5_q5_prospective360_winner_manifest_360_v1_audit.json"
)

EXPECTED_EXEC_SHA = (
    "76f8e5c0598bd8a9312a2c49c41a3177c9d01b9518fe3bb8bfc95ad0b8baaee9"
)

EXPECTED_FIREWALL_SHA = (
    "2bbac884a16b5a90d1d4d3b91684555fff6a598a8267f0732fb40a0751556de4"
)

EXPECTED_RESIDUAL_SHA = (
    "5084368629a8f1fed3bdfa052c755f21aed767875d5d7d6f8601c115a867c4ca"
)

Q5 = {
    "keff_lower": 1.2555488951630267,
    "keff_upper": 1.167078988515213,
    "Fa_upper": -0.00824025190735769,
    "Fq_r_upper": -0.15313887213725857,
}

ATOL = 1e-6


# ======================================================================
# HELPERS
# ======================================================================

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(PROJECT))


def exact_key(pattern):
    return tuple(int(x) for x in pattern)


def d4_key(pattern):
    return tuple(canonical_d4_key(pattern))


def d4_sha(pattern):
    key = d4_key(pattern)
    return hashlib.sha256(
        bytes(int(x) for x in key)
    ).hexdigest()


def close(a, b):
    try:
        return math.isclose(
            float(a),
            float(b),
            rel_tol=0.0,
            abs_tol=ATOL,
        )
    except Exception:
        return False


def frozen_write_json(path, obj):
    data = (
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(
                f"Frozen artifact already exists with different bytes:\n{path}"
            )
        return

    path.write_bytes(data)


def require_file(path):
    if not path.is_file():
        raise RuntimeError(f"Missing file:\n{path}")


# ======================================================================
# 1. VERIFY FROZEN SCIENTIFIC INPUTS
# ======================================================================

for p in (
    EXEC_MANIFEST,
    SCIENTIFIC_FIREWALL,
    RESIDUAL_METADATA,
):
    require_file(p)

if sha(EXEC_MANIFEST) != EXPECTED_EXEC_SHA:
    raise RuntimeError("Execution manifest SHA changed")

if sha(SCIENTIFIC_FIREWALL) != EXPECTED_FIREWALL_SHA:
    raise RuntimeError("Scientific prospective firewall SHA changed")

if sha(RESIDUAL_METADATA) != EXPECTED_RESIDUAL_SHA:
    raise RuntimeError("Residual5 q5 metadata SHA changed")


manifest = json.loads(EXEC_MANIFEST.read_text())
runs = manifest["runs"]

if len(runs) != 360:
    raise RuntimeError(
        f"Expected 360 manifest runs, got {len(runs)}"
    )

if [int(r["task_id"]) for r in runs] != list(range(360)):
    raise RuntimeError("Task IDs are not exactly 0..359")


# ======================================================================
# 2. FIREWALL D4 SET
# ======================================================================

firewall_rows = json.loads(SCIENTIFIC_FIREWALL.read_text())

if not isinstance(firewall_rows, list):
    raise RuntimeError("Scientific firewall must be a JSON list")

firewall_exact = {
    exact_key(r["pattern"])
    for r in firewall_rows
}

firewall_d4 = {
    d4_key(r["pattern"])
    for r in firewall_rows
}

if len(firewall_d4) != 32333:
    raise RuntimeError(
        f"Expected 32333 scientific firewall D4 classes, "
        f"got {len(firewall_d4)}"
    )


# ======================================================================
# 3. EXACT HISTORICAL WINNER MATCH POLICY
#
# One runner-selected winner per independent GA run.
#
# Match summary -> archive using:
#   exact:
#       ga_seed
#       calibrated_safe
#       d4_novel_vs_labelled_10000
#
#   numeric:
#       calibrated_Fa_upper
#       calibrated_Fq_r_upper
#       calibrated_keff_lower
#       calibrated_keff_upper
#       ensemble_mean_Fa
#       ensemble_mean_Fq_r
#       ensemble_mean_keff
#       fitness
#       objective
#
# Required:
#   exactly one archive match
#   matching archive index == 0
#
# No retrospective winner reselection.
# ======================================================================

NUMERIC_MATCH = [
    ("calibrated_Fa_upper",   "best_calibrated_Fa_upper"),
    ("calibrated_Fq_r_upper", "best_calibrated_Fq_r_upper"),
    ("calibrated_keff_lower", "best_calibrated_keff_lower"),
    ("calibrated_keff_upper", "best_calibrated_keff_upper"),
    ("ensemble_mean_Fa",      "best_ensemble_mean_Fa"),
    ("ensemble_mean_Fq_r",    "best_ensemble_mean_Fq_r"),
    ("ensemble_mean_keff",    "best_ensemble_mean_keff"),
    ("fitness",               "best_fitness"),
    ("objective",             "best_objective"),
]


def archive_matches_summary(row, summary):

    if int(row["ga_seed"]) != int(summary["ga_seed"]):
        return False

    if bool(row["calibrated_safe"]) != bool(
        summary["best_calibrated_safe"]
    ):
        return False

    if bool(row["d4_novel_vs_labelled_10000"]) != bool(
        summary["best_d4_novel_vs_labelled_10000"]
    ):
        return False

    for archive_field, summary_field in NUMERIC_MATCH:

        if archive_field not in row:
            return False

        if summary_field not in summary:
            return False

        if not close(
            row[archive_field],
            summary[summary_field],
        ):
            return False

    return True


# ======================================================================
# 4. COLLECT 360 RUNNER-SELECTED WINNERS
# ======================================================================

winners = []

arm_counts = Counter()
summary_archive_matches = 0
hard_feasible_count = 0
calibrated_safe_count = 0
firewall_exact_overlap = 0
firewall_d4_overlap = 0

summary_hashes = {}
archive_hashes = {}

for run in runs:

    task = int(run["task_id"])
    rep = int(run["replicate"])
    solver = run["solver"]
    initializer = run["initializer"]
    seed = int(run["paired_ga_seed"])

    summary_path = PROJECT / run["expected_summary"]
    archive_path = PROJECT / run["expected_archive"]

    require_file(summary_path)
    require_file(archive_path)

    summary = json.loads(summary_path.read_text())
    archive = json.loads(archive_path.read_text())

    if not isinstance(summary, dict):
        raise RuntimeError(
            f"Task {task}: summary is not a dict"
        )

    if summary.get("status") != "COMPLETED":
        raise RuntimeError(
            f"Task {task}: summary status={summary.get('status')!r}"
        )

    if not isinstance(archive, list) or not archive:
        raise RuntimeError(
            f"Task {task}: archive is empty/not a list"
        )

    if int(summary["ga_seed"]) != seed:
        raise RuntimeError(
            f"Task {task}: summary seed mismatch"
        )

    context = summary.get("context", {})

    if int(context.get("ga_seed", seed)) != seed:
        raise RuntimeError(
            f"Task {task}: context GA seed mismatch"
        )

    # Frozen q5 must be the runtime q for every run.
    context_q = context.get("q")

    if context_q is None:
        raise RuntimeError(
            f"Task {task}: summary context missing q"
        )

    for k, v in Q5.items():
        if abs(float(context_q[k]) - float(v)) > 1e-15:
            raise RuntimeError(
                f"Task {task}: q5 mismatch for {k}"
            )

    if (
        context.get("residual_model_metadata_sha256")
        != EXPECTED_RESIDUAL_SHA
    ):
        raise RuntimeError(
            f"Task {task}: Residual5 metadata SHA mismatch"
        )

    if (
        context.get("labelled_exclusion_sha256")
        != EXPECTED_FIREWALL_SHA
    ):
        raise RuntimeError(
            f"Task {task}: prospective firewall SHA mismatch"
        )

    # Anchor identity must agree for SA/SQA.
    if initializer == "random":

        if run["anchor_json"] is not None:
            raise RuntimeError(
                f"Task {task}: random run unexpectedly has anchor bank"
            )

    else:

        if run["anchor_json"] is None:
            raise RuntimeError(
                f"Task {task}: {initializer} missing anchor bank"
            )

        if (
            context.get("anchor_sha256")
            != run["anchor_sha256"]
        ):
            raise RuntimeError(
                f"Task {task}: runtime anchor SHA mismatch"
            )

    # Runner hard-constraint result.
    hard_audit = summary.get("hard_constraint_audit")

    if not isinstance(hard_audit, dict):
        raise RuntimeError(
            f"Task {task}: missing hard_constraint_audit"
        )

    if int(hard_audit.get("total", -1)) != 0:
        raise RuntimeError(
            f"Task {task}: runner hard constraint audit failed"
        )

    # Exactly one archive row must match the summary-selected winner.
    matches = [
        i
        for i, row in enumerate(archive)
        if archive_matches_summary(row, summary)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Task {task}: expected exactly one summary/archive "
            f"winner match, got {matches}"
        )

    idx = matches[0]

    # Exact historical parity requirement.
    if idx != 0:
        raise RuntimeError(
            f"Task {task}: runner-selected winner matched "
            f"archive index {idx}, expected index 0"
        )

    summary_archive_matches += 1

    row = archive[idx]

    pattern = [int(x) for x in row["pattern"]]

    if len(pattern) != 193:
        raise RuntimeError(
            f"Task {task}: winner pattern length != 193"
        )

    hard_ok = bool(is_hard_feasible(pattern))

    if not hard_ok:
        raise RuntimeError(
            f"Task {task}: independently non-hard-feasible winner"
        )

    hard_feasible_count += 1

    calibrated_safe = bool(row["calibrated_safe"])

    if calibrated_safe:
        calibrated_safe_count += 1

    ex_key = exact_key(pattern)
    dk = d4_key(pattern)

    exact_novel = ex_key not in firewall_exact
    d4_novel = dk not in firewall_d4

    if not exact_novel:
        firewall_exact_overlap += 1

    if not d4_novel:
        firewall_d4_overlap += 1

    if not d4_novel:
        raise RuntimeError(
            f"Task {task}: winner overlaps prospective firewall by D4"
        )

    if not bool(row["d4_novel_vs_labelled_10000"]):
        raise RuntimeError(
            f"Task {task}: runner says winner is not D4 novel"
        )

    s_sha = sha(summary_path)
    a_sha = sha(archive_path)

    summary_hashes[str(task)] = s_sha
    archive_hashes[str(task)] = a_sha

    arm = f"{solver}_{initializer}"
    arm_counts[arm] += 1

    winner = {
        "openmc_case_index":
            task,

        "winner_id":
            f"cycle5_q5_prospective360_task_{task:03d}",

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
            seed,

        "pattern":
            pattern,

        "pattern_length":
            193,

        "archive_index":
            idx,

        "archive_path":
            rel(archive_path),

        "archive_sha256":
            a_sha,

        "summary_path":
            rel(summary_path),

        "summary_sha256":
            s_sha,

        "winner_generation":
            int(row["generation"]),

        "runner_mode":
            context.get("runner_mode"),

        "best_selection_mode":
            summary.get("best_selection_mode"),

        "surrogate_fitness":
            float(row["fitness"]),

        "surrogate_objective_J":
            float(row["objective"]),

        "normalized_violation":
            (
                None
                if row.get("normalized_violation") is None
                else float(row["normalized_violation"])
            ),

        "ensemble_mean_keff":
            float(row["ensemble_mean_keff"]),

        "ensemble_mean_Fa":
            float(row["ensemble_mean_Fa"]),

        "ensemble_mean_Fq_r":
            float(row["ensemble_mean_Fq_r"]),

        "ensemble_std_keff":
            float(row["ensemble_std_keff"]),

        "ensemble_std_Fa":
            float(row["ensemble_std_Fa"]),

        "ensemble_std_Fq_r":
            float(row["ensemble_std_Fq_r"]),

        "calibrated_keff_lower":
            float(row["calibrated_keff_lower"]),

        "calibrated_keff_upper":
            float(row["calibrated_keff_upper"]),

        "calibrated_Fa_upper":
            float(row["calibrated_Fa_upper"]),

        "calibrated_Fq_r_upper":
            float(row["calibrated_Fq_r_upper"]),

        "calibrated_safe":
            calibrated_safe,

        "hard_feasible_independent":
            hard_ok,

        "hard_constraint_audit_independent":
            hard_audit,

        "exact_novel_vs_prospective_firewall":
            exact_novel,

        "d4_novel_vs_prospective_firewall":
            d4_novel,

        "archive_d4_key_sha256":
            (
                row.get("d4_key_sha256")
                or d4_sha(pattern)
            ),
    }

    winners.append(winner)


# ======================================================================
# 5. EXPERIMENTAL-UNIT / DUPLICATE ACCOUNTING
#
# Natural duplicate winners are preserved.
# They are NOT deleted and NOT replaced.
# ======================================================================

if len(winners) != 360:
    raise RuntimeError(
        f"Expected 360 winners, got {len(winners)}"
    )

if summary_archive_matches != 360:
    raise RuntimeError(
        "Not all runs had exactly one summary/archive match"
    )

expected_arms = {
    f"{s}_{i}": 30
    for s in ("v2", "v2m", "v3", "v3m")
    for i in ("random", "sa", "sqa")
}

if dict(arm_counts) != expected_arms:
    raise RuntimeError(
        f"Arm balance failed:\n{dict(arm_counts)}"
    )


exact_groups = defaultdict(list)
d4_groups = defaultdict(list)

for w in winners:

    exact_groups[
        exact_key(w["pattern"])
    ].append(w["task_id"])

    d4_groups[
        d4_key(w["pattern"])
    ].append(w["task_id"])


exact_duplicate_task_groups = [
    ids
    for ids in exact_groups.values()
    if len(ids) > 1
]

d4_duplicate_task_groups = [
    ids
    for ids in d4_groups.values()
    if len(ids) > 1
]

exact_unique = len(exact_groups)
d4_unique = len(d4_groups)

exact_duplicate_rows = 360 - exact_unique
d4_duplicate_rows = 360 - d4_unique


# ======================================================================
# 6. FREEZE WINNER MANIFEST
# ======================================================================

winner_manifest = {
    "stage":
        "CYCLE5_Q5_PROSPECTIVE360_WINNER_FREEZE",

    "status":
        "FROZEN_OPENMC_READY",

    "experimental_unit":
        "one_final_winner_per_independent_GA_run",

    "selection_policy": (
        "One runner-selected final winner per independent GA run. "
        "Winner identified only by matching frozen summary metrics "
        "to exactly one archive record. No retrospective reselection."
    ),

    "selection_match_implementation": {
        "required_archive_index":
            0,

        "exact_match_fields": [
            "ga_seed",
            "calibrated_safe",
            "d4_novel_vs_labelled_10000",
        ],

        "matched_numeric_fields": [
            {
                "archive": a,
                "summary": s,
            }
            for a, s in NUMERIC_MATCH
        ],

        "numeric_absolute_tolerance":
            ATOL,

        "numeric_relative_tolerance":
            0.0,

        "required_matches_per_run":
            1,
    },

    "hard_feasibility_implementation":
        "pwr_geometry.is_hard_feasible",

    "d4_implementation":
        "pwr_geometry.canonical_d4_key",

    "source_execution_manifest":
        rel(EXEC_MANIFEST),

    "source_execution_manifest_sha256":
        sha(EXEC_MANIFEST),

    "prospective_firewall":
        rel(SCIENTIFIC_FIREWALL),

    "prospective_firewall_sha256":
        sha(SCIENTIFIC_FIREWALL),

    "residual_metadata":
        rel(RESIDUAL_METADATA),

    "residual_metadata_sha256":
        sha(RESIDUAL_METADATA),

    "q5":
        Q5,

    "q5_refit":
        False,

    "residual5_refit":
        False,

    "openmc_used_for_selection":
        False,

    "retrospective_reselection":
        False,

    "duplicates_deleted":
        False,

    "duplicates_replaced":
        False,

    "replicates_per_arm":
        30,

    "arms":
        sorted(expected_arms),

    "arm_counts":
        dict(sorted(arm_counts.items())),

    "total_winners":
        360,

    "openmc_cases_planned":
        360,

    "hard_feasible_winners":
        hard_feasible_count,

    "calibrated_safe_winners":
        calibrated_safe_count,

    "exact_unique_winners":
        exact_unique,

    "exact_duplicate_rows":
        exact_duplicate_rows,

    "exact_duplicate_task_groups":
        exact_duplicate_task_groups,

    "d4_unique_winners":
        d4_unique,

    "d4_duplicate_rows":
        d4_duplicate_rows,

    "d4_duplicate_task_groups":
        d4_duplicate_task_groups,

    "exact_firewall_overlap":
        firewall_exact_overlap,

    "d4_firewall_overlap":
        firewall_d4_overlap,

    "summary_archive_unique_matches":
        summary_archive_matches,

    "summary_sha256_by_task":
        summary_hashes,

    "archive_sha256_by_task":
        archive_hashes,

    "winners":
        winners,
}

frozen_write_json(
    WINNER_MANIFEST,
    winner_manifest,
)


# ======================================================================
# 7. TSV COMPANION
# ======================================================================

tsv_fields = [
    "openmc_case_index",
    "task_id",
    "replicate",
    "solver",
    "initializer",
    "arm",
    "paired_ga_seed",
    "winner_generation",
    "surrogate_fitness",
    "surrogate_objective_J",
    "calibrated_keff_lower",
    "calibrated_keff_upper",
    "calibrated_Fa_upper",
    "calibrated_Fq_r_upper",
    "calibrated_safe",
    "hard_feasible_independent",
    "exact_novel_vs_prospective_firewall",
    "d4_novel_vs_prospective_firewall",
    "summary_sha256",
    "archive_sha256",
]

rows = []

for w in winners:
    rows.append({
        k: w[k]
        for k in tsv_fields
    })

from io import StringIO

buf = StringIO()
writer = csv.DictWriter(
    buf,
    fieldnames=tsv_fields,
    delimiter="\t",
    lineterminator="\n",
)
writer.writeheader()
writer.writerows(rows)

tsv_bytes = buf.getvalue().encode()

if WINNER_TSV.exists():
    if WINNER_TSV.read_bytes() != tsv_bytes:
        raise RuntimeError(
            "Frozen winner TSV already exists with different bytes"
        )
else:
    WINNER_TSV.write_bytes(tsv_bytes)


# ======================================================================
# 8. AUTOMATIC FREEZE RECORD
# ======================================================================

audit = {
    "status":
        "PASS",

    "winners":
        360,

    "unique_summary_archive_matches":
        summary_archive_matches,

    "hard_feasible":
        hard_feasible_count,

    "calibrated_safe":
        calibrated_safe_count,

    "exact_unique":
        exact_unique,

    "exact_duplicate_rows":
        exact_duplicate_rows,

    "d4_unique":
        d4_unique,

    "d4_duplicate_rows":
        d4_duplicate_rows,

    "exact_firewall_overlap":
        firewall_exact_overlap,

    "d4_firewall_overlap":
        firewall_d4_overlap,

    "duplicates_deleted":
        False,

    "duplicates_replaced":
        False,

    "openmc_used_for_selection":
        False,

    "retrospective_reselection":
        False,

    "q5_refit":
        False,

    "residual5_refit":
        False,

    "winner_manifest":
        rel(WINNER_MANIFEST),

    "winner_manifest_sha256":
        sha(WINNER_MANIFEST),

    "winner_tsv":
        rel(WINNER_TSV),

    "winner_tsv_sha256":
        sha(WINNER_TSV),
}

frozen_write_json(
    WINNER_AUDIT,
    audit,
)


# ======================================================================
# 9. OUTPUT
# ======================================================================

print("=" * 100)
print("CYCLE 5 STEP 12E — PROSPECTIVE360 WINNERS FROZEN")
print("=" * 100)

print()
print("GA")
print("  completed GA runs              = 360")
print("  runner-selected winners        =", len(winners))
print("  unique summary/archive matches =", summary_archive_matches)
print("  arm balance                    = 30 per arm")

print()
print("WINNER INTEGRITY")
print("  hard-feasible                  =", hard_feasible_count)
print("  q5 calibrated-safe             =", calibrated_safe_count)
print("  exact unique                   =", exact_unique)
print("  natural exact duplicate rows   =", exact_duplicate_rows)
print("  D4 unique                      =", d4_unique)
print("  natural D4 duplicate rows      =", d4_duplicate_rows)
print("  duplicates deleted             = NO")
print("  duplicates replaced            = NO")

print()
print("FIREWALL")
print("  exact overlap                  =", firewall_exact_overlap)
print("  D4 overlap                     =", firewall_d4_overlap)

print()
print("SCIENTIFIC FIREWALLS")
print("  OpenMC used for selection      = NO")
print("  retrospective reselection      = NO")
print("  q5 refit                       = NO")
print("  Residual5 refit                = NO")

print()
print("FROZEN ARTIFACTS")
print("  winner manifest                =", WINNER_MANIFEST)
print("  winner manifest SHA256         =", sha(WINNER_MANIFEST))
print()
print("  winner TSV                     =", WINNER_TSV)
print("  winner TSV SHA256              =", sha(WINNER_TSV))
print()
print("  winner freeze record           =", WINNER_AUDIT)
print("  winner freeze SHA256           =", sha(WINNER_AUDIT))

print()
print("CYCLE5_STEP12E_PROSPECTIVE360_WINNERS_FROZEN_OPENMC_READY")
