#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math

PROJECT = Path(__import__("os").environ["PWR_PROJECT"])

PROS = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360"
)

PROTOCOL = PROS / "protocol"
OPENMC = PROS / "openmc360"

WINNERS = (
    PROTOCOL
    / "cycle5_q5_prospective360_winner_manifest_360_v1.json"
)

WINNER_AUDIT = (
    PROTOCOL
    / "cycle5_q5_prospective360_winner_manifest_360_v1_audit.json"
)

OPENMC_INPUT = (
    PROTOCOL
    / "cycle5_q5_prospective360_openmc_input_360_v1.json"
)

OPENMC_EXECUTION = (
    PROTOCOL
    / "cycle5_q5_prospective360_openmc360_execution_manifest_v1.json"
)

Q5_CANDIDATE = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/"
      "q5_calibration/protocol/cycle5_q5_candidate_balanced960_v1.json"
)

MERGED = (
    PROTOCOL
    / "cycle5_q5_prospective360_openmc_merged_360_v1.json"
)

EVALUATION = (
    PROTOCOL
    / "cycle5_q5_prospective360_evaluation_v1.json"
)

EXPECTED = {
    "winner":
        "73e4049c0c1c89f0306537a07ece92603ae6b25f6afc3af46f4e7c87c6321ac0",

    "winner_audit":
        "e33a0cddc91368adbe0a975d243ced8bcd9c1437d84b225079867d75f4bf0cc6",

    "openmc_input":
        "a3676f37877f5864fdfded62f5f2c72841f6c476a4659f8fe887ad8ad4f6cae0",

    "openmc_execution":
        "a474e33a7f5e55fce4ca0cde5a42d253bff9393bc9336a2bd2b8dccb30735a6d",

    "q5_candidate":
        "21c1f0ac8ee5b7546b4264e81d54dc6f0c68a74605bbb5e527c207e6f2fb6760",
}

Q5_EXPECTED = {
    "keff_lower": 1.2555488951630267,
    "keff_upper": 1.167078988515213,
    "Fa_upper": -0.00824025190735769,
    "Fq_r_upper": -0.15313887213725857,
}

PHYSICS = {
    "keff_lower": 1.0,
    "keff_upper": 1.2,
    "Fa_upper": 2.0,
    "Fq_r_upper": 2.5,
}

JOINT_TARGET = 0.95


# ======================================================================
# HELPERS
# ======================================================================

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(PROJECT))


def frozen_write(path, obj):
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


def exact_key(pattern):
    return tuple(int(x) for x in pattern)


def pct(n, d=360):
    return 100.0 * n / d


def finite(*values):
    return all(math.isfinite(float(x)) for x in values)


# ======================================================================
# 1. VERIFY ALL PRE-OPENMC FROZEN ARTIFACTS
# ======================================================================

for p in (
    WINNERS,
    WINNER_AUDIT,
    OPENMC_INPUT,
    OPENMC_EXECUTION,
    Q5_CANDIDATE,
):
    if not p.is_file():
        raise RuntimeError(f"Missing frozen dependency:\n{p}")

checks = [
    ("winner manifest", WINNERS, EXPECTED["winner"]),
    ("winner audit", WINNER_AUDIT, EXPECTED["winner_audit"]),
    ("OpenMC input", OPENMC_INPUT, EXPECTED["openmc_input"]),
    ("OpenMC execution", OPENMC_EXECUTION, EXPECTED["openmc_execution"]),
    ("q5 candidate", Q5_CANDIDATE, EXPECTED["q5_candidate"]),
]

for name, path, expected in checks:
    actual = sha(path)

    if actual != expected:
        raise RuntimeError(
            f"{name} SHA changed:\n"
            f"expected={expected}\n"
            f"actual  ={actual}"
        )


winner_obj = json.loads(WINNERS.read_text())
winner_rows = winner_obj["winners"]

input_rows = json.loads(OPENMC_INPUT.read_text())
execution = json.loads(OPENMC_EXECUTION.read_text())
q5_obj = json.loads(Q5_CANDIDATE.read_text())

if len(winner_rows) != 360:
    raise RuntimeError("Winner manifest no longer contains 360 units")

if len(input_rows) != 360:
    raise RuntimeError("OpenMC input no longer contains 360 units")

if int(execution["experimental_units"]) != 360:
    raise RuntimeError("OpenMC execution manifest unit count changed")

if int(execution["natural_duplicate_copies"]) != 7:
    raise RuntimeError("Expected seven natural duplicate copies")

if execution["duplicates_deleted"] or execution["duplicates_replaced"]:
    raise RuntimeError("Duplicate-preservation protocol changed")


# Verify exact frozen q5 candidate.
q5_found = None

for key in (
    "q5_candidate",
    "q5",
    "candidate_q5",
):
    if isinstance(q5_obj, dict) and isinstance(q5_obj.get(key), dict):
        q5_found = q5_obj[key]
        break

if q5_found is None:
    # Some certificate versions store q directly at top level.
    if all(k in q5_obj for k in Q5_EXPECTED):
        q5_found = q5_obj
    else:
        raise RuntimeError("Could not locate q5 values in frozen candidate")

for k, expected in Q5_EXPECTED.items():
    if abs(float(q5_found[k]) - expected) > 1e-15:
        raise RuntimeError(
            f"Frozen q5 changed for {k}: "
            f"{q5_found[k]} != {expected}"
        )


# ======================================================================
# 2. LOAD + VERIFY ALL 360 OPENMC CHUNKS
# ======================================================================

truth_by_id = {}
chunk_hashes = {}
metrics_hashes = {}

for idx in range(360):

    chunk = (
        OPENMC
        / "chunks"
        / f"chunk_{idx:03d}.jsonl"
    )

    metrics = (
        OPENMC
        / "metrics"
        / f"chunk_{idx:03d}_metrics.json"
    )

    if not chunk.is_file():
        raise RuntimeError(
            f"Missing OpenMC chunk {idx}:\n{chunk}"
        )

    if not metrics.is_file():
        raise RuntimeError(
            f"Missing OpenMC metrics {idx}:\n{metrics}"
        )

    chunk_hashes[str(idx)] = sha(chunk)
    metrics_hashes[str(idx)] = sha(metrics)

    lines = [
        line.strip()
        for line in chunk.read_text().splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        raise RuntimeError(
            f"Chunk {idx}: expected exactly one JSONL row, "
            f"got {len(lines)}"
        )

    row = json.loads(lines[0])

    if "error" in row:
        raise RuntimeError(
            f"Chunk {idx} contains OpenMC error:\n{row['error']}"
        )

    sid = int(row["id"])

    if sid != idx:
        raise RuntimeError(
            f"Chunk {idx}: output ID={sid}, expected {idx}"
        )

    if sid in truth_by_id:
        raise RuntimeError(
            f"Duplicate OpenMC output ID {sid}"
        )

    required = (
        "keff",
        "peaking_assembly",
        "peaking_pin",
        "pattern",
    )

    for key in required:
        if key not in row:
            raise RuntimeError(
                f"OpenMC row {sid}: missing {key}"
            )

    if not finite(
        row["keff"],
        row["peaking_assembly"],
        row["peaking_pin"],
    ):
        raise RuntimeError(
            f"OpenMC row {sid}: non-finite physics result"
        )

    if int(row.get("batches", -1)) != 500:
        raise RuntimeError(
            f"OpenMC row {sid}: batches != 500"
        )

    if int(row.get("inactive", -1)) != 50:
        raise RuntimeError(
            f"OpenMC row {sid}: inactive != 50"
        )

    if int(row.get("particles", -1)) != 500000:
        raise RuntimeError(
            f"OpenMC row {sid}: particles != 500000"
        )

    if int(row.get("openmc_threads", -1)) != 48:
        raise RuntimeError(
            f"OpenMC row {sid}: threads != 48"
        )

    truth_by_id[sid] = row


if sorted(truth_by_id) != list(range(360)):
    raise RuntimeError(
        "OpenMC truth IDs are not exactly 0..359"
    )


# ======================================================================
# 3. MERGE WINNER RECORD + INPUT + OPENMC TRUTH
# ======================================================================

merged_rows = []

for idx in range(360):

    w = winner_rows[idx]
    inp = input_rows[idx]
    truth = truth_by_id[idx]

    if int(w["task_id"]) != idx:
        raise RuntimeError(
            f"Winner ordering mismatch at {idx}"
        )

    if int(w["openmc_case_index"]) != idx:
        raise RuntimeError(
            f"Winner OpenMC index mismatch at {idx}"
        )

    if int(inp["id"]) != idx:
        raise RuntimeError(
            f"OpenMC input ID mismatch at {idx}"
        )

    wp = [int(x) for x in w["pattern"]]
    ip = [int(x) for x in inp["pattern"]]
    tp = [int(x) for x in truth["pattern"]]

    if wp != ip:
        raise RuntimeError(
            f"Winner/OpenMC-input pattern mismatch at {idx}"
        )

    if ip != tp:
        raise RuntimeError(
            f"OpenMC-input/truth pattern mismatch at {idx}"
        )

    if len(tp) != 193:
        raise RuntimeError(
            f"OpenMC pattern {idx}: length != 193"
        )

    # Check metadata propagation where available.
    if truth.get("solver") is not None:
        if truth["solver"] != inp["solver"]:
            raise RuntimeError(
                f"Solver mismatch at OpenMC case {idx}"
            )

    if truth.get("strategy") is not None:
        if truth["strategy"] != inp["strategy"]:
            raise RuntimeError(
                f"Initializer mismatch at OpenMC case {idx}"
            )

    if truth.get("replicate") is not None:
        if int(truth["replicate"]) != int(inp["replicate"]):
            raise RuntimeError(
                f"Replicate mismatch at OpenMC case {idx}"
            )

    if truth.get("ga_seed") is not None:
        if int(truth["ga_seed"]) != int(inp["ga_seed"]):
            raise RuntimeError(
                f"GA seed mismatch at OpenMC case {idx}"
            )

    k = float(truth["keff"])
    fa = float(truth["peaking_assembly"])
    fq = float(truth["peaking_pin"])

    actual_J = (
        0.5 * (fa / PHYSICS["Fa_upper"])
        + 0.5 * (fq / PHYSICS["Fq_r_upper"])
    )

    pred_k = float(w["ensemble_mean_keff"])
    pred_fa = float(w["ensemble_mean_Fa"])
    pred_fq = float(w["ensemble_mean_Fq_r"])

    pred_J = (
        0.5 * (pred_fa / PHYSICS["Fa_upper"])
        + 0.5 * (pred_fq / PHYSICS["Fq_r_upper"])
    )

    b_kl = float(w["calibrated_keff_lower"])
    b_ku = float(w["calibrated_keff_upper"])
    b_fa = float(w["calibrated_Fa_upper"])
    b_fq = float(w["calibrated_Fq_r_upper"])

    # Recompute q5 certification independently from stored flag.
    q5_safe_recalc = (
        b_kl >= PHYSICS["keff_lower"]
        and b_ku <= PHYSICS["keff_upper"]
        and b_fa <= PHYSICS["Fa_upper"]
        and b_fq <= PHYSICS["Fq_r_upper"]
    )

    if q5_safe_recalc != bool(w["calibrated_safe"]):
        raise RuntimeError(
            f"Stored q5-safe flag disagrees with bounds at case {idx}"
        )

    actual_safe = (
        k >= PHYSICS["keff_lower"]
        and k <= PHYSICS["keff_upper"]
        and fa <= PHYSICS["Fa_upper"]
        and fq <= PHYSICS["Fq_r_upper"]
    )

    cov_kl = k >= b_kl
    cov_ku = k <= b_ku
    cov_fa = fa <= b_fa
    cov_fq = fq <= b_fq

    cov_joint = (
        cov_kl
        and cov_ku
        and cov_fa
        and cov_fq
    )

    false_safe = (
        q5_safe_recalc
        and not actual_safe
    )

    false_unsafe = (
        (not q5_safe_recalc)
        and actual_safe
    )

    merged_rows.append({
        "id":
            idx,

        "task_id":
            idx,

        "winner_id":
            w["winner_id"],

        "replicate":
            int(w["replicate"]),

        "solver":
            w["solver"],

        "initializer":
            w["initializer"],

        "arm":
            w["arm"],

        "paired_ga_seed":
            int(w["paired_ga_seed"]),

        "pattern":
            wp,

        "prediction": {
            "keff":
                pred_k,

            "Fa":
                pred_fa,

            "Fq_r":
                pred_fq,

            "J":
                pred_J,
        },

        "ensemble_std": {
            "keff":
                float(w["ensemble_std_keff"]),

            "Fa":
                float(w["ensemble_std_Fa"]),

            "Fq_r":
                float(w["ensemble_std_Fq_r"]),
        },

        "frozen_q5_bounds": {
            "keff_lower":
                b_kl,

            "keff_upper":
                b_ku,

            "Fa_upper":
                b_fa,

            "Fq_r_upper":
                b_fq,
        },

        "q5_certified_safe":
            q5_safe_recalc,

        "openmc": {
            "keff":
                k,

            "Fa":
                fa,

            "Fq_r":
                fq,

            "J":
                actual_J,

            "batches":
                int(truth["batches"]),

            "inactive":
                int(truth["inactive"]),

            "particles":
                int(truth["particles"]),

            "threads":
                int(truth["openmc_threads"]),
        },

        "actual_constraints_pass":
            actual_safe,

        "coverage": {
            "keff_lower":
                cov_kl,

            "keff_upper":
                cov_ku,

            "Fa_upper":
                cov_fa,

            "Fq_r_upper":
                cov_fq,

            "joint_all_four":
                cov_joint,
        },

        "false_safe":
            false_safe,

        "false_unsafe":
            false_unsafe,

        "source_openmc_chunk":
            rel(
                OPENMC
                / "chunks"
                / f"chunk_{idx:03d}.jsonl"
            ),

        "source_openmc_chunk_sha256":
            chunk_hashes[str(idx)],
    })


# ======================================================================
# 4. DUPLICATE-PRESERVATION CHECK
# ======================================================================

exact_groups = defaultdict(list)

for r in merged_rows:
    exact_groups[
        exact_key(r["pattern"])
    ].append(r["id"])

exact_unique = len(exact_groups)

duplicate_groups = [
    ids
    for ids in exact_groups.values()
    if len(ids) > 1
]

duplicate_copies = 360 - exact_unique

if exact_unique != 353:
    raise RuntimeError(
        f"Expected 353 exact-unique patterns, got {exact_unique}"
    )

if duplicate_copies != 7:
    raise RuntimeError(
        f"Expected seven natural duplicate copies, "
        f"got {duplicate_copies}"
    )


# ======================================================================
# 5. GLOBAL PROSPECTIVE q5 EVALUATION
# ======================================================================

coverage_keys = (
    "keff_lower",
    "keff_upper",
    "Fa_upper",
    "Fq_r_upper",
    "joint_all_four",
)

coverage_counts = {
    key: sum(
        bool(r["coverage"][key])
        for r in merged_rows
    )
    for key in coverage_keys
}

q5_safe_count = sum(
    bool(r["q5_certified_safe"])
    for r in merged_rows
)

actual_safe_count = sum(
    bool(r["actual_constraints_pass"])
    for r in merged_rows
)

false_safe_count = sum(
    bool(r["false_safe"])
    for r in merged_rows
)

false_unsafe_count = sum(
    bool(r["false_unsafe"])
    for r in merged_rows
)


# q5 search was intentionally restricted to certified-safe winners.
if q5_safe_count != 360:
    raise RuntimeError(
        f"Expected all 360 frozen winners to be q5-certified; "
        f"got {q5_safe_count}"
    )


# ======================================================================
# 6. BALANCED 12-ARM BREAKDOWN
# ======================================================================

arm_rows = defaultdict(list)

for r in merged_rows:
    arm_rows[r["arm"]].append(r)

expected_arms = [
    f"{solver}_{initializer}"
    for solver in ("v2", "v2m", "v3", "v3m")
    for initializer in ("random", "sa", "sqa")
]

if sorted(arm_rows) != sorted(expected_arms):
    raise RuntimeError(
        "Unexpected prospective arm set"
    )

arm_evaluation = {}

for arm in expected_arms:

    rr = arm_rows[arm]

    if len(rr) != 30:
        raise RuntimeError(
            f"Arm {arm}: expected 30 units, got {len(rr)}"
        )

    arm_evaluation[arm] = {
        "n":
            30,

        "actual_constraints_pass":
            sum(r["actual_constraints_pass"] for r in rr),

        "q5_certified_safe":
            sum(r["q5_certified_safe"] for r in rr),

        "false_safe":
            sum(r["false_safe"] for r in rr),

        "false_unsafe":
            sum(r["false_unsafe"] for r in rr),

        "coverage": {
            key: {
                "covered":
                    sum(r["coverage"][key] for r in rr),

                "rate":
                    sum(r["coverage"][key] for r in rr) / 30.0,
            }
            for key in coverage_keys
        },

        "mean_openmc": {
            "keff":
                sum(r["openmc"]["keff"] for r in rr) / 30.0,

            "Fa":
                sum(r["openmc"]["Fa"] for r in rr) / 30.0,

            "Fq_r":
                sum(r["openmc"]["Fq_r"] for r in rr) / 30.0,

            "J":
                sum(r["openmc"]["J"] for r in rr) / 30.0,
        },

        "mean_surrogate_objective_J":
            sum(
                r["prediction"]["J"]
                for r in rr
            ) / 30.0,
    }


# ======================================================================
# 7. PAIRED-REPLICATE CHECK
# ======================================================================

replicate_evaluation = {}

for rep in range(30):

    rr = [
        r
        for r in merged_rows
        if r["replicate"] == rep
    ]

    if len(rr) != 12:
        raise RuntimeError(
            f"Replicate {rep}: expected 12 arms, got {len(rr)}"
        )

    seeds = {
        r["paired_ga_seed"]
        for r in rr
    }

    if len(seeds) != 1:
        raise RuntimeError(
            f"Replicate {rep}: paired-seed property broken"
        )

    replicate_evaluation[str(rep)] = {
        "n":
            12,

        "paired_ga_seed":
            next(iter(seeds)),

        "actual_constraints_pass":
            sum(r["actual_constraints_pass"] for r in rr),

        "false_safe":
            sum(r["false_safe"] for r in rr),

        "joint_coverage":
            sum(
                r["coverage"]["joint_all_four"]
                for r in rr
            ),
    }


# ======================================================================
# 8. SCIENTIFIC PASS/FAIL
#
# This DOES NOT refit q5.
#
# Pass criterion mirrors the earlier prospective certificate rule:
#   - joint four-bound coverage >= 95%
#   - zero false-safe selections
# ======================================================================

joint_rate = (
    coverage_counts["joint_all_four"]
    / 360.0
)

prospective_pass = (
    joint_rate >= JOINT_TARGET
    and false_safe_count == 0
)

status = (
    "PASS_FROZEN_Q5_PROSPECTIVE_VALIDATION"
    if prospective_pass
    else "FAIL_FROZEN_Q5_PROSPECTIVE_VALIDATION"
)


# ======================================================================
# 9. FREEZE MERGED TRUTH
# ======================================================================

merged_artifact = {
    "stage":
        "CYCLE5_Q5_PROSPECTIVE360_OPENMC_MERGE",

    "status":
        "FROZEN_COMPLETE_360_OF_360",

    "openmc_job_id":
        20983367,

    "experimental_units":
        360,

    "exact_unique_patterns":
        exact_unique,

    "natural_duplicate_copies":
        duplicate_copies,

    "natural_duplicate_task_groups":
        duplicate_groups,

    "duplicates_deleted":
        False,

    "duplicates_replaced":
        False,

    "source_winner_manifest":
        rel(WINNERS),

    "source_winner_manifest_sha256":
        sha(WINNERS),

    "source_openmc_input":
        rel(OPENMC_INPUT),

    "source_openmc_input_sha256":
        sha(OPENMC_INPUT),

    "source_openmc_execution":
        rel(OPENMC_EXECUTION),

    "source_openmc_execution_sha256":
        sha(OPENMC_EXECUTION),

    "openmc_chunk_sha256":
        chunk_hashes,

    "openmc_metrics_sha256":
        metrics_hashes,

    "q5_refit":
        False,

    "residual5_refit":
        False,

    "retrospective_reselection":
        False,

    "rows":
        merged_rows,
}

frozen_write(
    MERGED,
    merged_artifact,
)


# ======================================================================
# 10. FREEZE PROSPECTIVE EVALUATION
# ======================================================================

evaluation = {
    "stage":
        "CYCLE5_Q5_PROSPECTIVE360_EVALUATION",

    "status":
        status,

    "prospective_validation_pass":
        prospective_pass,

    "pass_criterion": {
        "joint_all_four_coverage_at_least":
            JOINT_TARGET,

        "false_safe_required":
            0,
    },

    "statistical_unit":
        "one frozen final winner per independent GA run",

    "balanced_design": {
        "replicates":
            30,

        "arms":
            12,

        "units_per_arm":
            30,

        "total_units":
            360,

        "paired_seed_within_replicate":
            True,
    },

    "q5_candidate":
        Q5_EXPECTED,

    "q5_source":
        rel(Q5_CANDIDATE),

    "q5_source_sha256":
        sha(Q5_CANDIDATE),

    "q5_refitted_on_prospective_truth":
        False,

    "residual5_refit":
        False,

    "winner_reselection_after_openmc":
        False,

    "duplicates_deleted":
        False,

    "duplicates_replaced":
        False,

    "experimental_units":
        360,

    "exact_unique_patterns":
        exact_unique,

    "natural_duplicate_copies":
        duplicate_copies,

    "actual_constraints": {
        "passed":
            actual_safe_count,

        "failed":
            360 - actual_safe_count,

        "rate":
            actual_safe_count / 360.0,
    },

    "q5_certification": {
        "certified_safe":
            q5_safe_count,

        "certified_unsafe":
            360 - q5_safe_count,
    },

    "coverage": {
        key: {
            "covered":
                coverage_counts[key],

            "missed":
                360 - coverage_counts[key],

            "rate":
                coverage_counts[key] / 360.0,
        }
        for key in coverage_keys
    },

    "classification_against_actual_constraints": {
        "false_safe":
            false_safe_count,

        "false_unsafe":
            false_unsafe_count,

        "true_safe":
            sum(
                r["q5_certified_safe"]
                and r["actual_constraints_pass"]
                for r in merged_rows
            ),

        "true_unsafe":
            sum(
                (not r["q5_certified_safe"])
                and (not r["actual_constraints_pass"])
                for r in merged_rows
            ),
    },

    "per_arm":
        arm_evaluation,

    "per_replicate":
        replicate_evaluation,

    "source_merged_openmc":
        rel(MERGED),

    "source_merged_openmc_sha256":
        sha(MERGED),
}

frozen_write(
    EVALUATION,
    evaluation,
)


# ======================================================================
# 11. PRINT RESULT
# ======================================================================

print("=" * 104)
print("CYCLE 5 STEP 12G — FROZEN q5 PROSPECTIVE360 EVALUATION")
print("=" * 104)

print()
print("OPENMC MERGE")
print("  OpenMC cases                    = 360/360")
print("  exact unique patterns           =", exact_unique)
print("  natural duplicate copies        =", duplicate_copies)
print("  duplicates deleted              = NO")
print("  duplicates replaced             = NO")

print()
print("ACTUAL PHYSICAL CONSTRAINTS")
print(
    f"  pass                            = "
    f"{actual_safe_count}/360 "
    f"({pct(actual_safe_count):.3f}%)"
)
print(
    f"  fail                            = "
    f"{360-actual_safe_count}/360"
)

print()
print("FROZEN q5 FOUR-BOUND COVERAGE")

labels = {
    "keff_lower": "keff lower",
    "keff_upper": "keff upper",
    "Fa_upper": "Fa upper",
    "Fq_r_upper": "Fq_r upper",
    "joint_all_four": "JOINT",
}

for key in coverage_keys:
    n = coverage_counts[key]
    print(
        f"  {labels[key]:15s}                 = "
        f"{n}/360 ({pct(n):.3f}%)"
    )

print()
print("SAFETY CLASSIFICATION")
print("  q5-certified safe               =", q5_safe_count)
print("  false-safe                      =", false_safe_count)
print("  false-unsafe                    =", false_unsafe_count)

print()
print("BALANCED ARM RESULTS")
for arm in expected_arms:
    a = arm_evaluation[arm]
    j = a["coverage"]["joint_all_four"]["covered"]
    fs = a["false_safe"]
    aps = a["actual_constraints_pass"]

    print(
        f"  {arm:12s} "
        f"joint={j:2d}/30 "
        f"actual_safe={aps:2d}/30 "
        f"false_safe={fs}"
    )

print()
print("PROSPECTIVE DECISION")
print(
    f"  required joint coverage         >= "
    f"{100*JOINT_TARGET:.1f}%"
)
print("  required false-safe             = 0")
print("  observed joint coverage         =", f"{100*joint_rate:.3f}%")
print("  observed false-safe             =", false_safe_count)
print("  STATUS                          =", status)

print()
print("FROZEN ARTIFACTS")
print("  merged OpenMC                   =", MERGED)
print("  merged OpenMC SHA256            =", sha(MERGED))
print()
print("  prospective evaluation          =", EVALUATION)
print("  prospective evaluation SHA256   =", sha(EVALUATION))

print()
print("Q5 REFIT                          = NO")
print("RESIDUAL5 REFIT                   = NO")
print("RETROSPECTIVE RESELECTION         = NO")
print("PROSPECTIVE TRUTH USED IN SEARCH  = NO")

print()
print("CYCLE5_STEP12G_Q5_PROSPECTIVE360_EVALUATION_FROZEN")
