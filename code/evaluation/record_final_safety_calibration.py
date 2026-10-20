#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import sys

PROJECT = Path(__import__("os").environ["PWR_PROJECT"])
sys.path.insert(0, str(PROJECT / "scripts"))

from geometry import canonical_d4_key


# ======================================================================
# PATHS
# ======================================================================

CAL = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration"
)

PROS = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360"
)

CAL_PROTOCOL = CAL / "protocol"
PROS_PROTOCOL = PROS / "protocol"

CAL_MERGED = (
    CAL_PROTOCOL
    / "cycle5_q5_calibration_openmc_merged_960_v1.json"
)

CAL_SCORES = (
    CAL_PROTOCOL
    / "cycle5_q5_calibration_scores_960_v1.json"
)

Q5_CANDIDATE = (
    CAL_PROTOCOL
    / "cycle5_q5_candidate_balanced960_v1.json"
)

PROS_WINNERS = (
    PROS_PROTOCOL
    / "cycle5_q5_prospective360_winner_manifest_360_v1.json"
)

PROS_MERGED = (
    PROS_PROTOCOL
    / "cycle5_q5_prospective360_openmc_merged_360_v1.json"
)

PROS_EVAL = (
    PROS_PROTOCOL
    / "cycle5_q5_prospective360_evaluation_v1.json"
)

PRE_PROS_FIREWALL = (
    PROS_PROTOCOL
    / "cycle5_q5_prospective360_known_openmc_plus_calibration_d4_exclusion_v1.json"
)

PRE_PROS_GENERATOR_FIREWALL = (
    PROS_PROTOCOL
    / "cycle5_q5_prospective360_generator_exact_unique_exclusion_v1.json"
)

RESIDUAL_DIR = (
    PROJECT
    / "run/optimizer_tail_v2/models/"
      "learned_residual_safety_round5_q5_prospective_search"
)

RESIDUAL_METADATA = (
    RESIDUAL_DIR
    / "residual_safety_metadata.json"
)

FINAL_CERTIFICATE = (
    RESIDUAL_DIR
    / "deployment_certificate_final_q5.json"
)

FINAL_SCIENTIFIC_FIREWALL = (
    PROS_PROTOCOL
    / "cycle5_q5_final_known_openmc_plus_prospective360_d4_exclusion_v1.json"
)

FINAL_GENERATOR_FIREWALL = (
    PROS_PROTOCOL
    / "cycle5_q5_final_generator_exact_unique_exclusion_v1.json"
)

FIREWALL_AMENDMENT = (
    PROS_PROTOCOL
    / "cycle5_q5_prospective360_generator_firewall_amendment_v1.json"
)

FINAL_FREEZE_RECORD = (
    PROS_PROTOCOL
    / "cycle5_q5_final_deployment_freeze_v1.json"
)


# ======================================================================
# FROZEN HASHES
# ======================================================================

EXPECTED = {
    "cal_merged":
        "ec6923101e0d15a616978bf627ee9bac60db7a85185d4a1a0e6cde34a3157184",

    "cal_scores":
        "587074189576c318ceaa7b61ecadcaeb4fee939e85fd1a626a49f61270ab53a6",

    "q5_candidate":
        "21c1f0ac8ee5b7546b4264e81d54dc6f0c68a74605bbb5e527c207e6f2fb6760",

    "pros_winners":
        "73e4049c0c1c89f0306537a07ece92603ae6b25f6afc3af46f4e7c87c6321ac0",

    "pros_merged":
        "575c90a18e1ec20c00099c91af3e066d5d6f6742ff4f2c87c0040aa4084e640c",

    "pros_eval":
        "7a0edf92848ff57df26bbf027f29f679f830a1adad54d8a1a26c0d1986fa894c",

    "pre_pros_firewall":
        "2bbac884a16b5a90d1d4d3b91684555fff6a598a8267f0732fb40a0751556de4",

    "pre_pros_generator_firewall":
        "092544e1d417eb588d92ee2dd51b10b82c2396861b327d89b3a2074063047f73",

    "residual_metadata":
        "5084368629a8f1fed3bdfa052c755f21aed767875d5d7d6f8601c115a867c4ca",
}


Q5 = {
    "keff_lower": 1.2555488951630267,
    "keff_upper": 1.167078988515213,
    "Fa_upper": -0.00824025190735769,
    "Fq_r_upper": -0.15313887213725857,
}

PHYSICS_LIMITS = {
    "keff_lower": 1.0,
    "keff_upper": 1.2,
    "Fa_upper": 2.0,
    "Fq_r_upper": 2.5,
}


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


# ======================================================================
# 1. VERIFY EVERY FROZEN INPUT
# ======================================================================

dependencies = [
    ("calibration OpenMC960", CAL_MERGED, EXPECTED["cal_merged"]),
    ("calibration scores960", CAL_SCORES, EXPECTED["cal_scores"]),
    ("candidate q5", Q5_CANDIDATE, EXPECTED["q5_candidate"]),
    ("prospective winners", PROS_WINNERS, EXPECTED["pros_winners"]),
    ("prospective OpenMC360", PROS_MERGED, EXPECTED["pros_merged"]),
    ("prospective evaluation", PROS_EVAL, EXPECTED["pros_eval"]),
    (
        "pre-prospective scientific firewall",
        PRE_PROS_FIREWALL,
        EXPECTED["pre_pros_firewall"],
    ),
    (
        "pre-prospective generator firewall",
        PRE_PROS_GENERATOR_FIREWALL,
        EXPECTED["pre_pros_generator_firewall"],
    ),
    (
        "Residual5 q5 metadata",
        RESIDUAL_METADATA,
        EXPECTED["residual_metadata"],
    ),
]

for name, path, expected in dependencies:

    if not path.is_file():
        raise RuntimeError(
            f"Missing frozen dependency {name}:\n{path}"
        )

    actual = sha(path)

    if actual != expected:
        raise RuntimeError(
            f"{name} SHA changed:\n"
            f"expected={expected}\n"
            f"actual  ={actual}"
        )


# ======================================================================
# 2. VERIFY q5 WAS NOT ALTERED
# ======================================================================

metadata = json.loads(RESIDUAL_METADATA.read_text())

metadata_q = metadata.get("joint95_conformal_q")

if not isinstance(metadata_q, dict):
    raise RuntimeError(
        "Residual metadata missing joint95_conformal_q"
    )

for key, expected in Q5.items():

    actual = float(metadata_q[key])

    if abs(actual - expected) > 1e-15:
        raise RuntimeError(
            f"Residual package q5 changed for {key}: "
            f"{actual} != {expected}"
        )


# ======================================================================
# 3. VERIFY THE FROZEN PROSPECTIVE RESULT
# ======================================================================

evaluation = json.loads(PROS_EVAL.read_text())

if (
    evaluation["status"]
    != "PASS_FROZEN_Q5_PROSPECTIVE_VALIDATION"
):
    raise RuntimeError(
        "Prospective q5 evaluation did not pass"
    )

if not bool(evaluation["prospective_validation_pass"]):
    raise RuntimeError(
        "Prospective-validation pass flag is false"
    )

if int(evaluation["experimental_units"]) != 360:
    raise RuntimeError(
        "Prospective unit count changed"
    )

if int(evaluation["exact_unique_patterns"]) != 353:
    raise RuntimeError(
        "Prospective exact-unique count changed"
    )

if int(evaluation["natural_duplicate_copies"]) != 7:
    raise RuntimeError(
        "Prospective duplicate count changed"
    )

coverage = evaluation["coverage"]

if int(coverage["keff_lower"]["covered"]) != 353:
    raise RuntimeError("keff-lower coverage changed")

if int(coverage["keff_upper"]["covered"]) != 359:
    raise RuntimeError("keff-upper coverage changed")

if int(coverage["Fa_upper"]["covered"]) != 359:
    raise RuntimeError("Fa coverage changed")

if int(coverage["Fq_r_upper"]["covered"]) != 355:
    raise RuntimeError("Fq_r coverage changed")

if int(coverage["joint_all_four"]["covered"]) != 347:
    raise RuntimeError("Joint coverage changed")

joint_rate = float(
    coverage["joint_all_four"]["rate"]
)

if joint_rate < 0.95:
    raise RuntimeError(
        f"Prospective joint coverage below frozen gate: {joint_rate}"
    )

cls = evaluation[
    "classification_against_actual_constraints"
]

if int(cls["false_safe"]) != 0:
    raise RuntimeError(
        "Prospective validation has false-safe cases"
    )

actual = evaluation["actual_constraints"]

if int(actual["passed"]) != 360:
    raise RuntimeError(
        "Not all prospective OpenMC designs passed physical constraints"
    )

if bool(evaluation["q5_refitted_on_prospective_truth"]):
    raise RuntimeError(
        "Prospective evaluation reports q5 refit"
    )

if bool(evaluation["residual5_refit"]):
    raise RuntimeError(
        "Prospective evaluation reports Residual5 refit"
    )

if bool(evaluation["winner_reselection_after_openmc"]):
    raise RuntimeError(
        "Prospective evaluation reports winner reselection"
    )


# ======================================================================
# 4. LOAD PRE-PROSPECTIVE SCIENTIFIC FIREWALL
#
# It intentionally contains the 13 natural duplicate calibration copies.
# ======================================================================

old_rows = json.loads(
    PRE_PROS_FIREWALL.read_text()
)

if not isinstance(old_rows, list):
    raise RuntimeError(
        "Pre-prospective scientific firewall must be a list"
    )

if len(old_rows) != 32346:
    raise RuntimeError(
        f"Expected 32346 scientific-firewall rows, "
        f"got {len(old_rows)}"
    )

old_exact = {
    exact_key(row["pattern"])
    for row in old_rows
}

old_d4 = {
    d4_key(row["pattern"])
    for row in old_rows
}

if len(old_exact) != 32333:
    raise RuntimeError(
        f"Expected 32333 pre-prospective exact-unique, "
        f"got {len(old_exact)}"
    )

if len(old_d4) != 32333:
    raise RuntimeError(
        f"Expected 32333 pre-prospective D4-unique, "
        f"got {len(old_d4)}"
    )


# ======================================================================
# 5. LOAD ALL 360 PROSPECTIVE OPENMC EXPERIMENTAL UNITS
#
# Preserve all seven natural duplicate copies.
# ======================================================================

pros_obj = json.loads(
    PROS_MERGED.read_text()
)

pros_rows = pros_obj["rows"]

if len(pros_rows) != 360:
    raise RuntimeError(
        f"Expected 360 prospective OpenMC rows, got {len(pros_rows)}"
    )

pros_exact = {
    exact_key(row["pattern"])
    for row in pros_rows
}

pros_d4 = {
    d4_key(row["pattern"])
    for row in pros_rows
}

if len(pros_exact) != 353:
    raise RuntimeError(
        f"Expected 353 prospective exact-unique patterns, "
        f"got {len(pros_exact)}"
    )

if len(pros_d4) != 353:
    raise RuntimeError(
        f"Expected 353 prospective D4-unique patterns, "
        f"got {len(pros_d4)}"
    )

if old_exact & pros_exact:
    raise RuntimeError(
        "Prospective truth unexpectedly overlaps old firewall exactly"
    )

if old_d4 & pros_d4:
    raise RuntimeError(
        "Prospective truth unexpectedly overlaps old firewall by D4"
    )


# ======================================================================
# 6. BUILD FINAL SCIENTIFIC FIREWALL
#
# Original 32346 rows remain byte-logically represented as the same
# JSON objects. Then append all 360 prospective experimental units.
#
# 32346 + 360 = 32706 raw rows
# unique D4 = 32333 + 353 = 32686
#
# NO deduplication in the scientific record.
# ======================================================================

final_scientific_rows = list(old_rows)

for row in pros_rows:

    final_scientific_rows.append({
        "source":
            "cycle5_q5_prospective360_final_validation",

        "source_case_id":
            int(row["id"]),

        "winner_id":
            row["winner_id"],

        "replicate":
            int(row["replicate"]),

        "solver":
            row["solver"],

        "initializer":
            row["initializer"],

        "arm":
            row["arm"],

        "pattern":
            [int(x) for x in row["pattern"]],

        "openmc": {
            "keff":
                float(row["openmc"]["keff"]),

            "Fa":
                float(row["openmc"]["Fa"]),

            "Fq_r":
                float(row["openmc"]["Fq_r"]),
        },

        "actual_constraints_pass":
            bool(row["actual_constraints_pass"]),
    })


if len(final_scientific_rows) != 32706:
    raise RuntimeError(
        f"Expected final scientific-firewall raw rows=32706, "
        f"got {len(final_scientific_rows)}"
    )

final_exact = {
    exact_key(row["pattern"])
    for row in final_scientific_rows
}

final_d4 = {
    d4_key(row["pattern"])
    for row in final_scientific_rows
}

if len(final_exact) != 32686:
    raise RuntimeError(
        f"Expected final exact-unique=32686, got {len(final_exact)}"
    )

if len(final_d4) != 32686:
    raise RuntimeError(
        f"Expected final D4-unique=32686, got {len(final_d4)}"
    )

final_duplicate_copies = (
    len(final_scientific_rows)
    - len(final_exact)
)

if final_duplicate_copies != 20:
    raise RuntimeError(
        f"Expected 20 total natural duplicate copies "
        f"(13 calibration + 7 prospective), "
        f"got {final_duplicate_copies}"
    )

frozen_write(
    FINAL_SCIENTIFIC_FIREWALL,
    final_scientific_rows,
)


# ======================================================================
# 7. GENERATOR-COMPATIBLE FINAL PROJECTION
#
# Implementation-only projection:
# retain earliest occurrence of each exact pattern.
#
# Because exact-unique == canonical-D4-unique here, this projection
# preserves the complete scientific D4 exclusion set exactly.
# ======================================================================

seen_exact = set()
generator_rows = []

for row in final_scientific_rows:

    key = exact_key(row["pattern"])

    if key in seen_exact:
        continue

    seen_exact.add(key)
    generator_rows.append(row)


if len(generator_rows) != 32686:
    raise RuntimeError(
        f"Final generator projection rows={len(generator_rows)}, "
        f"expected 32686"
    )

generator_exact = {
    exact_key(row["pattern"])
    for row in generator_rows
}

generator_d4 = {
    d4_key(row["pattern"])
    for row in generator_rows
}

if len(generator_exact) != 32686:
    raise RuntimeError(
        "Final generator projection is not exact-unique"
    )

if len(generator_d4) != 32686:
    raise RuntimeError(
        "Final generator projection is not D4-unique"
    )

if generator_d4 != final_d4:
    raise RuntimeError(
        "Final generator projection changed the scientific D4 set"
    )

frozen_write(
    FINAL_GENERATOR_FIREWALL,
    generator_rows,
)


# ======================================================================
# 8. CLOSE THE EARLIER GENERATOR-FIREWALL PROVENANCE AMENDMENT
# ======================================================================

amendment = {
    "stage":
        "CYCLE5_Q5_PROSPECTIVE360_GENERATOR_FIREWALL_AMENDMENT",

    "status":
        "FROZEN_PROVENANCE_ONLY",

    "purpose":
        "Document the implementation-only exact-unique projection "
        "required by the SA/SQA generator while preserving the "
        "scientific calibration record unchanged.",

    "source_scientific_firewall":
        rel(PRE_PROS_FIREWALL),

    "source_scientific_firewall_sha256":
        EXPECTED["pre_pros_firewall"],

    "source_raw_rows":
        32346,

    "source_exact_unique":
        32333,

    "source_d4_unique":
        32333,

    "derived_generator_projection":
        rel(PRE_PROS_GENERATOR_FIREWALL),

    "derived_generator_projection_sha256":
        EXPECTED["pre_pros_generator_firewall"],

    "derived_rows":
        32333,

    "derived_exact_unique":
        32333,

    "derived_d4_unique":
        32333,

    "removed_exact_duplicate_copies":
        13,

    "scientific_d4_set_identical":
        True,

    "calibration_merged960_unchanged":
        True,

    "q5_unchanged":
        True,

    "residual5_unchanged":
        True,

    "prospective_truth_used_to_create_projection":
        False,

    "decision_temporally_preceded_successful_prospective_warmstart_generation":
        True,

    "scientific_interpretation":
        "The 13 removed copies were removed only from the generator "
        "compatibility projection. They remain part of calibration960 "
        "and of the scientific firewall provenance.",
}

frozen_write(
    FIREWALL_AMENDMENT,
    amendment,
)


# ======================================================================
# 9. FREEZE FINAL q5 DEPLOYMENT CERTIFICATE
# ======================================================================

certificate = {
    "version":
        "cycle5_q5_final_deployment_certificate_v1",

    "status":
        "FINAL_Q5_DEPLOYMENT_CERTIFICATE",

    "decision":
        "ACCEPT_FROZEN_Q5_FOR_DEPLOYMENT",

    "q5":
        Q5,

    "physics_limits":
        PHYSICS_LIMITS,

    "statistical_design": {
        "calibration": {
            "experimental_units":
                960,

            "arms":
                12,

            "units_per_arm":
                80,

            "exact_unique":
                947,

            "natural_duplicate_copies":
                13,

            "duplicates_deleted":
                False,

            "duplicates_replaced":
                False,

            "finite_sample_rule":
                "12th-largest / rank 949 of 960 per bound",

            "family_alpha":
                0.05,

            "alpha_per_bound":
                0.0125,

            "marginal_coverage_each":
                {
                    "covered": 949,
                    "total": 960,
                    "rate": 949 / 960,
                },

            "joint_coverage":
                {
                    "covered": 923,
                    "total": 960,
                    "rate": 923 / 960,
                },

            "actual_constraints_pass":
                {
                    "passed": 960,
                    "total": 960,
                    "rate": 1.0,
                },

            "false_safe":
                0,

            "false_unsafe":
                0,
        },

        "prospective_validation": {
            "experimental_units":
                360,

            "replicates":
                30,

            "arms":
                12,

            "units_per_arm":
                30,

            "paired_seed_design":
                True,

            "exact_unique":
                353,

            "natural_duplicate_copies":
                7,

            "duplicates_deleted":
                False,

            "duplicates_replaced":
                False,

            "actual_constraints_pass":
                {
                    "passed": 360,
                    "total": 360,
                    "rate": 1.0,
                },

            "coverage": {
                "keff_lower": {
                    "covered": 353,
                    "total": 360,
                    "rate": 353 / 360,
                },

                "keff_upper": {
                    "covered": 359,
                    "total": 360,
                    "rate": 359 / 360,
                },

                "Fa_upper": {
                    "covered": 359,
                    "total": 360,
                    "rate": 359 / 360,
                },

                "Fq_r_upper": {
                    "covered": 355,
                    "total": 360,
                    "rate": 355 / 360,
                },

                "joint_all_four": {
                    "covered": 347,
                    "total": 360,
                    "rate": 347 / 360,
                },
            },

            "false_safe":
                0,

            "false_unsafe":
                0,

            "pre_specified_acceptance_rule": {
                "joint_coverage_at_least":
                    0.95,

                "false_safe":
                    0,
            },

            "acceptance_rule_pass":
                True,

            "armwise_results_role":
                "descriptive_secondary_analysis_not_separate_acceptance_gate",
        },
    },

    "scientific_firewalls": {
        "q5_fit_uses_prospective_truth":
            False,

        "q5_refitted_after_prospective_validation":
            False,

        "Residual5_refitted_after_prospective_validation":
            False,

        "winner_reselected_after_OpenMC":
            False,

        "prospective_truth_available_during_GA_search":
            False,

        "prospective_truth_available_during_warmstart_generation":
            False,
    },

    "frozen_sources": {
        "calibration_openmc960": {
            "file":
                rel(CAL_MERGED),

            "sha256":
                sha(CAL_MERGED),
        },

        "calibration_scores960": {
            "file":
                rel(CAL_SCORES),

            "sha256":
                sha(CAL_SCORES),
        },

        "candidate_q5": {
            "file":
                rel(Q5_CANDIDATE),

            "sha256":
                sha(Q5_CANDIDATE),
        },

        "prospective_winner_manifest360": {
            "file":
                rel(PROS_WINNERS),

            "sha256":
                sha(PROS_WINNERS),
        },

        "prospective_openmc360": {
            "file":
                rel(PROS_MERGED),

            "sha256":
                sha(PROS_MERGED),
        },

        "prospective_evaluation": {
            "file":
                rel(PROS_EVAL),

            "sha256":
                sha(PROS_EVAL),
        },

        "deployment_residual_metadata": {
            "file":
                rel(RESIDUAL_METADATA),

            "sha256":
                sha(RESIDUAL_METADATA),
        },
    },

    "post_validation_firewall": {
        "scientific_firewall":
            rel(FINAL_SCIENTIFIC_FIREWALL),

        "scientific_firewall_sha256":
            sha(FINAL_SCIENTIFIC_FIREWALL),

        "raw_rows":
            32706,

        "exact_unique":
            32686,

        "d4_unique":
            32686,

        "natural_duplicate_copies":
            20,

        "generator_exact_unique_projection":
            rel(FINAL_GENERATOR_FIREWALL),

        "generator_exact_unique_projection_sha256":
            sha(FINAL_GENERATOR_FIREWALL),

        "generator_projection_rows":
            32686,

        "generator_projection_d4_identical":
            True,
    },

    "deployment_interpretation":
        "The q5 multipliers were fitted only on the frozen balanced "
        "960-unit calibration experiment and were then evaluated, "
        "without refitting, on a fresh balanced 360-unit prospective "
        "experiment. The prospective experiment achieved 347/360 "
        "(96.389%) joint four-bound coverage with zero false-safe "
        "selections and 360/360 actual OpenMC physical-feasibility "
        "passes. The pre-specified prospective acceptance rule is "
        "therefore satisfied. q5 is frozen for deployment.",
}

frozen_write(
    FINAL_CERTIFICATE,
    certificate,
)


# ======================================================================
# 10. FINAL IMMUTABLE FREEZE RECORD
# ======================================================================

freeze_record = {
    "stage":
        "CYCLE5_Q5_FINAL_DEPLOYMENT_FREEZE",

    "status":
        "COMPLETE",

    "q5":
        Q5,

    "prospective_validation":
        "PASS",

    "prospective_joint_coverage":
        347 / 360,

    "prospective_false_safe":
        0,

    "prospective_actual_safe":
        360,

    "q5_refit":
        False,

    "Residual5_refit":
        False,

    "retrospective_reselection":
        False,

    "deployment_certificate":
        rel(FINAL_CERTIFICATE),

    "deployment_certificate_sha256":
        sha(FINAL_CERTIFICATE),

    "final_scientific_firewall":
        rel(FINAL_SCIENTIFIC_FIREWALL),

    "final_scientific_firewall_sha256":
        sha(FINAL_SCIENTIFIC_FIREWALL),

    "final_generator_firewall":
        rel(FINAL_GENERATOR_FIREWALL),

    "final_generator_firewall_sha256":
        sha(FINAL_GENERATOR_FIREWALL),

    "generator_firewall_amendment":
        rel(FIREWALL_AMENDMENT),

    "generator_firewall_amendment_sha256":
        sha(FIREWALL_AMENDMENT),

    "residual_metadata":
        rel(RESIDUAL_METADATA),

    "residual_metadata_sha256":
        sha(RESIDUAL_METADATA),
}

frozen_write(
    FINAL_FREEZE_RECORD,
    freeze_record,
)


# ======================================================================
# 11. OUTPUT
# ======================================================================

print("=" * 104)
print("CYCLE 5 STEP 12H — FINAL q5 DEPLOYMENT CERTIFICATE FROZEN")
print("=" * 104)

print()
print("FINAL q5")
print("  keff_lower                    =", Q5["keff_lower"])
print("  keff_upper                    =", Q5["keff_upper"])
print("  Fa_upper                      =", Q5["Fa_upper"])
print("  Fq_r_upper                    =", Q5["Fq_r_upper"])

print()
print("CALIBRATION960")
print("  marginal coverage / bound     = 949/960 = 98.854%")
print("  joint coverage                = 923/960 = 96.146%")
print("  actual constraints pass       = 960/960")
print("  false-safe                    = 0")

print()
print("PROSPECTIVE360")
print("  keff lower                    = 353/360 = 98.056%")
print("  keff upper                    = 359/360 = 99.722%")
print("  Fa upper                      = 359/360 = 99.722%")
print("  Fq_r upper                    = 355/360 = 98.611%")
print("  joint                         = 347/360 = 96.389%")
print("  actual constraints pass       = 360/360")
print("  false-safe                    = 0")
print("  acceptance gate               = PASS")

print()
print("POST-VALIDATION FIREWALL")
print("  previous D4                   = 32333")
print("  prospective D4 added          = 353")
print("  final D4                      =", len(final_d4))
print("  final raw rows                =", len(final_scientific_rows))
print("  natural duplicate copies      =", final_duplicate_copies)
print("  scientific duplicates kept    = YES")
print("  generator projection rows     =", len(generator_rows))
print("  generator D4 identical        = YES")

print()
print("FINAL ARTIFACTS")
print("  deployment certificate        =", FINAL_CERTIFICATE)
print("  certificate SHA256            =", sha(FINAL_CERTIFICATE))
print()
print("  final scientific firewall     =", FINAL_SCIENTIFIC_FIREWALL)
print("  scientific firewall SHA256    =", sha(FINAL_SCIENTIFIC_FIREWALL))
print()
print("  final generator firewall      =", FINAL_GENERATOR_FIREWALL)
print("  generator firewall SHA256     =", sha(FINAL_GENERATOR_FIREWALL))
print()
print("  firewall amendment            =", FIREWALL_AMENDMENT)
print("  firewall amendment SHA256     =", sha(FIREWALL_AMENDMENT))
print()
print("  final freeze record           =", FINAL_FREEZE_RECORD)
print("  final freeze SHA256           =", sha(FINAL_FREEZE_RECORD))

print()
print("Q5 REFIT                       = NO")
print("RESIDUAL5 REFIT                = NO")
print("RETROSPECTIVE RESELECTION      = NO")
print("PROSPECTIVE TRUTH IN SEARCH    = NO")

print()
print("FINAL DEPLOYMENT STATUS        = ACCEPTED")
print()
print("CYCLE5_FINAL_Q5_DEPLOYMENT_CERTIFICATE_FROZEN")
