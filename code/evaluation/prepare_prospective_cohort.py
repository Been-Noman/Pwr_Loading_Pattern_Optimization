#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path


# ======================================================================
# IMPORT PROJECT GEOMETRY
# ======================================================================

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[1]

sys.path.insert(0, str(SCRIPTS))

from geometry import canonical_d4_key


# ======================================================================
# CONFIG
# ======================================================================

PROJECT = Path(
    os.environ.get(
        "PWR_PROJECT",
        __import__("os").environ["PWR_PROJECT"],
    )
).resolve()

Q5CAL = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration"
)

PROS = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360"
)

PROTOCOL = PROS / "protocol"

SOURCE_FIREWALL = (
    Q5CAL
    / "protocol/cycle5_q5_known_openmc_d4_exclusion_31386_v1.json"
)

SOURCE_RESIDUAL_DIR = (
    PROJECT
    / "run/optimizer_tail_v2/models/"
      "learned_residual_safety_round5_bridge_cycle4_commonq"
)

SOURCE_RESIDUAL_METADATA = (
    SOURCE_RESIDUAL_DIR / "residual_safety_metadata.json"
)

Q5_OVERLAY_DIR = (
    PROJECT
    / "run/optimizer_tail_v2/models/"
      "learned_residual_safety_round5_q5_prospective_search"
)

OVERLAY_METADATA = (
    Q5_OVERLAY_DIR / "residual_safety_metadata.json"
)


# ======================================================================
# FROZEN STEP-11 VALUES
# ======================================================================

Q5 = {
    "keff_lower": 1.2555488951630267,
    "keff_upper": 1.167078988515213,
    "Fa_upper": -0.00824025190735769,
    "Fq_r_upper": -0.15313887213725857,
}

OPENMC_TRUTH_SHA = (
    "f9c2f63cfe3ad3e75bb934a968057b2459eddc5a0b713251ba6ed7be5a05b792"
)

MERGED960_SHA = (
    "ec6923101e0d15a616978bf627ee9bac60db7a85185d4a1a0e6cde34a3157184"
)

SCORE_SHA = (
    "587074189576c318ceaa7b61ecadcaeb4fee939e85fd1a626a49f61270ab53a6"
)

CANDIDATE_Q5_SHA = (
    "21c1f0ac8ee5b7546b4264e81d54dc6f0c68a74605bbb5e527c207e6f2fb6760"
)


# ======================================================================
# PROSPECTIVE DESIGN
# ======================================================================

N_REPLICATES = 30

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
    f"{solver}_{init}"
    for solver in SOLVERS
    for init in INITIALIZERS
]

assert len(ARMS) == 12

TOTAL_UNITS = N_REPLICATES * len(ARMS)

assert TOTAL_UNITS == 360


# Keep same warm-start density used by prior common-q bridge:
#
#     10 raw shards per replicate
#     40 rows per shard
#
SHARDS_PER_REPLICATE = 10
N_WARMSTART_SHARDS = N_REPLICATES * SHARDS_PER_REPLICATE

assert N_WARMSTART_SHARDS == 300


# Fresh namespace.
#
# Calibration used the 2050... namespace.
# Prospective uses disjoint 2060... namespace.
#
PROSPECTIVE_SEED_BASE = 2_060_000_001
SEED_STRIDE = 1009

PROSPECTIVE_GA_SEEDS = [
    PROSPECTIVE_SEED_BASE + SEED_STRIDE * rep
    for rep in range(N_REPLICATES)
]

# Fresh generator base seed for SA/SQA raw candidates.
WARMSTART_BASE_SEED = 2026082812


# ======================================================================
# HELPERS
# ======================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def pattern_ok(row) -> bool:
    return (
        isinstance(row, dict)
        and isinstance(row.get("pattern"), list)
        and len(row["pattern"]) == 193
    )


def collect_pattern_lists(obj, out):
    """
    Find list-valued JSON containers whose rows are pattern records.
    """

    if isinstance(obj, list):

        if obj and all(
            pattern_ok(x)
            for x in obj
        ):
            out.append(obj)

        for x in obj:
            if isinstance(x, (dict, list)):
                collect_pattern_lists(x, out)

    elif isinstance(obj, dict):

        for x in obj.values():
            if isinstance(x, (dict, list)):
                collect_pattern_lists(x, out)


def get_pattern_list(
    path: Path,
    expected_rows: int | None = None,
):
    obj = json.loads(
        path.read_text()
    )

    candidates = []
    collect_pattern_lists(
        obj,
        candidates,
    )

    if expected_rows is not None:

        exact = [
            rows
            for rows in candidates
            if len(rows) == expected_rows
        ]

        if len(exact) != 1:
            raise RuntimeError(
                f"{path}: expected exactly one "
                f"{expected_rows}-row pattern list; "
                f"found {len(exact)}"
            )

        return exact[0]

    if not candidates:
        raise RuntimeError(
            f"No pattern-row list found in {path}"
        )

    # Firewall should be the largest pattern list.
    return max(
        candidates,
        key=len,
    )


def d4_key(pattern):
    return tuple(
        int(x)
        for x in canonical_d4_key(pattern)
    )


def exact_key(pattern):
    return tuple(
        int(x)
        for x in pattern
    )


def find_matching_q(obj):
    """
    Search candidate-q5 artifact without assuming exact wrapper schema.
    """

    if isinstance(obj, dict):

        required = set(Q5)

        if required.issubset(obj.keys()):

            good = True

            for key, expected in Q5.items():
                try:
                    actual = float(obj[key])
                except Exception:
                    good = False
                    break

                if not math.isclose(
                    actual,
                    expected,
                    rel_tol=1e-13,
                    abs_tol=1e-13,
                ):
                    good = False
                    break

            if good:
                return {
                    key: float(obj[key])
                    for key in Q5
                }

        for value in obj.values():

            found = find_matching_q(value)

            if found is not None:
                return found

    elif isinstance(obj, list):

        for value in obj:

            found = find_matching_q(value)

            if found is not None:
                return found

    return None


def find_frozen_artifacts(root: Path):
    """
    Locate Step-11 artifacts by SHA instead of guessing filenames.

    Large raw GA/warmstart directories are intentionally skipped.
    """

    targets = {
        MERGED960_SHA: "merged960",
        SCORE_SHA: "score_artifact",
        CANDIDATE_Q5_SHA: "candidate_q5",
    }

    found = {}

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        rel = path.relative_to(root)
        parts = set(rel.parts)

        # Skip enormous raw search directories.
        if (
            "runs" in parts
            or "raw" in parts
            or "tmp" in parts
        ):
            continue

        # Do not hash pathological giant files.
        try:
            size = path.stat().st_size
        except OSError:
            continue

        if size > 500_000_000:
            continue

        digest = sha256_file(path)

        if digest in targets:
            name = targets[digest]
            found[name] = path

            print(
                f"FOUND {name:16s} "
                f"{rel}"
            )

    missing = [
        name
        for name in targets.values()
        if name not in found
    ]

    if missing:
        raise RuntimeError(
            "Could not locate frozen Step-11 artifacts by SHA: "
            + ", ".join(missing)
        )

    return found


# ======================================================================
# PRECONDITIONS
# ======================================================================

if not Q5CAL.is_dir():
    raise RuntimeError(
        f"Missing q5 calibration directory: {Q5CAL}"
    )

if not SOURCE_FIREWALL.is_file():
    raise RuntimeError(
        f"Missing original q5 firewall: {SOURCE_FIREWALL}"
    )

if not SOURCE_RESIDUAL_METADATA.is_file():
    raise RuntimeError(
        f"Missing Residual5 metadata: {SOURCE_RESIDUAL_METADATA}"
    )

PROTOCOL.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# 1. RECOVER / VERIFY STEP-11 FROZEN ARTIFACTS
# ======================================================================

artifacts = find_frozen_artifacts(
    Q5CAL
)

merged960_path = artifacts["merged960"]
score_path = artifacts["score_artifact"]
candidate_q5_path = artifacts["candidate_q5"]

assert (
    sha256_file(merged960_path)
    == MERGED960_SHA
)

assert (
    sha256_file(score_path)
    == SCORE_SHA
)

assert (
    sha256_file(candidate_q5_path)
    == CANDIDATE_Q5_SHA
)


# Verify q itself.
candidate_obj = json.loads(
    candidate_q5_path.read_text()
)

artifact_q = find_matching_q(
    candidate_obj
)

if artifact_q is None:
    raise RuntimeError(
        "Candidate q5 artifact has correct SHA, "
        "but expected q5 values were not found."
    )

print()
print("FROZEN q5 VERIFIED")
print(json.dumps(
    artifact_q,
    indent=2,
    sort_keys=True,
))


# ======================================================================
# 2. VERIFY MERGED960 STRUCTURE
# ======================================================================

merged960_rows = get_pattern_list(
    merged960_path,
    expected_rows=960,
)

merged_exact_keys = [
    exact_key(r["pattern"])
    for r in merged960_rows
]

merged_exact_unique = len(
    set(merged_exact_keys)
)

merged_exact_duplicates = (
    len(merged_exact_keys)
    - merged_exact_unique
)

if merged_exact_unique != 947:
    raise RuntimeError(
        f"Expected 947 exact unique calibration patterns, "
        f"found {merged_exact_unique}"
    )

if merged_exact_duplicates != 13:
    raise RuntimeError(
        f"Expected 13 natural duplicate units, "
        f"found {merged_exact_duplicates}"
    )


# ======================================================================
# 3. LOAD PRE-CALIBRATION FIREWALL
# ======================================================================

base_firewall_rows = get_pattern_list(
    SOURCE_FIREWALL
)

base_d4 = {
    d4_key(r["pattern"])
    for r in base_firewall_rows
}

if len(base_d4) != 31_386:
    raise RuntimeError(
        "Expected original firewall to contain "
        f"31,386 canonical D4 patterns; got {len(base_d4)}"
    )


# ======================================================================
# 4. ADD ALL CALIBRATION-960 UNITS TO PROSPECTIVE FIREWALL
# ======================================================================

calibration_d4 = {
    d4_key(r["pattern"])
    for r in merged960_rows
}

preexisting_overlap = (
    base_d4
    & calibration_d4
)

if preexisting_overlap:
    raise RuntimeError(
        "Calibration population overlaps the pre-calibration "
        f"known-OpenMC firewall in {len(preexisting_overlap)} D4 keys."
    )

prospective_firewall_rows = (
    list(base_firewall_rows)
    + list(merged960_rows)
)

prospective_d4 = {
    d4_key(r["pattern"])
    for r in prospective_firewall_rows
}

EXPECTED_PROSPECTIVE_D4 = len(
    prospective_d4
)

FIREWALL_OUT = (
    PROTOCOL
    / "cycle5_q5_prospective360_known_openmc_plus_calibration_d4_exclusion_v1.json"
)

write_json(
    FIREWALL_OUT,
    prospective_firewall_rows,
)

FIREWALL_SHA = sha256_file(
    FIREWALL_OUT
)


# ======================================================================
# 5. BUILD RESIDUAL5 + q5 SEARCH OVERLAY
#
# IMPORTANT:
#   - residual models are NOT refit
#   - residual model files are symlinked byte-for-byte
#   - only joint95_conformal_q is changed
#
# This makes q5 active INSIDE all four GA families without changing
# their solver mathematics.
# ======================================================================

source_meta = json.loads(
    SOURCE_RESIDUAL_METADATA.read_text()
)

source_meta_sha = sha256_file(
    SOURCE_RESIDUAL_METADATA
)

if "joint95_conformal_q" not in source_meta:
    raise RuntimeError(
        "Residual5 metadata has no joint95_conformal_q."
    )

Q5_OVERLAY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for src in SOURCE_RESIDUAL_DIR.iterdir():

    if src.name == "residual_safety_metadata.json":
        continue

    dst = Q5_OVERLAY_DIR / src.name

    if dst.exists() or dst.is_symlink():

        if dst.resolve() != src.resolve():
            raise RuntimeError(
                f"Overlay collision: {dst}"
            )

        continue

    dst.symlink_to(
        src.resolve(),
        target_is_directory=src.is_dir(),
    )


overlay_meta = dict(
    source_meta
)

overlay_meta["joint95_conformal_q"] = dict(
    Q5
)

overlay_meta[
    "cycle5_q5_prospective_search_override"
] = {
    "status":
        "FROZEN_Q5_SEARCH_MULTIPLIER_ONLY",

    "source_residual_model_dir":
        str(
            SOURCE_RESIDUAL_DIR.relative_to(PROJECT)
        ),

    "source_residual_metadata_sha256":
        source_meta_sha,

    "candidate_q5_artifact":
        str(
            candidate_q5_path.relative_to(PROJECT)
        ),

    "candidate_q5_sha256":
        CANDIDATE_Q5_SHA,

    "q5":
        dict(Q5),

    "residual5_refit":
        False,

    "residual5_calibration_reprediction":
        False,

    "prospective_validation_used_for_q5_fit":
        False,

    "q5_may_be_refit_on_prospective_validation":
        False,

    "role":
        "prospective-search bound multiplier only",
}

write_json(
    OVERLAY_METADATA,
    overlay_meta,
)

OVERLAY_METADATA_SHA = sha256_file(
    OVERLAY_METADATA
)


# ======================================================================
# 6. FRESH PROSPECTIVE PAIRED-SEED REGISTRY
# ======================================================================

# Explicitly prove no overlap with previous Cycle-5 calibration namespace.
calibration_seed_set = {
    2_050_000_001 + SEED_STRIDE * rep
    for rep in range(80)
}

prospective_seed_set = set(
    PROSPECTIVE_GA_SEEDS
)

if not calibration_seed_set.isdisjoint(
    prospective_seed_set
):
    raise RuntimeError(
        "Prospective GA seeds overlap Cycle-5 calibration seeds."
    )


seed_rows = []

for rep, seed in enumerate(
    PROSPECTIVE_GA_SEEDS
):

    for solver in SOLVERS:

        for initializer in INITIALIZERS:

            seed_rows.append({
                "replicate": rep,
                "solver": solver,
                "initializer": initializer,
                "arm":
                    f"{solver}_{initializer}",
                "paired_ga_seed": seed,
                "runner_seed_index": rep,
            })

if len(seed_rows) != 360:
    raise RuntimeError(
        f"Expected 360 seed rows; got {len(seed_rows)}"
    )


SEED_REGISTRY = (
    PROTOCOL
    / "cycle5_q5_prospective360_paired_seed_registry_v1.json"
)

write_json(
    SEED_REGISTRY,
    {
        "status":
            "FROZEN_BEFORE_PROSPECTIVE_SEARCH",

        "experimental_unit":
            "replicate_x_solver_x_initializer",

        "replicates":
            N_REPLICATES,

        "solvers":
            SOLVERS,

        "initializers":
            INITIALIZERS,

        "arms_per_replicate":
            12,

        "total_runs":
            360,

        "paired_within_replicate":
            True,

        "seed_base":
            PROSPECTIVE_SEED_BASE,

        "seed_stride":
            SEED_STRIDE,

        "cycle5_calibration_seed_overlap":
            0,

        "runs":
            seed_rows,
    },
)

SEED_REGISTRY_SHA = sha256_file(
    SEED_REGISTRY
)


# ======================================================================
# 7. WARMSTART-SHARD REGISTRY
# ======================================================================

warmstart_rows = []

for rep in range(
    N_REPLICATES
):

    first = (
        rep
        * SHARDS_PER_REPLICATE
    )

    shard_ids = list(
        range(
            first,
            first + SHARDS_PER_REPLICATE,
        )
    )

    warmstart_rows.append({
        "replicate":
            rep,

        "shard_ids":
            shard_ids,

        "shards_per_replicate":
            SHARDS_PER_REPLICATE,
    })


WARMSTART_REGISTRY = (
    PROTOCOL
    / "cycle5_q5_prospective360_warmstart_shard_registry_v1.json"
)

write_json(
    WARMSTART_REGISTRY,
    {
        "status":
            "FROZEN_BEFORE_WARMSTART_GENERATION",

        "replicates":
            N_REPLICATES,

        "shards_per_replicate":
            SHARDS_PER_REPLICATE,

        "total_shards":
            N_WARMSTART_SHARDS,

        "rows_expected_per_shard":
            40,

        "warmstart_generator_base_seed":
            WARMSTART_BASE_SEED,

        "replicate_shards":
            warmstart_rows,
    },
)

WARMSTART_REGISTRY_SHA = sha256_file(
    WARMSTART_REGISTRY
)


# ======================================================================
# 8. FREEZE MASTER PROTOCOL
# ======================================================================

MASTER = (
    PROTOCOL
    / "cycle5_q5_prospective360_protocol_v1.json"
)

master_obj = {
    "status":
        "FROZEN_BEFORE_FRESH_PROSPECTIVE360",

    "purpose":
        "fresh balanced prospective validation of frozen q5",

    "candidate_q5_final_certificate":
        False,

    "prospective_validation_used":
        False,

    "q5_refit_allowed":
        False,

    "residual5_refit":
        False,

    "residual5_calibration_reprediction":
        False,

    "design": {
        "replicates":
            N_REPLICATES,

        "arms":
            len(ARMS),

        "units_per_arm":
            N_REPLICATES,

        "total_experimental_units":
            TOTAL_UNITS,

        "paired_ga_seed_within_replicate":
            True,

        "solvers":
            SOLVERS,

        "initializers":
            INITIALIZERS,

        "arms_list":
            ARMS,
    },

    "frozen_q5": {
        **Q5,

        "candidate_artifact":
            str(
                candidate_q5_path.relative_to(PROJECT)
            ),

        "candidate_artifact_sha256":
            CANDIDATE_Q5_SHA,
    },

    "step11_lineage": {
        "openmc_truth_sha256":
            OPENMC_TRUTH_SHA,

        "merged960":
            str(
                merged960_path.relative_to(PROJECT)
            ),

        "merged960_sha256":
            MERGED960_SHA,

        "score_artifact":
            str(
                score_path.relative_to(PROJECT)
            ),

        "score_artifact_sha256":
            SCORE_SHA,

        "calibration_units":
            960,

        "calibration_exact_unique":
            merged_exact_unique,

        "calibration_natural_duplicates":
            merged_exact_duplicates,

        "calibration_d4_unique":
            len(calibration_d4),
    },

    "prospective_firewall": {
        "source_firewall":
            str(
                SOURCE_FIREWALL.relative_to(PROJECT)
            ),

        "source_firewall_d4":
            len(base_d4),

        "calibration_d4_added":
            len(calibration_d4),

        "source_calibration_d4_overlap":
            0,

        "prospective_firewall":
            str(
                FIREWALL_OUT.relative_to(PROJECT)
            ),

        "prospective_firewall_d4":
            EXPECTED_PROSPECTIVE_D4,

        "prospective_firewall_sha256":
            FIREWALL_SHA,
    },

    "residual5_search_overlay": {
        "source_model_dir":
            str(
                SOURCE_RESIDUAL_DIR.relative_to(PROJECT)
            ),

        "source_metadata_sha256":
            source_meta_sha,

        "overlay_model_dir":
            str(
                Q5_OVERLAY_DIR.relative_to(PROJECT)
            ),

        "overlay_metadata_sha256":
            OVERLAY_METADATA_SHA,

        "model_files_refit":
            False,

        "only_q_changed":
            True,
    },

    "paired_seed_registry":
        str(
            SEED_REGISTRY.relative_to(PROJECT)
        ),

    "paired_seed_registry_sha256":
        SEED_REGISTRY_SHA,

    "warmstart": {
        "total_shards":
            N_WARMSTART_SHARDS,

        "shards_per_replicate":
            SHARDS_PER_REPLICATE,

        "expected_rows_per_shard":
            40,

        "base_seed":
            WARMSTART_BASE_SEED,

        "registry":
            str(
                WARMSTART_REGISTRY.relative_to(PROJECT)
            ),

        "registry_sha256":
            WARMSTART_REGISTRY_SHA,
    },

    "scientific_rules": {
        "prospective_truth_available_during_search":
            False,

        "openmc_used_for_ga_selection":
            False,

        "q5_changed_after_validation":
            False,

        "validation_used_to_choose_q5":
            False,

        "calibration_units_reused_as_prospective_units":
            False,

        "prospective_search_must_use_q5_inside_search":
            True,
    },
}

write_json(
    MASTER,
    master_obj,
)

MASTER_SHA = sha256_file(
    MASTER
)


# ======================================================================
# 9. SHELL ENV FOR SUBSEQUENT SLURM STAGES
# ======================================================================

ENV_FILE = (
    PROTOCOL / "step12a.env"
)

ENV_FILE.write_text(
    "\n".join([
        "# Cycle 5 Step 12A frozen prospective360 environment",

        f'export CYCLE5_Q5_PROSPECTIVE_FIREWALL='
        f'"{FIREWALL_OUT.relative_to(PROJECT)}"',

        f'export CYCLE5_Q5_PROSPECTIVE_EXPECTED_D4='
        f'"{EXPECTED_PROSPECTIVE_D4}"',

        f'export CYCLE5_Q5_PROSPECTIVE_Q5_OVERLAY='
        f'"{Q5_OVERLAY_DIR.relative_to(PROJECT)}"',

        f'export CYCLE5_Q5_PROSPECTIVE_MASTER='
        f'"{MASTER.relative_to(PROJECT)}"',

        f'export CYCLE5_Q5_PROSPECTIVE_MASTER_SHA='
        f'"{MASTER_SHA}"',

        f'export CYCLE5_Q5_PROSPECTIVE_WARMSTART_BASE_SEED='
        f'"{WARMSTART_BASE_SEED}"',

        "",
    ])
)


# ======================================================================
# REPORT
# ======================================================================

print()
print("=" * 100)
print(
    "CYCLE 5 STEP 12A — "
    "FRESH BALANCED PROSPECTIVE360 PREPARATION"
)
print("=" * 100)

print()
print("DESIGN")
print("  replicates                    =", N_REPLICATES)
print("  arms                          =", len(ARMS))
print("  units / arm                   =", N_REPLICATES)
print("  prospective units             =", TOTAL_UNITS)

print()
print("WARMSTART")
print("  shards / replicate            =", SHARDS_PER_REPLICATE)
print("  total fresh shards            =", N_WARMSTART_SHARDS)
print("  expected rows / shard         = 40")

print()
print("CALIBRATION FIREWALL")
print("  original known D4             =", len(base_d4))
print("  calibration D4 unique         =", len(calibration_d4))
print("  overlap                       =", len(preexisting_overlap))
print("  prospective firewall D4       =", EXPECTED_PROSPECTIVE_D4)

print()
print("q5")
for k, v in Q5.items():
    print(f"  {k:12s} = {v}")

print()
print("RESIDUAL5")
print("  refit                         = NO")
print("  old calibration re-prediction= NO")
print("  q5 active inside search       = YES")
print("  search overlay metadata       =", OVERLAY_METADATA)
print("  overlay metadata SHA256       =", OVERLAY_METADATA_SHA)

print()
print("FROZEN LINEAGE")
print("  OpenMC truth SHA256           =", OPENMC_TRUTH_SHA)
print("  merged960 SHA256              =", MERGED960_SHA)
print("  score artifact SHA256         =", SCORE_SHA)
print("  candidate q5 SHA256           =", CANDIDATE_Q5_SHA)

print()
print("NEW PROSPECTIVE ARTIFACTS")
print("  firewall                      =", FIREWALL_OUT)
print("  firewall SHA256               =", FIREWALL_SHA)
print("  paired seed registry          =", SEED_REGISTRY)
print("  paired seed registry SHA256   =", SEED_REGISTRY_SHA)
print("  master protocol               =", MASTER)
print("  master protocol SHA256        =", MASTER_SHA)
print("  env file                      =", ENV_FILE)

print()
print(
    "CYCLE5_STEP12A_PROSPECTIVE360_PROTOCOL_FROZEN_"
    "READY_FOR_FRESH_WARMSTART300"
)
