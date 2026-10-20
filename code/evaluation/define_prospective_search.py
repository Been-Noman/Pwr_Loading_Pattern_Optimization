#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import hashlib
import json
import textwrap

PROJECT = Path(__import__("os").environ["PWR_PROJECT"])

PROS = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360"
)

PROTOCOL = PROS / "protocol"
FINAL = PROS / "warmstarts/final_banks"
GA_ROOT = PROS / "ga360"

SEED_REGISTRY = (
    PROTOCOL
    / "cycle5_q5_prospective360_paired_seed_registry_v1.json"
)

BANK_MANIFEST = (
    FINAL
    / "cycle5_q5_prospective360_anchor_bank_manifest_v1.json"
)

WARMSTART_AUDIT = (
    FINAL
    / "cycle5_q5_prospective360_warmstart_strict_audit_final_v1.json"
)

Q5_CANDIDATE = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/"
      "q5_calibration/protocol/cycle5_q5_candidate_balanced960_v1.json"
)

SCIENTIFIC_FIREWALL = (
    PROTOCOL
    / "cycle5_q5_prospective360_known_openmc_plus_calibration_d4_exclusion_v1.json"
)

ROUND_DIR = (
    PROJECT
    / "run/optimizer_tail_v2/models/round5_cnn"
)

RESIDUAL_DIR = (
    PROJECT
    / "run/optimizer_tail_v2/models/"
      "learned_residual_safety_round5_q5_prospective_search"
)

RESIDUAL_METADATA = (
    RESIDUAL_DIR / "residual_safety_metadata.json"
)

DISPATCHER = (
    PROJECT
    / "scripts/optimizer_tail_v2/run_paired_search_task.py"
)

ADAPTER = (
    PROJECT
    / "scripts/optimizer_tail_v2/"
      "run_prospective_optimization.py"
)

SHARED_RUNTIME = (
    PROTOCOL
    / "cycle5_q5_prospective360_shared_runtime_inputs_v1.json"
)

EXEC_MANIFEST = (
    PROTOCOL
    / "cycle5_q5_prospective360_execution_manifest_360_v1.json"
)

EXPECTED = {
    "seed_registry":
        "5a6d748859acffdec5db256263c34acba05a994c2330ab57ccb73de7bac1931e",

    "bank_manifest":
        "ee4d6babc17bc0c9808f5ed377bb25159c5dcba675569da466d178da9fd9bc2b",

    "warmstart_audit":
        "62cb30d94e539bb47abf39653ebee148e91739b6bdef53eb15517bb1e1ad98f1",

    "q5_candidate":
        "21c1f0ac8ee5b7546b4264e81d54dc6f0c68a74605bbb5e527c207e6f2fb6760",

    "scientific_firewall":
        "2bbac884a16b5a90d1d4d3b91684555fff6a598a8267f0732fb40a0751556de4",

    "residual_metadata":
        "5084368629a8f1fed3bdfa052c755f21aed767875d5d7d6f8601c115a867c4ca",
}

Q5 = {
    "keff_lower": 1.2555488951630267,
    "keff_upper": 1.167078988515213,
    "Fa_upper": -0.00824025190735769,
    "Fq_r_upper": -0.15313887213725857,
}

RUNNERS = {
    "v2": (
        PROJECT
        / "scripts/optimizer_tail_v2/penalty_based_search_300.py",

        "5573751893c2d13e741a65f149e95a23faaa1ab0a7e9ac8939550fc66fc488cd",
    ),

    "v2m": (
        PROJECT
        / "scripts/optimizer_tail_v2/penalty_based_memetic_search.py",

        "3fe75f2078ef278be32989a3a8e8fa8014c0d3d35b5b544bd31e8c3d307b729f",
    ),

    "v3": (
        PROJECT
        / "scripts/optimizer_tail_v2/feasibility_first_search.py",

        "ef547aaa9871936a34818285fbfac4e26e42e0044117bc0318b79cf0bf3efe5d",
    ),

    "v3m": (
        PROJECT
        / "scripts/optimizer_tail_v2/feasibility_first_memetic_search.py",

        "419ad99d4996cb52b3b5465057c21ae6a714ebfc57966ae15990359a46a62504",
    ),
}

# Correct historical V3 SHA.
RUNNERS["v3"] = (
    RUNNERS["v3"][0],
    "ef547aaa9871936a34818285fbfac4e26e42e0044117bc0318b79c7088dcfbc6",
)

SOLVERS = ("v2", "v2m", "v3", "v3m")
INITIALIZERS = ("random", "sa", "sqa")

N_REPLICATES = 30
ARMS_PER_REPLICATE = 12
TOTAL_RUNS = 360


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


def frozen_write_text(path, text):
    data = text.encode()

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(
                f"Frozen file already exists with different bytes:\n{path}"
            )
        return

    path.write_bytes(data)


# ================================================================
# 1. VERIFY ALL ALREADY-FROZEN SCIENTIFIC INPUTS
# ================================================================

required = [
    SEED_REGISTRY,
    BANK_MANIFEST,
    WARMSTART_AUDIT,
    Q5_CANDIDATE,
    SCIENTIFIC_FIREWALL,
    ROUND_DIR,
    RESIDUAL_DIR,
    RESIDUAL_METADATA,
    DISPATCHER,
]

for p in required:
    if not p.exists():
        raise RuntimeError(f"Missing required frozen dependency:\n{p}")

checks = [
    ("seed registry", SEED_REGISTRY, EXPECTED["seed_registry"]),
    ("bank manifest", BANK_MANIFEST, EXPECTED["bank_manifest"]),
    ("warmstart audit", WARMSTART_AUDIT, EXPECTED["warmstart_audit"]),
    ("q5 candidate", Q5_CANDIDATE, EXPECTED["q5_candidate"]),
    (
        "scientific firewall",
        SCIENTIFIC_FIREWALL,
        EXPECTED["scientific_firewall"],
    ),
    (
        "Residual5 q5 metadata",
        RESIDUAL_METADATA,
        EXPECTED["residual_metadata"],
    ),
]

for name, path, expected_sha in checks:
    actual = sha(path)

    if actual != expected_sha:
        raise RuntimeError(
            f"{name} SHA changed:\n"
            f"expected={expected_sha}\n"
            f"actual  ={actual}"
        )


# ================================================================
# 2. VERIFY FROZEN RUNNER IDENTITIES
# ================================================================

runner_shas = {}

for solver, (path, expected_sha) in RUNNERS.items():

    if not path.is_file():
        raise RuntimeError(f"Missing runner: {path}")

    actual = sha(path)

    if actual != expected_sha:
        raise RuntimeError(
            f"Frozen {solver} runner SHA changed:\n"
            f"expected={expected_sha}\n"
            f"actual  ={actual}"
        )

    runner_shas[solver] = actual


# ================================================================
# 3. VERIFY q5 OVERLAY
# ================================================================

meta = json.loads(RESIDUAL_METADATA.read_text())

q_meta = meta.get("joint95_conformal_q")

if q_meta is None:
    raise RuntimeError(
        "Residual5 prospective metadata has no joint95_conformal_q"
    )

for key, expected in Q5.items():

    actual = float(q_meta[key])

    if abs(actual - expected) > 1e-15:
        raise RuntimeError(
            f"q5 mismatch for {key}: {actual} != {expected}"
        )


# ================================================================
# 4. RECOVER + VERIFY THE PRE-FROZEN 30 PAIRED SEEDS
# ================================================================

seed_obj = json.loads(SEED_REGISTRY.read_text())

found = defaultdict(set)


def walk(x):

    if isinstance(x, dict):

        if "replicate" in x:

            try:
                rep = int(x["replicate"])
            except Exception:
                rep = None

            if rep is not None:

                for key in (
                    "paired_ga_seed",
                    "ga_seed",
                    "seed",
                ):
                    if key in x:
                        try:
                            found[rep].add(int(x[key]))
                        except Exception:
                            pass

        if (
            "replicate_seeds" in x
            and isinstance(x["replicate_seeds"], dict)
        ):
            for k, v in x["replicate_seeds"].items():
                found[int(k)].add(int(v))

        for value in x.values():
            walk(value)

    elif isinstance(x, list):

        for value in x:
            walk(value)


walk(seed_obj)

paired_seeds = {
    rep: 2060000001 + 1009 * rep
    for rep in range(N_REPLICATES)
}

for rep, seed in paired_seeds.items():

    if rep in found and found[rep] != {seed}:
        raise RuntimeError(
            f"Frozen seed-registry mismatch at replicate {rep}: "
            f"{found[rep]} != {{{seed}}}"
        )


# ================================================================
# 5. VERIFY 60 FROZEN ANCHOR BANKS AGAINST THEIR MANIFEST
# ================================================================

bank_rows = json.loads(BANK_MANIFEST.read_text())

if not isinstance(bank_rows, list) or len(bank_rows) != 60:
    raise RuntimeError(
        f"Expected 60 frozen bank records, got {len(bank_rows)}"
    )

bank_lookup = {}

for row in bank_rows:

    rep = int(row["replicate"])
    strategy = row["strategy"]

    if strategy not in ("sa", "sqa"):
        raise RuntimeError(
            f"Unexpected bank strategy: {strategy}"
        )

    key = (rep, strategy)

    if key in bank_lookup:
        raise RuntimeError(
            f"Duplicate bank-manifest entry: {key}"
        )

    p = PROJECT / row["file"]

    if not p.is_file():
        raise RuntimeError(
            f"Missing frozen anchor bank: {p}"
        )

    if sha(p) != row["sha256"]:
        raise RuntimeError(
            f"Frozen anchor SHA mismatch: {p}"
        )

    if int(row["rows"]) != 200:
        raise RuntimeError(
            f"Bank {key} does not contain frozen 200-row design"
        )

    bank_lookup[key] = row


for rep in range(N_REPLICATES):
    for strategy in ("sa", "sqa"):
        if (rep, strategy) not in bank_lookup:
            raise RuntimeError(
                f"Missing final anchor bank rep={rep} {strategy}"
            )


# ================================================================
# 6. CREATE PROSPECTIVE-SPECIFIC EXPLICIT-SEED ADAPTER
#
# Historical bridge adapter supported only seed indices 0..9.
# This new adapter changes NO solver code. It only extends the
# injection vector to the pre-frozen 30 prospective replicates.
# ================================================================

adapter_text = r'''#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import importlib.util
import sys


PROJECT = Path(__file__).resolve().parents[2]

RUNNERS = {
    "v2": (
        PROJECT / "scripts/optimizer_tail_v2/penalty_based_search_300.py",
        "5573751893c2d13e741a65f149e95a23faaa1ab0a7e9ac8939550fc66fc488cd",
    ),
    "v2m": (
        PROJECT / "scripts/optimizer_tail_v2/penalty_based_memetic_search.py",
        "3fe75f2078ef278be32989a3a8e8fa8014c0d3d35b5b544bd31e8c3d307b729f",
    ),
    "v3": (
        PROJECT / "scripts/optimizer_tail_v2/feasibility_first_search.py",
        "ef547aaa9871936a34818285fbfac4e26e42e0044117bc0318b79c7088dcfbc6",
    ),
    "v3m": (
        PROJECT / "scripts/optimizer_tail_v2/feasibility_first_memetic_search.py",
        "419ad99d4996cb52b3b5465057c21ae6a714ebfc57966ae15990359a46a62504",
    ),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Cycle-5 prospective360 adapter: inject one explicit "
            "paired GA seed into an unchanged frozen solver runner."
        )
    )

    parser.add_argument(
        "--bridge-runner",
        required=True,
        choices=tuple(RUNNERS),
    )

    parser.add_argument(
        "--bridge-replicate",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--bridge-ga-seed",
        required=True,
        type=int,
    )

    args, runner_args = parser.parse_known_args()

    if not 0 <= args.bridge_replicate <= 29:
        raise SystemExit(
            "--bridge-replicate must be 0..29"
        )

    for token in runner_args:
        if (
            token == "--seed-index"
            or token.startswith("--seed-index=")
        ):
            raise SystemExit(
                "Do not pass --seed-index manually; "
                "the prospective adapter controls it."
            )

    runner_path, expected_sha = RUNNERS[
        args.bridge_runner
    ]

    actual_sha = sha256(runner_path)

    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Frozen runner SHA changed for "
            f"{args.bridge_runner}: {actual_sha}"
        )

    sys.path.insert(
        0,
        str(runner_path.parent.resolve()),
    )

    spec = importlib.util.spec_from_file_location(
        f"cycle5_q5_prospective_{args.bridge_runner}",
        runner_path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rep = int(args.bridge_replicate)
    seed = int(args.bridge_ga_seed)

    injected = [0] * 30
    injected[rep] = seed

    module.GA_SEEDS = tuple(injected)

    # Preserve historical adapter behavior for any fallback seed path.
    module.GA_BASE_SEED = seed

    sys.argv = [
        str(runner_path),
        *runner_args,
        "--seed-index",
        str(rep),
    ]

    module.main()


if __name__ == "__main__":
    main()
'''

frozen_write_text(ADAPTER, adapter_text)

adapter_sha = sha(ADAPTER)
dispatcher_sha = sha(DISPATCHER)


# ================================================================
# 7. FREEZE SHARED RUNTIME INPUTS
# ================================================================

shared = {
    "stage":
        "CYCLE5_Q5_PROSPECTIVE360_GA_RUNTIME",

    "status":
        "FROZEN_BEFORE_PROSPECTIVE_GA360",

    "round_dir":
        rel(ROUND_DIR),

    "residual_model_dir":
        rel(RESIDUAL_DIR),

    "residual_metadata":
        rel(RESIDUAL_METADATA),

    "residual_metadata_sha256":
        sha(RESIDUAL_METADATA),

    "q5":
        Q5,

    "q5_candidate":
        rel(Q5_CANDIDATE),

    "q5_candidate_sha256":
        sha(Q5_CANDIDATE),

    "q5_refit_allowed":
        False,

    "residual5_refit_allowed":
        False,

    "prospective_truth_used_for_search":
        False,

    "openmc_during_ga":
        False,

    # Use the original scientific prospective firewall.
    # The generator-only exact-unique projection remains only an
    # implementation compatibility artifact for SA/SQA generation.
    "scientific_firewall":
        rel(SCIENTIFIC_FIREWALL),

    "scientific_firewall_sha256":
        sha(SCIENTIFIC_FIREWALL),

    "paired_seed_registry":
        rel(SEED_REGISTRY),

    "paired_seed_registry_sha256":
        sha(SEED_REGISTRY),

    "final_anchor_manifest":
        rel(BANK_MANIFEST),

    "final_anchor_manifest_sha256":
        sha(BANK_MANIFEST),

    "warmstart_strict_audit":
        rel(WARMSTART_AUDIT),

    "warmstart_strict_audit_sha256":
        sha(WARMSTART_AUDIT),

    "explicit_seed_adapter":
        rel(ADAPTER),

    "explicit_seed_adapter_sha256":
        adapter_sha,

    "dispatcher":
        rel(DISPATCHER),

    "dispatcher_sha256":
        dispatcher_sha,

    "population":
        100,

    "parents":
        40,

    "generations":
        300,

    "threads_per_run":
        48,

    "solvers":
        list(SOLVERS),

    "initializers":
        list(INITIALIZERS),

    "replicates":
        N_REPLICATES,

    "arms_per_replicate":
        ARMS_PER_REPLICATE,

    "total_ga_runs":
        TOTAL_RUNS,
}

frozen_write(SHARED_RUNTIME, shared)

shared_sha = sha(SHARED_RUNTIME)


# ================================================================
# 8. FREEZE EXACT 360-RUN MANIFEST
#
# Task order:
# rep0:
#   v2 random,sa,sqa
#   v2m random,sa,sqa
#   v3 random,sa,sqa
#   v3m random,sa,sqa
# then rep1 ... rep29.
# ================================================================

runs = []

task_id = 0

for rep in range(N_REPLICATES):

    seed = paired_seeds[rep]

    for solver in SOLVERS:

        runner_path, runner_sha = RUNNERS[solver]

        for initializer in INITIALIZERS:

            anchor_path = None
            anchor_sha = None

            if initializer != "random":

                bank = bank_lookup[
                    (rep, initializer)
                ]

                anchor = PROJECT / bank["file"]

                anchor_path = rel(anchor)
                anchor_sha = sha(anchor)

            out_dir = (
                GA_ROOT
                / "runs"
                / f"rep_{rep:02d}"
                / solver
                / initializer
            )

            runs.append({
                "task_id":
                    task_id,

                "run_id":
                    f"rep_{rep:02d}__{solver}__{initializer}",

                "replicate":
                    rep,

                "solver":
                    solver,

                "initializer":
                    initializer,

                "paired_ga_seed":
                    seed,

                "runner_seed_index":
                    rep,

                "adapter":
                    rel(ADAPTER),

                "adapter_sha256":
                    adapter_sha,

                "runner_file":
                    rel(runner_path),

                "runner_sha256":
                    runner_sha,

                "shared_runtime_manifest":
                    rel(SHARED_RUNTIME),

                "shared_runtime_manifest_sha256":
                    shared_sha,

                "round_dir":
                    rel(ROUND_DIR),

                "residual_model_dir":
                    rel(RESIDUAL_DIR),

                "residual_metadata":
                    rel(RESIDUAL_METADATA),

                "residual_metadata_sha256":
                    sha(RESIDUAL_METADATA),

                "labelled_json":
                    rel(SCIENTIFIC_FIREWALL),

                "labelled_json_sha256":
                    sha(SCIENTIFIC_FIREWALL),

                "anchor_json":
                    anchor_path,

                "anchor_sha256":
                    anchor_sha,

                "output_dir":
                    rel(out_dir),

                "expected_summary":
                    rel(
                        out_dir
                        / f"ga_seed_{seed}_summary.json"
                    ),

                "expected_archive":
                    rel(
                        out_dir
                        / f"ga_seed_{seed}_archive.json"
                    ),

                "population":
                    100,

                "parents":
                    40,

                "generations":
                    300,

                "threads":
                    48,

                "q5_role":
                    "frozen prospective search certificate",

                "prospective_truth_available":
                    False,
            })

            task_id += 1


if len(runs) != TOTAL_RUNS:
    raise RuntimeError(
        f"Expected 360 GA runs, got {len(runs)}"
    )

if [r["task_id"] for r in runs] != list(range(360)):
    raise RuntimeError(
        "GA task IDs are not contiguous 0..359"
    )


# Exact paired-seed property.
for rep in range(N_REPLICATES):

    ss = {
        int(r["paired_ga_seed"])
        for r in runs
        if int(r["replicate"]) == rep
    }

    if ss != {paired_seeds[rep]}:
        raise RuntimeError(
            f"Replicate {rep} does not have one common paired seed"
        )

    rr = [
        r
        for r in runs
        if int(r["replicate"]) == rep
    ]

    if len(rr) != 12:
        raise RuntimeError(
            f"Replicate {rep} has {len(rr)} runs instead of 12"
        )


manifest = {
    "stage":
        "CYCLE5_Q5_PROSPECTIVE360_GA360",

    "status":
        "FROZEN_READY_TO_DISPATCH",

    "purpose": (
        "Fresh balanced 30-replicate x 12-arm prospective "
        "validation search with frozen q5 active inside search."
    ),

    "experimental_unit":
        "one solver x initializer run within a paired replicate",

    "replicates":
        30,

    "arms_per_replicate":
        12,

    "total_ga_runs":
        360,

    "solvers":
        list(SOLVERS),

    "initializers":
        list(INITIALIZERS),

    "paired_seed_rule":
        "Within replicate r, all 12 arms use exactly seed "
        "2060000001 + 1009*r.",

    "replicate_seeds": {
        str(rep): paired_seeds[rep]
        for rep in range(N_REPLICATES)
    },

    "q5":
        Q5,

    "q5_candidate":
        rel(Q5_CANDIDATE),

    "q5_candidate_sha256":
        sha(Q5_CANDIDATE),

    "q5_refit_allowed":
        False,

    "residual5_refit_allowed":
        False,

    "prospective_truth_used_for_search":
        False,

    "openmc_during_ga":
        False,

    "population":
        100,

    "parents":
        40,

    "generations":
        300,

    "threads_per_run":
        48,

    "adapter":
        rel(ADAPTER),

    "adapter_sha256":
        adapter_sha,

    "adapter_replicate_capacity":
        "0..29",

    "adapter_reason": (
        "Prospective-specific extension of the historical explicit-seed "
        "adapter. Frozen solver runners are unchanged; only the injected "
        "GA_SEEDS vector length is extended from 10 to 30."
    ),

    "dispatcher":
        rel(DISPATCHER),

    "dispatcher_sha256":
        dispatcher_sha,

    "shared_runtime_manifest":
        rel(SHARED_RUNTIME),

    "shared_runtime_manifest_sha256":
        shared_sha,

    "paired_seed_registry":
        rel(SEED_REGISTRY),

    "paired_seed_registry_sha256":
        sha(SEED_REGISTRY),

    "final_anchor_manifest":
        rel(BANK_MANIFEST),

    "final_anchor_manifest_sha256":
        sha(BANK_MANIFEST),

    "warmstart_strict_audit":
        rel(WARMSTART_AUDIT),

    "warmstart_strict_audit_sha256":
        sha(WARMSTART_AUDIT),

    "scientific_firewall":
        rel(SCIENTIFIC_FIREWALL),

    "scientific_firewall_sha256":
        sha(SCIENTIFIC_FIREWALL),

    "residual_metadata":
        rel(RESIDUAL_METADATA),

    "residual_metadata_sha256":
        sha(RESIDUAL_METADATA),

    "frozen_runner_sha256":
        runner_shas,

    "runs":
        runs,
}

frozen_write(EXEC_MANIFEST, manifest)

manifest_sha = sha(EXEC_MANIFEST)


# ================================================================
# 9. RESULT
# ================================================================

print("=" * 100)
print("CYCLE 5 STEP 12C — PROSPECTIVE360 GA MANIFEST FROZEN")
print("=" * 100)

print()
print("PAIRED DESIGN")
print("  replicates                    = 30")
print("  arms / replicate              = 12")
print("  total GA runs                 = 360")
print("  first paired seed             =", paired_seeds[0])
print("  last paired seed              =", paired_seeds[29])
print("  same seed across 12 arms      = YES")

print()
print("FROZEN SEARCH")
print("  population                    = 100")
print("  parents                       = 40")
print("  generations                   = 300")
print("  q5 active inside search       = YES")
print("  q5 refit                      = NO")
print("  Residual5 refit               = NO")
print("  prospective OpenMC used       = NO")

print()
print("EXECUTION SUPPORT")
print("  adapter                       =", ADAPTER)
print("  adapter replicate range       = 0..29")
print("  adapter SHA256                =", adapter_sha)
print("  frozen solver runners changed = NO")

print()
print("FROZEN ARTIFACTS")
print("  shared runtime                =", SHARED_RUNTIME)
print("  shared runtime SHA256         =", shared_sha)
print()
print("  execution manifest            =", EXEC_MANIFEST)
print("  execution manifest SHA256     =", manifest_sha)

print()
print("GA SUBMITTED                    = NO")
print()
print("CYCLE5_STEP12C_GA360_MANIFEST_FROZEN_READY_TO_SUBMIT")
