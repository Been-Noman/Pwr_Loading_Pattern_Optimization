#!/usr/bin/env python3

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__import__("os").environ["PWR_PROJECT"])
sys.path.insert(0, str(PROJECT / "scripts"))

from geometry import canonical_d4_key, is_hard_feasible


# ======================================================================
# PATHS
# ======================================================================

PROS = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360"
)

PROTOCOL = PROS / "protocol"

RAW = PROS / "warmstarts/fresh300/raw"
REPAIR_RAW = PROS / "warmstarts/repair_pool/raw"
FINAL = PROS / "warmstarts/final_banks"

FIREWALL = (
    PROTOCOL
    / "cycle5_q5_prospective360_generator_exact_unique_exclusion_v1.json"
)

PLAN = (
    PROTOCOL
    / "cycle5_q5_prospective360_warmstart_repair_plan_v1.json"
)

TARGETS = (
    PROTOCOL
    / "cycle5_q5_prospective360_warmstart_repair_targets_v1.json"
)

REPAIR_DESIGN = (
    PROTOCOL
    / "cycle5_q5_prospective360_repair_pool_design_v1.json"
)

POOL_AUDIT = (
    PROTOCOL
    / "cycle5_q5_prospective360_repair40_pool_audit_v1.json"
)

SELECTION = (
    PROTOCOL
    / "cycle5_q5_prospective360_repair_selection_219_v1.json"
)

APPLICATION = (
    FINAL
    / "cycle5_q5_prospective360_repair_application_219_v1.json"
)

BANK_MANIFEST = (
    FINAL
    / "cycle5_q5_prospective360_anchor_bank_manifest_v1.json"
)

STRICT_AUDIT = (
    FINAL
    / "cycle5_q5_prospective360_warmstart_strict_audit_final_v1.json"
)


# Frozen identities already established before repair generation.
EXPECTED_PLAN_SHA = (
    "a2b7c058acf1bee16c109b44da03fc343f03c72871d5f50d4e39d4043772c970"
)

EXPECTED_TARGETS_SHA = (
    "9e0be820b051f0adb1d6041a2215ff8d91389499c0256498e96c65b7490b849b"
)

EXPECTED_FIREWALL_SHA = (
    "092544e1d417eb588d92ee2dd51b10b82c2396861b327d89b3a2074063047f73"
)

REPAIR_JOB_ID = 20981775


# These are the same external anchor populations frozen when the
# repair targets were constructed.
EXTERNAL_BANK_DIRS = [
    PROJECT / "run/formal_random_sa_sqa/manifests/anchor_banks",

    PROJECT
    / "run/optimizer_tail_v2/fair_hybrid_benchmark/"
      "bridge_warmstarts/parallel100/final_banks",

    PROJECT
    / "run/optimizer_tail_v2/fair_hybrid_benchmark/"
      "commonq_calibration/warmstarts/final_banks",

    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/"
      "q5_calibration/warmstarts/final_banks",
]


# ======================================================================
# HELPERS
# ======================================================================

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def pattern_sha(pattern):
    return hashlib.sha256(
        bytes(int(x) for x in pattern)
    ).hexdigest()


def exact_key(pattern):
    return tuple(int(x) for x in pattern)


def d4_key(pattern):
    return tuple(canonical_d4_key(pattern))


def load_list(path):
    obj = json.loads(path.read_text())
    if not isinstance(obj, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    return obj


def frozen_write(path, obj):
    """
    Create a deterministic frozen artifact.

    If the artifact already exists, a rerun is allowed only when the
    new bytes are exactly identical.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = (
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()

    if path.exists():
        old = path.read_bytes()
        if old != data:
            raise RuntimeError(
                f"Frozen artifact already exists with different bytes:\n{path}"
            )
        return

    path.write_bytes(data)


def source_order(record):
    return (
        int(record["source_shard"]),
        int(record["source_local_index"]),
    )


# ======================================================================
# 1. VERIFY FROZEN INPUTS
# ======================================================================

for p in [FIREWALL, PLAN, TARGETS, REPAIR_DESIGN]:
    if not p.is_file():
        raise RuntimeError(f"Missing frozen input: {p}")

if sha256_file(FIREWALL) != EXPECTED_FIREWALL_SHA:
    raise RuntimeError("Prospective generator firewall SHA changed")

if sha256_file(PLAN) != EXPECTED_PLAN_SHA:
    raise RuntimeError("Frozen repair-plan SHA changed")

if sha256_file(TARGETS) != EXPECTED_TARGETS_SHA:
    raise RuntimeError("Frozen repair-target SHA changed")

plan = json.loads(PLAN.read_text())
target_obj = json.loads(TARGETS.read_text())

targets = list(target_obj["targets"])

if len(targets) != 219:
    raise RuntimeError(
        f"Expected 219 repair targets, got {len(targets)}"
    )

targets.sort(key=lambda x: int(x["target_id"]))

if [int(x["target_id"]) for x in targets] != list(range(219)):
    raise RuntimeError("Repair target IDs are not contiguous 0..218")

expected_groups = {
    "sa|sym=True": 69,
    "sqa|sym=True": 150,
}

if target_obj["target_class_counts"] != expected_groups:
    raise RuntimeError(
        "Frozen target-class counts changed"
    )


# ======================================================================
# 2. LOAD PROSPECTIVE FIREWALL
# ======================================================================

firewall_rows = load_list(FIREWALL)

if len(firewall_rows) != 32333:
    raise RuntimeError(
        f"Expected 32333 firewall rows, got {len(firewall_rows)}"
    )

firewall_exact = {
    exact_key(r["pattern"])
    for r in firewall_rows
}

firewall_d4 = {
    d4_key(r["pattern"])
    for r in firewall_rows
}

if len(firewall_exact) != 32333:
    raise RuntimeError("Firewall exact uniqueness failed")

if len(firewall_d4) != 32333:
    raise RuntimeError("Firewall D4 uniqueness failed")


# ======================================================================
# 3. LOAD ALL PRIOR ANCHOR POPULATIONS
# ======================================================================

prior_exact = set()
prior_d4 = set()

prior_rows_checked = 0
prior_files = []

for root in EXTERNAL_BANK_DIRS:

    if not root.is_dir():
        raise RuntimeError(
            f"Required prior anchor directory missing:\n{root}"
        )

    files = (
        sorted(root.glob("sa/anchors_*.json"))
        + sorted(root.glob("sqa/anchors_*.json"))
    )

    if not files:
        raise RuntimeError(
            f"No prior anchor files found under:\n{root}"
        )

    for p in files:

        rows = load_list(p)

        prior_files.append({
            "file": str(p.relative_to(PROJECT)),
            "rows": len(rows),
            "sha256": sha256_file(p),
        })

        for row in rows:
            pat = row["pattern"]

            prior_exact.add(exact_key(pat))
            prior_d4.add(d4_key(pat))

            prior_rows_checked += 1


# ======================================================================
# 4. LOAD ORIGINAL 300 PROSPECTIVE SHARDS
# ======================================================================

raw_records = []
raw_file_hashes = {}

for shard in range(300):

    p = (
        RAW
        / f"shard_{shard}"
        / f"candidate_shard_{shard}.json"
    )

    if not p.is_file():
        raise RuntimeError(f"Missing raw shard: {p}")

    rows = load_list(p)

    if len(rows) != 40:
        raise RuntimeError(
            f"Raw shard {shard}: expected 40 rows, got {len(rows)}"
        )

    raw_file_hashes[str(shard)] = sha256_file(p)

    rep = shard // 10

    for local_idx, row in enumerate(rows):

        strategy = row.get("solver")

        if strategy not in {"sa", "sqa"}:
            raise RuntimeError(
                f"Raw shard {shard}, row {local_idx}: "
                f"bad solver={strategy!r}"
            )

        pat = row["pattern"]

        if len(pat) != 193:
            raise RuntimeError(
                f"Raw shard {shard}, row {local_idx}: "
                "pattern length != 193"
            )

        if not is_hard_feasible(pat):
            raise RuntimeError(
                f"Raw shard {shard}, row {local_idx}: "
                "not hard-feasible"
            )

        raw_records.append({
            "replicate": rep,
            "source_shard": shard,
            "source_local_index": local_idx,
            "strategy": strategy,
            "symmetry_enabled":
                bool(row["symmetry_enabled"]),
            "pattern_sha256": pattern_sha(pat),
            "exact_key": exact_key(pat),
            "d4_key": d4_key(pat),

            # Preserve the original complete row.
            "row": row,
        })

if len(raw_records) != 12000:
    raise RuntimeError(
        f"Expected 12000 original rows, got {len(raw_records)}"
    )

raw_records.sort(key=source_order)

raw_aggregate_sha = sha256_bytes(
    json.dumps(
        raw_file_hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
)


# ======================================================================
# 5. IDENTIFY FROZEN TARGET SLOTS
# ======================================================================

target_by_slot = {}

for target in targets:

    slot = (
        int(target["source_shard"]),
        int(target["source_local_index"]),
    )

    if slot in target_by_slot:
        raise RuntimeError(
            f"Duplicate frozen repair slot: {slot}"
        )

    target_by_slot[slot] = target


if len(target_by_slot) != 219:
    raise RuntimeError("Target slot count != 219")


raw_by_slot = {
    (
        r["source_shard"],
        r["source_local_index"],
    ): r
    for r in raw_records
}

if len(raw_by_slot) != 12000:
    raise RuntimeError("Raw source-slot uniqueness failed")


# Verify every frozen target against the exact original row.
for slot, target in target_by_slot.items():

    if slot not in raw_by_slot:
        raise RuntimeError(
            f"Frozen target slot missing from raw population: {slot}"
        )

    rec = raw_by_slot[slot]

    if rec["pattern_sha256"] != target["pattern_sha256"]:
        raise RuntimeError(
            f"Target pattern SHA mismatch at slot {slot}"
        )

    if rec["strategy"] != target["strategy"]:
        raise RuntimeError(
            f"Target strategy mismatch at slot {slot}"
        )

    if (
        rec["symmetry_enabled"]
        != bool(target["symmetry_enabled"])
    ):
        raise RuntimeError(
            f"Target symmetry mismatch at slot {slot}"
        )


# ======================================================================
# 6. RETAIN ALL NON-TARGET ORIGINAL ROWS
# ======================================================================

retained = [
    r
    for r in raw_records
    if (
        r["source_shard"],
        r["source_local_index"],
    ) not in target_by_slot
]

if len(retained) != 11781:
    raise RuntimeError(
        f"Expected 11781 retained originals, got {len(retained)}"
    )

retained_exact = {
    r["exact_key"]
    for r in retained
}

retained_d4 = {
    r["d4_key"]
    for r in retained
}

if len(retained_exact) != 11781:
    raise RuntimeError(
        "Retained originals are not exact-unique"
    )

if len(retained_d4) != 11781:
    raise RuntimeError(
        "Retained originals are not D4-unique"
    )

if retained_exact & firewall_exact:
    raise RuntimeError(
        "Retained originals overlap prospective firewall exactly"
    )

if retained_d4 & firewall_d4:
    raise RuntimeError(
        "Retained originals overlap prospective firewall by D4"
    )

if retained_exact & prior_exact:
    raise RuntimeError(
        "Retained originals overlap prior anchors exactly"
    )

if retained_d4 & prior_d4:
    raise RuntimeError(
        "Retained originals overlap prior anchors by D4"
    )


# ======================================================================
# 7. LOAD + AUDIT REPAIR40 POOL
# ======================================================================

repair_candidates = []
repair_file_hashes = {}

pool_group_counts = Counter()

for shard in range(300, 340):

    d = REPAIR_RAW / f"shard_{shard}"

    cand = d / f"candidate_shard_{shard}.json"
    checkpoint = d / "checkpoint.json"
    summary = d / "summary.json"

    for p in (cand, checkpoint, summary):
        if not p.is_file():
            raise RuntimeError(
                f"Repair shard {shard} incomplete: missing {p.name}"
            )

    rows = load_list(cand)

    if len(rows) != 40:
        raise RuntimeError(
            f"Repair shard {shard}: expected 40 rows, got {len(rows)}"
        )

    repair_file_hashes[str(shard)] = sha256_file(cand)

    for row_idx, row in enumerate(rows):

        strategy = row.get("solver")

        if strategy not in {"sa", "sqa"}:
            raise RuntimeError(
                f"Repair shard {shard}, row {row_idx}: "
                f"bad solver={strategy!r}"
            )

        symmetry = bool(row["symmetry_enabled"])

        pat = row["pattern"]

        if len(pat) != 193:
            raise RuntimeError(
                f"Repair shard {shard}, row {row_idx}: "
                "pattern length != 193"
            )

        if not is_hard_feasible(pat):
            raise RuntimeError(
                f"Repair shard {shard}, row {row_idx}: "
                "not hard-feasible"
            )

        pool_group_counts[
            (strategy, symmetry)
        ] += 1

        repair_candidates.append({
            "repair_shard": shard,
            "repair_row_index": row_idx,
            "strategy": strategy,
            "symmetry_enabled": symmetry,
            "pattern_sha256": pattern_sha(pat),
            "exact_key": exact_key(pat),
            "d4_key": d4_key(pat),

            # Preserve the complete repair row unchanged.
            "row": row,
        })


if len(repair_candidates) != 1600:
    raise RuntimeError(
        f"Expected 1600 repair rows, got {len(repair_candidates)}"
    )

expected_pool_groups = {
    ("sa", False): 400,
    ("sa", True): 400,
    ("sqa", False): 400,
    ("sqa", True): 400,
}

if dict(pool_group_counts) != expected_pool_groups:
    raise RuntimeError(
        "Unexpected repair-pool class balance:\n"
        f"{dict(pool_group_counts)}"
    )

repair_candidates.sort(
    key=lambda x: (
        x["repair_shard"],
        x["repair_row_index"],
    )
)


pool_audit = {
    "stage":
        "CYCLE5_Q5_PROSPECTIVE360_REPAIR40_POOL_AUDIT",

    "status":
        "COMPLETE_FROZEN_REPAIR_POOL",

    "repair_generation_job":
        REPAIR_JOB_ID,

    "repair_shards":
        40,

    "repair_shard_first":
        300,

    "repair_shard_last":
        339,

    "rows_per_shard":
        40,

    "total_rows":
        len(repair_candidates),

    "group_counts": {
        f"{strategy}|sym={sym}": n
        for (strategy, sym), n
        in sorted(pool_group_counts.items())
    },

    "candidate_shard_sha256":
        repair_file_hashes,

    "repair_pool_files_modified":
        False,

    "performance_based_filtering":
        False,

    "q5_fit_occurs_here":
        False,

    "prospective_openmc_truth_used":
        False,
}

frozen_write(POOL_AUDIT, pool_audit)


# ======================================================================
# 8. EXACT STAGE-7 REPAIR SELECTION
#
# Target order:
#     ascending target_id
#
# Candidate order FOR EACH TARGET:
#     ascending repair shard
#     ascending repair row index
#
# Candidate must:
#   - match strategy
#   - match symmetry_enabled
#   - be hard-feasible
#   - avoid firewall exact/D4
#   - avoid all prior-anchor exact/D4
#   - avoid all retained-original exact/D4
#   - avoid all previously selected repair exact/D4
#
# NO QUBO / CNN / objective / q5 / OpenMC selection.
# ======================================================================

blocked_exact = (
    set(firewall_exact)
    | set(prior_exact)
    | set(retained_exact)
)

blocked_d4 = (
    set(firewall_d4)
    | set(prior_d4)
    | set(retained_d4)
)

selected = []
selected_by_target = {}

selected_exact = set()
selected_d4 = set()

selected_group_counts = Counter()


for target in targets:

    tid = int(target["target_id"])

    strategy = target["strategy"]
    symmetry = bool(target["symmetry_enabled"])

    chosen = None

    # IMPORTANT:
    # Restart scan at beginning for every target.
    # Previously accepted candidates are excluded through blocked sets.
    for cand in repair_candidates:

        if cand["strategy"] != strategy:
            continue

        if cand["symmetry_enabled"] != symmetry:
            continue

        if cand["exact_key"] in blocked_exact:
            continue

        if cand["d4_key"] in blocked_d4:
            continue

        chosen = cand
        break

    if chosen is None:
        raise RuntimeError(
            f"No eligible repair candidate for target {tid}: "
            f"{strategy}, symmetry={symmetry}"
        )

    blocked_exact.add(chosen["exact_key"])
    blocked_d4.add(chosen["d4_key"])

    selected_exact.add(chosen["exact_key"])
    selected_d4.add(chosen["d4_key"])

    selected_group_counts[
        (strategy, symmetry)
    ] += 1

    selected_by_target[tid] = chosen

    selected.append({
        "target_id": tid,

        "target_replicate":
            int(target["replicate"]),

        "target_source_shard":
            int(target["source_shard"]),

        "target_source_local_index":
            int(target["source_local_index"]),

        "target_strategy":
            target["strategy"],

        "target_symmetry_enabled":
            bool(target["symmetry_enabled"]),

        "target_pattern_sha256":
            target["pattern_sha256"],

        "target_primary_reason":
            target["primary_reason"],

        "target_reasons":
            list(target["reasons"]),

        "replacement_shard":
            chosen["repair_shard"],

        "replacement_row_index":
            chosen["repair_row_index"],

        "replacement_strategy":
            chosen["strategy"],

        "replacement_symmetry_enabled":
            chosen["symmetry_enabled"],

        "replacement_pattern_sha256":
            chosen["pattern_sha256"],
    })


if len(selected) != 219:
    raise RuntimeError(
        f"Expected 219 selected repairs, got {len(selected)}"
    )

if len(selected_exact) != 219:
    raise RuntimeError(
        "Selected repairs are not exact-unique"
    )

if len(selected_d4) != 219:
    raise RuntimeError(
        "Selected repairs are not D4-unique"
    )

if selected_group_counts[("sa", True)] != 69:
    raise RuntimeError("Expected 69 selected SA symmetry=True")

if selected_group_counts[("sqa", True)] != 150:
    raise RuntimeError("Expected 150 selected SQA symmetry=True")

if selected_group_counts[("sa", False)] != 0:
    raise RuntimeError("Unexpected SA symmetry=False repair")

if selected_group_counts[("sqa", False)] != 0:
    raise RuntimeError("Unexpected SQA symmetry=False repair")


# Project the final universe before writing files.
projected_exact = set(retained_exact) | set(selected_exact)
projected_d4 = set(retained_d4) | set(selected_d4)

if len(projected_exact) != 12000:
    raise RuntimeError(
        f"Projected final exact unique != 12000: "
        f"{len(projected_exact)}"
    )

if len(projected_d4) != 12000:
    raise RuntimeError(
        f"Projected final D4 unique != 12000: "
        f"{len(projected_d4)}"
    )


selection_record = {
    "stage":
        "CYCLE5_Q5_PROSPECTIVE360_REPAIR_SELECTION",

    "status":
        "FROZEN_DETERMINISTIC_REPAIR_SELECTION",

    "source_repair_plan":
        str(PLAN.relative_to(PROJECT)),

    "source_repair_plan_sha256":
        sha256_file(PLAN),

    "source_target_manifest":
        str(TARGETS.relative_to(PROJECT)),

    "source_target_manifest_sha256":
        sha256_file(TARGETS),

    "source_repair_pool_design":
        str(REPAIR_DESIGN.relative_to(PROJECT)),

    "source_repair_pool_design_sha256":
        sha256_file(REPAIR_DESIGN),

    "repair_pool_audit":
        str(POOL_AUDIT.relative_to(PROJECT)),

    "repair_pool_audit_sha256":
        sha256_file(POOL_AUDIT),

    "repair_generation_job":
        REPAIR_JOB_ID,

    "repair_targets":
        219,

    "selected_replacements":
        219,

    "selected_exact_unique":
        len(selected_exact),

    "selected_d4_unique":
        len(selected_d4),

    "selected_group_counts": {
        "sa|sym=True":
            selected_group_counts[("sa", True)],

        "sqa|sym=True":
            selected_group_counts[("sqa", True)],
    },

    "selection_rule": (
        "Process frozen targets in ascending target_id. "
        "For each target, scan repair candidates in ascending "
        "repair shard ID then repair row index. Candidate must "
        "match strategy and symmetry_enabled exactly, be "
        "hard-feasible, and have exact and canonical-D4 keys "
        "absent from the prospective OpenMC/calibration firewall, "
        "all prior anchor populations, retained original "
        "prospective anchors, and all previously selected repair "
        "candidates. No QUBO energy, CNN prediction, q5 value, "
        "objective value, or OpenMC response enters selection."
    ),

    "performance_based_selection":
        False,

    "q5_based_selection":
        False,

    "prospective_openmc_used":
        False,

    "repair_pool_files_modified":
        False,

    "original_raw_files_modified":
        False,

    "projected_final_rows":
        12000,

    "projected_final_exact_unique":
        len(projected_exact),

    "projected_final_d4_unique":
        len(projected_d4),

    "projected_final_firewall_exact_overlap":
        len(projected_exact & firewall_exact),

    "projected_final_firewall_d4_overlap":
        len(projected_d4 & firewall_d4),

    "projected_final_prior_anchor_exact_overlap":
        len(projected_exact & prior_exact),

    "projected_final_prior_anchor_d4_overlap":
        len(projected_d4 & prior_d4),

    "repair_pool_shard_sha256":
        repair_file_hashes,

    "selections":
        selected,
}

frozen_write(SELECTION, selection_record)


# ======================================================================
# 9. APPLY REPLACEMENTS IN-PLACE
#
# Complete selected repair-candidate row is preserved unchanged.
# ======================================================================

replacement_by_slot = {}

applications = []

for target in targets:

    tid = int(target["target_id"])

    slot = (
        int(target["source_shard"]),
        int(target["source_local_index"]),
    )

    chosen = selected_by_target[tid]

    replacement_by_slot[slot] = chosen["row"]

    applications.append({
        "target_id":
            tid,

        "replicate":
            int(target["replicate"]),

        "target_source_shard":
            int(target["source_shard"]),

        "target_source_local_index":
            int(target["source_local_index"]),

        "target_strategy":
            target["strategy"],

        "target_symmetry_enabled":
            bool(target["symmetry_enabled"]),

        "target_pattern_sha256":
            target["pattern_sha256"],

        "target_reasons":
            list(target["reasons"]),

        "replacement_shard":
            chosen["repair_shard"],

        "replacement_row_index":
            chosen["repair_row_index"],

        "replacement_pattern_sha256":
            chosen["pattern_sha256"],
    })


application_record = {
    "status":
        "FINAL_DETERMINISTIC_REPAIR_APPLIED",

    "source_repair_plan":
        str(PLAN.relative_to(PROJECT)),

    "source_repair_plan_sha256":
        sha256_file(PLAN),

    "source_selection_record":
        str(SELECTION.relative_to(PROJECT)),

    "source_selection_record_sha256":
        sha256_file(SELECTION),

    "repair_generation_job":
        REPAIR_JOB_ID,

    "original_rows":
        12000,

    "repair_targets":
        219,

    "selected_replacements":
        219,

    "target_class_counts":
        {
            "sa|sym=True": 69,
            "sqa|sym=True": 150,
        },

    "selection_rule":
        selection_record["selection_rule"],

    "application_rule": (
        "Each frozen replacement is substituted at the "
        "corresponding original logical source position. "
        "The complete selected repair-candidate row is preserved "
        "unchanged. No target or replacement is chosen using "
        "QUBO energy, CNN predictions, q5 values, objective value, "
        "or OpenMC response."
    ),

    "original_raw_files_modified":
        False,

    "repair_pool_files_modified":
        False,

    "applications":
        applications,
}

frozen_write(APPLICATION, application_record)


# ======================================================================
# 10. BUILD 30 SA + 30 SQA BANKS
# ======================================================================

FINAL.mkdir(parents=True, exist_ok=True)
(FINAL / "sa").mkdir(parents=True, exist_ok=True)
(FINAL / "sqa").mkdir(parents=True, exist_ok=True)

bank_manifest = []

all_final_patterns = []
all_final_rows = []

global_class_counts = Counter()

for rep in range(30):

    rep_records = [
        r
        for r in raw_records
        if r["replicate"] == rep
    ]

    if len(rep_records) != 400:
        raise RuntimeError(
            f"Replicate {rep}: expected 400 raw logical slots, "
            f"got {len(rep_records)}"
        )

    for strategy in ("sa", "sqa"):

        logical = [
            r
            for r in rep_records
            if r["strategy"] == strategy
        ]

        logical.sort(key=source_order)

        if len(logical) != 200:
            raise RuntimeError(
                f"rep={rep} strategy={strategy}: "
                f"expected 200 slots, got {len(logical)}"
            )

        bank_rows = []

        for rec in logical:

            slot = (
                rec["source_shard"],
                rec["source_local_index"],
            )

            if slot in replacement_by_slot:
                row = replacement_by_slot[slot]
            else:
                row = rec["row"]

            # Replacement must preserve experimental initializer class.
            if row["solver"] != strategy:
                raise RuntimeError(
                    f"rep={rep} strategy={strategy}: "
                    "replacement changed solver class"
                )

            bank_rows.append(row)

        if len(bank_rows) != 200:
            raise RuntimeError("Bank row count != 200")

        sym_true = sum(
            bool(r["symmetry_enabled"])
            for r in bank_rows
        )

        sym_false = 200 - sym_true

        if sym_true != 100 or sym_false != 100:
            raise RuntimeError(
                f"rep={rep} strategy={strategy}: "
                f"symmetry balance = {sym_false}/{sym_true}, "
                "expected 100/100"
            )

        for r in bank_rows:

            if not is_hard_feasible(r["pattern"]):
                raise RuntimeError(
                    f"rep={rep} strategy={strategy}: "
                    "final bank contains non-hard-feasible row"
                )

            global_class_counts[
                (
                    strategy,
                    bool(r["symmetry_enabled"]),
                )
            ] += 1

            all_final_patterns.append(r["pattern"])
            all_final_rows.append(r)

        out = (
            FINAL
            / strategy
            / f"anchors_{rep:02d}.json"
        )

        frozen_write(out, bank_rows)

        source_generator_shards = sorted({
            int(r["generation_shard"])
            for r in bank_rows
        })

        bank_manifest.append({
            "replicate":
                rep,

            "strategy":
                strategy,

            "file":
                str(out.relative_to(PROJECT)),

            "rows":
                200,

            "symmetry_false":
                sym_false,

            "symmetry_true":
                sym_true,

            "source_generator_shards":
                source_generator_shards,

            "sha256":
                sha256_file(out),
        })


if len(bank_manifest) != 60:
    raise RuntimeError(
        f"Expected 60 final banks, got {len(bank_manifest)}"
    )

frozen_write(BANK_MANIFEST, bank_manifest)


# ======================================================================
# 11. GLOBAL FINAL-BANK SCIENTIFIC INTEGRITY
# ======================================================================

if len(all_final_rows) != 12000:
    raise RuntimeError(
        f"Expected 12000 final rows, got {len(all_final_rows)}"
    )

final_exact = {
    exact_key(p)
    for p in all_final_patterns
}

final_d4 = {
    d4_key(p)
    for p in all_final_patterns
}

if len(final_exact) != 12000:
    raise RuntimeError(
        f"Final exact uniqueness failed: {len(final_exact)}"
    )

if len(final_d4) != 12000:
    raise RuntimeError(
        f"Final D4 uniqueness failed: {len(final_d4)}"
    )

final_hard_feasible = sum(
    bool(is_hard_feasible(p))
    for p in all_final_patterns
)

if final_hard_feasible != 12000:
    raise RuntimeError(
        f"Only {final_hard_feasible}/12000 final rows hard-feasible"
    )

firewall_exact_overlap = len(
    final_exact & firewall_exact
)

firewall_d4_overlap = len(
    final_d4 & firewall_d4
)

prior_exact_overlap = len(
    final_exact & prior_exact
)

prior_d4_overlap = len(
    final_d4 & prior_d4
)

if firewall_exact_overlap != 0:
    raise RuntimeError(
        "Final banks overlap prospective firewall exactly"
    )

if firewall_d4_overlap != 0:
    raise RuntimeError(
        "Final banks overlap prospective firewall by D4"
    )

if prior_exact_overlap != 0:
    raise RuntimeError(
        "Final banks overlap prior anchors exactly"
    )

if prior_d4_overlap != 0:
    raise RuntimeError(
        "Final banks overlap prior anchors by D4"
    )

expected_final_classes = {
    ("sa", False): 3000,
    ("sa", True): 3000,
    ("sqa", False): 3000,
    ("sqa", True): 3000,
}

if dict(global_class_counts) != expected_final_classes:
    raise RuntimeError(
        "Final global initializer/symmetry balance failed:\n"
        f"{dict(global_class_counts)}"
    )


strict_audit = {
    "stage":
        "CYCLE5_Q5_PROSPECTIVE360_WARMSTART_FINAL_BANK_FREEZE",

    "status":
        "FROZEN_ELIGIBLE_FOR_PROSPECTIVE360_GA",

    "final_replicates":
        30,

    "sa_banks":
        30,

    "sqa_banks":
        30,

    "final_banks":
        60,

    "rows_per_bank":
        200,

    "final_rows":
        12000,

    "symmetry_false_per_bank":
        100,

    "symmetry_true_per_bank":
        100,

    "global_class_counts": {
        f"{strategy}|sym={sym}": n
        for (strategy, sym), n
        in sorted(global_class_counts.items())
    },

    "repair_generation_job":
        REPAIR_JOB_ID,

    "repair_targets":
        219,

    "repair_sa_symmetry_true":
        69,

    "repair_sqa_symmetry_true":
        150,

    "final_exact_unique":
        len(final_exact),

    "final_d4_unique":
        len(final_d4),

    "final_hard_feasible":
        final_hard_feasible,

    "final_firewall_exact_overlap":
        firewall_exact_overlap,

    "final_firewall_d4_overlap":
        firewall_d4_overlap,

    "final_prior_anchor_exact_overlap":
        prior_exact_overlap,

    "final_prior_anchor_d4_overlap":
        prior_d4_overlap,

    "prospective_firewall":
        str(FIREWALL.relative_to(PROJECT)),

    "prospective_firewall_sha256":
        sha256_file(FIREWALL),

    "prior_anchor_rows_checked":
        prior_rows_checked,

    "prior_anchor_files_checked":
        len(prior_files),

    "prior_anchor_exact_union":
        len(prior_exact),

    "prior_anchor_d4_union":
        len(prior_d4),

    "original_raw_candidate_aggregate_sha256":
        raw_aggregate_sha,

    "repair_pool_audit":
        str(POOL_AUDIT.relative_to(PROJECT)),

    "repair_pool_audit_sha256":
        sha256_file(POOL_AUDIT),

    "repair_selection_record":
        str(SELECTION.relative_to(PROJECT)),

    "repair_selection_sha256":
        sha256_file(SELECTION),

    "repair_application":
        str(APPLICATION.relative_to(PROJECT)),

    "repair_application_sha256":
        sha256_file(APPLICATION),

    "bank_manifest":
        str(BANK_MANIFEST.relative_to(PROJECT)),

    "bank_manifest_sha256":
        sha256_file(BANK_MANIFEST),

    "original_raw_files_modified":
        False,

    "repair_pool_files_modified":
        False,

    "performance_based_repair_selection":
        False,

    "q5_fit_occurred_at_this_stage":
        False,

    "residual5_refit_occurred_at_this_stage":
        False,

    "prospective_openmc_truth_used":
        False,

    "GA_runs_occurred_at_this_stage":
        False,

    "ready_for_prospective360_seed_and_manifest_freeze":
        True,
}

frozen_write(STRICT_AUDIT, strict_audit)


# ======================================================================
# 12. FINAL COMPACT RESULT
# ======================================================================

print("=" * 100)
print("CYCLE 5 STEP 12B — FINAL PROSPECTIVE360 WARMSTART BANKS FROZEN")
print("=" * 100)

print()
print("REPAIR")
print("  frozen targets                 = 219")
print("  selected replacements          =", len(selected))
print("  SA symmetry=True               =", selected_group_counts[("sa", True)])
print("  SQA symmetry=True              =", selected_group_counts[("sqa", True)])

print()
print("FINAL BANKS")
print("  replicates                     = 30")
print("  SA banks                       = 30")
print("  SQA banks                      = 30")
print("  rows / bank                    = 200")
print("  final rows                     =", len(all_final_rows))
print("  final exact unique             =", len(final_exact))
print("  final D4 unique                =", len(final_d4))
print("  final hard-feasible            =", final_hard_feasible)

print()
print("FINAL EXCLUSION CHECKS")
print("  prospective firewall exact     =", firewall_exact_overlap)
print("  prospective firewall D4        =", firewall_d4_overlap)
print("  prior-anchor exact             =", prior_exact_overlap)
print("  prior-anchor D4                =", prior_d4_overlap)

print()
print("FROZEN ARTIFACTS")
print("  repair pool audit              =", POOL_AUDIT)
print("  repair pool audit SHA256       =", sha256_file(POOL_AUDIT))
print()
print("  repair selection               =", SELECTION)
print("  repair selection SHA256        =", sha256_file(SELECTION))
print()
print("  repair application             =", APPLICATION)
print("  repair application SHA256      =", sha256_file(APPLICATION))
print()
print("  bank manifest                  =", BANK_MANIFEST)
print("  bank manifest SHA256           =", sha256_file(BANK_MANIFEST))
print()
print("  strict final record            =", STRICT_AUDIT)
print("  strict final SHA256            =", sha256_file(STRICT_AUDIT))

print()
print("Q5 REFIT                        = NO")
print("RESIDUAL5 REFIT                 = NO")
print("PROSPECTIVE OPENMC USED         = NO")
print("GA RUNS                         = NO")

print()
print(
    "CYCLE5_STEP12B_FINAL_BANKS_FROZEN_"
    "READY_FOR_PROSPECTIVE360_MANIFEST"
)
