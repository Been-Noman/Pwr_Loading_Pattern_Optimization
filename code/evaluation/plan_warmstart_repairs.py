#!/usr/bin/env python3

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__import__("os").environ["PWR_PROJECT"])
sys.path.insert(0, str(PROJECT / "scripts"))

from geometry import canonical_d4_key

PROS = (
    PROJECT
    / "run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360"
)
RAW = PROS / "warmstarts/fresh300/raw"
PROTOCOL = PROS / "protocol"

FIREWALL = (
    PROTOCOL
    / "cycle5_q5_prospective360_generator_exact_unique_exclusion_v1.json"
)

OUT = PROTOCOL / "cycle5_q5_prospective360_warmstart_repair_plan_v1.json"
TARGETS_OUT = (
    PROTOCOL
    / "cycle5_q5_prospective360_warmstart_repair_targets_v1.json"
)

# ------------------------------------------------------------------
# Previous anchor populations that must not be reused in this fresh
# prospective experiment.
#
# This extends the exact Stage-7 rule:
#   historical anchors
#   bridge anchors
#   common-q calibration anchors
#   q5 calibration anchors
# ------------------------------------------------------------------

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


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def hard_feasible(pattern):
    # Same geometry validity is enforced by the generator itself.
    # Here we require the exact standard 193-gene legal representation.
    if len(pattern) != 193:
        return False
    return all(int(x) in (1, 2, 3) for x in pattern)


# ==================================================================
# 1. Frozen prospective OpenMC/calibration firewall
# ==================================================================

if not FIREWALL.is_file():
    raise RuntimeError(f"Missing prospective firewall: {FIREWALL}")

firewall_rows = load_list(FIREWALL)

if len(firewall_rows) != 32333:
    raise RuntimeError(
        f"Expected 32333 generator-firewall rows, got {len(firewall_rows)}"
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
    raise RuntimeError("Prospective generator firewall is not exact-unique")

if len(firewall_d4) != 32333:
    raise RuntimeError("Prospective generator firewall is not D4-unique")


# ==================================================================
# 2. Load every earlier frozen anchor bank
# ==================================================================

external_exact = set()
external_d4 = set()
external_files = []
external_source_counts = {}

for root in EXTERNAL_BANK_DIRS:

    label = str(root.relative_to(PROJECT))

    if not root.is_dir():
        raise RuntimeError(
            f"Required prior anchor directory missing:\n{root}"
        )

    files = sorted(root.glob("sa/anchors_*.json")) + \
            sorted(root.glob("sqa/anchors_*.json"))

    if not files:
        raise RuntimeError(f"No anchor banks found under {root}")

    nrows = 0

    for p in files:
        rows = load_list(p)

        external_files.append({
            "file": str(p.relative_to(PROJECT)),
            "rows": len(rows),
            "sha256": sha256_file(p),
        })

        for r in rows:
            pat = r["pattern"]

            external_exact.add(exact_key(pat))
            external_d4.add(d4_key(pat))
            nrows += 1

    external_source_counts[label] = {
        "files": len(files),
        "rows": nrows,
    }


# ==================================================================
# 3. Load fresh prospective raw population in EXACT source order
#
# replicate r owns shards:
#     10*r ... 10*r+9
#
# Each shard has 40 rows.
# Final logical banks:
#     200 SA
#     200 SQA
#
# Ordering preserved by:
#     (source_shard, source_local_index)
# ==================================================================

raw_records = []

for shard in range(300):

    p = RAW / f"shard_{shard}" / f"candidate_shard_{shard}.json"

    if not p.is_file():
        raise RuntimeError(f"Missing raw shard: {p}")

    rows = load_list(p)

    if len(rows) != 40:
        raise RuntimeError(
            f"Shard {shard}: expected 40 rows, got {len(rows)}"
        )

    rep = shard // 10

    for local_idx, row in enumerate(rows):

        strategy = row.get("solver")

        if strategy not in {"sa", "sqa"}:
            raise RuntimeError(
                f"Shard {shard} row {local_idx}: bad solver={strategy}"
            )

        pat = row["pattern"]

        if not hard_feasible(pat):
            raise RuntimeError(
                f"Shard {shard} row {local_idx}: malformed pattern"
            )

        raw_records.append({
            "replicate": rep,
            "source_shard": shard,
            "source_local_index": local_idx,
            "strategy": strategy,
            "symmetry_enabled": bool(row["symmetry_enabled"]),
            "pattern": pat,
            "pattern_sha256": pattern_sha(pat),
            "exact_key": exact_key(pat),
            "d4_key": d4_key(pat),
        })


if len(raw_records) != 12000:
    raise RuntimeError(
        f"Expected 12000 raw candidates, got {len(raw_records)}"
    )


# ==================================================================
# 4. Exact Stage-7 keeper rule
#
# For each raw D4 class:
#
# - if externally excluded -> retain none
# - otherwise retain earliest occurrence ordered by
#       (source_shard, source_local_index)
#
# Any later member of same raw D4 class becomes a repair target.
# ==================================================================

raw_records.sort(
    key=lambda r: (
        r["source_shard"],
        r["source_local_index"],
    )
)

seen_raw_d4 = set()

retained = []
invalid = []

reason_counts = Counter()

for rec in raw_records:

    reasons = []

    if rec["d4_key"] in firewall_d4:
        reasons.append("prospective_openmc_firewall_d4_overlap")

    if rec["d4_key"] in external_d4:
        reasons.append("prior_anchor_d4_overlap")

    if rec["d4_key"] in seen_raw_d4:
        reasons.append("internal_d4_duplicate")

    # Important:
    # first occurrence establishes the raw-class keeper position.
    # This matches the prior deterministic source-order rule.
    if rec["d4_key"] not in seen_raw_d4:
        seen_raw_d4.add(rec["d4_key"])

    if reasons:

        if "prospective_openmc_firewall_d4_overlap" in reasons:
            primary = "prospective_openmc_firewall_d4_overlap"
        elif "prior_anchor_d4_overlap" in reasons:
            primary = "prior_anchor_d4_overlap"
        else:
            primary = "internal_d4_duplicate"

        x = {
            "replicate": rec["replicate"],
            "source_shard": rec["source_shard"],
            "source_local_index": rec["source_local_index"],
            "strategy": rec["strategy"],
            "symmetry_enabled": rec["symmetry_enabled"],
            "pattern_sha256": rec["pattern_sha256"],
            "primary_reason": primary,
            "reasons": reasons,
        }

        invalid.append(x)

        for reason in reasons:
            reason_counts[reason] += 1

    else:
        retained.append(rec)


# ==================================================================
# 5. Check replicate bank structure
# ==================================================================

raw_by_rep_strategy = Counter(
    (r["replicate"], r["strategy"])
    for r in raw_records
)

for rep in range(30):
    for strategy in ("sa", "sqa"):
        n = raw_by_rep_strategy[(rep, strategy)]
        if n != 200:
            raise RuntimeError(
                f"rep={rep} strategy={strategy}: "
                f"expected 200 raw rows, got {n}"
            )


# ==================================================================
# 6. Freeze replacement targets in deterministic target order
# ==================================================================

invalid.sort(
    key=lambda r: (
        r["source_shard"],
        r["source_local_index"],
    )
)

targets = []

for tid, x in enumerate(invalid):
    targets.append({
        "target_id": tid,
        **x,
    })


target_class_counts = Counter(
    (
        t["strategy"],
        bool(t["symmetry_enabled"]),
    )
    for t in targets
)

targets_by_rep = Counter(
    t["replicate"]
    for t in targets
)


# ==================================================================
# 7. Freeze repair plan
# ==================================================================

plan = {
    "stage":
        "CYCLE5_Q5_PROSPECTIVE360_WARMSTART_REPAIR_PLAN",

    "status":
        "FROZEN_BEFORE_REPAIR_GENERATION",

    "methodology":
        "Exact Stage-7 deterministic source-order keeper/repair rule",

    "GA_runs_occur_at_this_stage": False,
    "q5_refit_occurs_at_this_stage": False,
    "OpenMC_prospective_truth_used": False,
    "performance_based_keeper_selection": False,
    "performance_based_repair_selection": False,

    "raw_candidate_files": 300,
    "raw_rows": len(raw_records),

    "replicates": 30,
    "shards_per_replicate": 10,
    "rows_per_replicate": 400,
    "sa_rows_per_replicate": 200,
    "sqa_rows_per_replicate": 200,

    "keeper_rule": (
        "For every canonical-D4 class absent from every frozen "
        "external exclusion set, retain the earliest source occurrence "
        "ordered by (source_shard, source_local_index). "
        "Externally excluded D4 classes are retained zero times."
    ),

    "replacement_rule": (
        "Each invalid logical slot must receive a fresh hard-feasible "
        "candidate matching strategy and symmetry_enabled exactly. "
        "Repair candidates are considered in ascending repair shard ID "
        "then ascending repair row index. Candidate exact and D4 keys "
        "must be absent from the prospective firewall, every prior "
        "anchor population, every retained original prospective anchor, "
        "and every previously accepted repair candidate. No QUBO energy, "
        "CNN score, q5 score, objective value, or OpenMC response may "
        "affect replacement selection."
    ),

    "prospective_firewall": str(FIREWALL.relative_to(PROJECT)),
    "prospective_firewall_sha256": sha256_file(FIREWALL),
    "prospective_firewall_exact_unique": len(firewall_exact),
    "prospective_firewall_d4_unique": len(firewall_d4),

    "prior_anchor_sources": external_source_counts,
    "prior_anchor_files": len(external_files),
    "prior_anchor_exact_union": len(external_exact),
    "prior_anchor_d4_union": len(external_d4),

    "raw_exact_unique": len({
        r["exact_key"]
        for r in raw_records
    }),

    "raw_d4_unique": len({
        r["d4_key"]
        for r in raw_records
    }),

    "retained_original_rows": len(retained),

    "total_replacement_slots": len(targets),

    "replicates_requiring_repair": len(targets_by_rep),

    "maximum_replacements_in_one_replicate":
        max(targets_by_rep.values(), default=0),

    "reason_membership_counts": dict(reason_counts),

    "replacement_requirements_by_group": {
        f"{strategy}|sym={sym}": n
        for (strategy, sym), n
        in sorted(target_class_counts.items())
    },

    "replacement_requirements_by_replicate": {
        f"{rep:02d}": targets_by_rep.get(rep, 0)
        for rep in range(30)
    },

    "invalid_slots": targets,
}

PROTOCOL.mkdir(parents=True, exist_ok=True)

OUT.write_text(
    json.dumps(plan, indent=2, sort_keys=True) + "\n"
)

targets_obj = {
    "status": "FROZEN_REPAIR_TARGETS",

    "source_repair_plan": str(OUT.relative_to(PROJECT)),
    "source_repair_plan_sha256": sha256_file(OUT),

    "selection_order":
        "ascending (source_shard, source_local_index)",

    "replacement_selection_rule":
        plan["replacement_rule"],

    "total_repair_targets": len(targets),

    "target_class_counts":
        plan["replacement_requirements_by_group"],

    "targets": targets,
}

TARGETS_OUT.write_text(
    json.dumps(targets_obj, indent=2, sort_keys=True) + "\n"
)


# ==================================================================
# 8. Compact output only
# ==================================================================

print("=" * 96)
print("CYCLE 5 STEP 12B — PROSPECTIVE WARMSTART REPAIR PLAN FROZEN")
print("=" * 96)

print()
print("RAW")
print("  rows                         =", len(raw_records))
print("  exact unique                 =", plan["raw_exact_unique"])
print("  D4 unique                    =", plan["raw_d4_unique"])

print()
print("EXCLUSIONS")
print("  prospective firewall D4      =", len(firewall_d4))
print("  previous-anchor D4 union     =", len(external_d4))

print()
print("REPAIR")
print("  retained original rows       =", len(retained))
print("  replacement slots            =", len(targets))
print("  replicates needing repair    =", len(targets_by_rep))
print(
    "  max replacements / replicate =",
    plan["maximum_replacements_in_one_replicate"],
)

print()
print("REASONS")
for k, v in sorted(reason_counts.items()):
    print(f"  {k:32s} = {v}")

print()
print("GROUP REQUIREMENTS")
for k, v in sorted(
    plan["replacement_requirements_by_group"].items()
):
    print(f"  {k:18s} = {v}")

print()
print("FROZEN")
print("  repair plan                  =", OUT)
print("  repair plan SHA256           =", sha256_file(OUT))
print("  repair targets               =", TARGETS_OUT)
print("  repair targets SHA256        =", sha256_file(TARGETS_OUT))

print()
print("Q5 REFIT                      = NO")
print("PROSPECTIVE OPENMC USED       = NO")
print("GA RUNS                       = NO")
print()
print("CYCLE5_STEP12B_REPAIR_PLAN_FROZEN")
