#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry import FA_LIMIT, canonical_d4_key


# Frozen Round-2 acquisition allocation approved by user.
N_LOW_J = 400
N_FA_BOUNDARY = 350
N_DISAGREEMENT = 50
N_REVISITED = 150
N_REPRESENTATIVE = 300
N_TOTAL = 1250

SELECTION_SEED = 2026


def strategy_name(mask: int) -> str:
    parts = []
    if mask & 1:
        parts.append("random")
    if mask & 2:
        parts.append("sa")
    if mask & 4:
        parts.append("sqa")
    return "+".join(parts) if parts else "unknown"


def choose_sorted(indices, keys, n):
    """Deterministic lexicographic selection."""
    order = np.lexsort(tuple(np.asarray(k)[indices] for k in keys[::-1]))
    return indices[order[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--candidate-bank", required=True)
    ap.add_argument("--round1-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    scores_path = Path(args.scores)
    bank_path = Path(args.candidate_bank)
    round1_path = Path(args.round1_manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(scores_path, allow_pickle=False)
    bank = np.load(bank_path, allow_pickle=False)

    patterns = np.asarray(z["pattern"], dtype=np.int8)
    mu = np.asarray(z["ensemble_mean"], dtype=float)
    sd = np.asarray(z["ensemble_std"], dtype=float)
    J = np.asarray(z["predicted_J"], dtype=float)
    visit_count = np.asarray(z["visit_count"], dtype=int)
    max_generation = np.asarray(z["max_generation"], dtype=int)
    strategy_mask = np.asarray(z["strategy_mask"], dtype=int)
    seed_count = np.asarray(z["seed_count"], dtype=int)
    source_index = np.asarray(z["source_index"], dtype=int)
    raw_feasible = np.asarray(
        z["raw_predicted_feasible"], dtype=bool
    )

    if len(patterns) != 84250:
        print(
            f"NOTE: scored eligible population contains {len(patterns)} "
            "candidates."
        )

    # ------------------------------------------------------------
    # Exclude every Round-1 OpenMC-labelled pattern.
    # This includes Round-1 train, validation and calibration.
    # ------------------------------------------------------------
    round1 = json.loads(round1_path.read_text())

    round1_keys = {
        tuple(canonical_d4_key(r["pattern"]))
        for r in round1
    }

    keep = np.array([
        tuple(canonical_d4_key(p)) not in round1_keys
        for p in patterns
    ], dtype=bool)

    available = np.flatnonzero(keep)

    if len(available) < N_TOTAL:
        raise RuntimeError("Insufficient candidates after Round-1 exclusion.")

    # ------------------------------------------------------------
    # Diagnostic acquisition quantities.
    # No physical limit is changed.
    # ------------------------------------------------------------

    # User-approved two-sided Fa-boundary distance.
    fa_boundary_distance = np.abs(mu[:, 1] - FA_LIMIT)

    # Unitless disagreement score:
    # average percentile rank of ensemble SD in the three outputs.
    disagreement_components = np.column_stack([
        rankdata(sd[:, j], method="average") / len(sd)
        for j in range(3)
    ])
    disagreement_score = disagreement_components.mean(axis=1)

    selected = []
    stratum = {}

    def take(idx, label):
        for i in idx:
            ii = int(i)
            if ii in stratum:
                raise RuntimeError(f"Duplicate selection: {ii}")
            stratum[ii] = label
            selected.append(ii)

    def remaining():
        if not selected:
            return available.copy()
        used = np.zeros(len(patterns), dtype=bool)
        used[np.asarray(selected, dtype=int)] = True
        return available[~used[available]]

    # 1) 350 lowest predicted-J candidates.
    rem = remaining()
    idx = choose_sorted(
        rem,
        keys=(J, fa_boundary_distance),
        n=N_LOW_J,
    )
    take(idx, "low_J")

    # 2) 300 closest to Fa=2.0 from BOTH sides.
    rem = remaining()
    idx = choose_sorted(
        rem,
        keys=(fa_boundary_distance, J),
        n=N_FA_BOUNDARY,
    )
    take(idx, "Fa_boundary_two_sided")

    # 3) 200 highest ensemble disagreement.
    rem = remaining()
    idx = choose_sorted(
        rem,
        keys=(-disagreement_score, J),
        n=N_DISAGREEMENT,
    )
    take(idx, "high_disagreement")

    # 4) 150 most revisited optimizer candidates.
    rem = remaining()
    idx = choose_sorted(
        rem,
        keys=(-visit_count, -seed_count, -max_generation, J),
        n=N_REVISITED,
    )
    take(idx, "optimizer_revisited")

    # 5) 250 representative/diverse candidates from everything remaining.
    # MiniBatchKMeans is used ONLY as a geometric diversity mechanism
    # on the 193-cell loading vectors. No physics threshold is introduced.
    rem = remaining()

    X = patterns[rem].astype(np.float32)

    km = MiniBatchKMeans(
        n_clusters=N_REPRESENTATIVE,
        random_state=SELECTION_SEED,
        batch_size=4096,
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    reps = []
    for c in range(N_REPRESENTATIVE):
        members_local = np.flatnonzero(labels == c)

        if len(members_local) == 0:
            continue

        Xm = X[members_local]
        d2 = np.sum((Xm - centers[c]) ** 2, axis=1)

        # Tie-break toward lower J only when geometrically equal.
        best_d = np.min(d2)
        tied_local = members_local[
            np.isclose(d2, best_d, rtol=0, atol=1e-12)
        ]

        if len(tied_local) > 1:
            candidate_global = rem[tied_local]
            chosen = candidate_global[
                np.argmin(J[candidate_global])
            ]
        else:
            chosen = rem[tied_local[0]]

        reps.append(int(chosen))

    # Extremely unlikely empty-cluster fallback.
    if len(reps) < N_REPRESENTATIVE:
        already = set(reps)
        pool = [int(i) for i in rem if int(i) not in already]
        need = N_REPRESENTATIVE - len(reps)

        # Deterministic fill using evenly spaced positions after sorting by J.
        pool = np.asarray(pool, dtype=int)
        pool = pool[np.argsort(J[pool], kind="stable")]
        positions = np.linspace(
            0, len(pool) - 1, need, dtype=int
        )
        reps.extend(pool[positions].tolist())

    if len(set(reps)) != N_REPRESENTATIVE:
        raise RuntimeError("Representative selection is not unique.")

    take(np.asarray(reps, dtype=int), "representative_diverse")

    if len(selected) != N_TOTAL:
        raise RuntimeError(
            f"Expected {N_TOTAL}, selected {len(selected)}"
        )

    if len(set(selected)) != N_TOTAL:
        raise RuntimeError("Round-2 selections are not unique.")

    # ------------------------------------------------------------
    # Build OpenMC manifest.
    # Every Round-2 case is TRAINING data.
    # ------------------------------------------------------------
    rows = []

    for new_id, i in enumerate(selected):
        mask = int(strategy_mask[i])

        row = {
            "id": int(new_id),
            "split": "train",
            "acquisition_stratum": stratum[i],

            "pattern": patterns[i].astype(int).tolist(),

            "strategy": strategy_name(mask),
            "strategy_mask": mask,

            # Exact replicate/GA seed are not preserved in the D4-dedup
            # candidate bank. Do not invent them.
            "replicate": None,
            "ga_seed": None,

            "source_generation": int(max_generation[i]),
            "predicted_J": float(J[i]),

            "ensemble_mean": mu[i].astype(float).tolist(),
            "ensemble_std": sd[i].astype(float).tolist(),

            # V2 conformal certification is deliberately NOT assigned here.
            "calibrated_safe": None,

            "raw_predicted_feasible": bool(raw_feasible[i]),

            "uncertainty_score": float(disagreement_score[i]),

            # For Round 2 this explicitly means two-sided |Fa_pred - 2.0|.
            "boundary_margin": float(fa_boundary_distance[i]),

            "visit_count": int(visit_count[i]),
            "seed_count": int(seed_count[i]),

            "candidate_bank_source_index": int(source_index[i]),
        }

        rows.append(row)

    manifest_path = out_dir / "round2_openmc_1250.json"
    manifest_path.write_text(json.dumps(rows, indent=2))

    # Round 2 has no new validation/calibration split.
    (out_dir / "round2_train_openmc_1250.json").write_text(
        json.dumps(rows, indent=2)
    )

    # ------------------------------------------------------------
    # Hard audit.
    # ------------------------------------------------------------
    selected_keys = [
        tuple(canonical_d4_key(r["pattern"]))
        for r in rows
    ]

    round1_overlap = sum(k in round1_keys for k in selected_keys)

    source_eligible = np.asarray(
        bank["eligible_new_AL"][source_index[np.asarray(selected)]],
        dtype=bool,
    )

    strata_counts = Counter(
        r["acquisition_stratum"] for r in rows
    )

    strategy_counts = Counter(
        r["strategy"] for r in rows
    )

    audit = {
        "version": "optimizer_tail_v2_round2",
        "source": "existing_V1_optimizer_bank_rescored_by_V2_round1_ensemble",

        "selection_seed": SELECTION_SEED,

        "allocation_frozen": {
            "low_J": N_LOW_J,
            "Fa_boundary_two_sided": N_FA_BOUNDARY,
            "high_disagreement": N_DISAGREEMENT,
            "optimizer_revisited": N_REVISITED,
            "representative_diverse": N_REPRESENTATIVE,
            "total": N_TOTAL,
        },

        "Fa_boundary_definition": "abs(V2_predicted_Fa - 2.0)",
        "Fa_physics_limit_changed": False,

        "round2_all_split": "train",

        "available_after_round1_exclusion": int(len(available)),
        "selected_count": int(len(rows)),
        "unique_D4_selected": int(len(set(selected_keys))),
        "D4_overlap_round1_1250": int(round1_overlap),
        "all_source_candidates_marked_eligible_new_AL": bool(
            np.all(source_eligible)
        ),

        "stratum_counts": dict(strata_counts),
        "strategy_counts": dict(strategy_counts),

        "raw_predicted_feasible_selected": int(sum(
            r["raw_predicted_feasible"] for r in rows
        )),

        "predicted_J": {
            "min": float(np.min(J[selected])),
            "median": float(np.median(J[selected])),
            "max": float(np.max(J[selected])),
        },

        "Fa_boundary_distance": {
            "min": float(np.min(fa_boundary_distance[selected])),
            "median": float(np.median(fa_boundary_distance[selected])),
            "max": float(np.max(fa_boundary_distance[selected])),
        },

        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),

        "safeguards": [
            "No GA mechanics changed.",
            "No physical limit changed.",
            "No conformal-safe label invented.",
            "Round-1 train/validation/calibration patterns excluded.",
            "All 1250 Round-2 labels are reserved for training.",
        ],
    }

    if audit["selected_count"] != N_TOTAL:
        raise RuntimeError("Wrong total selection count.")
    if audit["unique_D4_selected"] != N_TOTAL:
        raise RuntimeError("D4 duplication detected.")
    if audit["D4_overlap_round1_1250"] != 0:
        raise RuntimeError("Round-1 leakage detected.")
    if not audit["all_source_candidates_marked_eligible_new_AL"]:
        raise RuntimeError("Non-eligible candidate selected.")

    expected = {
        "low_J": 400,
        "Fa_boundary_two_sided": 350,
        "high_disagreement": 50,
        "optimizer_revisited": 150,
        "representative_diverse": 300,
    }

    if dict(strata_counts) != expected:
        raise RuntimeError(
            f"Stratum mismatch: {dict(strata_counts)}"
        )

    audit_path = out_dir / "round2_selection_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))

    print(json.dumps(audit, indent=2))
    print("ROUND2_EXISTING_BANK_SELECTION_OK")


if __name__ == "__main__":
    main()
