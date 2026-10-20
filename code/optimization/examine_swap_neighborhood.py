#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

import feasibility_first_search as v3


def exact_k_feasible_swaps(base, k, rng, max_attempts=None):
    """Apply exactly k accepted cross-type swaps while preserving hard feasibility."""
    x = np.asarray(base, dtype=int).copy()

    if not v3.is_hard_feasible(x):
        raise RuntimeError("Base pattern is hard-infeasible.")

    k = int(k)
    max_attempts = (
        max(500, 100 * k)
        if max_attempts is None
        else int(max_attempts)
    )

    accepted = 0
    attempts = 0

    while accepted < k and attempts < max_attempts:
        attempts += 1

        i, j = rng.choice(
            v3.N_CELLS,
            size=2,
            replace=False,
        )

        i = int(i)
        j = int(j)

        if x[i] == x[j]:
            continue

        x[i], x[j] = x[j], x[i]

        if v3.is_hard_feasible(x):
            accepted += 1
        else:
            x[i], x[j] = x[j], x[i]

    if accepted != k:
        return None

    return x


def exhaustive_one_swap(base, forbidden_d4):
    """Enumerate every D4-unique hard-feasible single-swap neighbor."""
    base = np.asarray(base, dtype=int)
    base_key = v3.v3_d4_key(base)

    rows = []
    seen = set()

    for i in range(v3.N_CELLS - 1):
        for j in range(i + 1, v3.N_CELLS):
            if base[i] == base[j]:
                continue

            x = base.copy()
            x[i], x[j] = x[j], x[i]

            if not v3.is_hard_feasible(x):
                continue

            key = v3.v3_d4_key(x)

            if (
                key == base_key
                or key in forbidden_d4
                or key in seen
            ):
                continue

            rows.append(x)
            seen.add(key)

    if not rows:
        return np.empty((0, v3.N_CELLS), dtype=int)

    return np.stack(rows).astype(int)


def sample_k_swap_neighbors(
    base,
    k,
    target,
    rng,
    forbidden_d4,
    blocked_d4,
):
    """Sample target D4-unique exact-k-swap hard-feasible neighbors."""
    target = int(target)

    rows = []
    seen = set(blocked_d4)

    attempts = 0
    max_attempts = target * 100

    while len(rows) < target and attempts < max_attempts:
        attempts += 1

        x = exact_k_feasible_swaps(
            base,
            k,
            rng,
        )

        if x is None:
            continue

        key = v3.v3_d4_key(x)

        if (
            key in forbidden_d4
            or key in seen
        ):
            continue

        rows.append(x)
        seen.add(key)

    if len(rows) != target:
        raise RuntimeError(
            f"Generated only {len(rows)}/{target} "
            f"unique exact-{k}-swap neighbors."
        )

    return np.stack(rows).astype(int)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--round-dir", required=True)
    ap.add_argument("--residual-model-dir", required=True)
    ap.add_argument("--near-archive-json", required=True)
    ap.add_argument("--labelled-json", required=True)
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--sample-2", type=int, default=20000)
    ap.add_argument("--sample-4", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=93017)
    ap.add_argument("--threads", type=int, default=48)

    args = ap.parse_args()

    v3.configure_tensorflow(args.threads)
    tf.keras.utils.set_random_seed(args.seed)

    rng = np.random.default_rng(args.seed)

    round_dir = Path(args.round_dir)
    residual_dir = Path(args.residual_model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Frozen known-OpenMC exclusion.
    # ------------------------------------------------------------
    labelled_path = Path(args.labelled_json)
    labelled_rows = json.loads(labelled_path.read_text())

    forbidden_d4 = {
        tuple(v3.canonical_d4_key(row["pattern"]))
        for row in labelled_rows
    }

    # ------------------------------------------------------------
    # Best current V3 near-feasible pattern.
    # ------------------------------------------------------------
    archive_path = Path(args.near_archive_json)
    archive_rows = json.loads(archive_path.read_text())

    if not archive_rows:
        raise RuntimeError("Near-feasible archive is empty.")

    archive_rows = sorted(
        archive_rows,
        key=lambda r: (
            float(r["normalized_violation"]),
            float(r["objective"]),
        ),
    )

    base_row = archive_rows[0]
    base = np.asarray(base_row["pattern"], dtype=int)

    if not v3.is_hard_feasible(base):
        raise RuntimeError("Best near-feasible archive pattern is hard-infeasible.")

    base_key = v3.v3_d4_key(base)

    if base_key in forbidden_d4:
        raise RuntimeError("Best near-feasible pattern is not prospectively D4-novel.")

    # ------------------------------------------------------------
    # Exact local-neighborhood generation.
    # ------------------------------------------------------------
    one = exhaustive_one_swap(
        base,
        forbidden_d4,
    )

    blocked = {base_key}
    blocked.update(v3.v3_d4_key(x) for x in one)

    two = sample_k_swap_neighbors(
        base,
        k=2,
        target=args.sample_2,
        rng=rng,
        forbidden_d4=forbidden_d4,
        blocked_d4=blocked,
    )

    blocked.update(v3.v3_d4_key(x) for x in two)

    four = sample_k_swap_neighbors(
        base,
        k=4,
        target=args.sample_4,
        rng=rng,
        forbidden_d4=forbidden_d4,
        blocked_d4=blocked,
    )

    groups = [
        ("base", base.reshape(1, -1)),
        ("swap1", one),
        ("swap2", two),
        ("swap4", four),
    ]

    labels = []
    populations = []

    for name, arr in groups:
        arr = np.asarray(arr, dtype=int)

        if len(arr):
            populations.append(arr)
            labels.extend([name] * len(arr))

    population = np.concatenate(
        populations,
        axis=0,
    )

    labels = np.asarray(labels)

    # ------------------------------------------------------------
    # Frozen Round-2 CNN ensemble.
    # ------------------------------------------------------------
    scalers = {
        name: joblib.load(
            round_dir / f"{name}_scaler.pkl"
        )
        for name in ("keff", "fa", "fqr")
    }

    models = [
        tf.keras.models.load_model(
            round_dir / "models" / f"cnn_seed_{seed}.keras",
            custom_objects=v3.CUSTOM_OBJECTS,
            compile=False,
        )
        for seed in v3.CNN_TRAINING_SEEDS
    ]

    residual_metadata_path = (
        residual_dir / "residual_safety_metadata.json"
    )

    residual_metadata = json.loads(
        residual_metadata_path.read_text()
    )

    q = residual_metadata["joint95_conformal_q"]

    residual_models = {
        "keff": joblib.load(
            residual_dir / "residual_keff.joblib"
        ),
        "Fa": joblib.load(
            residual_dir / "residual_Fa.joblib"
        ),
        "Fq_r": joblib.load(
            residual_dir / "residual_Fq_r.joblib"
        ),
    }

    def inverse_outputs(model, x):
        outputs = model(
            tf.constant(x),
            training=False,
        )

        return np.column_stack([
            scalers["keff"].inverse_transform(
                outputs[0].numpy().reshape(-1, 1)
            ).ravel(),
            scalers["fa"].inverse_transform(
                outputs[1].numpy().reshape(-1, 1)
            ).ravel(),
            scalers["fqr"].inverse_transform(
                outputs[2].numpy().reshape(-1, 1)
            ).ravel(),
        ]).astype(np.float64)

    def make_residual_features(pop, mu, sd):
        pop = np.asarray(
            pop,
            dtype=np.float32,
        )

        predicted_j = (
            v3.FA_WEIGHT * mu[:, 1] / v3.FA_LIMIT
            + v3.FQR_WEIGHT * mu[:, 2] / v3.FQR_LIMIT
        )[:, None]

        fa_margin = (
            v3.FA_LIMIT - mu[:, 1]
        )[:, None]

        fqr_margin = (
            v3.FQR_LIMIT - mu[:, 2]
        )[:, None]

        return np.concatenate(
            [
                pop,
                mu.astype(np.float32),
                sd.astype(np.float32),
                predicted_j.astype(np.float32),
                fa_margin.astype(np.float32),
                fqr_margin.astype(np.float32),
            ],
            axis=1,
        )

    # ------------------------------------------------------------
    # Batched exact V3 scoring.
    # ------------------------------------------------------------
    all_mu = []
    all_sd = []
    all_scale = []
    all_kL = []
    all_kU = []
    all_faU = []
    all_fqU = []

    for start in range(
        0,
        len(population),
        args.batch_size,
    ):
        stop = min(
            start + args.batch_size,
            len(population),
        )

        pop = population[start:stop]

        x = v3.patterns_to_onehot_grid(pop)

        per_seed = np.stack(
            [
                inverse_outputs(model, x)
                for model in models
            ],
            axis=0,
        )

        mu = per_seed.mean(axis=0)
        sd = per_seed.std(axis=0, ddof=1)

        residual_x = make_residual_features(
            pop,
            mu,
            sd,
        )

        learned_scale = np.column_stack([
            np.maximum(
                residual_models["keff"].predict(residual_x),
                1e-12,
            ),
            np.maximum(
                residual_models["Fa"].predict(residual_x),
                1e-12,
            ),
            np.maximum(
                residual_models["Fq_r"].predict(residual_x),
                1e-12,
            ),
        ]).astype(float)

        lower_k = (
            mu[:, 0]
            - float(q["keff_lower"])
            * learned_scale[:, 0]
        )

        upper_k = (
            mu[:, 0]
            + float(q["keff_upper"])
            * learned_scale[:, 0]
        )

        upper_fa = (
            mu[:, 1]
            + float(q["Fa_upper"])
            * learned_scale[:, 1]
        )

        upper_fq = (
            mu[:, 2]
            + float(q["Fq_r_upper"])
            * learned_scale[:, 2]
        )

        all_mu.append(mu)
        all_sd.append(sd)
        all_scale.append(learned_scale)
        all_kL.append(lower_k)
        all_kU.append(upper_k)
        all_faU.append(upper_fa)
        all_fqU.append(upper_fq)

    mu = np.concatenate(all_mu)
    sd = np.concatenate(all_sd)
    scale = np.concatenate(all_scale)
    kL = np.concatenate(all_kL)
    kU = np.concatenate(all_kU)
    faU = np.concatenate(all_faU)
    fqU = np.concatenate(all_fqU)

    violation = v3.v3_normalized_physics_violation(
        kL,
        kU,
        faU,
        fqU,
    )

    J = (
        v3.FA_WEIGHT * mu[:, 1] / v3.FA_LIMIT
        + v3.FQR_WEIGHT * mu[:, 2] / v3.FQR_LIMIT
    )

    safe = (
        (violation <= 0.0)
        & (kL >= v3.KEFF_LO)
        & (kU <= v3.KEFF_HI)
        & (faU <= v3.FA_LIMIT)
        & (fqU <= v3.FQR_LIMIT)
    )

    # ------------------------------------------------------------
    # Summaries.
    # ------------------------------------------------------------
    base_violation = float(violation[0])
    base_faU = float(faU[0])

    summary = {
        "status": "COMPLETED",
        "probe_mode": (
            "v3_best_near_local_exact1_sampled2_sampled4"
        ),
        "seed": int(args.seed),
        "base_archive_generation": int(
            base_row["generation"]
        ),
        "base_archive_reported_violation": float(
            base_row["normalized_violation"]
        ),
        "base_rescored_violation": base_violation,
        "base_rescored_Fa_upper": base_faU,
        "counts": {},
        "best_overall": None,
        "provenance": {
            "round_arrays_sha256": hashlib.sha256(
                (round_dir / "round_arrays.npz").read_bytes()
            ).hexdigest(),
            "residual_metadata_sha256": hashlib.sha256(
                residual_metadata_path.read_bytes()
            ).hexdigest(),
            "labelled_exclusion_sha256": hashlib.sha256(
                labelled_path.read_bytes()
            ).hexdigest(),
            "near_archive_sha256": hashlib.sha256(
                archive_path.read_bytes()
            ).hexdigest(),
        },
    }

    def row_dict(i):
        return {
            "index": int(i),
            "group": str(labels[i]),
            "normalized_violation": float(violation[i]),
            "objective": float(J[i]),
            "ensemble_mean_keff": float(mu[i, 0]),
            "ensemble_mean_Fa": float(mu[i, 1]),
            "ensemble_mean_Fq_r": float(mu[i, 2]),
            "learned_scale_keff": float(scale[i, 0]),
            "learned_scale_Fa": float(scale[i, 1]),
            "learned_scale_Fq_r": float(scale[i, 2]),
            "calibrated_keff_lower": float(kL[i]),
            "calibrated_keff_upper": float(kU[i]),
            "calibrated_Fa_upper": float(faU[i]),
            "calibrated_Fq_r_upper": float(fqU[i]),
            "safe": bool(safe[i]),
            "FaU_improvement_vs_base": float(
                base_faU - faU[i]
            ),
            "violation_improvement_vs_base": float(
                base_violation - violation[i]
            ),
            "changed_positions_vs_base": int(
                np.sum(population[i] != base)
            ),
            "pattern": population[i].tolist(),
        }

    for group in ("base", "swap1", "swap2", "swap4"):
        idx = np.flatnonzero(labels == group)

        if not len(idx):
            continue

        order = idx[
            np.lexsort(
                (
                    J[idx],
                    violation[idx],
                    (~safe[idx]).astype(int),
                )
            )
        ]

        best = int(order[0])

        summary["counts"][group] = {
            "n": int(len(idx)),
            "safe": int(np.sum(safe[idx])),
            "better_violation_than_base": int(
                np.sum(
                    violation[idx]
                    < base_violation - 1e-12
                )
            ),
            "better_FaU_than_base": int(
                np.sum(
                    faU[idx]
                    < base_faU - 1e-12
                )
            ),
            "min_violation": float(
                np.min(violation[idx])
            ),
            "median_violation": float(
                np.median(violation[idx])
            ),
            "min_Fa_upper": float(
                np.min(faU[idx])
            ),
            "median_Fa_upper": float(
                np.median(faU[idx])
            ),
            "best": row_dict(best),
        }

    global_order = np.lexsort(
        (
            J,
            violation,
            (~safe).astype(int),
        )
    )

    summary["best_overall"] = row_dict(
        int(global_order[0])
    )

    # Save top 100 local candidates under feasibility-first ranking.
    top_n = min(100, len(population))
    top_rows = [
        row_dict(int(i))
        for i in global_order[:top_n]
    ]

    (output_dir / "local_probe_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    (output_dir / "local_probe_top100.json").write_text(
        json.dumps(
            top_rows,
            indent=2,
        )
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
