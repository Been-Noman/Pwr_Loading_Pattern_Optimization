#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

import feasibility_first_search as v3


def exact_k_feasible_swaps(base, k, rng, max_attempts=None):
    x = np.asarray(base, dtype=int).copy()

    if not v3.is_hard_feasible(x):
        raise RuntimeError("Base pattern is hard-infeasible.")

    k = int(k)
    max_attempts = max(1000, 200 * k) if max_attempts is None else int(max_attempts)

    accepted = 0
    attempts = 0

    while accepted < k and attempts < max_attempts:
        attempts += 1

        i, j = rng.choice(v3.N_CELLS, size=2, replace=False)
        i, j = int(i), int(j)

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


def sample_neighbors(base, k, target, rng, forbidden_d4, blocked):
    rows = []
    seen = set(blocked)

    attempts = 0
    max_attempts = int(target) * 200

    while len(rows) < int(target) and attempts < max_attempts:
        attempts += 1

        x = exact_k_feasible_swaps(base, k, rng)

        if x is None:
            continue

        key = v3.v3_d4_key(x)

        if key in forbidden_d4 or key in seen:
            continue

        rows.append(x)
        seen.add(key)

    if len(rows) != int(target):
        raise RuntimeError(
            f"Only generated {len(rows)}/{target} unique {k}-swap neighbors."
        )

    return np.stack(rows).astype(int)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--round-dir", required=True)
    ap.add_argument("--residual-model-dir", required=True)
    ap.add_argument("--descent-json", required=True)
    ap.add_argument("--labelled-json", required=True)
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--sample-2", type=int, default=50000)
    ap.add_argument("--sample-4", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=93018)
    ap.add_argument("--threads", type=int, default=48)

    args = ap.parse_args()

    v3.configure_tensorflow(args.threads)
    tf.keras.utils.set_random_seed(args.seed)

    rng = np.random.default_rng(args.seed)

    round_dir = Path(args.round_dir)
    residual_dir = Path(args.residual_model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labelled_rows = json.loads(Path(args.labelled_json).read_text())

    forbidden_d4 = {
        tuple(v3.canonical_d4_key(row["pattern"]))
        for row in labelled_rows
    }

    descent = json.loads(Path(args.descent_json).read_text())

    base = np.asarray(
        descent["final"]["pattern"],
        dtype=int,
    )

    if not v3.is_hard_feasible(base):
        raise RuntimeError("Descent final pattern is hard-infeasible.")

    base_key = v3.v3_d4_key(base)

    if base_key in forbidden_d4:
        raise RuntimeError("Descent final pattern is not D4-novel.")

    blocked = {base_key}

    swap2 = sample_neighbors(
        base,
        2,
        args.sample_2,
        rng,
        forbidden_d4,
        blocked,
    )

    blocked.update(v3.v3_d4_key(x) for x in swap2)

    swap4 = sample_neighbors(
        base,
        4,
        args.sample_4,
        rng,
        forbidden_d4,
        blocked,
    )

    populations = [
        base.reshape(1, -1),
        swap2,
        swap4,
    ]

    labels = np.asarray(
        ["base"]
        + ["swap2"] * len(swap2)
        + ["swap4"] * len(swap4)
    )

    population = np.concatenate(populations, axis=0)

    scalers = {
        name: joblib.load(round_dir / f"{name}_scaler.pkl")
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

    metadata = json.loads(
        (residual_dir / "residual_safety_metadata.json").read_text()
    )

    q = metadata["joint95_conformal_q"]

    residual_models = {
        "keff": joblib.load(residual_dir / "residual_keff.joblib"),
        "Fa": joblib.load(residual_dir / "residual_Fa.joblib"),
        "Fq_r": joblib.load(residual_dir / "residual_Fq_r.joblib"),
    }

    def inverse_outputs(model, x):
        outputs = model(tf.constant(x), training=False)

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

    def residual_features(pop, mu, sd):
        pop = np.asarray(pop, dtype=np.float32)

        J = (
            v3.FA_WEIGHT * mu[:, 1] / v3.FA_LIMIT
            + v3.FQR_WEIGHT * mu[:, 2] / v3.FQR_LIMIT
        )[:, None]

        return np.concatenate(
            [
                pop,
                mu.astype(np.float32),
                sd.astype(np.float32),
                J.astype(np.float32),
                (v3.FA_LIMIT - mu[:, 1])[:, None].astype(np.float32),
                (v3.FQR_LIMIT - mu[:, 2])[:, None].astype(np.float32),
            ],
            axis=1,
        )

    mus = []
    sds = []
    scales = []
    kLs = []
    kUs = []
    faUs = []
    fqUs = []

    for start in range(0, len(population), args.batch_size):
        stop = min(start + args.batch_size, len(population))

        pop = population[start:stop]
        x = v3.patterns_to_onehot_grid(pop)

        per_seed = np.stack(
            [inverse_outputs(model, x) for model in models],
            axis=0,
        )

        mu = per_seed.mean(axis=0)
        sd = per_seed.std(axis=0, ddof=1)

        rx = residual_features(pop, mu, sd)

        scale = np.column_stack([
            np.maximum(residual_models["keff"].predict(rx), 1e-12),
            np.maximum(residual_models["Fa"].predict(rx), 1e-12),
            np.maximum(residual_models["Fq_r"].predict(rx), 1e-12),
        ])

        kL = mu[:, 0] - float(q["keff_lower"]) * scale[:, 0]
        kU = mu[:, 0] + float(q["keff_upper"]) * scale[:, 0]
        faU = mu[:, 1] + float(q["Fa_upper"]) * scale[:, 1]
        fqU = mu[:, 2] + float(q["Fq_r_upper"]) * scale[:, 2]

        mus.append(mu)
        sds.append(sd)
        scales.append(scale)
        kLs.append(kL)
        kUs.append(kU)
        faUs.append(faU)
        fqUs.append(fqU)

    mu = np.concatenate(mus)
    sd = np.concatenate(sds)
    scale = np.concatenate(scales)
    kL = np.concatenate(kLs)
    kU = np.concatenate(kUs)
    faU = np.concatenate(faUs)
    fqU = np.concatenate(fqUs)

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

    base_v = float(violation[0])
    base_fau = float(faU[0])

    def row(i):
        i = int(i)
        return {
            "group": str(labels[i]),
            "normalized_violation": float(violation[i]),
            "objective": float(J[i]),
            "ensemble_mean_Fa": float(mu[i, 1]),
            "learned_scale_Fa": float(scale[i, 1]),
            "calibrated_Fa_upper": float(faU[i]),
            "calibrated_Fq_r_upper": float(fqU[i]),
            "calibrated_keff_lower": float(kL[i]),
            "calibrated_keff_upper": float(kU[i]),
            "safe": bool(safe[i]),
            "changed_positions_vs_base": int(
                np.sum(population[i] != base)
            ),
            "violation_improvement_vs_base": float(
                base_v - violation[i]
            ),
            "FaU_improvement_vs_base": float(
                base_fau - faU[i]
            ),
            "pattern": population[i].tolist(),
        }

    summary = {
        "status": "COMPLETED",
        "base": row(0),
        "groups": {},
    }

    for group in ("swap2", "swap4"):
        idx = np.flatnonzero(labels == group)

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

        summary["groups"][group] = {
            "n": int(len(idx)),
            "safe_count": int(np.sum(safe[idx])),
            "better_violation_than_base": int(
                np.sum(violation[idx] < base_v - 1e-12)
            ),
            "better_FaU_than_base": int(
                np.sum(faU[idx] < base_fau - 1e-12)
            ),
            "min_violation": float(np.min(violation[idx])),
            "median_violation": float(np.median(violation[idx])),
            "min_Fa_upper": float(np.min(faU[idx])),
            "best": row(best),
        }

    global_order = np.lexsort(
        (
            J,
            violation,
            (~safe).astype(int),
        )
    )

    summary["best_overall"] = row(int(global_order[0]))

    (out_dir / "local_escape_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    top = [
        row(int(i))
        for i in global_order[:100]
    ]

    (out_dir / "local_escape_top100.json").write_text(
        json.dumps(top, indent=2)
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
