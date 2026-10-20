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


def exhaustive_one_swap(base, forbidden_d4):
    """
    Enumerate every D4-unique, hard-feasible single cross-type swap
    around base, excluding all previously OpenMC-labelled D4 classes.
    """
    base = np.asarray(base, dtype=int)

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

            if key in forbidden_d4 or key in seen:
                continue

            rows.append(x)
            seen.add(key)

    if not rows:
        return np.empty((0, v3.N_CELLS), dtype=int)

    return np.stack(rows).astype(int)


class FrozenV3Scorer:
    def __init__(self, round_dir, residual_dir, threads):
        self.round_dir = Path(round_dir)
        self.residual_dir = Path(residual_dir)

        v3.configure_tensorflow(int(threads))

        self.scalers = {
            name: joblib.load(
                self.round_dir / f"{name}_scaler.pkl"
            )
            for name in ("keff", "fa", "fqr")
        }

        self.models = [
            tf.keras.models.load_model(
                self.round_dir
                / "models"
                / f"cnn_seed_{seed}.keras",
                custom_objects=v3.CUSTOM_OBJECTS,
                compile=False,
            )
            for seed in v3.CNN_TRAINING_SEEDS
        ]

        metadata_path = (
            self.residual_dir
            / "residual_safety_metadata.json"
        )

        self.residual_metadata = json.loads(
            metadata_path.read_text()
        )

        self.q = self.residual_metadata[
            "joint95_conformal_q"
        ]

        self.residual_models = {
            "keff": joblib.load(
                self.residual_dir
                / "residual_keff.joblib"
            ),
            "Fa": joblib.load(
                self.residual_dir
                / "residual_Fa.joblib"
            ),
            "Fq_r": joblib.load(
                self.residual_dir
                / "residual_Fq_r.joblib"
            ),
        }

    def inverse_outputs(self, model, x):
        outputs = model(
            tf.constant(x),
            training=False,
        )

        return np.column_stack([
            self.scalers["keff"].inverse_transform(
                outputs[0].numpy().reshape(-1, 1)
            ).ravel(),

            self.scalers["fa"].inverse_transform(
                outputs[1].numpy().reshape(-1, 1)
            ).ravel(),

            self.scalers["fqr"].inverse_transform(
                outputs[2].numpy().reshape(-1, 1)
            ).ravel(),
        ]).astype(np.float64)

    def make_residual_features(self, population, mu, sd):
        population = np.asarray(
            population,
            dtype=np.float32,
        )

        predicted_j = (
            v3.FA_WEIGHT
            * mu[:, 1]
            / v3.FA_LIMIT
            +
            v3.FQR_WEIGHT
            * mu[:, 2]
            / v3.FQR_LIMIT
        )[:, None]

        fa_margin = (
            v3.FA_LIMIT - mu[:, 1]
        )[:, None]

        fqr_margin = (
            v3.FQR_LIMIT - mu[:, 2]
        )[:, None]

        return np.concatenate(
            [
                population,
                mu.astype(np.float32),
                sd.astype(np.float32),
                predicted_j.astype(np.float32),
                fa_margin.astype(np.float32),
                fqr_margin.astype(np.float32),
            ],
            axis=1,
        )

    def score(self, population, batch_size=2048):
        population = np.asarray(
            population,
            dtype=int,
        )

        if population.ndim == 1:
            population = population.reshape(1, -1)

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
            int(batch_size),
        ):
            stop = min(
                start + int(batch_size),
                len(population),
            )

            pop = population[start:stop]

            x = v3.patterns_to_onehot_grid(pop)

            per_seed = np.stack(
                [
                    self.inverse_outputs(model, x)
                    for model in self.models
                ],
                axis=0,
            )

            mu = per_seed.mean(axis=0)
            sd = per_seed.std(axis=0, ddof=1)

            residual_x = self.make_residual_features(
                pop,
                mu,
                sd,
            )

            learned_scale = np.column_stack([
                np.maximum(
                    self.residual_models[
                        "keff"
                    ].predict(residual_x),
                    1e-12,
                ),
                np.maximum(
                    self.residual_models[
                        "Fa"
                    ].predict(residual_x),
                    1e-12,
                ),
                np.maximum(
                    self.residual_models[
                        "Fq_r"
                    ].predict(residual_x),
                    1e-12,
                ),
            ]).astype(float)

            lower_k = (
                mu[:, 0]
                - float(
                    self.q["keff_lower"]
                )
                * learned_scale[:, 0]
            )

            upper_k = (
                mu[:, 0]
                + float(
                    self.q["keff_upper"]
                )
                * learned_scale[:, 0]
            )

            upper_fa = (
                mu[:, 1]
                + float(
                    self.q["Fa_upper"]
                )
                * learned_scale[:, 1]
            )

            upper_fq = (
                mu[:, 2]
                + float(
                    self.q["Fq_r_upper"]
                )
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

        violation = (
            v3.v3_normalized_physics_violation(
                kL,
                kU,
                faU,
                fqU,
            )
        )

        J = (
            v3.FA_WEIGHT
            * mu[:, 1]
            / v3.FA_LIMIT
            +
            v3.FQR_WEIGHT
            * mu[:, 2]
            / v3.FQR_LIMIT
        )

        safe = (
            (violation <= 0.0)
            & (kL >= v3.KEFF_LO)
            & (kU <= v3.KEFF_HI)
            & (faU <= v3.FA_LIMIT)
            & (fqU <= v3.FQR_LIMIT)
        )

        return {
            "mu": mu,
            "sd": sd,
            "scale": scale,
            "kL": kL,
            "kU": kU,
            "faU": faU,
            "fqU": fqU,
            "violation": violation,
            "J": J,
            "safe": safe,
        }


def record_row(pattern, score, index=0):
    i = int(index)

    return {
        "normalized_violation": float(
            score["violation"][i]
        ),
        "objective": float(
            score["J"][i]
        ),
        "ensemble_mean_keff": float(
            score["mu"][i, 0]
        ),
        "ensemble_mean_Fa": float(
            score["mu"][i, 1]
        ),
        "ensemble_mean_Fq_r": float(
            score["mu"][i, 2]
        ),
        "learned_scale_keff": float(
            score["scale"][i, 0]
        ),
        "learned_scale_Fa": float(
            score["scale"][i, 1]
        ),
        "learned_scale_Fq_r": float(
            score["scale"][i, 2]
        ),
        "calibrated_keff_lower": float(
            score["kL"][i]
        ),
        "calibrated_keff_upper": float(
            score["kU"][i]
        ),
        "calibrated_Fa_upper": float(
            score["faU"][i]
        ),
        "calibrated_Fq_r_upper": float(
            score["fqU"][i]
        ),
        "safe": bool(
            score["safe"][i]
        ),
        "pattern": np.asarray(
            pattern,
            dtype=int,
        ).tolist(),
    }


def dominates(current, candidate, tol):
    """
    Strict V3-style local improvement:
      feasible beats infeasible;
      among infeasible, lower normalized violation;
      among numerically tied violations, lower J;
      among feasible, lower J.
    """
    csafe = bool(current["safe"])
    nsafe = bool(candidate["safe"])

    if nsafe and not csafe:
        return True

    if csafe and not nsafe:
        return False

    if csafe and nsafe:
        return (
            candidate["objective"]
            < current["objective"] - tol
        )

    cv = current["normalized_violation"]
    nv = candidate["normalized_violation"]

    if nv < cv - tol:
        return True

    if abs(nv - cv) <= tol:
        return (
            candidate["objective"]
            < current["objective"] - tol
        )

    return False


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--round-dir",
        required=True,
    )

    ap.add_argument(
        "--residual-model-dir",
        required=True,
    )

    ap.add_argument(
        "--near-archive-json",
        required=True,
    )

    ap.add_argument(
        "--labelled-json",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    ap.add_argument(
        "--threads",
        type=int,
        default=48,
    )

    ap.add_argument(
        "--max-steps",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--improvement-tol",
        type=float,
        default=1e-12,
    )

    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labelled_path = Path(
        args.labelled_json
    )

    labelled_rows = json.loads(
        labelled_path.read_text()
    )

    forbidden_d4 = {
        tuple(
            v3.canonical_d4_key(
                row["pattern"]
            )
        )
        for row in labelled_rows
    }

    archive_path = Path(
        args.near_archive_json
    )

    archive_rows = json.loads(
        archive_path.read_text()
    )

    if not archive_rows:
        raise RuntimeError(
            "Near-feasible archive is empty."
        )

    archive_rows = sorted(
        archive_rows,
        key=lambda r: (
            float(
                r["normalized_violation"]
            ),
            float(
                r["objective"]
            ),
        ),
    )

    base_row = archive_rows[0]

    current_pattern = np.asarray(
        base_row["pattern"],
        dtype=int,
    )

    if not v3.is_hard_feasible(
        current_pattern
    ):
        raise RuntimeError(
            "Starting pattern is hard-infeasible."
        )

    if (
        v3.v3_d4_key(current_pattern)
        in forbidden_d4
    ):
        raise RuntimeError(
            "Starting pattern is not "
            "D4-novel versus known OpenMC."
        )

    scorer = FrozenV3Scorer(
        args.round_dir,
        args.residual_model_dir,
        args.threads,
    )

    current_score_raw = scorer.score(
        current_pattern.reshape(1, -1),
        batch_size=args.batch_size,
    )

    current = record_row(
        current_pattern,
        current_score_raw,
        0,
    )

    history = []

    print(
        "START "
        f"V={current['normalized_violation']:.8f} "
        f"FaU={current['calibrated_Fa_upper']:.8f} "
        f"J={current['objective']:.8f} "
        f"safe={current['safe']}",
        flush=True,
    )

    stop_reason = None

    for step in range(
        1,
        int(args.max_steps) + 1,
    ):
        if current["safe"]:
            stop_reason = (
                "ROBUST_FEASIBILITY_REACHED"
            )
            break

        neighbors = exhaustive_one_swap(
            current_pattern,
            forbidden_d4,
        )

        if len(neighbors) == 0:
            stop_reason = (
                "NO_VALID_ONE_SWAP_NEIGHBORS"
            )
            break

        score = scorer.score(
            neighbors,
            batch_size=args.batch_size,
        )

        order = np.lexsort(
            (
                score["J"],
                score["violation"],
                (~score["safe"]).astype(int),
            )
        )

        best_i = int(order[0])

        best_neighbor = record_row(
            neighbors[best_i],
            score,
            best_i,
        )

        better_count = 0

        for i in range(len(neighbors)):
            candidate = record_row(
                neighbors[i],
                score,
                i,
            )

            if dominates(
                current,
                candidate,
                args.improvement_tol,
            ):
                better_count += 1

        step_record = {
            "step": int(step),
            "neighbors_evaluated": int(
                len(neighbors)
            ),
            "improving_neighbors": int(
                better_count
            ),
            "current": current,
            "best_neighbor": best_neighbor,
            "violation_improvement": float(
                current[
                    "normalized_violation"
                ]
                - best_neighbor[
                    "normalized_violation"
                ]
            ),
            "FaU_improvement": float(
                current[
                    "calibrated_Fa_upper"
                ]
                - best_neighbor[
                    "calibrated_Fa_upper"
                ]
            ),
            "objective_improvement": float(
                current["objective"]
                - best_neighbor["objective"]
            ),
        }

        history.append(step_record)

        print(
            f"STEP {step:02d} "
            f"neighbors={len(neighbors)} "
            f"improving={better_count} "
            f"currentV={current['normalized_violation']:.8f} "
            f"bestV={best_neighbor['normalized_violation']:.8f} "
            f"currentFaU={current['calibrated_Fa_upper']:.8f} "
            f"bestFaU={best_neighbor['calibrated_Fa_upper']:.8f} "
            f"bestJ={best_neighbor['objective']:.8f} "
            f"safe={best_neighbor['safe']}",
            flush=True,
        )

        if not dominates(
            current,
            best_neighbor,
            args.improvement_tol,
        ):
            stop_reason = (
                "ONE_SWAP_LOCAL_OPTIMUM"
            )
            break

        current_pattern = np.asarray(
            best_neighbor["pattern"],
            dtype=int,
        )

        current = best_neighbor

        if current["safe"]:
            stop_reason = (
                "ROBUST_FEASIBILITY_REACHED"
            )
            break

    if stop_reason is None:
        stop_reason = "MAX_STEPS_REACHED"

    result = {
        "status": "COMPLETED",
        "probe_mode": (
            "iterative_exhaustive_"
            "one_swap_v3_descent"
        ),
        "stop_reason": stop_reason,
        "starting_archive_generation": int(
            base_row["generation"]
        ),
        "starting_archive_reported_violation": float(
            base_row["normalized_violation"]
        ),
        "steps_attempted": int(
            len(history)
        ),
        "accepted_moves": int(
            sum(
                1
                for h in history
                if dominates(
                    h["current"],
                    h["best_neighbor"],
                    args.improvement_tol,
                )
            )
        ),
        "start": (
            history[0]["current"]
            if history
            else current
        ),
        "final": current,
        "history": history,
        "provenance": {
            "round_arrays_sha256": hashlib.sha256(
                (
                    Path(args.round_dir)
                    / "round_arrays.npz"
                ).read_bytes()
            ).hexdigest(),

            "residual_metadata_sha256": hashlib.sha256(
                (
                    Path(
                        args.residual_model_dir
                    )
                    / "residual_safety_metadata.json"
                ).read_bytes()
            ).hexdigest(),

            "labelled_exclusion_sha256": hashlib.sha256(
                labelled_path.read_bytes()
            ).hexdigest(),

            "near_archive_sha256": hashlib.sha256(
                archive_path.read_bytes()
            ).hexdigest(),

            "known_openmc_d4_exclusion_count": int(
                len(forbidden_d4)
            ),

            "improvement_tolerance": float(
                args.improvement_tol
            ),
        },
    }

    (
        output_dir
        / "iterative_local_descent.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        "\nFINAL "
        f"stop={stop_reason} "
        f"moves={result['accepted_moves']} "
        f"V={current['normalized_violation']:.8f} "
        f"FaU={current['calibrated_Fa_upper']:.8f} "
        f"FqU={current['calibrated_Fq_r_upper']:.8f} "
        f"kL={current['calibrated_keff_lower']:.8f} "
        f"kU={current['calibrated_keff_upper']:.8f} "
        f"J={current['objective']:.8f} "
        f"safe={current['safe']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
