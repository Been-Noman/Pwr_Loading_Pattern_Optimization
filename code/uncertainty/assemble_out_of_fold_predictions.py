#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cnn_architecture import (
    CNN_TRAINING_SEEDS,
    CUSTOM_OBJECTS,
    configure_tensorflow,
)
from evaluate_cnn_ensemble import inverse_predictions


OUTPUTS = ("keff", "Fa", "Fq_r")


def safe_rho(a, b):
    r = spearmanr(a, b).statistic
    return None if not np.isfinite(r) else float(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--threads", type=int, default=48)
    args = ap.parse_args()

    configure_tensorflow(args.threads)

    fold_root = Path(args.fold_root)
    manifest = json.loads(Path(args.manifest).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(manifest) != 2250:
        raise RuntimeError(
            f"Expected 2250 manifest rows, got {len(manifest)}"
        )

    all_ids = []
    all_patterns = []
    all_truth = []
    all_mu = []
    all_sd = []
    all_fold = []
    all_source_round = []

    fold_reports = {}

    for fold in range(5):
        fold_dir = fold_root / f"fold_{fold}"
        arrays = np.load(
            fold_dir / "round_arrays.npz",
            allow_pickle=False,
        )

        X = np.asarray(
            arrays["X_crossfit_holdout"],
            dtype=np.float32,
        )
        ids = np.asarray(
            arrays["crossfit_holdout_id"],
            dtype=np.int32,
        )
        patterns = np.asarray(
            arrays["crossfit_holdout_patterns"],
            dtype=np.int8,
        )

        truth = np.column_stack([
            arrays["yk_crossfit_holdout_raw"],
            arrays["yfa_crossfit_holdout_raw"],
            arrays["yfqr_crossfit_holdout_raw"],
        ]).astype(np.float64)

        if len(X) != 450:
            raise RuntimeError(
                f"Fold {fold}: expected 450 holdout rows, got {len(X)}"
            )

        scalers = {
            name: joblib.load(
                fold_dir / f"{name}_scaler.pkl"
            )
            for name in ("keff", "fa", "fqr")
        }

        seed_preds = []

        for seed in CNN_TRAINING_SEEDS:
            model_path = (
                fold_dir / "models" / f"cnn_seed_{seed}.keras"
            )

            if not model_path.exists():
                raise RuntimeError(
                    f"Missing model: {model_path}"
                )

            model = tf.keras.models.load_model(
                model_path,
                custom_objects=CUSTOM_OBJECTS,
                compile=False,
            )

            pred = inverse_predictions(
                model,
                X,
                scalers,
            )

            seed_preds.append(pred)

            del model
            tf.keras.backend.clear_session()

        seed_preds = np.stack(seed_preds, axis=0)

        mu = seed_preds.mean(axis=0)
        sd = seed_preds.std(axis=0, ddof=1)

        # Verify IDs, patterns, truths and fold membership against
        # the master manifest.
        for local_i, cfid in enumerate(ids):
            row = manifest[int(cfid)]

            if int(row["crossfit_id"]) != int(cfid):
                raise RuntimeError(
                    f"Manifest ID mismatch at {cfid}"
                )

            if int(row["fold"]) != fold:
                raise RuntimeError(
                    f"Fold mismatch for crossfit_id={cfid}"
                )

            if not np.array_equal(
                np.asarray(row["pattern"], dtype=np.int8),
                patterns[local_i],
            ):
                raise RuntimeError(
                    f"Pattern mismatch for crossfit_id={cfid}"
                )

            expected_truth = np.asarray([
                row["openmc_keff"],
                row["openmc_Fa"],
                row["openmc_Fq_r"],
            ], dtype=np.float64)

            if not np.allclose(
                expected_truth,
                truth[local_i],
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"Truth mismatch for crossfit_id={cfid}"
                )

        err = np.abs(truth - mu)

        fold_reports[str(fold)] = {
            "n": int(len(ids)),
            "MAE": {
                OUTPUTS[j]: float(
                    mean_absolute_error(
                        truth[:, j], mu[:, j]
                    )
                )
                for j in range(3)
            },
            "rho_actual_error_vs_ensemble_sd": {
                OUTPUTS[j]: safe_rho(
                    err[:, j], sd[:, j]
                )
                for j in range(3)
            },
        }

        all_ids.append(ids)
        all_patterns.append(patterns)
        all_truth.append(truth)
        all_mu.append(mu)
        all_sd.append(sd)
        all_fold.append(
            np.full(len(ids), fold, dtype=np.int8)
        )
        all_source_round.extend(
            [manifest[int(i)]["source_round"] for i in ids]
        )

        print(
            f"fold {fold}: assembled {len(ids)} OOF predictions"
        )

    ids = np.concatenate(all_ids)
    patterns = np.concatenate(all_patterns, axis=0)
    truth = np.concatenate(all_truth, axis=0)
    mu = np.concatenate(all_mu, axis=0)
    sd = np.concatenate(all_sd, axis=0)
    folds = np.concatenate(all_fold)

    if len(ids) != 2250:
        raise RuntimeError(
            f"Expected 2250 assembled rows, got {len(ids)}"
        )

    if len(np.unique(ids)) != 2250:
        raise RuntimeError("Duplicate crossfit IDs detected")

    if set(ids.tolist()) != set(range(2250)):
        missing = sorted(set(range(2250)) - set(ids.tolist()))
        extra = sorted(set(ids.tolist()) - set(range(2250)))
        raise RuntimeError(
            f"ID coverage failure. missing={missing[:10]} "
            f"extra={extra[:10]}"
        )

    # Canonical output ordering by crossfit_id.
    order = np.argsort(ids)

    ids = ids[order]
    patterns = patterns[order]
    truth = truth[order]
    mu = mu[order]
    sd = sd[order]
    folds = folds[order]
    source_round = np.asarray(
        all_source_round,
        dtype="U16",
    )[order]

    err = np.abs(truth - mu)

    report = {
        "purpose": (
            "Current-generation 5-fold out-of-fold predictions "
            "for residual/uncertainty modeling."
        ),
        "n": 2250,
        "folds": 5,
        "cnn_seeds_per_fold": list(CNN_TRAINING_SEEDS),
        "strict_oof": True,
        "every_crossfit_id_exactly_once": True,
        "fold_reports": fold_reports,
        "overall": {
            "MAE": {
                OUTPUTS[j]: float(
                    mean_absolute_error(
                        truth[:, j], mu[:, j]
                    )
                )
                for j in range(3)
            },
            "bias_true_minus_pred": {
                OUTPUTS[j]: float(
                    np.mean(truth[:, j] - mu[:, j])
                )
                for j in range(3)
            },
            "rho_actual_error_vs_ensemble_sd": {
                OUTPUTS[j]: safe_rho(
                    err[:, j], sd[:, j]
                )
                for j in range(3)
            },
            "median_abs_error": {
                OUTPUTS[j]: float(
                    np.median(err[:, j])
                )
                for j in range(3)
            },
            "median_ensemble_sd": {
                OUTPUTS[j]: float(
                    np.median(sd[:, j])
                )
                for j in range(3)
            },
        },
    }

    np.savez_compressed(
        output_dir / "crossfit_round2_oof_predictions.npz",
        crossfit_id=ids,
        fold=folds,
        source_round=source_round,
        pattern=patterns,
        truth=truth,
        ensemble_mean=mu,
        ensemble_std=sd,
        abs_error=err,
    )

    (
        output_dir / "crossfit_round2_oof_audit.json"
    ).write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print("\nCROSSFIT_ROUND2_OOF_ASSEMBLY_OK")


if __name__ == "__main__":
    main()
