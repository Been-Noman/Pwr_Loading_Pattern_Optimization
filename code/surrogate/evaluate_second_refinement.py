#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cnn_architecture import (
    CNN_TRAINING_SEEDS, CUSTOM_OBJECTS, configure_tensorflow
)
from geometry import FA_LIMIT, FQR_LIMIT, KEFF_HI, KEFF_LO
from evaluate_cnn_ensemble import (
    inverse_predictions,
    feasible,
    fit_adaptive_conformal,
    adaptive_bounds,
    calibrated_safe,
    coverage_audit,
    selection_audit,
)
from data_io import atomic_json


OUTPUT_NAMES = ("keff", "Fa", "Fq_r")


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)) if len(y) > 1 else None,
        "Spearman_rho": (
            float(spearmanr(y, p).statistic) if len(y) > 1 else None
        ),
        "Pearson_r": (
            float(pearsonr(y, p).statistic) if len(y) > 1 else None
        ),
        "bias_true_minus_pred": float(np.mean(y - p)),
    }


def objective(y):
    return (
        0.5 * y[:, 1] / FA_LIMIT
        + 0.5 * y[:, 2] / FQR_LIMIT
    )


def prediction_metrics(y, mu):
    out = {
        name: metrics(y[:, j], mu[:, j])
        for j, name in enumerate(OUTPUT_NAMES)
    }

    out["J"] = metrics(objective(y), objective(mu))
    return out


def evaluate_split(y, mu, sd, q_marginal, q_joint):
    true_safe = feasible(y)
    raw_safe = feasible(mu)

    marginal_bounds = adaptive_bounds(mu, sd, q_marginal)
    joint_bounds = adaptive_bounds(mu, sd, q_joint)

    marginal_safe = calibrated_safe(marginal_bounds)
    joint_safe = calibrated_safe(joint_bounds)

    return {
        "n": int(len(y)),
        "true_feasible_count": int(true_safe.sum()),
        "true_feasible_fraction": float(true_safe.mean()),

        "ensemble_metrics": prediction_metrics(y, mu),

        "mean_ensemble_std": {
            "keff": float(sd[:, 0].mean()),
            "Fa": float(sd[:, 1].mean()),
            "Fq_r": float(sd[:, 2].mean()),
        },

        "raw_safe_selection": selection_audit(
            raw_safe, true_safe
        ),

        "marginal95": {
            "coverage": coverage_audit(y, marginal_bounds),
            "safe_selection": selection_audit(
                marginal_safe, true_safe
            ),
        },

        "joint95_bonferroni": {
            "coverage": coverage_audit(y, joint_bounds),
            "safe_selection": selection_audit(
                joint_safe, true_safe
            ),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--threads", type=int, default=48)
    args = ap.parse_args()

    configure_tensorflow(args.threads)

    round_dir = Path(args.round_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = np.load(
        round_dir / "round_arrays.npz",
        allow_pickle=False
    )

    X_val = arrays["X_val"]
    X_cal = arrays["X_calibration"]
    X_test = arrays["X_test"]

    y_val = np.column_stack([
        arrays["yk_val_raw"],
        arrays["yfa_val_raw"],
        arrays["yfqr_val_raw"],
    ]).astype(np.float64)

    y_cal = np.column_stack([
        arrays["yk_calibration_raw"],
        arrays["yfa_calibration_raw"],
        arrays["yfqr_calibration_raw"],
    ]).astype(np.float64)

    y_test = np.column_stack([
        arrays["yk_test_raw"],
        arrays["yfa_test_raw"],
        arrays["yfqr_test_raw"],
    ]).astype(np.float64)

    scalers = {
        name: joblib.load(round_dir / f"{name}_scaler.pkl")
        for name in ("keff", "fa", "fqr")
    }

    val_seed = []
    cal_seed = []
    test_seed = []

    for seed in CNN_TRAINING_SEEDS:
        model_path = (
            round_dir / "models" / f"cnn_seed_{seed}.keras"
        )

        if not model_path.exists():
            raise SystemExit(f"Missing model: {model_path}")

        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )

        val_seed.append(
            inverse_predictions(model, X_val, scalers)
        )
        cal_seed.append(
            inverse_predictions(model, X_cal, scalers)
        )
        test_seed.append(
            inverse_predictions(model, X_test, scalers)
        )

        del model
        tf.keras.backend.clear_session()

    val_seed = np.stack(val_seed, axis=0)
    cal_seed = np.stack(cal_seed, axis=0)
    test_seed = np.stack(test_seed, axis=0)

    val_mu = val_seed.mean(axis=0)
    cal_mu = cal_seed.mean(axis=0)
    test_mu = test_seed.mean(axis=0)

    val_sd = val_seed.std(axis=0, ddof=1)
    cal_sd = cal_seed.std(axis=0, ddof=1)
    test_sd = test_seed.std(axis=0, ddof=1)

    # Fit ONLY on the dedicated tail calibration set.
    q_marginal = fit_adaptive_conformal(
        y_cal, cal_mu, cal_sd,
        alpha_each=0.05,
    )

    q_joint = fit_adaptive_conformal(
        y_cal, cal_mu, cal_sd,
        alpha_each=0.05 / 4.0,
    )

    report = {
        "version": "optimizer_tail_v2_round2",

        "cnn_seeds": list(CNN_TRAINING_SEEDS),

        "roles": {
            "tail_calibration": int(len(y_cal)),
            "tail_validation": int(len(y_val)),
            "untouched_broad_test": int(len(y_test)),
        },

        "adaptive_conformal_marginal_95": {
            "alpha_per_bound": 0.05,
            "q": q_marginal,
        },

        "adaptive_conformal_joint_95_bonferroni": {
            "family_alpha": 0.05,
            "alpha_per_bound": 0.05 / 4.0,
            "q": q_joint,
        },

        "tail_calibration_fit_set": evaluate_split(
            y_cal, cal_mu, cal_sd,
            q_marginal, q_joint,
        ),

        "tail_validation_independent": evaluate_split(
            y_val, val_mu, val_sd,
            q_marginal, q_joint,
        ),

        "broad_test_untouched": evaluate_split(
            y_test, test_mu, test_sd,
            q_marginal, q_joint,
        ),

        "physics_limits": {
            "keff_low": KEFF_LO,
            "keff_high": KEFF_HI,
            "Fa": FA_LIMIT,
            "Fq_r": FQR_LIMIT,
        },

        "interpretation": (
            "Conformal multipliers are fit only on the 125-case "
            "optimizer-tail calibration set. The 125-case optimizer-tail "
            "validation set is independent and is the primary Round-1 "
            "optimizer-domain audit. The original 1500-case V1 broad test "
            "set remains untouched and is used only to detect degradation "
            "of broad-domain surrogate performance."
        ),
    }

    atomic_json(
        report,
        output_dir / "round2_ensemble_audit.json",
    )

    np.savez_compressed(
        output_dir / "round2_ensemble_predictions.npz",

        tail_validation_truth=y_val,
        tail_validation_per_seed=val_seed,
        tail_validation_mean=val_mu,
        tail_validation_std=val_sd,

        tail_calibration_truth=y_cal,
        tail_calibration_per_seed=cal_seed,
        tail_calibration_mean=cal_mu,
        tail_calibration_std=cal_sd,

        broad_test_truth=y_test,
        broad_test_per_seed=test_seed,
        broad_test_mean=test_mu,
        broad_test_std=test_sd,
    )

    print(json.dumps(report, indent=2))
    print("V2_ROUND1_ENSEMBLE_EVALUATION_OK")


if __name__ == "__main__":
    main()
