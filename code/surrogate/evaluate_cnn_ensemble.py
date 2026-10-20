#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import beta, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

from cnn_architecture import CNN_TRAINING_SEEDS, CUSTOM_OBJECTS, configure_tensorflow
from geometry import FA_LIMIT, FQR_LIMIT, KEFF_HI, KEFF_LO
from data_io import atomic_json


OUTPUT_NAMES = ("keff", "Fa", "Fq_r")


def metric_block(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)) if len(y) > 1 else None,
        "Spearman_rho": float(spearmanr(y, p).statistic) if len(y) > 1 else None,
    }


def inverse_predictions(model, x, scalers, batch_size=256):
    out = model.predict(x, batch_size=batch_size, verbose=0)
    return np.column_stack([
        scalers["keff"].inverse_transform(out[0]).ravel(),
        scalers["fa"].inverse_transform(out[1]).ravel(),
        scalers["fqr"].inverse_transform(out[2]).ravel(),
    ]).astype(np.float64)


def feasible(y):
    return (
        (y[:, 0] >= KEFF_LO)
        & (y[:, 0] <= KEFF_HI)
        & (y[:, 1] <= FA_LIMIT)
        & (y[:, 2] <= FQR_LIMIT)
    )


def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    try:
        q = np.quantile(scores, level, method="higher")
    except TypeError:
        q = np.quantile(scores, level, interpolation="higher")
    return float(max(0.0, q))


def cp_interval(x, n, confidence=0.95):
    if n == 0:
        return [None, None]
    a = 1.0 - confidence
    lo = 0.0 if x == 0 else float(beta.ppf(a / 2.0, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1.0 - a / 2.0, x + 1, n - x))
    return [lo, hi]


def selection_audit(mask, truth):
    n = int(mask.sum())
    false = int(np.sum(mask & ~truth))
    true = int(np.sum(mask & truth))
    return {
        "selected_count": n,
        "true_feasible_selected": true,
        "false_feasible_selected": false,
        "false_feasible_rate": None if n == 0 else float(false / n),
        "false_feasible_rate_95pct_clopper_pearson": cp_interval(false, n),
    }


def adaptive_bounds(mu, sd, q):
    scale = np.maximum(sd, 1e-12)
    return {
        "keff_lower": mu[:, 0] - q["keff_lower"] * scale[:, 0],
        "keff_upper": mu[:, 0] + q["keff_upper"] * scale[:, 0],
        "Fa_upper": mu[:, 1] + q["Fa_upper"] * scale[:, 1],
        "Fq_r_upper": mu[:, 2] + q["Fq_r_upper"] * scale[:, 2],
    }


def calibrated_safe(bounds):
    return (
        (bounds["keff_lower"] >= KEFF_LO)
        & (bounds["keff_upper"] <= KEFF_HI)
        & (bounds["Fa_upper"] <= FA_LIMIT)
        & (bounds["Fq_r_upper"] <= FQR_LIMIT)
    )


def fit_adaptive_conformal(y, mu, sd, alpha_each):
    scale = np.maximum(sd, 1e-12)

    scores = {
        "keff_lower": (mu[:, 0] - y[:, 0]) / scale[:, 0],
        "keff_upper": (y[:, 0] - mu[:, 0]) / scale[:, 0],
        "Fa_upper": (y[:, 1] - mu[:, 1]) / scale[:, 1],
        "Fq_r_upper": (y[:, 2] - mu[:, 2]) / scale[:, 2],
    }

    return {
        key: conformal_quantile(value, alpha_each)
        for key, value in scores.items()
    }


def coverage_audit(y, bounds):
    checks = {
        "keff_lower": y[:, 0] >= bounds["keff_lower"],
        "keff_upper": y[:, 0] <= bounds["keff_upper"],
        "Fa_upper": y[:, 1] <= bounds["Fa_upper"],
        "Fq_r_upper": y[:, 2] <= bounds["Fq_r_upper"],
    }
    joint = np.logical_and.reduce(list(checks.values()))
    result = {k: float(np.mean(v)) for k, v in checks.items()}
    result["joint_all_four_bounds"] = float(np.mean(joint))
    return result


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

    arrays = np.load(round_dir / "round_arrays.npz")

    x_cal = arrays["X_calibration"]
    x_test = arrays["X_test"]

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

    cal_seed = []
    test_seed = []

    for seed in CNN_TRAINING_SEEDS:
        path = round_dir / "models" / f"cnn_seed_{seed}.keras"
        if not path.exists():
            raise SystemExit(f"Missing model: {path}")

        model = tf.keras.models.load_model(
            path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )

        cal_seed.append(inverse_predictions(model, x_cal, scalers))
        test_seed.append(inverse_predictions(model, x_test, scalers))

        del model
        tf.keras.backend.clear_session()

    cal_seed = np.stack(cal_seed, axis=0)
    test_seed = np.stack(test_seed, axis=0)

    cal_mu = cal_seed.mean(axis=0)
    test_mu = test_seed.mean(axis=0)

    cal_sd = cal_seed.std(axis=0, ddof=1)
    test_sd = test_seed.std(axis=0, ddof=1)

    cal_true_feasible = feasible(y_cal)
    test_true_feasible = feasible(y_test)

    raw_cal_safe = feasible(cal_mu)
    raw_test_safe = feasible(test_mu)

    test_metrics = {
        name: metric_block(y_test[:, j], test_mu[:, j])
        for j, name in enumerate(OUTPUT_NAMES)
    }

    feasible_test_metrics = {
        name: metric_block(
            y_test[test_true_feasible, j],
            test_mu[test_true_feasible, j],
        )
        for j, name in enumerate(OUTPUT_NAMES)
    }

    # Two adaptive conformal audits:
    # 1) 95% marginal per one-sided physical bound.
    # 2) Bonferroni alpha/4, giving a conservative >=95% simultaneous target
    #    across the four one-sided bounds.
    q_marginal = fit_adaptive_conformal(
        y_cal, cal_mu, cal_sd, alpha_each=0.05
    )
    q_joint = fit_adaptive_conformal(
        y_cal, cal_mu, cal_sd, alpha_each=0.05 / 4.0
    )

    cal_bounds_marginal = adaptive_bounds(cal_mu, cal_sd, q_marginal)
    test_bounds_marginal = adaptive_bounds(test_mu, test_sd, q_marginal)

    cal_bounds_joint = adaptive_bounds(cal_mu, cal_sd, q_joint)
    test_bounds_joint = adaptive_bounds(test_mu, test_sd, q_joint)

    cal_safe_marginal = calibrated_safe(cal_bounds_marginal)
    test_safe_marginal = calibrated_safe(test_bounds_marginal)

    cal_safe_joint = calibrated_safe(cal_bounds_joint)
    test_safe_joint = calibrated_safe(test_bounds_joint)

    report = {
        "cnn_seeds": list(CNN_TRAINING_SEEDS),
        "calibration_count": int(len(y_cal)),
        "test_count": int(len(y_test)),
        "test_true_physics_feasible": int(test_true_feasible.sum()),
        "test_true_physics_feasible_fraction": float(test_true_feasible.mean()),

        "test_ensemble_metrics": test_metrics,
        "test_true_feasible_subset_metrics": feasible_test_metrics,

        "mean_ensemble_std": {
            "calibration": {
                "keff": float(cal_sd[:, 0].mean()),
                "Fa": float(cal_sd[:, 1].mean()),
                "Fq_r": float(cal_sd[:, 2].mean()),
            },
            "test": {
                "keff": float(test_sd[:, 0].mean()),
                "Fa": float(test_sd[:, 1].mean()),
                "Fq_r": float(test_sd[:, 2].mean()),
            },
        },

        "raw_ensemble": {
            "calibration": selection_audit(raw_cal_safe, cal_true_feasible),
            "test": selection_audit(raw_test_safe, test_true_feasible),
        },

        "adaptive_conformal_marginal_95": {
            "alpha_per_bound": 0.05,
            "q": q_marginal,
            "calibration_coverage": coverage_audit(y_cal, cal_bounds_marginal),
            "test_coverage": coverage_audit(y_test, test_bounds_marginal),
            "calibration_safe_selection": selection_audit(
                cal_safe_marginal, cal_true_feasible
            ),
            "test_safe_selection": selection_audit(
                test_safe_marginal, test_true_feasible
            ),
        },

        "adaptive_conformal_joint_95_bonferroni": {
            "family_alpha": 0.05,
            "alpha_per_bound": 0.05 / 4.0,
            "q": q_joint,
            "calibration_coverage": coverage_audit(y_cal, cal_bounds_joint),
            "test_coverage": coverage_audit(y_test, test_bounds_joint),
            "calibration_safe_selection": selection_audit(
                cal_safe_joint, cal_true_feasible
            ),
            "test_safe_selection": selection_audit(
                test_safe_joint, test_true_feasible
            ),
        },

        "physics_limits": {
            "keff_low": KEFF_LO,
            "keff_high": KEFF_HI,
            "Fa": FA_LIMIT,
            "Fq_r": FQR_LIMIT,
        },

        "interpretation": (
            "The 750-case calibration set is used only to fit adaptive conformal "
            "multipliers. The 1500-case test set remains untouched until this final "
            "audit. No GA go/no-go threshold is invented here; the report exposes "
            "ensemble accuracy, false-feasible behavior, and calibrated safe-set "
            "coverage so the GA decision can be made from observed evidence."
        ),
    }

    np.savez_compressed(
        output_dir / "final_ensemble_predictions.npz",
        calibration_truth=y_cal,
        calibration_per_seed=cal_seed,
        calibration_mean=cal_mu,
        calibration_std=cal_sd,
        test_truth=y_test,
        test_per_seed=test_seed,
        test_mean=test_mu,
        test_std=test_sd,
        test_true_feasible=test_true_feasible,
        raw_test_safe=raw_test_safe,
        marginal95_test_safe=test_safe_marginal,
        joint95_test_safe=test_safe_joint,
    )

    atomic_json(report, output_dir / "final_ensemble_audit.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
