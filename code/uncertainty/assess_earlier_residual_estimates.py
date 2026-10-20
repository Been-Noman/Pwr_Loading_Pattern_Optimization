#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error


OUTPUTS = ("keff", "Fa", "Fq_r")


def load_rows(path):
    return json.loads(Path(path).read_text())


def make_xy(rows):
    patterns = np.asarray([r["pattern"] for r in rows], dtype=np.float32)
    mu = np.asarray([r["ensemble_mean"] for r in rows], dtype=np.float32)
    sd = np.asarray([r["ensemble_std"] for r in rows], dtype=np.float32)
    J = np.asarray([r["predicted_J"] for r in rows], dtype=np.float32)[:, None]

    # No round/generation identifier is included, so the model cannot
    # simply memorize which AL round produced a residual.
    X = np.concatenate([patterns, mu, sd, J], axis=1)

    truth = np.asarray([
        [r["openmc_keff"], r["openmc_Fa"], r["openmc_Fq_r"]]
        for r in rows
    ], dtype=np.float64)

    residual_abs = np.abs(truth - mu.astype(np.float64))

    return X, residual_abs, sd.astype(np.float64)


def safe_rho(a, b):
    rho = spearmanr(a, b).statistic
    return None if not np.isfinite(rho) else float(rho)


def top20_audit(actual_error, score):
    n = len(actual_error)
    k = max(1, int(np.ceil(0.20 * n)))

    true_top = set(np.argsort(actual_error)[-k:].tolist())
    pred_top = set(np.argsort(score)[-k:].tolist())

    overlap = len(true_top & pred_top)

    return {
        "k": int(k),
        "overlap": int(overlap),
        "precision_recall_at_20pct": float(overlap / k),
        "random_expectation": 0.20,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1-labels", required=True)
    ap.add_argument("--round2-labels", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    r1 = load_rows(args.round1_labels)
    r2 = load_rows(args.round2_labels)

    if len(r1) != 1000:
        raise RuntimeError(f"Expected 1000 Round-1 rows, got {len(r1)}")
    if len(r2) != 1250:
        raise RuntimeError(f"Expected 1250 Round-2 rows, got {len(r2)}")

    X_train, e_train, sd_train = make_xy(r1)
    X_test, e_test, sd_test = make_xy(r2)

    report = {
        "purpose": (
            "Preliminary residual-model diagnostic only. "
            "Train on honest Round-1 residuals and test on later "
            "honest Round-2 residuals. Not a final safety model."
        ),
        "train_round": "round1",
        "test_round": "round2",
        "train_n": int(len(X_train)),
        "test_n": int(len(X_test)),
        "features": (
            "193 loading cells + ensemble mean(3) + ensemble std(3) "
            "+ predicted J; no round identifier"
        ),
        "models": {},
    }

    for j, name in enumerate(OUTPUTS):
        model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=2026 + j,
            n_jobs=-1,
        )

        model.fit(X_train, e_train[:, j])

        pred_error = np.maximum(
            model.predict(X_test),
            0.0,
        )

        actual = e_test[:, j]
        ensemble_sd = sd_test[:, j]

        report["models"][name] = {
            "actual_error_median": float(np.median(actual)),
            "actual_error_p95": float(np.quantile(actual, 0.95)),

            "residual_model_error_MAE": float(
                mean_absolute_error(actual, pred_error)
            ),

            "rho_actual_error_vs_predicted_error": safe_rho(
                actual, pred_error
            ),

            "rho_actual_error_vs_ensemble_sd": safe_rho(
                actual, ensemble_sd
            ),

            "top20_residual_model": top20_audit(
                actual, pred_error
            ),

            "top20_ensemble_sd": top20_audit(
                actual, ensemble_sd
            ),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print("\nHISTORICAL_RESIDUAL_MODEL_DIAGNOSTIC_OK")


if __name__ == "__main__":
    main()
