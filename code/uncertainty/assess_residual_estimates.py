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

FA_LIMIT = 2.0
FQR_LIMIT = 2.5


def safe_rho(a, b):
    r = spearmanr(a, b).statistic
    return None if not np.isfinite(r) else float(r)


def top_fraction_audit(actual, score, frac=0.20):
    n = len(actual)
    k = max(1, int(np.ceil(frac * n)))

    true_top = set(np.argsort(actual)[-k:])
    pred_top = set(np.argsort(score)[-k:])

    overlap = len(true_top & pred_top)

    return {
        "k": int(k),
        "overlap": int(overlap),
        "precision_recall": float(overlap / k),
        "random_expectation": float(frac),
    }


def make_features(pattern, mu, sd):
    J = (
        0.5 * mu[:, 1] / FA_LIMIT
        + 0.5 * mu[:, 2] / FQR_LIMIT
    )[:, None]

    # Distance from the two peaking-factor safety boundaries.
    fa_margin = (FA_LIMIT - mu[:, 1])[:, None]
    fqr_margin = (FQR_LIMIT - mu[:, 2])[:, None]

    # No fold ID, source-round ID, or OpenMC truth is included.
    return np.concatenate(
        [
            pattern.astype(np.float32),
            mu.astype(np.float32),
            sd.astype(np.float32),
            J.astype(np.float32),
            fa_margin.astype(np.float32),
            fqr_margin.astype(np.float32),
        ],
        axis=1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=False)

    ids = np.asarray(z["crossfit_id"], dtype=np.int32)
    folds = np.asarray(z["fold"], dtype=np.int32)
    source_round = np.asarray(z["source_round"])
    pattern = np.asarray(z["pattern"], dtype=np.int8)
    truth = np.asarray(z["truth"], dtype=np.float64)
    mu = np.asarray(z["ensemble_mean"], dtype=np.float64)
    sd = np.asarray(z["ensemble_std"], dtype=np.float64)
    actual_error = np.abs(truth - mu)

    if len(ids) != 2250:
        raise RuntimeError(f"Expected 2250 rows, found {len(ids)}")

    if set(np.unique(folds).tolist()) != set(range(5)):
        raise RuntimeError("Expected fold IDs 0..4")

    X = make_features(pattern, mu, sd)

    # Residual-model OOF predictions.
    pred_error = np.full_like(actual_error, np.nan)

    fold_report = {}

    for held_fold in range(5):
        train_idx = np.flatnonzero(folds != held_fold)
        test_idx = np.flatnonzero(folds == held_fold)

        if len(train_idx) != 1800 or len(test_idx) != 450:
            raise RuntimeError(
                f"Fold {held_fold}: expected 1800/450, got "
                f"{len(train_idx)}/{len(test_idx)}"
            )

        fold_report[str(held_fold)] = {}

        for j, name in enumerate(OUTPUTS):
            model = ExtraTreesRegressor(
                n_estimators=750,
                min_samples_leaf=5,
                max_features=0.7,
                random_state=20260 + held_fold * 10 + j,
                n_jobs=-1,
            )

            model.fit(
                X[train_idx],
                actual_error[train_idx, j],
            )

            p = np.maximum(
                model.predict(X[test_idx]),
                1e-12,
            )

            pred_error[test_idx, j] = p

            fold_report[str(held_fold)][name] = {
                "n": int(len(test_idx)),
                "error_scale_MAE": float(
                    mean_absolute_error(
                        actual_error[test_idx, j], p
                    )
                ),
                "rho_actual_vs_predicted_error": safe_rho(
                    actual_error[test_idx, j], p
                ),
                "rho_actual_vs_ensemble_sd": safe_rho(
                    actual_error[test_idx, j],
                    sd[test_idx, j],
                ),
            }

    if np.isnan(pred_error).any():
        raise RuntimeError(
            "Missing residual-model OOF predictions"
        )

    report = {
        "purpose": (
            "Leakage-safe current-generation residual-model diagnostic. "
            "Base CNN predictions are strict OOF; residual-model "
            "predictions are also generated fold-to-fold OOF."
        ),
        "n": 2250,
        "base_prediction_strict_oof": True,
        "residual_prediction_strict_oof": True,
        "residual_model": {
            "type": "ExtraTreesRegressor",
            "n_estimators": 750,
            "min_samples_leaf": 5,
            "max_features": 0.7,
        },
        "features": (
            "193 pattern cells + CNN ensemble mean(3) + ensemble SD(3) "
            "+ predicted J + Fa/Fq predicted safety margins. "
            "No fold/source-round/truth fields."
        ),
        "fold_report": fold_report,
        "overall": {},
        "by_source_round": {},
    }

    for j, name in enumerate(OUTPUTS):
        actual = actual_error[:, j]
        learned = pred_error[:, j]
        baseline = sd[:, j]

        report["overall"][name] = {
            "actual_error_median": float(np.median(actual)),
            "actual_error_p95": float(np.quantile(actual, 0.95)),

            "learned_error_scale_MAE": float(
                mean_absolute_error(actual, learned)
            ),

            "rho_actual_vs_learned_error": safe_rho(
                actual, learned
            ),
            "rho_actual_vs_ensemble_sd": safe_rho(
                actual, baseline
            ),

            "top20_learned_error": top_fraction_audit(
                actual, learned
            ),
            "top20_ensemble_sd": top_fraction_audit(
                actual, baseline
            ),

            "median_actual_over_learned": float(
                np.median(
                    actual / np.maximum(learned, 1e-12)
                )
            ),
            "p95_actual_over_learned": float(
                np.quantile(
                    actual / np.maximum(learned, 1e-12),
                    0.95,
                )
            ),
        }

    for sr in np.unique(source_round):
        idx = np.flatnonzero(source_round == sr)
        sr_key = str(sr)

        report["by_source_round"][sr_key] = {
            "n": int(len(idx))
        }

        for j, name in enumerate(OUTPUTS):
            report["by_source_round"][sr_key][name] = {
                "rho_actual_vs_learned_error": safe_rho(
                    actual_error[idx, j],
                    pred_error[idx, j],
                ),
                "rho_actual_vs_ensemble_sd": safe_rho(
                    actual_error[idx, j],
                    sd[idx, j],
                ),
            }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "current_residual_model_oof_predictions.npz",
        crossfit_id=ids,
        fold=folds,
        source_round=source_round,
        truth=truth,
        ensemble_mean=mu,
        ensemble_std=sd,
        actual_abs_error=actual_error,
        learned_abs_error=pred_error,
    )

    (
        out_dir / "current_residual_model_oof_audit.json"
    ).write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(
        "\nCURRENT_CROSSFIT_RESIDUAL_MODEL_DIAGNOSTIC_OK"
    )


if __name__ == "__main__":
    main()
