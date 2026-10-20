#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry import FA_LIMIT, FQR_LIMIT
from evaluate_cnn_ensemble import (
    feasible,
    fit_adaptive_conformal,
    adaptive_bounds,
    calibrated_safe,
    coverage_audit,
    selection_audit,
)


OUTPUTS = ("keff", "Fa", "Fq_r")


def safe_rho(a, b):
    r = spearmanr(a, b).statistic
    return None if not np.isfinite(r) else float(r)


def make_features(pattern, mu, sd):
    J = (
        0.5 * mu[:, 1] / FA_LIMIT
        + 0.5 * mu[:, 2] / FQR_LIMIT
    )[:, None]

    fa_margin = (FA_LIMIT - mu[:, 1])[:, None]
    fqr_margin = (FQR_LIMIT - mu[:, 2])[:, None]

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


def scale_quality(y, mu, learned_scale):
    err = np.abs(y - mu)

    return {
        OUTPUTS[j]: {
            "MAE_of_error_scale": float(
                mean_absolute_error(
                    err[:, j], learned_scale[:, j]
                )
            ),
            "rho_abs_error_vs_learned_scale": safe_rho(
                err[:, j], learned_scale[:, j]
            ),
            "median_abs_error": float(
                np.median(err[:, j])
            ),
            "median_learned_scale": float(
                np.median(learned_scale[:, j])
            ),
        }
        for j in range(3)
    }


def evaluate_safety(y, mu, scale, q_marginal, q_joint):
    true_safe = feasible(y)
    raw_safe = feasible(mu)

    marginal_bounds = adaptive_bounds(
        mu, scale, q_marginal
    )
    joint_bounds = adaptive_bounds(
        mu, scale, q_joint
    )

    marginal_safe = calibrated_safe(
        marginal_bounds
    )
    joint_safe = calibrated_safe(
        joint_bounds
    )

    return {
        "n": int(len(y)),
        "true_feasible_count": int(true_safe.sum()),

        "raw_safe_selection": selection_audit(
            raw_safe, true_safe
        ),

        "marginal95": {
            "coverage": coverage_audit(
                y, marginal_bounds
            ),
            "safe_selection": selection_audit(
                marginal_safe, true_safe
            ),
        },

        "joint95_bonferroni": {
            "coverage": coverage_audit(
                y, joint_bounds
            ),
            "safe_selection": selection_audit(
                joint_safe, true_safe
            ),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--round2-arrays", required=True)
    ap.add_argument("--round2-predictions", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    # ============================================================
    # 1. Honest current-generation OOF residual training data
    # ============================================================
    z = np.load(args.oof, allow_pickle=False)

    pattern_oof = np.asarray(
        z["pattern"], dtype=np.int8
    )
    truth_oof = np.asarray(
        z["truth"], dtype=np.float64
    )
    mu_oof = np.asarray(
        z["ensemble_mean"], dtype=np.float64
    )
    sd_oof = np.asarray(
        z["ensemble_std"], dtype=np.float64
    )

    if len(pattern_oof) != 2250:
        raise RuntimeError(
            f"Expected 2250 OOF rows, got {len(pattern_oof)}"
        )

    err_oof = np.abs(truth_oof - mu_oof)
    X_oof = make_features(
        pattern_oof, mu_oof, sd_oof
    )

    # ============================================================
    # 2. Current Round-2 calibration / external-validation roles
    # ============================================================
    arr = np.load(
        args.round2_arrays,
        allow_pickle=False,
    )
    pred = np.load(
        args.round2_predictions,
        allow_pickle=False,
    )

    cal_patterns = np.asarray(
        arr["calibration_patterns"],
        dtype=np.int8,
    )
    val_patterns = np.asarray(
        arr["validation_patterns"],
        dtype=np.int8,
    )

    cal_mu = np.asarray(
        pred["tail_calibration_mean"],
        dtype=np.float64,
    )
    cal_sd = np.asarray(
        pred["tail_calibration_std"],
        dtype=np.float64,
    )
    cal_truth = np.asarray(
        pred["tail_calibration_truth"],
        dtype=np.float64,
    )

    val_mu = np.asarray(
        pred["tail_validation_mean"],
        dtype=np.float64,
    )
    val_sd = np.asarray(
        pred["tail_validation_std"],
        dtype=np.float64,
    )
    val_truth = np.asarray(
        pred["tail_validation_truth"],
        dtype=np.float64,
    )

    if len(cal_patterns) != 125 or len(val_patterns) != 125:
        raise RuntimeError(
            "Expected 125 calibration and 125 validation patterns"
        )

    X_cal = make_features(
        cal_patterns, cal_mu, cal_sd
    )
    X_val = make_features(
        val_patterns, val_mu, val_sd
    )

    # ============================================================
    # 3. Fit residual/error-scale models ONLY on 2250 OOF rows
    # ============================================================
    learned_cal = np.zeros_like(cal_truth)
    learned_val = np.zeros_like(val_truth)

    models_meta = {}

    for j, name in enumerate(OUTPUTS):
        model = ExtraTreesRegressor(
            n_estimators=1000,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=30300 + j,
            n_jobs=-1,
        )

        model.fit(
            X_oof,
            err_oof[:, j],
        )

        learned_cal[:, j] = np.maximum(
            model.predict(X_cal),
            1e-12,
        )
        learned_val[:, j] = np.maximum(
            model.predict(X_val),
            1e-12,
        )

        models_meta[name] = {
            "training_rows": 2250,
            "target": "absolute OOF CNN error",
        }

    # ============================================================
    # 4. SAME conformal machinery as existing pipeline.
    #    Only the local scale changes:
    #       old = ensemble SD
    #       new = learned residual/error scale
    # ============================================================
    q_marginal = fit_adaptive_conformal(
        cal_truth,
        cal_mu,
        learned_cal,
        alpha_each=0.05,
    )

    q_joint = fit_adaptive_conformal(
        cal_truth,
        cal_mu,
        learned_cal,
        alpha_each=0.05 / 4.0,
    )

    report = {
        "purpose": (
            "Evaluate learned current-generation residual/error scale "
            "inside the existing conformal safety framework."
        ),

        "important_status": (
            "Diagnostic only. Does not authorize GA use."
        ),

        "physics_thresholds_changed": False,
        "conformal_confidence_lowered": False,

        "residual_model_training": {
            "n": 2250,
            "base_predictions": "strict current-generation CNN OOF",
            "models": models_meta,
        },

        "roles": {
            "tail_calibration": 125,
            "tail_external_validation": 125,
            "note": (
                "The 125-case calibration set is used only to fit "
                "conformal multipliers. The 125-case tail validation "
                "set is external to this residual-model fit/calibration, "
                "although it was historically used for early stopping "
                "of the Round-2 point CNN and is therefore not a pristine "
                "final test set."
            ),
        },

        "learned_scale_quality": {
            "calibration": scale_quality(
                cal_truth, cal_mu, learned_cal
            ),
            "external_validation": scale_quality(
                val_truth, val_mu, learned_val
            ),
        },

        "adaptive_conformal_marginal95": {
            "alpha_per_bound": 0.05,
            "q": q_marginal,
        },

        "adaptive_conformal_joint95_bonferroni": {
            "family_alpha": 0.05,
            "alpha_per_bound": 0.05 / 4.0,
            "q": q_joint,
        },

        "calibration_fit_audit": evaluate_safety(
            cal_truth,
            cal_mu,
            learned_cal,
            q_marginal,
            q_joint,
        ),

        "external_tail_validation_audit": evaluate_safety(
            val_truth,
            val_mu,
            learned_val,
            q_marginal,
            q_joint,
        ),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "learned_residual_conformal_predictions.npz",
        calibration_truth=cal_truth,
        calibration_mean=cal_mu,
        calibration_ensemble_std=cal_sd,
        calibration_learned_scale=learned_cal,
        validation_truth=val_truth,
        validation_mean=val_mu,
        validation_ensemble_std=val_sd,
        validation_learned_scale=learned_val,
    )

    (
        out_dir / "learned_residual_conformal_audit.json"
    ).write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print("\nLEARNED_RESIDUAL_CONFORMAL_AUDIT_OK")


if __name__ == "__main__":
    main()
