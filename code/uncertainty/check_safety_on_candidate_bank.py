#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


FA_LIMIT = 2.0
FQR_LIMIT = 2.5
KEFF_LO = 1.0
KEFF_HI = 1.2
OUTPUTS = ("keff", "Fa", "Fq_r")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--bank-scores", required=True)
    ap.add_argument("--conformal-audit", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    # ------------------------------------------------------------
    # 1) Residual-model training set:
    #    2250 strict current-generation OOF residuals
    # ------------------------------------------------------------
    oof = np.load(args.oof, allow_pickle=False)

    pattern_oof = np.asarray(oof["pattern"], dtype=np.int8)
    truth_oof = np.asarray(oof["truth"], dtype=np.float64)
    mu_oof = np.asarray(oof["ensemble_mean"], dtype=np.float64)
    sd_oof = np.asarray(oof["ensemble_std"], dtype=np.float64)

    err_oof = np.abs(truth_oof - mu_oof)
    X_oof = make_features(pattern_oof, mu_oof, sd_oof)

    if len(X_oof) != 2250:
        raise RuntimeError(
            f"Expected 2250 OOF rows, got {len(X_oof)}"
        )

    # ------------------------------------------------------------
    # 2) Current Round-2 bank predictions
    # ------------------------------------------------------------
    bank = np.load(args.bank_scores, allow_pickle=False)

    pattern_bank = np.asarray(bank["pattern"], dtype=np.int8)
    mu_bank = np.asarray(bank["ensemble_mean"], dtype=np.float64)
    sd_bank = np.asarray(bank["ensemble_std"], dtype=np.float64)
    J_bank = np.asarray(bank["predicted_J"], dtype=np.float64)

    source_index = np.asarray(bank["source_index"], dtype=np.int32)
    visit_count = np.asarray(bank["visit_count"])
    max_generation = np.asarray(bank["max_generation"])
    strategy_mask = np.asarray(bank["strategy_mask"])
    seed_count = np.asarray(bank["seed_count"])

    n = len(pattern_bank)

    if n != 84250:
        raise RuntimeError(
            f"Expected 84250 eligible bank candidates, got {n}"
        )

    X_bank = make_features(
        pattern_bank,
        mu_bank,
        sd_bank,
    )

    # ------------------------------------------------------------
    # 3) Fit learned residual scales on ALL 2250 honest OOF rows
    # ------------------------------------------------------------
    learned_scale = np.zeros_like(mu_bank)

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

        learned_scale[:, j] = np.maximum(
            model.predict(X_bank),
            1e-12,
        )

        print(
            f"{name}: predicted learned error scale for {n} candidates",
            flush=True,
        )

    # ------------------------------------------------------------
    # 4) Use EXACT joint95 multipliers from dedicated 125-case
    #    Round-2 calibration audit.
    # ------------------------------------------------------------
    audit = json.loads(
        Path(args.conformal_audit).read_text()
    )

    q = audit[
        "adaptive_conformal_joint95_bonferroni"
    ]["q"]

    q_keff_lo = float(q["keff_lower"])
    q_keff_hi = float(q["keff_upper"])
    q_fa = float(q["Fa_upper"])
    q_fqr = float(q["Fq_r_upper"])

    lower_keff = (
        mu_bank[:, 0]
        - q_keff_lo * learned_scale[:, 0]
    )
    upper_keff = (
        mu_bank[:, 0]
        + q_keff_hi * learned_scale[:, 0]
    )
    upper_fa = (
        mu_bank[:, 1]
        + q_fa * learned_scale[:, 1]
    )
    upper_fqr = (
        mu_bank[:, 2]
        + q_fqr * learned_scale[:, 2]
    )

    raw_safe = (
        (mu_bank[:, 0] >= KEFF_LO)
        & (mu_bank[:, 0] <= KEFF_HI)
        & (mu_bank[:, 1] <= FA_LIMIT)
        & (mu_bank[:, 2] <= FQR_LIMIT)
    )

    joint95_safe = (
        (lower_keff >= KEFF_LO)
        & (upper_keff <= KEFF_HI)
        & (upper_fa <= FA_LIMIT)
        & (upper_fqr <= FQR_LIMIT)
    )

    # Individual learned-bound pass masks.
    pass_keff_lo = lower_keff >= KEFF_LO
    pass_keff_hi = upper_keff <= KEFF_HI
    pass_fa = upper_fa <= FA_LIMIT
    pass_fqr = upper_fqr <= FQR_LIMIT

    robust_idx = np.flatnonzero(joint95_safe)

    report = {
        "mode": "diagnostic_only_no_GA_change",
        "candidate_count": int(n),
        "raw_predicted_feasible_count": int(raw_safe.sum()),
        "joint95_learned_residual_safe_count": int(
            joint95_safe.sum()
        ),
        "joint95_fraction_of_bank": float(
            joint95_safe.mean()
        ),
        "joint95_fraction_of_raw_safe": (
            float(joint95_safe.sum() / raw_safe.sum())
            if raw_safe.sum() else None
        ),

        "individual_bound_pass_counts": {
            "keff_lower": int(pass_keff_lo.sum()),
            "keff_upper": int(pass_keff_hi.sum()),
            "Fa_upper": int(pass_fa.sum()),
            "Fq_r_upper": int(pass_fqr.sum()),
        },

        "among_raw_safe": {
            "n": int(raw_safe.sum()),
            "keff_lower_pass": int(
                (raw_safe & pass_keff_lo).sum()
            ),
            "keff_upper_pass": int(
                (raw_safe & pass_keff_hi).sum()
            ),
            "Fa_upper_pass": int(
                (raw_safe & pass_fa).sum()
            ),
            "Fq_r_upper_pass": int(
                (raw_safe & pass_fqr).sum()
            ),
        },

        "joint95_conformal_q": q,

        "robust_safe_summary": None,

        "physics_thresholds_changed": False,
        "conformal_confidence_lowered": False,

        "note": (
            "This script only audits the existing candidate bank. "
            "It does not modify the GA, acquisition strategy, "
            "physics limits, or conformal confidence level."
        ),
    }

    if len(robust_idx):
        report["robust_safe_summary"] = {
            "count": int(len(robust_idx)),
            "predicted_J_min": float(
                np.min(J_bank[robust_idx])
            ),
            "predicted_J_median": float(
                np.median(J_bank[robust_idx])
            ),
            "predicted_J_max": float(
                np.max(J_bank[robust_idx])
            ),
            "Fa_pred_min": float(
                np.min(mu_bank[robust_idx, 1])
            ),
            "Fa_pred_median": float(
                np.median(mu_bank[robust_idx, 1])
            ),
            "Fa_pred_max": float(
                np.max(mu_bank[robust_idx, 1])
            ),
            "Fa_required_margin_median": float(
                np.median(
                    q_fa * learned_scale[robust_idx, 1]
                )
            ),
            "Fa_available_margin_median": float(
                np.median(
                    FA_LIMIT - mu_bank[robust_idx, 1]
                )
            ),
            "Fq_pred_median": float(
                np.median(mu_bank[robust_idx, 2])
            ),
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "existing_bank_learned_residual_safety.npz",

        source_index=source_index,
        pattern=pattern_bank,

        ensemble_mean=mu_bank,
        ensemble_std=sd_bank,
        learned_scale=learned_scale,
        predicted_J=J_bank,

        raw_safe=raw_safe.astype(np.uint8),
        joint95_safe=joint95_safe.astype(np.uint8),

        learned_lower_keff=lower_keff,
        learned_upper_keff=upper_keff,
        learned_upper_Fa=upper_fa,
        learned_upper_Fq_r=upper_fqr,

        pass_keff_lower=pass_keff_lo.astype(np.uint8),
        pass_keff_upper=pass_keff_hi.astype(np.uint8),
        pass_Fa_upper=pass_fa.astype(np.uint8),
        pass_Fq_r_upper=pass_fqr.astype(np.uint8),

        visit_count=visit_count,
        max_generation=max_generation,
        strategy_mask=strategy_mask,
        seed_count=seed_count,
    )

    (
        out_dir / "existing_bank_learned_residual_safety_audit.json"
    ).write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print("\nEXISTING_BANK_LEARNED_RESIDUAL_SAFETY_AUDIT_OK")


if __name__ == "__main__":
    main()
