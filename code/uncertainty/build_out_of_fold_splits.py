#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round2-dir", required=True)
    ap.add_argument("--crossfit-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    r2_dir = Path(args.round2_dir)
    manifest_path = Path(args.crossfit_manifest)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    a = np.load(r2_dir / "round_arrays.npz", allow_pickle=False)

    old_scalers = {
        name: joblib.load(r2_dir / f"{name}_scaler.pkl")
        for name in ("keff", "fa", "fqr")
    }

    # ---------- frozen broad/tail training pools ----------
    Xb = np.asarray(a["X_train_broad"], dtype=np.float32)
    Xt = np.asarray(a["X_train_tail"], dtype=np.float32)

    broad_patterns = np.asarray(
        a["broad_train_patterns"], dtype=np.int8
    )
    tail_patterns = np.asarray(
        a["tail_train_patterns"], dtype=np.int8
    )

    # Recover RAW targets from existing Round-2 scaled arrays.
    kb = old_scalers["keff"].inverse_transform(
        a["yk_train_broad"]
    ).ravel()
    fab = old_scalers["fa"].inverse_transform(
        a["yfa_train_broad"]
    ).ravel()
    fqrb = old_scalers["fqr"].inverse_transform(
        a["yfqr_train_broad"]
    ).ravel()

    kt = old_scalers["keff"].inverse_transform(
        a["yk_train_tail"]
    ).ravel()
    fat = old_scalers["fa"].inverse_transform(
        a["yfa_train_tail"]
    ).ravel()
    fqrt = old_scalers["fqr"].inverse_transform(
        a["yfqr_train_tail"]
    ).ravel()

    manifest = json.loads(manifest_path.read_text())

    if len(manifest) != 2250:
        raise RuntimeError(
            f"Expected 2250 cross-fit rows, got {len(manifest)}"
        )

    manifest_patterns = np.asarray(
        [r["pattern"] for r in manifest],
        dtype=np.int8,
    )

    if not np.array_equal(manifest_patterns, tail_patterns):
        raise RuntimeError(
            "Cross-fit manifest order does not match tail_train_patterns"
        )

    fold_id = np.asarray(
        [int(r["fold"]) for r in manifest],
        dtype=np.int32,
    )

    # Used ONLY to balance the internal validation split.
    strata = np.asarray([
        f'{r["source_round"]}_'
        f'{"feasible" if r["openmc_feasible"] else "infeasible"}'
        for r in manifest
    ])

    # ---------- external sets ----------
    # IMPORTANT:
    # Original 125-case tail validation is retained for external audit only.
    # It will NOT be placed in X_val and will NOT drive early stopping.
    X_external_tail_val = np.asarray(a["X_val"], dtype=np.float32)

    k_external_tail_val = np.asarray(a["yk_val_raw"], dtype=float)
    fa_external_tail_val = np.asarray(a["yfa_val_raw"], dtype=float)
    fqr_external_tail_val = np.asarray(a["yfqr_val_raw"], dtype=float)

    X_cal = np.asarray(a["X_calibration"], dtype=np.float32)
    kcal = np.asarray(a["yk_calibration_raw"], dtype=float)
    facal = np.asarray(a["yfa_calibration_raw"], dtype=float)
    fqrcal = np.asarray(a["yfqr_calibration_raw"], dtype=float)

    X_test = np.asarray(a["X_test"], dtype=np.float32)
    ktest = np.asarray(a["yk_test_raw"], dtype=float)
    fatest = np.asarray(a["yfa_test_raw"], dtype=float)
    fqrtest = np.asarray(a["yfqr_test_raw"], dtype=float)

    summary = {}

    for fold in range(5):
        fold_dir = out_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        hold_idx = np.flatnonzero(fold_id == fold)
        available_tail_idx = np.flatnonzero(fold_id != fold)

        if len(hold_idx) != 450:
            raise RuntimeError(
                f"Fold {fold}: expected 450 holdout, got {len(hold_idx)}"
            )

        if len(available_tail_idx) != 1800:
            raise RuntimeError(
                f"Fold {fold}: expected 1800 available tail rows, "
                f"got {len(available_tail_idx)}"
            )

        # ---------------------------------------------------------
        # Internal validation is drawn ONLY from the 1800 tail rows
        # available to this fold.
        #
        # 1620 tail -> fit
        #  180 tail -> internal validation / early stopping
        #
        # Cross-fit holdout 450 remains completely unseen.
        # ---------------------------------------------------------
        fit_tail_idx, val_tail_idx = train_test_split(
            available_tail_idx,
            test_size=180,
            random_state=2026 + fold,
            shuffle=True,
            stratify=strata[available_tail_idx],
        )

        fit_tail_idx = np.sort(fit_tail_idx)
        val_tail_idx = np.sort(val_tail_idx)

        if len(fit_tail_idx) != 1620:
            raise RuntimeError(
                f"Fold {fold}: expected 1620 fit-tail rows"
            )

        if len(val_tail_idx) != 180:
            raise RuntimeError(
                f"Fold {fold}: expected 180 internal validation rows"
            )

        if set(fit_tail_idx) & set(val_tail_idx):
            raise RuntimeError("Fit/internal-validation leakage")

        if (
            set(fit_tail_idx) & set(hold_idx)
            or set(val_tail_idx) & set(hold_idx)
        ):
            raise RuntimeError("Cross-fit holdout leakage")

        # ---------- raw fold-training targets ----------
        k_train_raw = np.concatenate([kb, kt[fit_tail_idx]])
        fa_train_raw = np.concatenate([fab, fat[fit_tail_idx]])
        fqr_train_raw = np.concatenate([fqrb, fqrt[fit_tail_idx]])

        # CRITICAL:
        # scalers fit ONLY on the 8620 actual model-fitting rows.
        scalers = {
            "keff": StandardScaler().fit(k_train_raw[:, None]),
            "fa": StandardScaler().fit(fa_train_raw[:, None]),
            "fqr": StandardScaler().fit(fqr_train_raw[:, None]),
        }

        for name, scaler in scalers.items():
            joblib.dump(
                scaler,
                fold_dir / f"{name}_scaler.pkl"
            )

        # ---------- actual fitting data ----------
        X_train_tail = Xt[fit_tail_idx]

        # ---------- internal early-stopping validation ----------
        X_internal_val = Xt[val_tail_idx]

        k_internal_val = kt[val_tail_idx]
        fa_internal_val = fat[val_tail_idx]
        fqr_internal_val = fqrt[val_tail_idx]

        # ---------- unseen cross-fit holdout ----------
        X_hold = Xt[hold_idx]

        k_hold = kt[hold_idx]
        fa_hold = fat[hold_idx]
        fqr_hold = fqrt[hold_idx]

        np.savez_compressed(
            fold_dir / "round_arrays.npz",

            # Actual training pool: 7000 broad + 1620 tail.
            X_train=np.concatenate(
                [Xb, X_train_tail],
                axis=0,
            ),
            X_train_broad=Xb,
            X_train_tail=X_train_tail,

            yk_train_broad=scalers["keff"].transform(
                kb[:, None]
            ),
            yfa_train_broad=scalers["fa"].transform(
                fab[:, None]
            ),
            yfqr_train_broad=scalers["fqr"].transform(
                fqrb[:, None]
            ),

            yk_train_tail=scalers["keff"].transform(
                kt[fit_tail_idx, None]
            ),
            yfa_train_tail=scalers["fa"].transform(
                fat[fit_tail_idx, None]
            ),
            yfqr_train_tail=scalers["fqr"].transform(
                fqrt[fit_tail_idx, None]
            ),

            # Internal validation ONLY.
            # The trainer will use these arrays for early stopping.
            X_val=X_internal_val,
            yk_val=scalers["keff"].transform(
                k_internal_val[:, None]
            ),
            yfa_val=scalers["fa"].transform(
                fa_internal_val[:, None]
            ),
            yfqr_val=scalers["fqr"].transform(
                fqr_internal_val[:, None]
            ),
            yk_val_raw=k_internal_val,
            yfa_val_raw=fa_internal_val,
            yfqr_val_raw=fqr_internal_val,
            internal_val_tail_index=val_tail_idx.astype(
                np.int32
            ),

            # Completely unseen cross-fit holdout.
            X_crossfit_holdout=X_hold,
            yk_crossfit_holdout_raw=k_hold,
            yfa_crossfit_holdout_raw=fa_hold,
            yfqr_crossfit_holdout_raw=fqr_hold,
            crossfit_holdout_tail_index=hold_idx.astype(
                np.int32
            ),
            crossfit_holdout_id=np.asarray(
                [manifest[i]["crossfit_id"] for i in hold_idx],
                dtype=np.int32,
            ),
            crossfit_holdout_patterns=tail_patterns[hold_idx],

            # Original Round-2 tail validation retained ONLY
            # as an explicitly named external audit set.
            X_external_tail_validation=X_external_tail_val,
            yk_external_tail_validation_raw=k_external_tail_val,
            yfa_external_tail_validation_raw=fa_external_tail_val,
            yfqr_external_tail_validation_raw=fqr_external_tail_val,

            # Frozen calibration — never used for fold training.
            X_calibration=X_cal,
            yk_calibration_raw=kcal,
            yfa_calibration_raw=facal,
            yfqr_calibration_raw=fqrcal,

            # Frozen broad test — never used for fold training.
            X_test=X_test,
            yk_test_raw=ktest,
            yfa_test_raw=fatest,
            yfqr_test_raw=fqrtest,

            broad_train_patterns=broad_patterns,
            tail_train_patterns=tail_patterns[fit_tail_idx],
            internal_val_patterns=tail_patterns[val_tail_idx],
        )

        # Balance audit for the 180-case internal validation.
        vals, counts = np.unique(
            strata[val_tail_idx],
            return_counts=True,
        )

        internal_val_balance = {
            str(v): int(c)
            for v, c in zip(vals, counts)
        }

        metadata = {
            "fold": fold,

            "broad_fit": 7000,
            "tail_fit": 1620,
            "actual_fit_total": 8620,

            "internal_tail_validation": 180,
            "crossfit_holdout": 450,

            "crossfit_holdout_used_for_fit": False,
            "original_125_tail_validation_used_for_fit": False,
            "calibration_used_for_fit": False,
            "broad_test_used_for_fit": False,

            "scaler_fit": (
                "fold-specific 8620 actual fitting rows only"
            ),

            "internal_validation_balance":
                internal_val_balance,

            "batch_plan": "50% broad / 50% tail",
            "cnn_seeds": [42, 314, 2718, 31415, 161803],
        }

        (fold_dir / "round_metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )

        summary[str(fold)] = metadata

    (out_root / "crossfit_dataset_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))
    print("\nCROSSFIT_5FOLD_DATASETS_LEAKAGE_CLEAN_BUILD_OK")


if __name__ == "__main__":
    main()
