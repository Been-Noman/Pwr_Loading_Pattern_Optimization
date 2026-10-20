#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry import canonical_d4_key, patterns_to_onehot_grid


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1-dir", required=True)
    ap.add_argument("--round2-labels", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    r1_dir = Path(args.round1_dir)
    r2_labels_path = Path(args.round2_labels)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    a = np.load(r1_dir / "round_arrays.npz", allow_pickle=False)

    # ------------------------------------------------------------
    # Recover the exact frozen Round-1 broad/tail raw targets.
    # Round-1 scalers were fit on training only, so inverse_transform
    # recovers the original OpenMC targets without leakage.
    # ------------------------------------------------------------
    old_scalers = {
        name: joblib.load(r1_dir / f"{name}_scaler.pkl")
        for name in ("keff", "fa", "fqr")
    }

    broad_patterns = np.asarray(a["broad_train_patterns"], dtype=np.int8)
    tail1_patterns = np.asarray(a["tail_train_patterns"], dtype=np.int8)

    kb = old_scalers["keff"].inverse_transform(
        a["yk_train_broad"]
    ).ravel()
    fab = old_scalers["fa"].inverse_transform(
        a["yfa_train_broad"]
    ).ravel()
    fqrb = old_scalers["fqr"].inverse_transform(
        a["yfqr_train_broad"]
    ).ravel()

    kt1 = old_scalers["keff"].inverse_transform(
        a["yk_train_tail"]
    ).ravel()
    fat1 = old_scalers["fa"].inverse_transform(
        a["yfa_train_tail"]
    ).ravel()
    fqrt1 = old_scalers["fqr"].inverse_transform(
        a["yfqr_train_tail"]
    ).ravel()

    # ------------------------------------------------------------
    # New Round-2 tail labels.
    # ------------------------------------------------------------
    rows2 = json.loads(r2_labels_path.read_text())

    if len(rows2) != 1250:
        raise RuntimeError(
            f"Expected 1250 Round-2 labels, found {len(rows2)}"
        )

    if any(r.get("split") != "train" for r in rows2):
        raise RuntimeError("Every Round-2 row must be split=train")

    tail2_patterns = np.asarray(
        [r["pattern"] for r in rows2],
        dtype=np.int8,
    )

    kt2 = np.asarray([r["openmc_keff"] for r in rows2], float)
    fat2 = np.asarray([r["openmc_Fa"] for r in rows2], float)
    fqrt2 = np.asarray([r["openmc_Fq_r"] for r in rows2], float)

    # ------------------------------------------------------------
    # Frozen holdouts copied exactly from Round 1.
    # ------------------------------------------------------------
    val_patterns = np.asarray(a["validation_patterns"], dtype=np.int8)
    cal_patterns = np.asarray(a["calibration_patterns"], dtype=np.int8)
    test_patterns = np.asarray(a["test_patterns"], dtype=np.int8)

    X_val = np.asarray(a["X_val"], dtype=np.float32)
    X_cal = np.asarray(a["X_calibration"], dtype=np.float32)
    X_test = np.asarray(a["X_test"], dtype=np.float32)

    kval = np.asarray(a["yk_val_raw"], float)
    faval = np.asarray(a["yfa_val_raw"], float)
    fqrval = np.asarray(a["yfqr_val_raw"], float)

    kcal = np.asarray(a["yk_calibration_raw"], float)
    facal = np.asarray(a["yfa_calibration_raw"], float)
    fqrcal = np.asarray(a["yfqr_calibration_raw"], float)

    ktest = np.asarray(a["yk_test_raw"], float)
    fatest = np.asarray(a["yfa_test_raw"], float)
    fqrtest = np.asarray(a["yfqr_test_raw"], float)

    # ------------------------------------------------------------
    # Expanded training pool: 7000 broad + 1000 R1 tail + 1250 R2 tail.
    # ------------------------------------------------------------
    tail_patterns = np.concatenate(
        [tail1_patterns, tail2_patterns],
        axis=0,
    )
    kt = np.concatenate([kt1, kt2])
    fat = np.concatenate([fat1, fat2])
    fqrt = np.concatenate([fqrt1, fqrt2])

    train_patterns = np.concatenate(
        [broad_patterns, tail_patterns],
        axis=0,
    )
    ktrain = np.concatenate([kb, kt])
    fatrain = np.concatenate([fab, fat])
    fqrtrain = np.concatenate([fqrb, fqrt])

    if train_patterns.shape != (9250, 193):
        raise RuntimeError(
            f"Unexpected train pattern shape: {train_patterns.shape}"
        )

    if tail_patterns.shape != (2250, 193):
        raise RuntimeError(
            f"Unexpected tail pattern shape: {tail_patterns.shape}"
        )

    # ------------------------------------------------------------
    # D4 leakage audit across training / frozen holdouts.
    # ------------------------------------------------------------
    def keys(patterns):
        return {tuple(canonical_d4_key(p)) for p in patterns}

    role_keys = {
        "train": keys(train_patterns),
        "validation": keys(val_patterns),
        "calibration": keys(cal_patterns),
        "test": keys(test_patterns),
    }

    if len(role_keys["train"]) != len(train_patterns):
        raise RuntimeError("D4 duplicate inside expanded training set")

    roles = list(role_keys)
    leakage = {}

    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            n = len(role_keys[left] & role_keys[right])
            leakage[f"{left}__{right}"] = n
            if n:
                raise RuntimeError(
                    f"D4 leakage {left} vs {right}: {n}"
                )

    # ------------------------------------------------------------
    # New scalers fit ONLY on the 9250 expanded training rows.
    # ------------------------------------------------------------
    scalers = {
        "keff": StandardScaler().fit(ktrain[:, None]),
        "fa": StandardScaler().fit(fatrain[:, None]),
        "fqr": StandardScaler().fit(fqrtrain[:, None]),
    }

    for name, scaler in scalers.items():
        joblib.dump(scaler, out / f"{name}_scaler.pkl")

    Xb = np.asarray(a["X_train_broad"], dtype=np.float32)
    Xt1 = np.asarray(a["X_train_tail"], dtype=np.float32)
    Xt2 = patterns_to_onehot_grid(tail2_patterns).astype(np.float32)
    Xt = np.concatenate([Xt1, Xt2], axis=0)
    Xtrain = np.concatenate([Xb, Xt], axis=0)

    np.savez_compressed(
        out / "round_arrays.npz",

        X_train=Xtrain,
        X_train_broad=Xb,
        X_train_tail=Xt,
        train_origin=np.concatenate([
            np.zeros(len(Xb), dtype=np.uint8),
            np.ones(len(Xt), dtype=np.uint8),
        ]),

        X_val=X_val,
        X_calibration=X_cal,
        X_test=X_test,

        yk_train=scalers["keff"].transform(ktrain[:, None]),
        yfa_train=scalers["fa"].transform(fatrain[:, None]),
        yfqr_train=scalers["fqr"].transform(fqrtrain[:, None]),

        yk_train_broad=scalers["keff"].transform(kb[:, None]),
        yfa_train_broad=scalers["fa"].transform(fab[:, None]),
        yfqr_train_broad=scalers["fqr"].transform(fqrb[:, None]),

        yk_train_tail=scalers["keff"].transform(kt[:, None]),
        yfa_train_tail=scalers["fa"].transform(fat[:, None]),
        yfqr_train_tail=scalers["fqr"].transform(fqrt[:, None]),

        yk_val=scalers["keff"].transform(kval[:, None]),
        yfa_val=scalers["fa"].transform(faval[:, None]),
        yfqr_val=scalers["fqr"].transform(fqrval[:, None]),

        yk_val_raw=kval,
        yfa_val_raw=faval,
        yfqr_val_raw=fqrval,

        yk_calibration_raw=kcal,
        yfa_calibration_raw=facal,
        yfqr_calibration_raw=fqrcal,

        yk_test_raw=ktest,
        yfa_test_raw=fatest,
        yfqr_test_raw=fqrtest,

        train_patterns=train_patterns,
        broad_train_patterns=broad_patterns,
        tail_train_patterns=tail_patterns,
        round1_tail_train_patterns=tail1_patterns,
        round2_tail_train_patterns=tail2_patterns,

        validation_patterns=val_patterns,
        calibration_patterns=cal_patterns,
        test_patterns=test_patterns,
    )

    metadata = {
        "version": "optimizer_tail_v2_round2",
        "training_unique_total": 9250,
        "broad_train": 7000,
        "optimizer_tail_train": 2250,
        "optimizer_tail_round1": 1000,
        "optimizer_tail_round2": 1250,
        "tail_validation": 125,
        "tail_calibration": 125,
        "untouched_broad_test": 1500,

        "training_sampling_plan": (
            "50% broad / 50% optimizer-tail minibatches"
        ),

        "scaler_fit": (
            "combined 9250 training rows only; "
            "validation/calibration/test excluded"
        ),

        "architecture_plan": (
            "unchanged V1 residual-attention 5-CNN architecture"
        ),

        "data_pipeline_plan": (
            "native tf.data balanced 16 broad + 16 tail per batch"
        ),

        "round1_dir": str(r1_dir),
        "round2_labels": str(r2_labels_path),
        "round2_labels_sha256": sha256_file(r2_labels_path),

        "d4_leakage": leakage,
        "d4_leakage_across_all_roles": int(sum(leakage.values())),

        "frozen_holdouts": True,
    }

    (out / "round_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print(json.dumps(metadata, indent=2))
    print("ROUND2_DATASET_BUILD_OK")


if __name__ == "__main__":
    main()
