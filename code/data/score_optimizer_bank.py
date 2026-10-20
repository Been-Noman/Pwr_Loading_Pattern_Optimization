#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cnn_architecture import (
    CNN_TRAINING_SEEDS,
    CUSTOM_OBJECTS,
    configure_tensorflow,
)
from geometry import (
    FA_LIMIT,
    FQR_LIMIT,
    KEFF_HI,
    KEFF_LO,
    patterns_to_onehot_grid,
)


def inverse_outputs(model, x, scalers):
    outputs = model(tf.constant(x), training=False)
    return np.column_stack([
        scalers["keff"].inverse_transform(
            outputs[0].numpy().reshape(-1, 1)
        ).ravel(),
        scalers["fa"].inverse_transform(
            outputs[1].numpy().reshape(-1, 1)
        ).ravel(),
        scalers["fqr"].inverse_transform(
            outputs[2].numpy().reshape(-1, 1)
        ).ravel(),
    ]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", required=True)
    ap.add_argument("--candidate-bank", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--chunk-size", type=int, default=2048)
    args = ap.parse_args()

    configure_tensorflow(args.threads)

    round_dir = Path(args.round_dir)
    bank_path = Path(args.candidate_bank)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = np.load(bank_path, allow_pickle=False)

    eligible = np.asarray(bank["eligible_new_AL"], dtype=bool)
    source_index = np.flatnonzero(eligible)
    patterns = np.asarray(bank["pattern"][eligible], dtype=np.int8)

    if len(patterns) == 0:
        raise RuntimeError("No eligible_new_AL candidates found.")

    scalers = {
        name: joblib.load(round_dir / f"{name}_scaler.pkl")
        for name in ("keff", "fa", "fqr")
    }

    n = len(patterns)
    per_seed = np.empty(
        (len(CNN_TRAINING_SEEDS), n, 3),
        dtype=np.float32,
    )

    for si, seed in enumerate(CNN_TRAINING_SEEDS):
        model_path = round_dir / "models" / f"cnn_seed_{seed}.keras"
        if not model_path.exists():
            raise RuntimeError(f"Missing model: {model_path}")

        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )

        for start in range(0, n, args.chunk_size):
            stop = min(start + args.chunk_size, n)
            x = patterns_to_onehot_grid(patterns[start:stop])
            per_seed[si, start:stop] = inverse_outputs(
                model, x, scalers
            )

        del model
        tf.keras.backend.clear_session()
        print(f"seed {seed}: scored {n} candidates", flush=True)

    mu = per_seed.mean(axis=0)
    sd = per_seed.std(axis=0, ddof=1)

    # Existing frozen objective; diagnostic only.
    J = (
        0.5 * mu[:, 1] / FA_LIMIT
        + 0.5 * mu[:, 2] / FQR_LIMIT
    ).astype(np.float32)

    # Signed distances to the existing physical limits.
    # Positive = predicted inside that individual limit.
    margin_keff_low = (mu[:, 0] - KEFF_LO).astype(np.float32)
    margin_keff_high = (KEFF_HI - mu[:, 0]).astype(np.float32)
    margin_Fa = (FA_LIMIT - mu[:, 1]).astype(np.float32)
    margin_Fq_r = (FQR_LIMIT - mu[:, 2]).astype(np.float32)

    raw_predicted_feasible = (
        (margin_keff_low >= 0)
        & (margin_keff_high >= 0)
        & (margin_Fa >= 0)
        & (margin_Fq_r >= 0)
    )

    np.savez_compressed(
        out_dir / "existing_bank_v2_scores.npz",

        source_index=source_index.astype(np.int32),
        pattern=patterns,

        per_seed=per_seed,
        ensemble_mean=mu.astype(np.float32),
        ensemble_std=sd.astype(np.float32),
        predicted_J=J,

        raw_predicted_feasible=raw_predicted_feasible.astype(np.uint8),

        margin_keff_low=margin_keff_low,
        margin_keff_high=margin_keff_high,
        margin_Fa=margin_Fa,
        margin_Fq_r=margin_Fq_r,

        visit_count=bank["visit_count"][eligible],
        min_generation=bank["min_generation"][eligible],
        max_generation=bank["max_generation"][eligible],
        strategy_mask=bank["strategy_mask"][eligible],
        seed_count=bank["seed_count"][eligible],
        overlap_legacy90=bank["overlap_legacy90"][eligible],
    )

    report = {
        "mode": "diagnostic_only_no_acquisition_selection",
        "candidate_bank": str(bank_path),
        "round_dir": str(round_dir),
        "eligible_candidate_count": int(n),
        "cnn_seeds": [int(x) for x in CNN_TRAINING_SEEDS],
        "physics_limits_used_only_for_reporting": {
            "keff_low": float(KEFF_LO),
            "keff_high": float(KEFF_HI),
            "Fa": float(FA_LIMIT),
            "Fq_r": float(FQR_LIMIT),
        },
        "raw_predicted_feasible_count": int(
            raw_predicted_feasible.sum()
        ),
        "note": (
            "No Round-2 candidate selection, safety threshold change, "
            "GA change, or physics decision is made by this script."
        ),
    }

    (out_dir / "existing_bank_v2_score_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print(json.dumps(report, indent=2))
    print("V2_EXISTING_BANK_SCORING_OK")


if __name__ == "__main__":
    main()
