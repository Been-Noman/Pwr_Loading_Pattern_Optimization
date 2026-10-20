#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cnn_architecture import (
    CNN_BATCH_SIZE, CNN_EPOCHS, CNN_PATIENCE, CNN_TRAINING_SEEDS,
    configure_tensorflow, create_model, set_tensorflow_seed,
)
from data_io import atomic_json


def metrics(y_true, y_pred):
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "Spearman_rho": float(rho),
        "bias_true_minus_pred": float(np.mean(np.asarray(y_true) - np.asarray(y_pred))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.seed not in CNN_TRAINING_SEEDS:
        raise SystemExit(
            f"Seed {args.seed} is not one of the frozen seeds {CNN_TRAINING_SEEDS}"
        )

    threads, inter = configure_tensorflow(args.threads)
    set_tensorflow_seed(args.seed)

    import tensorflow as tf

    round_dir = Path(args.round_dir)
    model_dir = round_dir / "models"
    history_dir = round_dir / "histories"
    model_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"cnn_seed_{args.seed}.keras"
    history_path = history_dir / f"cnn_seed_{args.seed}_history.json"
    record_path = history_dir / f"cnn_seed_{args.seed}_record.json"

    arrays = np.load(round_dir / "round_arrays.npz", allow_pickle=False)

    Xb = arrays["X_train_broad"]
    Xt = arrays["X_train_tail"]

    yb = {
        "keff": arrays["yk_train_broad"],
        "fa": arrays["yfa_train_broad"],
        "fqr": arrays["yfqr_train_broad"],
    }
    yt = {
        "keff": arrays["yk_train_tail"],
        "fa": arrays["yfa_train_tail"],
        "fqr": arrays["yfqr_train_tail"],
    }

    X_val = arrays["X_val"]
    y_val = {
        "keff": arrays["yk_val"],
        "fa": arrays["yfa_val"],
        "fqr": arrays["yfqr_val"],
    }

    X_test = arrays["X_test"]

    if CNN_BATCH_SIZE % 2 != 0:
        raise RuntimeError("Balanced training requires an even CNN_BATCH_SIZE")

    half = CNN_BATCH_SIZE // 2
    unique_train_size = len(Xb) + len(Xt)
    steps_per_epoch = int(np.ceil(unique_train_size / CNN_BATCH_SIZE))

    broad_slots = steps_per_epoch * half
    tail_slots = steps_per_epoch * half

    if broad_slots > len(Xb):
        raise RuntimeError(
            "Current design expects broad sampling without replacement within an epoch"
        )

    if tail_slots % len(Xt) != 0:
        raise RuntimeError(
            "Current V2 Round-1 design expects an integer number of full tail passes"
        )

    tail_passes_per_epoch = tail_slots // len(Xt)

    class BalancedSequence(tf.keras.utils.Sequence):
        def __init__(self, seed):
            super().__init__()
            self.rng = np.random.default_rng(int(seed))
            self.epoch_index = 0
            self._prepare_epoch()

        def __len__(self):
            return steps_per_epoch

        def _prepare_epoch(self):
            # 4,000 distinct broad patterns per epoch.
            broad = self.rng.permutation(len(Xb))[:broad_slots]

            # Every tail pattern appears exactly four times per epoch,
            # with an independent permutation for each pass.
            tail = np.concatenate([
                self.rng.permutation(len(Xt))
                for _ in range(tail_passes_per_epoch)
            ])

            self.broad_batches = broad.reshape(steps_per_epoch, half)
            self.tail_batches = tail.reshape(steps_per_epoch, half)

            # Precompute deterministic within-batch permutations.
            self.batch_permutations = np.asarray([
                self.rng.permutation(CNN_BATCH_SIZE)
                for _ in range(steps_per_epoch)
            ])

        def __getitem__(self, idx):
            bi = self.broad_batches[idx]
            ti = self.tail_batches[idx]

            X = np.concatenate([Xb[bi], Xt[ti]], axis=0)
            targets = {
                name: np.concatenate([yb[name][bi], yt[name][ti]], axis=0)
                for name in ("keff", "fa", "fqr")
            }

            order = self.batch_permutations[idx]
            X = X[order]
            targets = {name: value[order] for name, value in targets.items()}

            return X, targets

        def on_epoch_end(self):
            self.epoch_index += 1
            self._prepare_epoch()

    train_sequence = BalancedSequence(args.seed)

    # Keep architecture/loss unchanged. Using train_size=8000 gives
    # exactly ceil(8000/32)=250 LR-schedule steps per epoch.
    model = create_model(
        train_size=unique_train_size,
        batch_size=CNN_BATCH_SIZE,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CNN_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    started = time.perf_counter()

    history = model.fit(
        train_sequence,
        validation_data=(X_val, y_val),
        epochs=CNN_EPOCHS,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )

    elapsed = time.perf_counter() - started

    history_data = {
        k: [float(v) for v in values]
        for k, values in history.history.items()
    }
    atomic_json(history_data, history_path)

    best_epoch_index = int(np.argmin(history_data["val_loss"]))
    best_epoch = best_epoch_index + 1
    best_val_loss = float(history_data["val_loss"][best_epoch_index])

    best_model = tf.keras.models.load_model(model_path, compile=False)

    scalers = {
        name: joblib.load(round_dir / f"{name}_scaler.pkl")
        for name in ("keff", "fa", "fqr")
    }

    # --------------------------------------------------------
    # Tail-validation audit
    # --------------------------------------------------------
    pred_val = best_model.predict(X_val, batch_size=128, verbose=0)

    pred_val_raw = {
        "keff": scalers["keff"].inverse_transform(pred_val[0]).ravel(),
        "fa": scalers["fa"].inverse_transform(pred_val[1]).ravel(),
        "fqr": scalers["fqr"].inverse_transform(pred_val[2]).ravel(),
    }

    truth_val = {
        "keff": arrays["yk_val_raw"],
        "fa": arrays["yfa_val_raw"],
        "fqr": arrays["yfqr_val_raw"],
    }

    validation_metrics = {
        name: metrics(truth_val[name], pred_val_raw[name])
        for name in truth_val
    }

    trueJ_val = (
        0.5 * truth_val["fa"] / 2.0
        + 0.5 * truth_val["fqr"] / 2.5
    )
    predJ_val = (
        0.5 * pred_val_raw["fa"] / 2.0
        + 0.5 * pred_val_raw["fqr"] / 2.5
    )
    validation_metrics["J"] = metrics(trueJ_val, predJ_val)

    # --------------------------------------------------------
    # Untouched broad V1-test audit
    # --------------------------------------------------------
    pred_test = best_model.predict(X_test, batch_size=128, verbose=0)

    pred_test_raw = {
        "keff": scalers["keff"].inverse_transform(pred_test[0]).ravel(),
        "fa": scalers["fa"].inverse_transform(pred_test[1]).ravel(),
        "fqr": scalers["fqr"].inverse_transform(pred_test[2]).ravel(),
    }

    truth_test = {
        "keff": arrays["yk_test_raw"],
        "fa": arrays["yfa_test_raw"],
        "fqr": arrays["yfqr_test_raw"],
    }

    broad_test_metrics = {
        name: metrics(truth_test[name], pred_test_raw[name])
        for name in truth_test
    }

    trueJ_test = (
        0.5 * truth_test["fa"] / 2.0
        + 0.5 * truth_test["fqr"] / 2.5
    )
    predJ_test = (
        0.5 * pred_test_raw["fa"] / 2.0
        + 0.5 * pred_test_raw["fqr"] / 2.5
    )
    broad_test_metrics["J"] = metrics(trueJ_test, predJ_test)

    record = {
        "seed": int(args.seed),
        "architecture": "unchanged V1 residual-attention multi-output CNN",
        "loss": "unchanged equal-weight MSE",
        "epochs_max": CNN_EPOCHS,
        "batch_size": CNN_BATCH_SIZE,
        "balanced_batch": {
            "broad_per_batch": half,
            "tail_per_batch": half,
            "steps_per_epoch": steps_per_epoch,
            "broad_exposures_per_epoch": broad_slots,
            "tail_exposures_per_epoch": tail_slots,
            "tail_full_passes_per_epoch": tail_passes_per_epoch,
            "unique_broad_training_patterns": int(len(Xb)),
            "unique_tail_training_patterns": int(len(Xt)),
        },
        "patience": CNN_PATIENCE,
        "epochs_completed": len(history_data["loss"]),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "training_wall_seconds": float(elapsed),
        "threads": int(threads),
        "inter_threads": int(inter),
        "model_path": str(model_path.resolve()),
        "tail_validation_metrics": validation_metrics,
        "untouched_broad_test_metrics": broad_test_metrics,
    }

    atomic_json(record, record_path)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
