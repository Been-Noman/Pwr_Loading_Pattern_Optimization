#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
    ),
)

from cnn_architecture import (
    CNN_BATCH_SIZE,
    CNN_EPOCHS,
    CNN_PATIENCE,
    CNN_TRAINING_SEEDS,
    configure_tensorflow,
    create_model,
    set_tensorflow_seed,
)

from data_io import atomic_json


# ============================================================
# FROZEN CNN5 DATASET HASHES
# ============================================================

EXPECTED_ARRAYS_SHA = (
    "240713fe0c24935cbd5936f20c1ec363"
    "8d51f0ae082914dafa244fc768a74b8a"
)

EXPECTED_METADATA_SHA = (
    "d8d31cb54b18af9bf88143d8d0310c924"
    "0b14515003a1598231e6317e68a82da"
)

EXPECTED_KEFF_SCALER_SHA = (
    "6693f7c46f118b0202cbf39a97603c372"
    "31a6f39f013d53e3e2ebefd8c49ae62"
)

EXPECTED_FA_SCALER_SHA = (
    "4f6c3de3a26d811edc86d6840db28116b"
    "0fbc07b3a303f5983bde5a74c2999b5"
)

EXPECTED_FQR_SCALER_SHA = (
    "83745d6598683663ca66e75476b66f4a6"
    "88cce8ea782e3e5aaada4f667263e38"
)


TAIL_DOMAIN_NAMES = [
    "historical_tail",
    "round3_acquisition",

    "v2_random",
    "v2_sa",
    "v2_sqa",

    "v2m_random",
    "v2m_sa",
    "v2m_sqa",

    "v3_random",
    "v3_sa",
    "v3_sqa",

    "v3m_random",
    "v3m_sa",
    "v3m_sqa",
]


EXPECTED_DOMAIN_COUNTS = {
    "historical_tail": 2250,
    "round3_acquisition": 2500,

    "v2_random": 270,
    "v2_sa": 270,
    "v2_sqa": 270,

    "v2m_random": 270,
    "v2m_sa": 270,
    "v2m_sqa": 270,

    "v3_random": 270,
    "v3_sa": 270,
    "v3_sqa": 270,

    "v3m_random": 270,
    "v3m_sa": 270,

    "v3m_sqa": 2530,
}


# ============================================================
# HELPERS
# ============================================================

def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    rho = (
        spearmanr(
            y_true,
            y_pred,
        ).statistic
        if len(y_true) > 1
        else float("nan")
    )

    return {
        "MAE":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),

        "RMSE":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                )
            ),

        "R2":
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            )
            if len(y_true) > 1
            else float("nan"),

        "Spearman_rho":
            float(
                rho
            ),

        "bias_true_minus_pred":
            float(
                np.mean(
                    y_true
                    - y_pred
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--round-dir",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
    )

    args = parser.parse_args()


    if (
        args.seed
        not in CNN_TRAINING_SEEDS
    ):

        raise SystemExit(
            f"Seed {args.seed} is not one of "
            f"the frozen seeds "
            f"{CNN_TRAINING_SEEDS}"
        )


    # ========================================================
    # FROZEN DATASET
    # ========================================================

    round_dir = Path(
        args.round_dir
    )

    arrays_path = (
        round_dir
        / "round_arrays.npz"
    )

    metadata_path = (
        round_dir
        / "round_metadata.json"
    )

    scaler_paths = {
        "keff":
            round_dir
            / "keff_scaler.pkl",

        "fa":
            round_dir
            / "fa_scaler.pkl",

        "fqr":
            round_dir
            / "fqr_scaler.pkl",
    }


    required = [
        arrays_path,
        metadata_path,
        *scaler_paths.values(),
    ]


    for path in required:

        if not path.is_file():

            raise RuntimeError(
                f"Missing frozen CNN5 input: "
                f"{path}"
            )


    if (
        sha256(
            arrays_path
        )
        != EXPECTED_ARRAYS_SHA
    ):

        raise RuntimeError(
            "CNN5 round_arrays SHA mismatch"
        )


    if (
        sha256(
            metadata_path
        )
        != EXPECTED_METADATA_SHA
    ):

        raise RuntimeError(
            "CNN5 metadata SHA mismatch"
        )


    if (
        sha256(
            scaler_paths["keff"]
        )
        != EXPECTED_KEFF_SCALER_SHA
    ):

        raise RuntimeError(
            "CNN5 keff scaler SHA mismatch"
        )


    if (
        sha256(
            scaler_paths["fa"]
        )
        != EXPECTED_FA_SCALER_SHA
    ):

        raise RuntimeError(
            "CNN5 Fa scaler SHA mismatch"
        )


    if (
        sha256(
            scaler_paths["fqr"]
        )
        != EXPECTED_FQR_SCALER_SHA
    ):

        raise RuntimeError(
            "CNN5 Fq_r scaler SHA mismatch"
        )


    metadata = json.loads(
        metadata_path.read_text()
    )


    if (
        int(
            metadata[
                "training_unique_total"
            ]
        )
        != 17250
    ):

        raise RuntimeError(
            "CNN5 training size != 17250"
        )


    if (
        int(
            metadata[
                "cycle5_locked_point_test"
            ]
        )
        != 600
    ):

        raise RuntimeError(
            "CNN5 locked test != 600"
        )


    if bool(
        metadata[
            "cycle5_locked_used_for_training"
        ]
    ):

        raise RuntimeError(
            "Locked test marked for training"
        )


    # ========================================================
    # CONFIGURE TENSORFLOW
    # ========================================================

    threads, inter = (
        configure_tensorflow(
            args.threads
        )
    )

    set_tensorflow_seed(
        args.seed
    )


    import tensorflow as tf


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    model_dir = (
        round_dir
        / "models"
    )

    history_dir = (
        round_dir
        / "histories"
    )


    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    model_path = (
        model_dir
        / f"cnn_seed_{args.seed}.keras"
    )

    history_path = (
        history_dir
        / (
            f"cnn_seed_{args.seed}"
            "_history.json"
        )
    )

    record_path = (
        history_dir
        / (
            f"cnn_seed_{args.seed}"
            "_record.json"
        )
    )


    # Production training is single-shot.
    if (
        model_path.exists()
        or history_path.exists()
        or record_path.exists()
    ):

        raise RuntimeError(
            f"Seed {args.seed}: "
            "CNN5 output already exists"
        )


    # ========================================================
    # LOAD ARRAYS
    # ========================================================

    arrays = np.load(
        arrays_path,
        allow_pickle=False,
    )


    Xb = np.asarray(
        arrays[
            "X_train_broad"
        ],
        dtype=np.float32,
    )

    Xt = np.asarray(
        arrays[
            "X_train_tail"
        ],
        dtype=np.float32,
    )


    yb = {
        "keff":
            np.asarray(
                arrays[
                    "yk_train_broad"
                ],
                dtype=np.float32,
            ),

        "fa":
            np.asarray(
                arrays[
                    "yfa_train_broad"
                ],
                dtype=np.float32,
            ),

        "fqr":
            np.asarray(
                arrays[
                    "yfqr_train_broad"
                ],
                dtype=np.float32,
            ),
    }


    yt = {
        "keff":
            np.asarray(
                arrays[
                    "yk_train_tail"
                ],
                dtype=np.float32,
            ),

        "fa":
            np.asarray(
                arrays[
                    "yfa_train_tail"
                ],
                dtype=np.float32,
            ),

        "fqr":
            np.asarray(
                arrays[
                    "yfqr_train_tail"
                ],
                dtype=np.float32,
            ),
    }


    tail_domain_id = np.asarray(
        arrays[
            "tail_domain_id"
        ],
        dtype=np.int64,
    )


    stored_domain_names = [
        str(x)
        for x in arrays[
            "tail_domain_names"
        ].tolist()
    ]


    if (
        stored_domain_names
        != TAIL_DOMAIN_NAMES
    ):

        raise RuntimeError(
            "CNN5 tail-domain names changed"
        )


    if len(
        Xb
    ) != 7000:

        raise RuntimeError(
            f"Broad training={len(Xb)}, "
            "expected 7000"
        )


    if len(
        Xt
    ) != 10250:

        raise RuntimeError(
            f"Tail training={len(Xt)}, "
            "expected 10250"
        )


    if len(
        tail_domain_id
    ) != len(
        Xt
    ):

        raise RuntimeError(
            "tail_domain_id length mismatch"
        )


    # ========================================================
    # VERIFY DOMAIN COUNTS
    # ========================================================

    observed_domain_counts = Counter(
        int(x)
        for x in tail_domain_id
    )


    for domain_id, name in enumerate(
        TAIL_DOMAIN_NAMES
    ):

        observed = int(
            observed_domain_counts[
                domain_id
            ]
        )

        expected = int(
            EXPECTED_DOMAIN_COUNTS[
                name
            ]
        )


        if (
            observed
            != expected
        ):

            raise RuntimeError(
                f"Tail domain {name}: "
                f"{observed} != {expected}"
            )


    if len(
        observed_domain_counts
    ) != 14:

        raise RuntimeError(
            "Tail-domain count != 14"
        )


    # ========================================================
    # VALIDATION ONLY
    #
    # Locked Cycle5 600 is deliberately not loaded or
    # evaluated during training.
    # ========================================================

    X_val = np.asarray(
        arrays[
            "X_val"
        ],
        dtype=np.float32,
    )


    y_val = {
        "keff":
            np.asarray(
                arrays[
                    "yk_val"
                ],
                dtype=np.float32,
            ),

        "fa":
            np.asarray(
                arrays[
                    "yfa_val"
                ],
                dtype=np.float32,
            ),

        "fqr":
            np.asarray(
                arrays[
                    "yfqr_val"
                ],
                dtype=np.float32,
            ),
    }


    if len(
        X_val
    ) != 125:

        raise RuntimeError(
            "CNN5 validation size != 125"
        )


    # ========================================================
    # FROZEN 50/50 BROAD-TAIL BATCH DESIGN
    # ========================================================

    if (
        CNN_BATCH_SIZE % 2
        != 0
    ):

        raise RuntimeError(
            "CNN batch size must be even"
        )


    half = (
        CNN_BATCH_SIZE
        // 2
    )


    if half != 16:

        raise RuntimeError(
            f"Expected half-batch 16, got {half}"
        )


    unique_train_size = (
        len(Xb)
        + len(Xt)
    )


    if unique_train_size != 17250:

        raise RuntimeError(
            "CNN5 unique train size != 17250"
        )


    nominal_steps = int(
        np.ceil(
            unique_train_size
            / CNN_BATCH_SIZE
        )
    )


    broad_capacity_steps = (
        len(Xb)
        // half
    )


    steps_per_epoch = min(
        nominal_steps,
        broad_capacity_steps,
    )


    # 17250 / 32 -> nominal 540
    # 7000 / 16 -> broad capacity 437
    #
    # therefore 437 production steps/epoch.

    if steps_per_epoch != 437:

        raise RuntimeError(
            f"Expected 437 steps/epoch, "
            f"got {steps_per_epoch}"
        )


    broad_slots = (
        steps_per_epoch
        * half
    )

    tail_slots = (
        steps_per_epoch
        * half
    )


    if broad_slots != 6992:

        raise RuntimeError(
            "Broad slots/epoch != 6992"
        )


    if tail_slots != 6992:

        raise RuntimeError(
            "Tail slots/epoch != 6992"
        )


    if broad_slots > len(Xb):

        raise RuntimeError(
            "Broad sampling would "
            "require replacement"
        )


    # ========================================================
    # BROAD DATASET
    #
    # 6992 / 7000 broad cases per epoch.
    # No replacement within an epoch.
    # ========================================================

    broad_base = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                Xb,
                yb["keff"],
                yb["fa"],
                yb["fqr"],
            )
        )
    )


    broad_epoch = (
        broad_base

        .shuffle(
            buffer_size=len(Xb),
            seed=int(
                args.seed
            ),
            reshuffle_each_iteration=True,
        )

        .take(
            broad_slots
        )

        .batch(
            half,
            drop_remainder=True,
        )
    )


    broad_batches = (
        broad_epoch
        .repeat()
    )


    # ========================================================
    # EXACTLY BALANCED 14-DOMAIN TAIL SAMPLER
    #
    # Each domain is an infinite independently shuffled stream.
    #
    # Selector:
    #   shuffle(0..13) -> repeat
    #
    # Therefore every complete 14-sample selector cycle uses
    # every domain exactly once.
    #
    # 6992 tail exposures:
    #
    #   6992 = 14 * 499 + 6
    #
    # Thus every tail domain receives either:
    #
    #   499 or 500 exposures per epoch.
    #
    # Raw count cannot control gradient exposure.
    # ========================================================

    domain_datasets = []


    for domain_id, name in enumerate(
        TAIL_DOMAIN_NAMES
    ):

        idx = np.flatnonzero(
            tail_domain_id
            == domain_id
        )


        if len(
            idx
        ) != EXPECTED_DOMAIN_COUNTS[
            name
        ]:

            raise RuntimeError(
                f"{name}: domain index count mismatch"
            )


        ds = (
            tf.data.Dataset
            .from_tensor_slices(
                (
                    Xt[
                        idx
                    ],

                    yt[
                        "keff"
                    ][
                        idx
                    ],

                    yt[
                        "fa"
                    ][
                        idx
                    ],

                    yt[
                        "fqr"
                    ][
                        idx
                    ],
                )
            )

            .shuffle(
                buffer_size=len(
                    idx
                ),
                seed=(
                    int(
                        args.seed
                    )
                    +
                    1000003
                    +
                    1009
                    * domain_id
                ),
                reshuffle_each_iteration=True,
            )

            .repeat()
        )


        domain_datasets.append(
            ds
        )


    selector = (
        tf.data.Dataset
        .range(
            len(
                TAIL_DOMAIN_NAMES
            )
        )

        .shuffle(
            buffer_size=len(
                TAIL_DOMAIN_NAMES
            ),
            seed=(
                int(
                    args.seed
                )
                +
                2000003
            ),
            reshuffle_each_iteration=True,
        )

        .repeat()
    )


    tail_balanced = (
        tf.data.Dataset
        .choose_from_datasets(
            domain_datasets,
            selector,
            stop_on_empty_dataset=True,
        )
    )


    tail_epoch = (
        tail_balanced

        .take(
            tail_slots
        )

        .batch(
            half,
            drop_remainder=True,
        )
    )


    tail_batches = (
        tail_epoch
        .repeat()
    )


    # ========================================================
    # MERGE EXACT 16 BROAD + 16 BALANCED-TAIL
    # ========================================================

    def merge_batches(
        broad,
        tail,
    ):

        X = tf.concat(
            [
                broad[0],
                tail[0],
            ],
            axis=0,
        )


        targets = {
            "keff":
                tf.concat(
                    [
                        broad[1],
                        tail[1],
                    ],
                    axis=0,
                ),

            "fa":
                tf.concat(
                    [
                        broad[2],
                        tail[2],
                    ],
                    axis=0,
                ),

            "fqr":
                tf.concat(
                    [
                        broad[3],
                        tail[3],
                    ],
                    axis=0,
                ),
        }


        return (
            X,
            targets,
        )


    train_dataset = (
        tf.data.Dataset
        .zip(
            (
                broad_batches,
                tail_batches,
            )
        )

        .map(
            merge_batches,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        .prefetch(
            tf.data.AUTOTUNE
        )
    )


    options = tf.data.Options()

    options.experimental_deterministic = True

    train_dataset = (
        train_dataset
        .with_options(
            options
        )
    )


    # ========================================================
    # UNCHANGED CNN ARCHITECTURE / LOSS
    # ========================================================

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


    # ========================================================
    # TRAIN
    # ========================================================

    started = time.perf_counter()


    history = model.fit(
        train_dataset,

        validation_data=(
            X_val,
            y_val,
        ),

        epochs=CNN_EPOCHS,

        steps_per_epoch=
            steps_per_epoch,

        callbacks=
            callbacks,

        verbose=2,
    )


    elapsed = (
        time.perf_counter()
        - started
    )


    history_data = {
        key: [
            float(v)
            for v in values
        ]
        for key, values
        in history.history.items()
    }


    atomic_json(
        history_data,
        history_path,
    )


    best_epoch_index = int(
        np.argmin(
            history_data[
                "val_loss"
            ]
        )
    )


    best_epoch = (
        best_epoch_index
        + 1
    )


    best_val_loss = float(
        history_data[
            "val_loss"
        ][
            best_epoch_index
        ]
    )


    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    best_model = (
        tf.keras.models
        .load_model(
            model_path,
            compile=False,
        )
    )


    scalers = {
        name:
            joblib.load(
                scaler_paths[
                    name
                ]
            )
        for name in (
            "keff",
            "fa",
            "fqr",
        )
    }


    # ========================================================
    # VALIDATION AUDIT ONLY
    #
    # No broad-test inspection.
    # No Cycle5 locked-test inspection.
    # ========================================================

    pred_val = (
        best_model.predict(
            X_val,
            batch_size=128,
            verbose=0,
        )
    )


    pred_val_raw = {
        "keff":
            scalers[
                "keff"
            ].inverse_transform(
                pred_val[0]
            ).ravel(),

        "fa":
            scalers[
                "fa"
            ].inverse_transform(
                pred_val[1]
            ).ravel(),

        "fqr":
            scalers[
                "fqr"
            ].inverse_transform(
                pred_val[2]
            ).ravel(),
    }


    truth_val = {
        "keff":
            np.asarray(
                arrays[
                    "yk_val_raw"
                ],
                dtype=float,
            ),

        "fa":
            np.asarray(
                arrays[
                    "yfa_val_raw"
                ],
                dtype=float,
            ),

        "fqr":
            np.asarray(
                arrays[
                    "yfqr_val_raw"
                ],
                dtype=float,
            ),
    }


    validation_metrics = {
        name:
            metrics(
                truth_val[
                    name
                ],
                pred_val_raw[
                    name
                ],
            )
        for name in truth_val
    }


    true_J = (
        0.5
        * truth_val[
            "fa"
        ]
        / 2.0

        +

        0.5
        * truth_val[
            "fqr"
        ]
        / 2.5
    )


    pred_J = (
        0.5
        * pred_val_raw[
            "fa"
        ]
        / 2.0

        +

        0.5
        * pred_val_raw[
            "fqr"
        ]
        / 2.5
    )


    validation_metrics[
        "J"
    ] = metrics(
        true_J,
        pred_J,
    )


    # ========================================================
    # PRODUCTION RECORD
    # ========================================================

    domain_raw_counts = {
        name:
            int(
                EXPECTED_DOMAIN_COUNTS[
                    name
                ]
            )
        for name in
        TAIL_DOMAIN_NAMES
    }


    record = {
        "seed":
            int(
                args.seed
            ),

        "architecture":
            (
                "unchanged V1 residual-attention "
                "multi-output CNN"
            ),

        "loss":
            "unchanged equal-weight MSE",

        "training_data":
            {
                "unique_total":
                    17250,

                "broad":
                    7000,

                "tail":
                    10250,

                "cycle5_new_development":
                    3000,

                "cycle5_locked600_used":
                    False,
            },

        "sampling":
            {
                "batch_size":
                    int(
                        CNN_BATCH_SIZE
                    ),

                "broad_per_batch":
                    int(
                        half
                    ),

                "tail_per_batch":
                    int(
                        half
                    ),

                "steps_per_epoch":
                    int(
                        steps_per_epoch
                    ),

                "broad_exposures_per_epoch":
                    int(
                        broad_slots
                    ),

                "tail_exposures_per_epoch":
                    int(
                        tail_slots
                    ),

                "broad_sampling":
                    (
                        "without replacement "
                        "within each epoch"
                    ),

                "tail_domain_sampling":
                    (
                        "exact cyclic equal-domain "
                        "selector"
                    ),

                "tail_domain_count":
                    14,

                "tail_exposure_floor_per_domain":
                    int(
                        tail_slots
                        //
                        len(
                            TAIL_DOMAIN_NAMES
                        )
                    ),

                "tail_extra_domain_exposures_per_epoch":
                    int(
                        tail_slots
                        %
                        len(
                            TAIL_DOMAIN_NAMES
                        )
                    ),

                "tail_domain_raw_counts":
                    domain_raw_counts,

                "v3m_sqa_bias_correction":
                    (
                        "Round4 2500 + Cycle5 30 "
                        "share one V3M-SQA domain; "
                        "raw count does not determine "
                        "sampling probability"
                    ),

                "pipeline":
                    "native_tf_data",
            },

        "epochs_max":
            int(
                CNN_EPOCHS
            ),

        "patience":
            int(
                CNN_PATIENCE
            ),

        "epochs_completed":
            int(
                len(
                    history_data[
                        "loss"
                    ]
                )
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_validation_loss":
            float(
                best_val_loss
            ),

        "training_wall_seconds":
            float(
                elapsed
            ),

        "threads":
            int(
                threads
            ),

        "inter_threads":
            int(
                inter
            ),

        "model_path":
            str(
                model_path.resolve()
            ),

        "tail_validation_metrics":
            validation_metrics,

        "broad_test_audit":
            "NOT_RUN",

        "cycle5_locked600_audit":
            "NOT_RUN",

        "round_arrays_sha256":
            EXPECTED_ARRAYS_SHA,

        "round_metadata_sha256":
            EXPECTED_METADATA_SHA,
    }


    atomic_json(
        record,
        record_path,
    )


    print(
        json.dumps(
            record,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
