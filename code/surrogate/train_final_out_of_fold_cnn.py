#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

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


EXPECTED_SEEDS = [
    42,
    314,
    2718,
    31415,
    161803,
]


def sha256(path):

    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--round-dir",
        required=True,
    )

    ap.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    ap.add_argument(
        "--threads",
        type=int,
        default=None,
    )

    args = ap.parse_args()


    round_dir = Path(
        args.round_dir
    )


    if list(
        CNN_TRAINING_SEEDS
    ) != EXPECTED_SEEDS:

        raise RuntimeError(
            f"Frozen seed list changed: "
            f"{CNN_TRAINING_SEEDS}"
        )


    if args.seed not in EXPECTED_SEEDS:

        raise RuntimeError(
            f"Invalid seed {args.seed}"
        )


    arrays_path = (
        round_dir
        / "round_arrays.npz"
    )

    metadata_path = (
        round_dir
        / "round_metadata.json"
    )


    for path in (
        arrays_path,
        metadata_path,
        round_dir / "keff_scaler.pkl",
        round_dir / "fa_scaler.pkl",
        round_dir / "fqr_scaler.pkl",
    ):

        if not path.is_file():

            raise RuntimeError(
                f"Missing fold artifact: {path}"
            )


    metadata = json.loads(
        metadata_path.read_text()
    )


    # ========================================================
    # STRICT FOLD CONTRACT
    # ========================================================

    required_metadata = {
        "broad_fit":
            7000,

        "tail_fit":
            7380,

        "actual_fit_total":
            14380,

        "internal_tail_validation":
            820,

        "crossfit_holdout":
            2050,

        "old_round4_holdout":
            1450,

        "new_cycle5_holdout":
            600,
    }


    for key, expected in required_metadata.items():

        actual = int(
            metadata[
                key
            ]
        )


        if actual != expected:

            raise RuntimeError(
                f"{round_dir.name}: "
                f"{key}={actual}, "
                f"expected={expected}"
            )


    if bool(
        metadata[
            "crossfit_holdout_used_for_fit"
        ]
    ):

        raise RuntimeError(
            "Crossfit holdout marked as fit data"
        )


    if bool(
        metadata[
            "cycle5_locked600_present"
        ]
    ):

        raise RuntimeError(
            "Locked600 unexpectedly present"
        )


    if bool(
        metadata[
            "cycle5_locked600_used_for_fit"
        ]
    ):

        raise RuntimeError(
            "Locked600 marked as fitting data"
        )


    # ========================================================
    # LOAD FOLD ARRAYS
    # ========================================================

    a = np.load(
        arrays_path,
        allow_pickle=False,
    )


    Xb = np.asarray(
        a[
            "X_train_broad"
        ],
        dtype=np.float32,
    )


    Xt = np.asarray(
        a[
            "X_train_tail"
        ],
        dtype=np.float32,
    )


    yb = {
        "keff":
            np.asarray(
                a[
                    "yk_train_broad"
                ],
                dtype=np.float32,
            ),

        "fa":
            np.asarray(
                a[
                    "yfa_train_broad"
                ],
                dtype=np.float32,
            ),

        "fqr":
            np.asarray(
                a[
                    "yfqr_train_broad"
                ],
                dtype=np.float32,
            ),
    }


    yt = {
        "keff":
            np.asarray(
                a[
                    "yk_train_tail"
                ],
                dtype=np.float32,
            ),

        "fa":
            np.asarray(
                a[
                    "yfa_train_tail"
                ],
                dtype=np.float32,
            ),

        "fqr":
            np.asarray(
                a[
                    "yfqr_train_tail"
                ],
                dtype=np.float32,
            ),
    }


    tail_domain_id = np.asarray(
        a[
            "tail_domain_id"
        ],
        dtype=np.int64,
    )


    stored_names = [
        str(x)
        for x in
        a[
            "tail_domain_names"
        ].tolist()
    ]


    if stored_names != TAIL_DOMAIN_NAMES:

        raise RuntimeError(
            "Tail-domain ordering changed"
        )


    if len(Xb) != 7000:

        raise RuntimeError(
            f"Broad rows={len(Xb)}"
        )


    if len(Xt) != 7380:

        raise RuntimeError(
            f"Tail rows={len(Xt)}"
        )


    if len(
        tail_domain_id
    ) != 7380:

        raise RuntimeError(
            "tail_domain_id length mismatch"
        )


    domain_counts = Counter(
        int(x)
        for x in tail_domain_id
    )


    if set(
        domain_counts
    ) != set(
        range(14)
    ):

        raise RuntimeError(
            f"Not all 14 domains available: "
            f"{dict(domain_counts)}"
        )


    if min(
        domain_counts.values()
    ) <= 0:

        raise RuntimeError(
            "Empty CNN5 sampling domain"
        )


    # ========================================================
    # INTERNAL VALIDATION ONLY
    #
    # Strict OOF holdout is NOT loaded during training.
    # ========================================================

    X_val = np.asarray(
        a[
            "X_val"
        ],
        dtype=np.float32,
    )


    y_val = {
        "keff":
            np.asarray(
                a[
                    "yk_val"
                ],
                dtype=np.float32,
            ),

        "fa":
            np.asarray(
                a[
                    "yfa_val"
                ],
                dtype=np.float32,
            ),

        "fqr":
            np.asarray(
                a[
                    "yfqr_val"
                ],
                dtype=np.float32,
            ),
    }


    if len(
        X_val
    ) != 820:

        raise RuntimeError(
            f"Internal validation={len(X_val)}"
        )


    # ========================================================
    # SAME CNN5 SAMPLER
    #
    # Batch = 32:
    #   16 broad
    #   16 tail
    #
    # 7000 broad rows limit the epoch:
    #
    # floor(7000 / 16) = 437 batches
    #
    # Hence:
    #   6992 broad exposures
    #   6992 tail exposures
    #
    # Tail domain exposure:
    #
    # 6992 = 14*499 + 6
    #
    # => every domain gets 499 or 500 exposures.
    # ========================================================

    if CNN_BATCH_SIZE != 32:

        raise RuntimeError(
            f"Expected batch size 32, "
            f"got {CNN_BATCH_SIZE}"
        )


    half_batch = 16

    steps_per_epoch = 437

    broad_slots = (
        steps_per_epoch
        * half_batch
    )

    tail_slots = broad_slots


    if broad_slots != 6992:

        raise RuntimeError(
            "Unexpected broad exposure count"
        )


    if tail_slots != 6992:

        raise RuntimeError(
            "Unexpected tail exposure count"
        )


    # ========================================================
    # TENSORFLOW CONFIGURATION
    # ========================================================

    threads, inter_threads = (
        configure_tensorflow(
            args.threads
        )
    )


    set_tensorflow_seed(
        args.seed
    )


    import tensorflow as tf


    # ========================================================
    # BROAD STREAM
    # ========================================================

    broad_stream = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                Xb,
                yb[
                    "keff"
                ],
                yb[
                    "fa"
                ],
                yb[
                    "fqr"
                ],
            )
        )

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
            half_batch,
            drop_remainder=True,
        )

        .repeat()
    )


    # ========================================================
    # 14 INDEPENDENT TAIL-DOMAIN STREAMS
    # ========================================================

    domain_datasets = []


    for domain_id in range(
        14
    ):

        idx = np.flatnonzero(
            tail_domain_id
            == domain_id
        )


        if len(
            idx
        ) == 0:

            raise RuntimeError(
                f"Empty domain {domain_id}"
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


    # Every complete selector cycle contains
    # all 14 domains exactly once.
    selector = (
        tf.data.Dataset
        .range(
            14
        )

        .shuffle(
            buffer_size=14,

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


    balanced_tail = (
        tf.data.Dataset
        .choose_from_datasets(
            domain_datasets,
            selector,
            stop_on_empty_dataset=True,
        )

        .take(
            tail_slots
        )

        .batch(
            half_batch,
            drop_remainder=True,
        )

        .repeat()
    )


    # ========================================================
    # MERGE 16 + 16
    # ========================================================

    def merge_batch(
        broad,
        tail,
    ):

        X = tf.concat(
            [
                broad[
                    0
                ],
                tail[
                    0
                ],
            ],
            axis=0,
        )


        y = {
            "keff":
                tf.concat(
                    [
                        broad[
                            1
                        ],
                        tail[
                            1
                        ],
                    ],
                    axis=0,
                ),

            "fa":
                tf.concat(
                    [
                        broad[
                            2
                        ],
                        tail[
                            2
                        ],
                    ],
                    axis=0,
                ),

            "fqr":
                tf.concat(
                    [
                        broad[
                            3
                        ],
                        tail[
                            3
                        ],
                    ],
                    axis=0,
                ),
        }


        return (
            X,
            y,
        )


    train_dataset = (
        tf.data.Dataset
        .zip(
            (
                broad_stream,
                balanced_tail,
            )
        )

        .map(
            merge_batch,
            num_parallel_calls=
                tf.data.AUTOTUNE,
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
    # UNCHANGED CNN ARCHITECTURE
    # ========================================================

    model = create_model(
        train_size=14380,
        batch_size=CNN_BATCH_SIZE,
    )


    models_dir = (
        round_dir
        / "models"
    )

    histories_dir = (
        round_dir
        / "histories"
    )


    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    histories_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    model_path = (
        models_dir
        / f"cnn_seed_{args.seed}.keras"
    )


    history_path = (
        histories_dir
        / (
            f"cnn_seed_{args.seed}"
            "_history.json"
        )
    )


    record_path = (
        histories_dir
        / (
            f"cnn_seed_{args.seed}"
            "_record.json"
        )
    )


    for path in (
        model_path,
        history_path,
        record_path,
    ):

        if path.exists():

            raise RuntimeError(
                f"Refusing to overwrite "
                f"existing output: {path}"
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

    started = (
        time.perf_counter()
    )


    history = model.fit(
        train_dataset,

        validation_data=(
            X_val,
            y_val,
        ),

        epochs=
            CNN_EPOCHS,

        steps_per_epoch=
            steps_per_epoch,

        callbacks=
            callbacks,

        verbose=2,
    )


    wall_seconds = (
        time.perf_counter()
        - started
    )


    history_json = {
        key: [
            float(v)
            for v in values
        ]
        for key, values
        in history.history.items()
    }


    atomic_json(
        history_json,
        history_path,
    )


    best_idx = int(
        np.argmin(
            history_json[
                "val_loss"
            ]
        )
    )


    best_epoch = (
        best_idx
        + 1
    )


    best_val_loss = float(
        history_json[
            "val_loss"
        ][
            best_idx
        ]
    )


    record = {
        "stage":
            "CYCLE5_STRICT_OOF_CNN_TRAINING",

        "fold":
            int(
                metadata[
                    "fold"
                ]
            ),

        "seed":
            int(
                args.seed
            ),

        "architecture":
            (
                "unchanged CNN5 "
                "residual-attention CNN"
            ),

        "training_rows":
            14380,

        "broad_fit":
            7000,

        "tail_fit":
            7380,

        "internal_validation":
            820,

        "strict_oof_holdout":
            2050,

        "strict_oof_holdout_used_for_training":
            False,

        "cycle5_locked600_used":
            False,

        "batch_size":
            32,

        "broad_per_batch":
            16,

        "tail_per_batch":
            16,

        "steps_per_epoch":
            437,

        "broad_exposures_per_epoch":
            6992,

        "tail_exposures_per_epoch":
            6992,

        "tail_domains":
            14,

        "tail_domain_exposure_floor":
            499,

        "tail_domains_receiving_one_extra_exposure":
            6,

        "domain_sampling":
            (
                "equal expected exposure; "
                "complete selector cycle "
                "contains all 14 domains"
            ),

        "raw_tail_domain_counts":
            {
                TAIL_DOMAIN_NAMES[
                    domain_id
                ]:
                    int(
                        domain_counts[
                            domain_id
                        ]
                    )
                for domain_id in range(
                    14
                )
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
                    history_json[
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

        "wall_seconds":
            float(
                wall_seconds
            ),

        "threads":
            int(
                threads
            ),

        "inter_threads":
            int(
                inter_threads
            ),

        "round_arrays_sha256":
            sha256(
                arrays_path
            ),

        "round_metadata_sha256":
            sha256(
                metadata_path
            ),

        "model_sha256":
            sha256(
                model_path
            ),
    }


    atomic_json(
        record,
        record_path,
    )


    print(
        "=" * 72
    )

    print(
        "CYCLE5 STRICT-OOF CNN COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "fold                 =",
        record[
            "fold"
        ],
    )

    print(
        "seed                 =",
        args.seed,
    )

    print(
        "training rows        = 14380"
    )

    print(
        "holdout used         = NO"
    )

    print(
        "locked600 used       = NO"
    )

    print(
        "best epoch           =",
        best_epoch,
    )

    print(
        "best val loss        =",
        best_val_loss,
    )

    print(
        "model SHA256         =",
        record[
            "model_sha256"
        ],
    )

    print(
        "CYCLE5_STRICT_OOF_CNN_DONE"
    )


if __name__ == "__main__":

    main()
