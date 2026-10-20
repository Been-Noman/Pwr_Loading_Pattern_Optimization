#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np

from scipy.stats import spearmanr

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error


OUTPUTS = (
    "keff",
    "Fa",
    "Fq_r",
)

FA_LIMIT = 2.0
FQR_LIMIT = 2.5

EXPECTED_ROWS = 10250
EXPECTED_FOLDS = 5
EXPECTED_PER_FOLD = 2050
EXPECTED_RESIDUAL_TRAIN = 8200

EXPECTED_OOF_SHA = (
    "c963028d84d0b5cb5fe710c27b2ed07a"
    "b181df749fe4bac9cc85212b5e13a72f"
)

EXPECTED_AUDIT_SHA = (
    "4f2881be9435881568ea39e2b2c1544d"
    "a73430c8daa427e63d702bd8976a4f48"
)


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(
                chunk
            )

    return h.hexdigest()


def write_json(
    path: Path,
    obj,
):

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def safe_rho(
    a,
    b,
):

    r = spearmanr(
        a,
        b,
    ).statistic

    if not np.isfinite(
        r
    ):

        return None

    return float(
        r
    )


def make_features(
    pattern,
    mu,
    sd,
):

    # EXACT established residual feature construction:
    #
    # 193 loading-pattern genes
    # + 3 CNN ensemble means
    # + 3 CNN ensemble SDs
    # + predicted J
    # + Fa margin
    # + Fq_r margin
    #
    # = 202 features.
    #
    # No fold ID.
    # No source-round ID.
    # No acquisition stratum.
    # No arm label.
    # No OpenMC truth.

    J = (
        0.5
        * mu[
            :,
            1
        ]
        / FA_LIMIT

        +

        0.5
        * mu[
            :,
            2
        ]
        / FQR_LIMIT
    )[
        :,
        None
    ]


    fa_margin = (
        FA_LIMIT
        -
        mu[
            :,
            1
        ]
    )[
        :,
        None
    ]


    fqr_margin = (
        FQR_LIMIT
        -
        mu[
            :,
            2
        ]
    )[
        :,
        None
    ]


    X = np.concatenate(
        [
            pattern.astype(
                np.float32
            ),

            mu.astype(
                np.float32
            ),

            sd.astype(
                np.float32
            ),

            J.astype(
                np.float32
            ),

            fa_margin.astype(
                np.float32
            ),

            fqr_margin.astype(
                np.float32
            ),
        ],
        axis=1,
    )


    if X.shape[
        1
    ] != 202:

        raise RuntimeError(
            f"Residual feature count "
            f"{X.shape[1]} != 202"
        )


    return X


def top_fraction_overlap(
    actual,
    predicted,
    frac=0.20,
):

    n = len(
        actual
    )

    k = max(
        1,
        int(
            np.ceil(
                frac
                * n
            )
        ),
    )


    actual_top = set(
        np.argsort(
            actual
        )[
            -k:
        ].tolist()
    )


    predicted_top = set(
        np.argsort(
            predicted
        )[
            -k:
        ].tolist()
    )


    overlap = len(
        actual_top
        &
        predicted_top
    )


    return {
        "fraction":
            float(
                frac
            ),

        "k":
            int(
                k
            ),

        "overlap":
            int(
                overlap
            ),

        "precision_recall":
            float(
                overlap
                / k
            ),

        "random_expectation":
            float(
                frac
            ),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--oof",
        required=True,
    )

    ap.add_argument(
        "--oof-audit",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()


    oof_path = Path(
        args.oof
    )

    audit_path = Path(
        args.oof_audit
    )

    output_dir = Path(
        args.output_dir
    )

    tmp_dir = Path(
        str(
            output_dir
        )
        + ".__building__"
    )


    # ========================================================
    # FROZEN INPUT CONTRACT
    # ========================================================

    if sha256_file(
        oof_path
    ) != EXPECTED_OOF_SHA:

        raise RuntimeError(
            "Round5 OOF SHA mismatch"
        )


    if sha256_file(
        audit_path
    ) != EXPECTED_AUDIT_SHA:

        raise RuntimeError(
            "Round5 OOF audit SHA mismatch"
        )


    if output_dir.exists():

        raise RuntimeError(
            f"Refusing to overwrite "
            f"{output_dir}"
        )


    if tmp_dir.exists():

        shutil.rmtree(
            tmp_dir
        )


    tmp_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


    z = np.load(
        oof_path,
        allow_pickle=False,
    )


    required_keys = {
        "crossfit_id",
        "fold",
        "source_round",
        "pattern",
        "truth",
        "ensemble_mean",
        "ensemble_std",
        "abs_error",
    }


    missing = (
        required_keys
        -
        set(
            z.files
        )
    )


    if missing:

        raise RuntimeError(
            f"Missing OOF keys: "
            f"{sorted(missing)}"
        )


    ids = np.asarray(
        z[
            "crossfit_id"
        ],
        dtype=np.int32,
    )


    folds = np.asarray(
        z[
            "fold"
        ],
        dtype=np.int32,
    )


    source_round = np.asarray(
        z[
            "source_round"
        ]
    ).astype(
        "U32"
    )


    pattern = np.asarray(
        z[
            "pattern"
        ],
        dtype=np.int8,
    )


    truth = np.asarray(
        z[
            "truth"
        ],
        dtype=np.float64,
    )


    mu = np.asarray(
        z[
            "ensemble_mean"
        ],
        dtype=np.float64,
    )


    sd = np.asarray(
        z[
            "ensemble_std"
        ],
        dtype=np.float64,
    )


    stored_error = np.asarray(
        z[
            "abs_error"
        ],
        dtype=np.float64,
    )


    # ========================================================
    # STRICT OOF INTEGRITY
    # ========================================================

    if not np.array_equal(
        ids,
        np.arange(
            EXPECTED_ROWS,
            dtype=np.int32,
        ),
    ):

        raise RuntimeError(
            "crossfit_id is not exactly "
            "0..10249"
        )


    if pattern.shape != (
        EXPECTED_ROWS,
        193,
    ):

        raise RuntimeError(
            f"Unexpected pattern shape "
            f"{pattern.shape}"
        )


    for name, arr in (
        (
            "truth",
            truth,
        ),
        (
            "ensemble_mean",
            mu,
        ),
        (
            "ensemble_std",
            sd,
        ),
        (
            "abs_error",
            stored_error,
        ),
    ):

        if arr.shape != (
            EXPECTED_ROWS,
            3,
        ):

            raise RuntimeError(
                f"{name} shape "
                f"{arr.shape} != "
                "(10250, 3)"
            )


        if not np.all(
            np.isfinite(
                arr
            )
        ):

            raise RuntimeError(
                f"{name} contains "
                "non-finite values"
            )


    actual_error = np.abs(
        truth
        -
        mu
    )


    if not np.allclose(
        stored_error,
        actual_error,
        rtol=0.0,
        atol=1e-12,
    ):

        raise RuntimeError(
            "Stored abs_error does "
            "not equal |truth - mean|"
        )


    unique_folds = set(
        np.unique(
            folds
        ).tolist()
    )


    if unique_folds != set(
        range(
            EXPECTED_FOLDS
        )
    ):

        raise RuntimeError(
            f"Unexpected fold IDs: "
            f"{sorted(unique_folds)}"
        )


    for fold in range(
        EXPECTED_FOLDS
    ):

        n = int(
            np.sum(
                folds
                == fold
            )
        )


        if n != EXPECTED_PER_FOLD:

            raise RuntimeError(
                f"Fold {fold}: "
                f"{n} != "
                f"{EXPECTED_PER_FOLD}"
            )


    X = make_features(
        pattern,
        mu,
        sd,
    )


    if X.shape != (
        EXPECTED_ROWS,
        202,
    ):

        raise RuntimeError(
            f"Residual design shape "
            f"{X.shape}"
        )


    if not np.all(
        np.isfinite(
            X
        )
    ):

        raise RuntimeError(
            "Residual features contain "
            "non-finite values"
        )


    # ========================================================
    # SECOND-LEVEL STRICT 5-FOLD OOF RESIDUAL DIAGNOSTIC
    #
    # CNN predictions are already strict OOF.
    #
    # For each held CNN fold:
    # residual model trains ONLY on the other four folds'
    # strict-OOF errors and predicts the held fold.
    #
    # Hyperparameters match the established diagnostic:
    #
    # n_estimators=750
    # min_samples_leaf=5
    # max_features=0.7
    # random_state=20260 + held_fold*10 + output
    # ========================================================

    predicted_error_oof = np.full_like(
        actual_error,
        np.nan,
        dtype=np.float64,
    )


    fold_report = {}


    for held_fold in range(
        EXPECTED_FOLDS
    ):

        train_idx = np.flatnonzero(
            folds
            != held_fold
        )


        test_idx = np.flatnonzero(
            folds
            == held_fold
        )


        if len(
            train_idx
        ) != EXPECTED_RESIDUAL_TRAIN:

            raise RuntimeError(
                f"Fold {held_fold}: "
                f"residual train="
                f"{len(train_idx)}"
            )


        if len(
            test_idx
        ) != EXPECTED_PER_FOLD:

            raise RuntimeError(
                f"Fold {held_fold}: "
                f"residual holdout="
                f"{len(test_idx)}"
            )


        fold_report[
            str(
                held_fold
            )
        ] = {}


        for j, name in enumerate(
            OUTPUTS
        ):

            model = ExtraTreesRegressor(
                n_estimators=750,
                min_samples_leaf=5,
                max_features=0.7,

                random_state=(
                    20260
                    +
                    held_fold
                    * 10
                    +
                    j
                ),

                n_jobs=-1,
            )


            model.fit(
                X[
                    train_idx
                ],
                actual_error[
                    train_idx,
                    j
                ],
            )


            pred = np.maximum(
                model.predict(
                    X[
                        test_idx
                    ]
                ),
                1e-12,
            )


            predicted_error_oof[
                test_idx,
                j
            ] = pred


            fold_report[
                str(
                    held_fold
                )
            ][
                name
            ] = {
                "n":
                    int(
                        len(
                            test_idx
                        )
                    ),

                "residual_training_rows":
                    int(
                        len(
                            train_idx
                        )
                    ),

                "error_scale_MAE":
                    float(
                        mean_absolute_error(
                            actual_error[
                                test_idx,
                                j
                            ],
                            pred,
                        )
                    ),

                "rho_actual_vs_learned_scale":
                    safe_rho(
                        actual_error[
                            test_idx,
                            j
                        ],
                        pred,
                    ),

                "rho_actual_vs_ensemble_sd":
                    safe_rho(
                        actual_error[
                            test_idx,
                            j
                        ],
                        sd[
                            test_idx,
                            j
                        ],
                    ),

                "top20_error_overlap":
                    top_fraction_overlap(
                        actual_error[
                            test_idx,
                            j
                        ],
                        pred,
                        frac=0.20,
                    ),
            }


            del model


    if np.isnan(
        predicted_error_oof
    ).any():

        raise RuntimeError(
            "Residual-model OOF predictions "
            "are incomplete"
        )


    # ========================================================
    # OVERALL STRICT RESIDUAL DIAGNOSTIC
    # ========================================================

    diagnostic_overall = {}


    for j, name in enumerate(
        OUTPUTS
    ):

        diagnostic_overall[
            name
        ] = {
            "n":
                EXPECTED_ROWS,

            "actual_error_MAE":
                float(
                    np.mean(
                        actual_error[
                            :,
                            j
                        ]
                    )
                ),

            "learned_scale_MAE":
                float(
                    mean_absolute_error(
                        actual_error[
                            :,
                            j
                        ],
                        predicted_error_oof[
                            :,
                            j
                        ],
                    )
                ),

            "rho_actual_vs_learned_scale":
                safe_rho(
                    actual_error[
                        :,
                        j
                    ],
                    predicted_error_oof[
                        :,
                        j
                    ],
                ),

            "rho_actual_vs_ensemble_sd":
                safe_rho(
                    actual_error[
                        :,
                        j
                    ],
                    sd[
                        :,
                        j
                    ],
                ),

            "top20_error_overlap":
                top_fraction_overlap(
                    actual_error[
                        :,
                        j
                    ],
                    predicted_error_oof[
                        :,
                        j
                    ],
                    frac=0.20,
                ),

            "median_actual_abs_error":
                float(
                    np.median(
                        actual_error[
                            :,
                            j
                        ]
                    )
                ),

            "median_predicted_error_scale":
                float(
                    np.median(
                        predicted_error_oof[
                            :,
                            j
                        ]
                    )
                ),
        }


    # ========================================================
    # SAVE STRICT RESIDUAL OOF DIAGNOSTIC
    # ========================================================

    diagnostic_npz = (
        tmp_dir
        / "round5_residual_oof_diagnostic.npz"
    )


    np.savez_compressed(
        diagnostic_npz,

        crossfit_id=
            ids,

        fold=
            folds,

        source_round=
            source_round,

        truth=
            truth,

        ensemble_mean=
            mu,

        ensemble_std=
            sd,

        actual_abs_error=
            actual_error,

        learned_abs_error_oof=
            predicted_error_oof,
    )


    diagnostic_report = {
        "stage":
            "CYCLE5_RESIDUAL5_STRICT_OOF_DIAGNOSTIC",

        "strict_base_cnn_oof":
            True,

        "strict_residual_model_oof":
            True,

        "rows":
            EXPECTED_ROWS,

        "folds":
            EXPECTED_FOLDS,

        "residual_train_per_fold":
            EXPECTED_RESIDUAL_TRAIN,

        "residual_holdout_per_fold":
            EXPECTED_PER_FOLD,

        "feature_count":
            202,

        "features":
            (
                "pattern193 + ensemble_mean3 + "
                "ensemble_sd3 + predicted_J1 + "
                "Fa_margin1 + Fq_r_margin1"
            ),

        "forbidden_features":
            [
                "fold_id",
                "source_round",
                "solver",
                "initializer",
                "arm",
                "selection_stratum",
                "OpenMC_truth",
            ],

        "diagnostic_model":
            {
                "type":
                    "ExtraTreesRegressor",

                "n_estimators":
                    750,

                "min_samples_leaf":
                    5,

                "max_features":
                    0.7,

                "random_state":
                    (
                        "20260 + held_fold*10 "
                        "+ output_index"
                    ),
            },

        "fold_report":
            fold_report,

        "overall":
            diagnostic_overall,

        "locked600_used":
            False,
    }


    diagnostic_json = (
        tmp_dir
        / "round5_residual_oof_audit.json"
    )


    write_json(
        diagnostic_json,
        diagnostic_report,
    )


    # ========================================================
    # PRODUCTION RESIDUAL5 MODELS
    #
    # Fit on ALL 10,250 strict CNN5 OOF residual rows.
    #
    # Same established production hyperparameters:
    #
    # 1000 trees
    # leaf 5
    # max_features 0.7
    # random_state 30300 + output
    # ========================================================

    production_hashes = {}


    for j, name in enumerate(
        OUTPUTS
    ):

        model = ExtraTreesRegressor(
            n_estimators=1000,
            min_samples_leaf=5,
            max_features=0.7,

            random_state=(
                30300
                +
                j
            ),

            n_jobs=-1,
        )


        model.fit(
            X,
            actual_error[
                :,
                j
            ],
        )


        model_path = (
            tmp_dir
            / f"residual_{name}.joblib"
        )


        joblib.dump(
            model,
            model_path,
        )


        production_hashes[
            name
        ] = sha256_file(
            model_path
        )


        del model


    # ========================================================
    # PRE-CALIBRATION METADATA
    #
    # q5 is intentionally NOT estimated here.
    # ========================================================

    metadata = {
        "version":
            "cycle5_residual5_precalibration_v1",

        "status":
            (
                "FROZEN_RESIDUAL5_MODELS_"
                "AWAITING_BALANCED_Q5_CALIBRATION"
            ),

        "training_rows":
            EXPECTED_ROWS,

        "training_target":
            (
                "absolute strict 5-fold OOF "
                "CNN5 error"
            ),

        "feature_count":
            202,

        "feature_definition":
            (
                "pattern193 + CNN5 ensemble_mean3 "
                "+ CNN5 ensemble_std3 + predicted_J "
                "+ Fa_margin + Fq_r_margin"
            ),

        "feature_excludes":
            [
                "fold_id",
                "source_round",
                "solver",
                "initializer",
                "arm",
                "selection_stratum",
                "OpenMC_truth",
            ],

        "residual_model":
            {
                "type":
                    "ExtraTreesRegressor",

                "n_estimators":
                    1000,

                "min_samples_leaf":
                    5,

                "max_features":
                    0.7,

                "random_state":
                    {
                        "keff":
                            30300,

                        "Fa":
                            30301,

                        "Fq_r":
                            30302,
                    },
            },

        "strict_residual_oof_diagnostic":
            {
                "folds":
                    5,

                "train_per_fold":
                    8200,

                "holdout_per_fold":
                    2050,

                "n_estimators":
                    750,

                "audit_file":
                    "round5_residual_oof_audit.json",

                "prediction_file":
                    "round5_residual_oof_diagnostic.npz",
            },

        "production_model_sha256":
            production_hashes,

        "source_cnn5_oof_file":
            str(
                oof_path
            ),

        "source_cnn5_oof_sha256":
            EXPECTED_OOF_SHA,

        "source_cnn5_oof_audit_sha256":
            EXPECTED_AUDIT_SHA,

        "locked600_used":
            False,

        "q5_calibrated":
            False,

        "joint95_conformal_q":
            None,

        "methodological_note":
            (
                "Residual5 is frozen before q5. "
                "The balanced 960-case q5 calibration "
                "must not alter or retrain these models."
            ),
    }


    metadata_path = (
        tmp_dir
        / "residual_safety_metadata.json"
    )


    write_json(
        metadata_path,
        metadata,
    )


    # ========================================================
    # COMPACT SUMMARY
    # ========================================================

    lines = []

    lines.append(
        "=" * 84
    )

    lines.append(
        "CYCLE 5 — RESIDUAL5 STRICT-OOF DIAGNOSTIC + PRODUCTION MODELS"
    )

    lines.append(
        "=" * 84
    )

    lines.append(
        "CNN5 strict OOF rows      = 10250"
    )

    lines.append(
        "residual features         = 202"
    )

    lines.append(
        "residual diagnostic folds = 5"
    )

    lines.append(
        "residual train / fold     = 8200"
    )

    lines.append(
        "residual holdout / fold   = 2050"
    )

    lines.append(
        "diagnostic trees          = 750"
    )

    lines.append(
        "production trees          = 1000"
    )

    lines.append(
        "locked600 used            = NO"
    )

    lines.append(
        "q5 calibrated             = NO"
    )

    lines.append("")

    lines.append(
        "STRICT RESIDUAL OOF DISCRIMINATION"
    )


    for name in OUTPUTS:

        x = diagnostic_overall[
            name
        ]


        rho_l = x[
            "rho_actual_vs_learned_scale"
        ]

        rho_s = x[
            "rho_actual_vs_ensemble_sd"
        ]


        lines.append(
            f"{name:6s} "
            f"scale_MAE="
            f"{x['learned_scale_MAE']:.8f}  "
            f"rho(learned,|err|)="
            f"{rho_l if rho_l is not None else 'NA'}  "
            f"rho(SD,|err|)="
            f"{rho_s if rho_s is not None else 'NA'}  "
            f"top20_overlap="
            f"{x['top20_error_overlap']['precision_recall']:.4f}"
        )


    lines.append("")

    lines.append(
        "PRODUCTION MODEL HASHES"
    )


    for name in OUTPUTS:

        lines.append(
            f"{name:6s} "
            f"{production_hashes[name]}"
        )


    summary_path = (
        tmp_dir
        / "round5_residual_summary.txt"
    )


    summary_path.write_text(
        "\n".join(
            lines
        )
        + "\n"
    )


    # ========================================================
    # ARTIFACT HASHES + ATOMIC FREEZE
    # ========================================================

    artifact_manifest = {
        "stage":
            "CYCLE5_RESIDUAL5_FREEZE",

        "status":
            "FROZEN_PRE_Q5_CALIBRATION",

        "files":
            {
                p.name:
                    sha256_file(
                        p
                    )
                for p in sorted(
                    tmp_dir.iterdir()
                )
                if p.is_file()
            },
    }


    freeze_path = (
        tmp_dir
        / "residual5_freeze_manifest.json"
    )


    write_json(
        freeze_path,
        artifact_manifest,
    )


    freeze_sha = sha256_file(
        freeze_path
    )


    tmp_dir.rename(
        output_dir
    )


    print(
        "\n".join(
            lines
        )
    )

    print()

    print(
        "residual metadata SHA256 =",
        sha256_file(
            output_dir
            / "residual_safety_metadata.json"
        ),
    )

    print(
        "residual audit SHA256    =",
        sha256_file(
            output_dir
            / "round5_residual_oof_audit.json"
        ),
    )

    print(
        "residual freeze SHA256   =",
        freeze_sha,
    )

    print()

    print(
        "CYCLE5_RESIDUAL5_MODELS_PRECALIBRATION_FROZEN"
    )


if __name__ == "__main__":

    main()
