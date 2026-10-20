#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(
    __import__("os").environ["PWR_PROJECT"]
)

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from geometry import canonical_d4_key


RUN = ROOT / "run/optimizer_tail_v2"

ROUND5 = (
    RUN
    / "models"
    / "round5_cnn"
)

OLD_OOF = (
    RUN
    / "crossfit_round4"
    / "oof"
    / "crossfit_round4_oof_predictions.npz"
)

CYCLE5_PROTO = (
    RUN
    / "cycle5_balanced_multiarm"
    / "protocol"
)

DEV3000 = (
    CYCLE5_PROTO
    / "cycle5_model_development_openmc_3000_v1.json"
)

FINAL_ROOT = (
    RUN
    / "crossfit_round5"
)

TMP_ROOT = (
    RUN
    / "crossfit_round5.__building__"
)

MANIFEST_NAME = (
    "tail_crossfit_5fold_manifest.json"
)

FREEZE_NAME = (
    "crossfit_round5_dataset_freeze_manifest_v1.json"
)


# ============================================================
# FROZEN CYCLE-5 HASHES
# ============================================================

EXPECTED_ARRAYS_SHA = (
    "240713fe0c24935cbd5936f20c1ec363"
    "8d51f0ae082914dafa244fc768a74b8a"
)

EXPECTED_METADATA_SHA = (
    "d8d31cb54b18af9bf88143d8d0310c924"
    "0b14515003a1598231e6317e68a82da"
)

EXPECTED_DEV_SHA = (
    "6e103543fc7e90398a6c0c5fe860b386"
    "2b6bc1ae06bb5120c64db9f147dd4e3f"
)


ARMS = [
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


STRATA = [
    "diversity",
    "boundary_uncertainty",
    "elite",
]


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


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):

            h.update(chunk)

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


def d4_sha(
    pattern,
):

    canonical = np.asarray(
        canonical_d4_key(
            np.asarray(
                pattern,
                dtype=np.int8,
            )
        ),
        dtype=np.uint8,
    )

    return hashlib.sha256(
        canonical.tobytes()
    ).hexdigest()


def physically_feasible(
    truth,
):

    k, fa, fq = map(
        float,
        truth,
    )

    return bool(
        (k >= 1.0)
        and
        (k <= 1.2)
        and
        (fa <= 2.0)
        and
        (fq <= 2.5)
    )


# ============================================================
# VERIFY FROZEN INPUTS
# ============================================================

arrays_path = (
    ROUND5
    / "round_arrays.npz"
)

metadata_path = (
    ROUND5
    / "round_metadata.json"
)


for path in (
    arrays_path,
    metadata_path,
    DEV3000,
    OLD_OOF,
):

    if not path.is_file():

        raise RuntimeError(
            f"Missing required input: {path}"
        )


if (
    sha256_file(arrays_path)
    != EXPECTED_ARRAYS_SHA
):

    raise RuntimeError(
        "CNN5 round_arrays SHA mismatch"
    )


if (
    sha256_file(metadata_path)
    != EXPECTED_METADATA_SHA
):

    raise RuntimeError(
        "CNN5 round_metadata SHA mismatch"
    )


if (
    sha256_file(DEV3000)
    != EXPECTED_DEV_SHA
):

    raise RuntimeError(
        "Cycle5 development3000 SHA mismatch"
    )


if FINAL_ROOT.exists():

    raise RuntimeError(
        f"Refusing to overwrite existing "
        f"crossfit_round5: {FINAL_ROOT}"
    )


if TMP_ROOT.exists():

    shutil.rmtree(
        TMP_ROOT
    )


TMP_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


# ============================================================
# LOAD CNN5 DATASET
# ============================================================

a = np.load(
    arrays_path,
    allow_pickle=False,
)


Xb = np.asarray(
    a["X_train_broad"],
    dtype=np.float32,
)

Xt = np.asarray(
    a["X_train_tail"],
    dtype=np.float32,
)


broad_patterns = np.asarray(
    a["broad_train_patterns"],
    dtype=np.int8,
)

tail_patterns = np.asarray(
    a["tail_train_patterns"],
    dtype=np.int8,
)


kb = np.asarray(
    a["yk_train_broad_raw"],
    dtype=np.float64,
)

fab = np.asarray(
    a["yfa_train_broad_raw"],
    dtype=np.float64,
)

fqrb = np.asarray(
    a["yfqr_train_broad_raw"],
    dtype=np.float64,
)


kt = np.asarray(
    a["yk_train_tail_raw"],
    dtype=np.float64,
)

fat = np.asarray(
    a["yfa_train_tail_raw"],
    dtype=np.float64,
)

fqrt = np.asarray(
    a["yfqr_train_tail_raw"],
    dtype=np.float64,
)


tail_domain_id = np.asarray(
    a["tail_domain_id"],
    dtype=np.int32,
)


stored_domain_names = [
    str(x)
    for x in
    a["tail_domain_names"].tolist()
]


if stored_domain_names != TAIL_DOMAIN_NAMES:

    raise RuntimeError(
        "CNN5 tail-domain ordering changed"
    )


if Xb.shape[0] != 7000:

    raise RuntimeError(
        "Expected 7000 broad rows"
    )


if Xt.shape[0] != 10250:

    raise RuntimeError(
        "Expected 10250 tail rows"
    )


if tail_patterns.shape != (
    10250,
    193,
):

    raise RuntimeError(
        f"Unexpected tail pattern shape "
        f"{tail_patterns.shape}"
    )


if len(
    tail_domain_id
) != 10250:

    raise RuntimeError(
        "tail_domain_id length mismatch"
    )


# ============================================================
# VERIFY ALL 10,250 TAIL CASES ARE D4 UNIQUE
# ============================================================

tail_d4 = np.asarray(
    [
        d4_sha(p)
        for p in tail_patterns
    ],
    dtype="U64",
)


if len(
    np.unique(
        tail_d4
    )
) != 10250:

    raise RuntimeError(
        "Tail10250 is not D4 unique"
    )


# ============================================================
# LOAD / VERIFY OLD ROUND-4 STRICT OOF
#
# We preserve its first 7,250 fold memberships EXACTLY.
# ============================================================

old = np.load(
    OLD_OOF,
    allow_pickle=False,
)


old_ids = np.asarray(
    old["crossfit_id"],
    dtype=np.int32,
)

old_fold = np.asarray(
    old["fold"],
    dtype=np.int32,
)

old_pattern = np.asarray(
    old["pattern"],
    dtype=np.int8,
)

old_truth = np.asarray(
    old["truth"],
    dtype=np.float64,
)

old_source = np.asarray(
    old["source_round"]
).astype(str)


if not np.array_equal(
    old_ids,
    np.arange(
        7250,
        dtype=np.int32,
    ),
):

    raise RuntimeError(
        "Round4 OOF IDs are not exactly 0..7249"
    )


if old_pattern.shape != (
    7250,
    193,
):

    raise RuntimeError(
        "Round4 OOF pattern shape mismatch"
    )


if old_truth.shape != (
    7250,
    3,
):

    raise RuntimeError(
        "Round4 OOF truth shape mismatch"
    )


if not np.array_equal(
    old_pattern,
    tail_patterns[:7250],
):

    raise RuntimeError(
        "Round4 OOF patterns do not match "
        "CNN5 first 7250 tail rows"
    )


existing_truth = np.column_stack(
    [
        kt[:7250],
        fat[:7250],
        fqrt[:7250],
    ]
)


if not np.allclose(
    old_truth,
    existing_truth,
    rtol=0.0,
    atol=1e-12,
):

    raise RuntimeError(
        "Round4 OOF truth does not match "
        "CNN5 historical tail truth"
    )


old_fold_counts = Counter(
    int(x)
    for x in old_fold
)


if dict(
    sorted(
        old_fold_counts.items()
    )
) != {
    0: 1450,
    1: 1450,
    2: 1450,
    3: 1450,
    4: 1450,
}:

    raise RuntimeError(
        "Round4 fold counts changed"
    )


# ============================================================
# LOAD NEW CYCLE-5 DEVELOPMENT3000
# ============================================================

dev = json.loads(
    DEV3000.read_text()
)


if len(dev) != 3000:

    raise RuntimeError(
        f"Cycle5 development rows={len(dev)}"
    )


dev_patterns = np.asarray(
    [
        r["pattern"]
        for r in dev
    ],
    dtype=np.int8,
)


if not np.array_equal(
    dev_patterns,
    tail_patterns[7250:],
):

    raise RuntimeError(
        "Cycle5 development order does not "
        "match CNN5 tail rows 7250..10249"
    )


dev_truth = np.column_stack(
    [
        [
            float(r["keff"])
            for r in dev
        ],
        [
            float(r["peaking_assembly"])
            for r in dev
        ],
        [
            float(r["peaking_pin"])
            for r in dev
        ],
    ]
)


cnn5_dev_truth = np.column_stack(
    [
        kt[7250:],
        fat[7250:],
        fqrt[7250:],
    ]
)


if not np.allclose(
    dev_truth,
    cnn5_dev_truth,
    rtol=0.0,
    atol=1e-12,
):

    raise RuntimeError(
        "Cycle5 development truth mismatch"
    )


for i, row in enumerate(dev):

    if (
        str(
            row["d4_key_sha256"]
        )
        != tail_d4[
            7250 + i
        ]
    ):

        raise RuntimeError(
            f"Cycle5 D4 SHA mismatch "
            f"at development row {i}"
        )


# ============================================================
# NEW 3000 FOLD ASSIGNMENT
#
# IMPORTANT:
# - NO OpenMC truth used.
# - Uses only pre-OpenMC arm, selection stratum and D4 identity.
# - Every normal arm: 270 -> 54/fold.
# - V3M-SQA:       30 ->  6/fold.
#
# Within each arm:
#
# normal:
# diversity 108 -> 22,22,22,21,21
# boundary   81 -> 16,16,16,17,16
# elite      81 -> 16,16,16,16,17
# total          -> 54,54,54,54,54
#
# V3M-SQA:
# diversity 12 -> 3,3,2,2,2
# boundary   9 -> 2,1,2,2,2
# elite      9 -> 1,2,2,2,2
# total         -> 6,6,6,6,6
# ============================================================

NORMAL_QUOTAS = {
    "diversity":
        [22, 22, 22, 21, 21],

    "boundary_uncertainty":
        [16, 16, 16, 17, 16],

    "elite":
        [16, 16, 16, 16, 17],
}


V3MSQA_QUOTAS = {
    "diversity":
        [3, 3, 2, 2, 2],

    "boundary_uncertainty":
        [2, 1, 2, 2, 2],

    "elite":
        [1, 2, 2, 2, 2],
}


new_fold = np.full(
    3000,
    -1,
    dtype=np.int32,
)


for arm in ARMS:

    quotas = (
        V3MSQA_QUOTAS
        if arm == "v3m_sqa"
        else NORMAL_QUOTAS
    )


    for stratum in STRATA:

        idx = [
            i
            for i, row in enumerate(dev)
            if (
                row["arm"] == arm
                and
                row["selection_stratum"]
                == stratum
            )
        ]


        expected_n = sum(
            quotas[
                stratum
            ]
        )


        if len(idx) != expected_n:

            raise RuntimeError(
                f"{arm}/{stratum}: "
                f"{len(idx)} != {expected_n}"
            )


        # Stable deterministic ordering;
        # d4_key_sha256 existed before OpenMC truth.
        idx = sorted(
            idx,
            key=lambda i: (
                str(
                    dev[i][
                        "d4_key_sha256"
                    ]
                ),
                int(
                    dev[i][
                        "id"
                    ]
                ),
            ),
        )


        cursor = 0


        for fold in range(5):

            n = int(
                quotas[
                    stratum
                ][
                    fold
                ]
            )


            chosen = idx[
                cursor:
                cursor + n
            ]


            for i in chosen:

                if (
                    new_fold[i]
                    != -1
                ):

                    raise RuntimeError(
                        "Duplicate new fold assignment"
                    )


                new_fold[i] = fold


            cursor += n


        if cursor != len(idx):

            raise RuntimeError(
                f"{arm}/{stratum}: "
                "quota assignment incomplete"
            )


if np.any(
    new_fold < 0
):

    raise RuntimeError(
        "Unassigned Cycle5 development rows"
    )


new_fold_counts = Counter(
    int(x)
    for x in new_fold
)


if dict(
    sorted(
        new_fold_counts.items()
    )
) != {
    0: 600,
    1: 600,
    2: 600,
    3: 600,
    4: 600,
}:

    raise RuntimeError(
        f"Cycle5 new fold counts invalid: "
        f"{new_fold_counts}"
    )


# Exact per-arm balance.
for arm in ARMS:

    idx = np.asarray(
        [
            i
            for i, row in enumerate(dev)
            if row["arm"] == arm
        ],
        dtype=int,
    )


    expected = (
        6
        if arm == "v3m_sqa"
        else 54
    )


    counts = Counter(
        int(x)
        for x in new_fold[
            idx
        ]
    )


    for fold in range(5):

        if counts[
            fold
        ] != expected:

            raise RuntimeError(
                f"{arm}/fold{fold}: "
                f"{counts[fold]} != {expected}"
            )


# ============================================================
# COMBINED 10,250 FOLDS
# ============================================================

fold_id = np.concatenate(
    [
        old_fold,
        new_fold,
    ]
).astype(
    np.int32
)


combined_fold_counts = Counter(
    int(x)
    for x in fold_id
)


if dict(
    sorted(
        combined_fold_counts.items()
    )
) != {
    0: 2050,
    1: 2050,
    2: 2050,
    3: 2050,
    4: 2050,
}:

    raise RuntimeError(
        "Combined 10250 fold counts invalid"
    )


# ============================================================
# MASTER CROSSFIT MANIFEST
# ============================================================

manifest = []


for i in range(
    7250
):

    truth = old_truth[
        i
    ]


    manifest.append(
        {
            "crossfit_id":
                int(i),

            "fold":
                int(
                    old_fold[
                        i
                    ]
                ),

            "fold_assignment_origin":
                "preserved_round4_strict_oof",

            "source_round":
                str(
                    old_source[
                        i
                    ]
                ),

            "arm":
                None,

            "selection_stratum":
                None,

            "openmc_feasible":
                physically_feasible(
                    truth
                ),

            "openmc_keff":
                float(
                    truth[
                        0
                    ]
                ),

            "openmc_Fa":
                float(
                    truth[
                        1
                    ]
                ),

            "openmc_Fq_r":
                float(
                    truth[
                        2
                    ]
                ),

            "d4_key_sha256":
                str(
                    tail_d4[
                        i
                    ]
                ),

            "pattern":
                tail_patterns[
                    i
                ].astype(
                    int
                ).tolist(),
        }
    )


for local_i, row in enumerate(
    dev
):

    i = (
        7250
        + local_i
    )


    truth = dev_truth[
        local_i
    ]


    manifest.append(
        {
            "crossfit_id":
                int(i),

            "fold":
                int(
                    new_fold[
                        local_i
                    ]
                ),

            "fold_assignment_origin":
                (
                    "cycle5_preopenmc_metadata_"
                    "balanced_arm_stratum_d4"
                ),

            "source_round":
                "cycle5",

            "arm":
                str(
                    row["arm"]
                ),

            "selection_stratum":
                str(
                    row[
                        "selection_stratum"
                    ]
                ),

            "cycle5_openmc_id":
                int(
                    row[
                        "id"
                    ]
                ),

            "openmc_feasible":
                physically_feasible(
                    truth
                ),

            "openmc_keff":
                float(
                    truth[
                        0
                    ]
                ),

            "openmc_Fa":
                float(
                    truth[
                        1
                    ]
                ),

            "openmc_Fq_r":
                float(
                    truth[
                        2
                    ]
                ),

            "d4_key_sha256":
                str(
                    tail_d4[
                        i
                    ]
                ),

            "pattern":
                tail_patterns[
                    i
                ].astype(
                    int
                ).tolist(),
        }
    )


if len(
    manifest
) != 10250:

    raise RuntimeError(
        "Crossfit manifest length != 10250"
    )


manifest_patterns = np.asarray(
    [
        r["pattern"]
        for r in manifest
    ],
    dtype=np.int8,
)


if not np.array_equal(
    manifest_patterns,
    tail_patterns,
):

    raise RuntimeError(
        "Manifest/tail pattern ordering mismatch"
    )


write_json(
    TMP_ROOT
    / MANIFEST_NAME,
    manifest,
)


manifest_sha = sha256_file(
    TMP_ROOT
    / MANIFEST_NAME
)


# ============================================================
# BUILD FIVE LEAKAGE-CLEAN FOLD DATASETS
#
# Per fold:
#
# 2050 = strict unseen holdout
# 8200 = available tail
#
# Of 8200:
#  820 = internal early-stopping validation
# 7380 = actual tail fitting
#
# Plus all 7000 frozen broad training cases:
#
# 14380 actual fitting rows.
#
# Internal validation is stratified ONLY by frozen CNN5
# sampling domain. Truth/feasibility is NOT used to choose
# validation membership.
# ============================================================

fold_summary = {}

fold_artifact_hashes = {}


for fold in range(
    5
):

    fold_dir = (
        TMP_ROOT
        / "folds"
        / f"fold_{fold}"
    )


    fold_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


    hold_idx = np.flatnonzero(
        fold_id
        == fold
    )


    available_idx = np.flatnonzero(
        fold_id
        != fold
    )


    if len(
        hold_idx
    ) != 2050:

        raise RuntimeError(
            f"Fold {fold}: "
            f"holdout={len(hold_idx)}"
        )


    if len(
        available_idx
    ) != 8200:

        raise RuntimeError(
            f"Fold {fold}: "
            f"available={len(available_idx)}"
        )


    fit_tail_idx, val_tail_idx = (
        train_test_split(
            available_idx,

            test_size=820,

            random_state=
                2026
                + fold,

            shuffle=True,

            stratify=
                tail_domain_id[
                    available_idx
                ],
        )
    )


    fit_tail_idx = np.sort(
        np.asarray(
            fit_tail_idx,
            dtype=np.int32,
        )
    )


    val_tail_idx = np.sort(
        np.asarray(
            val_tail_idx,
            dtype=np.int32,
        )
    )


    if len(
        fit_tail_idx
    ) != 7380:

        raise RuntimeError(
            f"Fold {fold}: "
            "fit tail != 7380"
        )


    if len(
        val_tail_idx
    ) != 820:

        raise RuntimeError(
            f"Fold {fold}: "
            "internal validation != 820"
        )


    A = set(
        fit_tail_idx.tolist()
    )

    B = set(
        val_tail_idx.tolist()
    )

    C = set(
        hold_idx.tolist()
    )


    if A & B:

        raise RuntimeError(
            f"Fold {fold}: fit/val leakage"
        )


    if A & C:

        raise RuntimeError(
            f"Fold {fold}: fit/hold leakage"
        )


    if B & C:

        raise RuntimeError(
            f"Fold {fold}: val/hold leakage"
        )


    if (
        len(
            A | B | C
        )
        != 10250
    ):

        raise RuntimeError(
            f"Fold {fold}: split "
            "does not cover 10250"
        )


    # Explicit D4 check.
    fit_d4 = set(
        tail_d4[
            fit_tail_idx
        ].tolist()
    )

    val_d4 = set(
        tail_d4[
            val_tail_idx
        ].tolist()
    )

    hold_d4 = set(
        tail_d4[
            hold_idx
        ].tolist()
    )


    if (
        fit_d4 & val_d4
        or
        fit_d4 & hold_d4
        or
        val_d4 & hold_d4
    ):

        raise RuntimeError(
            f"Fold {fold}: D4 leakage"
        )


    # ========================================================
    # FIT SCALERS ON ACTUAL FIT ROWS ONLY
    # ========================================================

    k_train_raw = np.concatenate(
        [
            kb,
            kt[
                fit_tail_idx
            ],
        ]
    )


    fa_train_raw = np.concatenate(
        [
            fab,
            fat[
                fit_tail_idx
            ],
        ]
    )


    fqr_train_raw = np.concatenate(
        [
            fqrb,
            fqrt[
                fit_tail_idx
            ],
        ]
    )


    if len(
        k_train_raw
    ) != 14380:

        raise RuntimeError(
            f"Fold {fold}: "
            "actual fit rows != 14380"
        )


    scalers = {
        "keff":
            StandardScaler().fit(
                k_train_raw[
                    :,
                    None,
                ]
            ),

        "fa":
            StandardScaler().fit(
                fa_train_raw[
                    :,
                    None,
                ]
            ),

        "fqr":
            StandardScaler().fit(
                fqr_train_raw[
                    :,
                    None,
                ]
            ),
    }


    for name, scaler in scalers.items():

        joblib.dump(
            scaler,
            fold_dir
            / f"{name}_scaler.pkl",
        )


    # ========================================================
    # SAVE FOLD ARRAYS
    #
    # Locked600 intentionally does NOT appear here.
    # ========================================================

    np.savez_compressed(
        fold_dir
        / "round_arrays.npz",

        # --------------------------------------------
        # Actual fitting
        # --------------------------------------------

        X_train=
            np.concatenate(
                [
                    Xb,
                    Xt[
                        fit_tail_idx
                    ],
                ],
                axis=0,
            ),

        X_train_broad=
            Xb,

        X_train_tail=
            Xt[
                fit_tail_idx
            ],

        yk_train_broad=
            scalers[
                "keff"
            ].transform(
                kb[
                    :,
                    None,
                ]
            ),

        yfa_train_broad=
            scalers[
                "fa"
            ].transform(
                fab[
                    :,
                    None,
                ]
            ),

        yfqr_train_broad=
            scalers[
                "fqr"
            ].transform(
                fqrb[
                    :,
                    None,
                ]
            ),

        yk_train_tail=
            scalers[
                "keff"
            ].transform(
                kt[
                    fit_tail_idx,
                    None,
                ]
            ),

        yfa_train_tail=
            scalers[
                "fa"
            ].transform(
                fat[
                    fit_tail_idx,
                    None,
                ]
            ),

        yfqr_train_tail=
            scalers[
                "fqr"
            ].transform(
                fqrt[
                    fit_tail_idx,
                    None,
                ]
            ),

        # --------------------------------------------
        # Sampling-domain information
        # --------------------------------------------

        tail_domain_id=
            tail_domain_id[
                fit_tail_idx
            ].astype(
                np.int32
            ),

        tail_domain_names=
            np.asarray(
                TAIL_DOMAIN_NAMES,
                dtype="U40",
            ),

        # --------------------------------------------
        # Internal validation / early stopping only
        # --------------------------------------------

        X_val=
            Xt[
                val_tail_idx
            ],

        yk_val=
            scalers[
                "keff"
            ].transform(
                kt[
                    val_tail_idx,
                    None,
                ]
            ),

        yfa_val=
            scalers[
                "fa"
            ].transform(
                fat[
                    val_tail_idx,
                    None,
                ]
            ),

        yfqr_val=
            scalers[
                "fqr"
            ].transform(
                fqrt[
                    val_tail_idx,
                    None,
                ]
            ),

        yk_val_raw=
            kt[
                val_tail_idx
            ],

        yfa_val_raw=
            fat[
                val_tail_idx
            ],

        yfqr_val_raw=
            fqrt[
                val_tail_idx
            ],

        internal_val_tail_index=
            val_tail_idx,

        internal_val_tail_domain_id=
            tail_domain_id[
                val_tail_idx
            ].astype(
                np.int32
            ),

        internal_val_patterns=
            tail_patterns[
                val_tail_idx
            ],

        # --------------------------------------------
        # STRICT OOF HOLDOUT
        # --------------------------------------------

        X_crossfit_holdout=
            Xt[
                hold_idx
            ],

        yk_crossfit_holdout_raw=
            kt[
                hold_idx
            ],

        yfa_crossfit_holdout_raw=
            fat[
                hold_idx
            ],

        yfqr_crossfit_holdout_raw=
            fqrt[
                hold_idx
            ],

        crossfit_holdout_tail_index=
            hold_idx.astype(
                np.int32
            ),

        crossfit_holdout_id=
            hold_idx.astype(
                np.int32
            ),

        crossfit_holdout_patterns=
            tail_patterns[
                hold_idx
            ],

        crossfit_holdout_domain_id=
            tail_domain_id[
                hold_idx
            ].astype(
                np.int32
            ),

        # --------------------------------------------
        # Pattern provenance
        # --------------------------------------------

        broad_train_patterns=
            broad_patterns,

        tail_train_patterns=
            tail_patterns[
                fit_tail_idx
            ],

        tail_train_index=
            fit_tail_idx,
    )


    # ========================================================
    # DOMAIN BALANCE AUDITS
    # ========================================================

    fit_domain_counts = Counter(
        int(x)
        for x in
        tail_domain_id[
            fit_tail_idx
        ]
    )


    val_domain_counts = Counter(
        int(x)
        for x in
        tail_domain_id[
            val_tail_idx
        ]
    )


    hold_domain_counts = Counter(
        int(x)
        for x in
        tail_domain_id[
            hold_idx
        ]
    )


    if set(
        fit_domain_counts
    ) != set(
        range(14)
    ):

        raise RuntimeError(
            f"Fold {fold}: "
            "not all 14 domains present in fit"
        )


    metadata = {
        "version":
            "cycle5_crossfit_round5",

        "fold":
            int(
                fold
            ),

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

        "old_round4_fold_assignments_preserved":
            True,

        "new_cycle5_fold_assignment_used_openmc_truth":
            False,

        "internal_validation_assignment_used_openmc_truth":
            False,

        "internal_validation_stratification":
            "CNN5 frozen tail_domain_id",

        "internal_validation_random_state":
            int(
                2026
                + fold
            ),

        "crossfit_holdout_used_for_fit":
            False,

        "cycle5_locked600_present":
            False,

        "cycle5_locked600_used_for_fit":
            False,

        "scaler_fit":
            (
                "fold-specific 14380 actual "
                "fitting rows only"
            ),

        "batch_plan":
            (
                "50% broad / 50% tail; "
                "equal expected exposure "
                "across 14 tail domains"
            ),

        "cnn_seeds":
            [
                42,
                314,
                2718,
                31415,
                161803,
            ],

        "tail_domain_names":
            TAIL_DOMAIN_NAMES,

        "fit_domain_counts":
            {
                TAIL_DOMAIN_NAMES[
                    k
                ]:
                    int(v)
                for k, v in sorted(
                    fit_domain_counts.items()
                )
            },

        "validation_domain_counts":
            {
                TAIL_DOMAIN_NAMES[
                    k
                ]:
                    int(v)
                for k, v in sorted(
                    val_domain_counts.items()
                )
            },

        "holdout_domain_counts":
            {
                TAIL_DOMAIN_NAMES[
                    k
                ]:
                    int(v)
                for k, v in sorted(
                    hold_domain_counts.items()
                )
            },

        "manifest_sha256":
            manifest_sha,
    }


    write_json(
        fold_dir
        / "round_metadata.json",
        metadata,
    )


    fold_summary[
        str(fold)
    ] = metadata


    fold_artifact_hashes[
        str(fold)
    ] = {
        "round_arrays_sha256":
            sha256_file(
                fold_dir
                / "round_arrays.npz"
            ),

        "round_metadata_sha256":
            sha256_file(
                fold_dir
                / "round_metadata.json"
            ),

        "keff_scaler_sha256":
            sha256_file(
                fold_dir
                / "keff_scaler.pkl"
            ),

        "fa_scaler_sha256":
            sha256_file(
                fold_dir
                / "fa_scaler.pkl"
            ),

        "fqr_scaler_sha256":
            sha256_file(
                fold_dir
                / "fqr_scaler.pkl"
            ),
    }


# ============================================================
# FREEZE DESIGN
# ============================================================

new_arm_fold_counts = {}


for arm in ARMS:

    idx = np.asarray(
        [
            i
            for i, r in enumerate(dev)
            if r["arm"] == arm
        ],
        dtype=int,
    )


    new_arm_fold_counts[
        arm
    ] = {
        str(fold):
            int(
                np.sum(
                    new_fold[
                        idx
                    ]
                    == fold
                )
            )
        for fold in range(5)
    }


freeze = {
    "stage":
        "CYCLE5_STRICT_CNN5_CROSSFIT_DATASET",

    "status":
        "FROZEN_BEFORE_CROSSFIT_CNN_TRAINING",

    "tail_rows":
        10250,

    "folds":
        5,

    "holdout_per_fold":
        2050,

    "old_round4_holdout_per_fold":
        1450,

    "cycle5_new_holdout_per_fold":
        600,

    "available_tail_per_fold":
        8200,

    "internal_validation_per_fold":
        820,

    "tail_fit_per_fold":
        7380,

    "broad_fit_per_fold":
        7000,

    "actual_fit_per_fold":
        14380,

    "old_round4_rows":
        7250,

    "old_round4_fold_assignment_preserved":
        True,

    "new_cycle5_rows":
        3000,

    "new_fold_assignment":
        (
            "pre-OpenMC arm + acquisition stratum "
            "+ D4 identity only"
        ),

    "new_fold_assignment_used_openmc_truth":
        False,

    "new_cycle5_fold_counts":
        {
            str(k):
                int(v)
            for k, v
            in sorted(
                new_fold_counts.items()
            )
        },

    "new_cycle5_arm_fold_counts":
        new_arm_fold_counts,

    "combined_fold_counts":
        {
            str(k):
                int(v)
            for k, v
            in sorted(
                combined_fold_counts.items()
            )
        },

    "internal_validation":
        {
            "n_per_fold":
                820,

            "fraction_of_available_tail":
                0.10,

            "stratification":
                "CNN5 tail_domain_id",

            "uses_truth_or_feasibility":
                False,

            "random_state":
                "2026 + fold",
        },

    "cycle5_locked600_used":
        False,

    "cycle5_locked600_present_in_fold_files":
        False,

    "all_10250_tail_D4_unique":
        True,

    "crossfit_role_D4_overlap":
        0,

    "sampling_policy_for_fold_CNNs":
        (
            "same CNN5 policy: 50% broad / "
            "50% tail with equal expected exposure "
            "across 14 tail domains"
        ),

    "source_round5_arrays_sha256":
        EXPECTED_ARRAYS_SHA,

    "source_round5_metadata_sha256":
        EXPECTED_METADATA_SHA,

    "source_cycle5_development_sha256":
        EXPECTED_DEV_SHA,

    "source_round4_oof":
        str(
            OLD_OOF.relative_to(
                ROOT
            )
        ),

    "source_round4_oof_sha256":
        sha256_file(
            OLD_OOF
        ),

    "crossfit_manifest_sha256":
        manifest_sha,

    "fold_artifacts":
        fold_artifact_hashes,
}


write_json(
    TMP_ROOT
    / FREEZE_NAME,
    freeze,
)


freeze_sha = sha256_file(
    TMP_ROOT
    / FREEZE_NAME
)


# ============================================================
# ATOMIC FREEZE
# ============================================================

TMP_ROOT.rename(
    FINAL_ROOT
)


# ============================================================
# COMPACT OUTPUT
# ============================================================

print("=" * 80)
print("CYCLE 5 STEP 5A — STRICT CNN5 CROSSFIT DATASET FROZEN")
print("=" * 80)

print("tail cases               = 10250")
print("old Round4 assignments   = 7250 PRESERVED")
print("new Cycle5 cases         = 3000")
print("new fold assignment truth= NOT USED")
print("fold count               = 5")
print("holdout / fold           = 2050")
print("  old Round4 / fold      = 1450")
print("  new Cycle5 / fold      = 600")
print("available tail / fold    = 8200")
print("internal val / fold      = 820")
print("tail fit / fold          = 7380")
print("broad fit / fold         = 7000")
print("actual fit / fold        = 14380")
print("tail domains             = 14")
print("all tail D4 unique       = 10250 /10250")
print("fit/val/hold D4 leakage  = 0")
print("locked600 present        = NO")
print("locked600 used           = NO")

print()
print("NEW CYCLE5 FOLD COUNTS")

for fold in range(5):

    print(
        f"fold {fold} = "
        f"{new_fold_counts[fold]}"
    )


print()
print("NEW CYCLE5 PER-ARM / PER-FOLD")

for arm in ARMS:

    counts = new_arm_fold_counts[
        arm
    ]

    print(
        f"{arm:14s} = "
        + " ".join(
            f"{counts[str(f)]}"
            for f in range(5)
        )
    )


print()
print(
    "Round4 OOF SHA256       =",
    sha256_file(
        OLD_OOF
    ),
)

print(
    "crossfit manifest SHA256=",
    manifest_sha,
)

print(
    "freeze manifest SHA256  =",
    freeze_sha,
)

print()
print(
    "CYCLE5_STEP5A_STRICT_CROSSFIT_DATASET_FROZEN"
)
