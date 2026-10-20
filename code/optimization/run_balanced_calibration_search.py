#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import importlib.util
import sys


MAX_REPLICATES = 80


RUNNERS = {
    "v2": (
        Path("scripts/optimizer_tail_v2/penalty_based_search_300.py"),
        "5573751893c2d13e741a65f149e95a23faaa1ab0a7e9ac8939550fc66fc488cd",
    ),

    "v2m": (
        Path("scripts/optimizer_tail_v2/penalty_based_memetic_search.py"),
        "3fe75f2078ef278be32989a3a8e8fa8014c0d3d35b5b544bd31e8c3d307b729f",
    ),

    "v3": (
        Path("scripts/optimizer_tail_v2/feasibility_first_search.py"),
        "ef547aaa9871936a34818285fbfac4e26e42e0044117bc0318b79c7088dcfbc6",
    ),

    "v3m": (
        Path("scripts/optimizer_tail_v2/feasibility_first_memetic_search.py"),
        "419ad99d4996cb52b3b5465057c21ae6a714ebfc57966ae15990359a46a62504",
    ),
}


def sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Cycle5 explicit-seed adapter for the "
            "80-replicate balanced q5 calibration."
        ),
        add_help=True,
    )

    parser.add_argument(
        "--bridge-runner",
        required=True,
        choices=tuple(
            RUNNERS
        ),
    )

    parser.add_argument(
        "--bridge-replicate",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--bridge-ga-seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--bridge-validate-only",
        action="store_true",
    )

    args, runner_args = (
        parser.parse_known_args()
    )


    rep = int(
        args.bridge_replicate
    )

    seed = int(
        args.bridge_ga_seed
    )


    if not (
        0
        <= rep
        < MAX_REPLICATES
    ):

        raise SystemExit(
            "--bridge-replicate must be 0..79"
        )


    if not (
        0
        < seed
        < 2**31 - 1
    ):

        raise SystemExit(
            "--bridge-ga-seed must be a positive "
            "signed-int32-compatible integer"
        )


    # Adapter owns seed-index completely.
    for token in runner_args:

        if (
            token
            == "--seed-index"
            or
            token.startswith(
                "--seed-index="
            )
        ):

            raise SystemExit(
                "Do not pass --seed-index manually; "
                "the Cycle5 adapter controls it."
            )


    runner_path, expected_sha = (
        RUNNERS[
            args.bridge_runner
        ]
    )


    actual_sha = sha256(
        runner_path
    )


    if actual_sha != expected_sha:

        raise RuntimeError(
            f"Frozen runner SHA changed for "
            f"{args.bridge_runner}: "
            f"{actual_sha}"
        )


    # Allow local imports such as memetic_common.
    sys.path.insert(
        0,
        str(
            runner_path.parent.resolve()
        ),
    )


    spec = (
        importlib.util.spec_from_file_location(
            (
                "cycle5_explicit_seed_"
                f"{args.bridge_runner}"
            ),
            runner_path,
        )
    )


    module = (
        importlib.util.module_from_spec(
            spec
        )
    )


    spec.loader.exec_module(
        module
    )


    # --------------------------------------------------------
    # Critical Cycle5 generalization:
    #
    # frozen runners validate:
    #
    #   0 <= seed_index < len(GA_SEEDS)
    #
    # Therefore expose exactly 80 seed slots and place the
    # experimental paired seed at the replicate's own index.
    #
    # The other entries are never selected by this invocation.
    # --------------------------------------------------------

    injected = [
        0
    ] * MAX_REPLICATES


    injected[
        rep
    ] = seed


    module.GA_SEEDS = tuple(
        injected
    )


    # Used only by fallback paths; keep consistent with actual
    # experimental seed.
    module.GA_BASE_SEED = seed


    if len(
        module.GA_SEEDS
    ) != MAX_REPLICATES:

        raise RuntimeError(
            "Injected GA_SEEDS length is not 80"
        )


    if int(
        module.GA_SEEDS[
            rep
        ]
    ) != seed:

        raise RuntimeError(
            "Injected paired GA seed mismatch"
        )


    if args.bridge_validate_only:

        print(
            (
                "CYCLE5_EXPLICIT_SEED_ADAPTER_VALIDATE_OK "
                f"runner={args.bridge_runner} "
                f"replicate={rep} "
                f"seed_index={rep} "
                f"seed={seed} "
                f"GA_SEEDS_LEN={len(module.GA_SEEDS)}"
            )
        )

        return


    # Invoke original frozen solver normally.
    sys.argv = [
        str(
            runner_path
        ),
        *runner_args,
        "--seed-index",
        str(
            rep
        ),
    ]


    module.main()


if __name__ == "__main__":
    main()
