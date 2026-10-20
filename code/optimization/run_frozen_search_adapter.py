#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import importlib.util
import sys


RUNNERS = {
    "v2": (
        Path("scripts/optimizer_tail_v2/run_robust_ga_v2_300.py"),
        "5573751893c2d13e741a65f149e95a23faaa1ab0a7e9ac8939550fc66fc488cd",
    ),
    "v2m": (
        Path("scripts/optimizer_tail_v2/run_robust_ga_v2m.py"),
        "3fe75f2078ef278be32989a3a8e8fa8014c0d3d35b5b544bd31e8c3d307b729f",
    ),
    "v3": (
        Path("scripts/optimizer_tail_v2/run_robust_ga_v3.py"),
        "ef547aaa9871936a34818285fbfac4e26e42e0044117bc0318b79c7088dcfbc6",
    ),
    "v3m": (
        Path("scripts/optimizer_tail_v2/run_robust_ga_v3m.py"),
        "419ad99d4996cb52b3b5465057c21ae6a714ebfc57966ae15990359a46a62504",
    ),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fair-bridge adapter: inject an explicit paired GA seed "
            "without modifying the frozen solver runner."
        ),
        add_help=True,
    )

    parser.add_argument(
        "--bridge-runner",
        required=True,
        choices=tuple(RUNNERS),
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

    args, runner_args = parser.parse_known_args()

    if not 0 <= args.bridge_replicate <= 9:
        raise SystemExit(
            "--bridge-replicate must be 0..9"
        )

    # The wrapper owns seed-index completely.
    for token in runner_args:
        if (
            token == "--seed-index"
            or token.startswith("--seed-index=")
        ):
            raise SystemExit(
                "Do not pass --seed-index manually; "
                "the fair-bridge adapter controls it."
            )

    runner_path, expected_sha = RUNNERS[
        args.bridge_runner
    ]

    actual_sha = sha256(runner_path)

    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Frozen runner SHA changed for "
            f"{args.bridge_runner}: {actual_sha}"
        )

    # Allow local imports such as memetic_common.
    sys.path.insert(
        0,
        str(runner_path.parent.resolve()),
    )

    spec = importlib.util.spec_from_file_location(
        f"fair_bridge_{args.bridge_runner}",
        runner_path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rep = int(args.bridge_replicate)
    seed = int(args.bridge_ga_seed)

    # Make runner's own recorded seed_index equal the paired
    # replicate index rather than forcing every run to index 0.
    injected = [0] * 10
    injected[rep] = seed

    module.GA_SEEDS = tuple(injected)

    # Used only by fallback code paths. Keep it consistent with
    # the actual experimental seed.
    module.GA_BASE_SEED = seed

    # Invoke the original, frozen runner normally.
    sys.argv = [
        str(runner_path),
        *runner_args,
        "--seed-index",
        str(rep),
    ]

    module.main()


if __name__ == "__main__":
    main()
