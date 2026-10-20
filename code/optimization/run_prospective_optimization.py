#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import importlib.util
import sys


PROJECT = Path(__file__).resolve().parents[2]

RUNNERS = {
    "v2": (
        PROJECT / "scripts/optimizer_tail_v2/penalty_based_search_300.py",
        "5573751893c2d13e741a65f149e95a23faaa1ab0a7e9ac8939550fc66fc488cd",
    ),
    "v2m": (
        PROJECT / "scripts/optimizer_tail_v2/penalty_based_memetic_search.py",
        "3fe75f2078ef278be32989a3a8e8fa8014c0d3d35b5b544bd31e8c3d307b729f",
    ),
    "v3": (
        PROJECT / "scripts/optimizer_tail_v2/feasibility_first_search.py",
        "ef547aaa9871936a34818285fbfac4e26e42e0044117bc0318b79c7088dcfbc6",
    ),
    "v3m": (
        PROJECT / "scripts/optimizer_tail_v2/feasibility_first_memetic_search.py",
        "419ad99d4996cb52b3b5465057c21ae6a714ebfc57966ae15990359a46a62504",
    ),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Cycle-5 prospective360 adapter: inject one explicit "
            "paired GA seed into an unchanged frozen solver runner."
        )
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

    if not 0 <= args.bridge_replicate <= 29:
        raise SystemExit(
            "--bridge-replicate must be 0..29"
        )

    for token in runner_args:
        if (
            token == "--seed-index"
            or token.startswith("--seed-index=")
        ):
            raise SystemExit(
                "Do not pass --seed-index manually; "
                "the prospective adapter controls it."
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

    sys.path.insert(
        0,
        str(runner_path.parent.resolve()),
    )

    spec = importlib.util.spec_from_file_location(
        f"cycle5_q5_prospective_{args.bridge_runner}",
        runner_path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rep = int(args.bridge_replicate)
    seed = int(args.bridge_ga_seed)

    injected = [0] * 30
    injected[rep] = seed

    module.GA_SEEDS = tuple(injected)

    # Preserve historical adapter behavior for any fallback seed path.
    module.GA_BASE_SEED = seed

    sys.argv = [
        str(runner_path),
        *runner_args,
        "--seed-index",
        str(rep),
    ]

    module.main()


if __name__ == "__main__":
    main()
