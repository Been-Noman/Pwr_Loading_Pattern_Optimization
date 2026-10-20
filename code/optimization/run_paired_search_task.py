#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--manifest",
        required=True,
    )

    ap.add_argument(
        "--expected-manifest-sha",
        required=True,
    )

    ap.add_argument(
        "--task-id",
        required=True,
        type=int,
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = ap.parse_args()

    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        raise RuntimeError(
            f"Missing execution manifest: {manifest_path}"
        )

    actual_manifest_sha = sha(manifest_path)

    if actual_manifest_sha != args.expected_manifest_sha:
        raise RuntimeError(
            "Execution manifest SHA mismatch:\n"
            f"expected={args.expected_manifest_sha}\n"
            f"actual  ={actual_manifest_sha}"
        )

    obj = json.loads(
        manifest_path.read_text()
    )

    runs = obj["runs"]

    task = int(args.task_id)

    if not 0 <= task < len(runs):
        raise RuntimeError(
            f"task-id {task} outside 0..{len(runs)-1}"
        )

    row = runs[task]

    if int(row["task_id"]) != task:
        raise RuntimeError(
            "Manifest task-index identity mismatch"
        )

    # ----------------------------------------------------------
    # Frozen file identity checks
    # ----------------------------------------------------------

    adapter = Path(
        row["adapter"]
    )

    if sha(adapter) != row["adapter_sha256"]:
        raise RuntimeError(
            "Explicit-seed adapter SHA changed"
        )

    runner = Path(
        row["runner_file"]
    )

    if sha(runner) != row["runner_sha256"]:
        raise RuntimeError(
            f"Frozen runner SHA changed: {runner}"
        )

    shared_manifest = Path(
        row["shared_runtime_manifest"]
    )

    if (
        sha(shared_manifest)
        != row["shared_runtime_manifest_sha256"]
    ):
        raise RuntimeError(
            "Shared-runtime manifest SHA changed"
        )

    labelled = Path(
        row["labelled_json"]
    )

    if (
        sha(labelled)
        != row["labelled_json_sha256"]
    ):
        raise RuntimeError(
            "Known-OpenMC firewall SHA changed"
        )

    residual_metadata = Path(
        row["residual_metadata"]
    )

    if (
        sha(residual_metadata)
        != row["residual_metadata_sha256"]
    ):
        raise RuntimeError(
            "Residual/q4 metadata SHA changed"
        )

    anchor = row["anchor_json"]

    if row["initializer"] == "random":

        if anchor is not None:
            raise RuntimeError(
                "Random initialization must not have anchor JSON"
            )

    else:

        if anchor is None:
            raise RuntimeError(
                f"{row['initializer']} run missing anchor JSON"
            )

        anchor_path = Path(anchor)

        if sha(anchor_path) != row["anchor_sha256"]:
            raise RuntimeError(
                f"Anchor SHA mismatch: {anchor_path}"
            )

    # ----------------------------------------------------------
    # Output protection
    # ----------------------------------------------------------

    output_dir = Path(
        row["output_dir"]
    )

    if not args.dry_run:

        if (
            output_dir.exists()
            and any(output_dir.iterdir())
        ):
            raise RuntimeError(
                "Refusing to overwrite non-empty bridge run:\n"
                f"{output_dir}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ----------------------------------------------------------
    # Exact command
    # ----------------------------------------------------------

    cmd = [
        sys.executable,
        str(adapter),

        "--bridge-runner",
        row["solver"],

        "--bridge-replicate",
        str(row["replicate"]),

        "--bridge-ga-seed",
        str(row["paired_ga_seed"]),

        "--round-dir",
        row["round_dir"],

        "--residual-model-dir",
        row["residual_model_dir"],

        "--output-dir",
        row["output_dir"],

        "--strategy",
        row["initializer"],

        "--labelled-json",
        row["labelled_json"],

        "--threads",
        str(row["threads"]),
    ]

    if anchor is not None:
        cmd.extend([
            "--anchor-json",
            anchor,
        ])

    print(
        f"TASK={task:03d} "
        f"REP={row['replicate']:02d} "
        f"SOLVER={row['solver']} "
        f"INIT={row['initializer']} "
        f"GA_SEED={row['paired_ga_seed']}",
        flush=True,
    )

    print(
        "COMMAND:",
        " ".join(cmd),
        flush=True,
    )

    if args.dry_run:
        return

    subprocess.run(
        cmd,
        check=True,
    )


if __name__ == "__main__":
    main()
