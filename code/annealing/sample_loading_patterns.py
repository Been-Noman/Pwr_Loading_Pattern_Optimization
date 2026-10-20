#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from joblib import Parallel, delayed

import qubo_and_reactor_model as pipe

GROUP_ORDER = [
    ("sa", True),
    ("sa", False),
    ("sqa", True),
    ("sqa", False),
]

SCHEMA_VERSION = 2


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def pkey(row):
    p = row.get("pattern", row.get("chromosome"))
    if p is None:
        raise ValueError("Missing pattern/chromosome.")
    p = tuple(int(x) for x in p)
    if len(p) != 193:
        raise ValueError(f"Expected 193 cells, got {len(p)}.")
    return p


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gname(method, symmetry):
    return f"{method}|symmetry={bool(symmetry)}"


def new_checkpoint(shard_id, target, base_seed, old_file, old_sha):
    method_offsets = {"sa": 0, "sqa": 2_000_000}
    groups = {}
    for method, symmetry in GROUP_ORDER:
        symmetry_index = list(pipe.SYMMETRY_SETTINGS).index(bool(symmetry))
        groups[gname(method, symmetry)] = {
            "method": method,
            "symmetry_enabled": bool(symmetry),
            "base_seed": int(
                base_seed
                + shard_id * 10_000_000
                + method_offsets[method]
                + symmetry_index * 10_000
            ),
            "next_batch": 0,
            "records": [],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "shard_id": int(shard_id),
        "target_per_group": int(target),
        "base_seed": int(base_seed),
        "old_pattern_file": str(Path(old_file).resolve()),
        "old_sha256": old_sha,
        "groups": groups,
        "complete": False,
    }


def load_old(path, expected):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Exclusion pattern bank must be a JSON list.")
    keys = [pkey(r) for r in rows]
    if len(rows) != expected:
        raise ValueError(f"Exclusion bank has {len(rows)} rows; expected {expected}.")
    if len(set(keys)) != expected:
        raise ValueError("Exclusion bank contains duplicate exact patterns.")
    return rows, set(keys)


def load_checkpoint(path, shard_id, target, base_seed, old_file, old_sha, resume):
    path = Path(path)
    if not resume or not path.exists():
        return new_checkpoint(shard_id, target, base_seed, old_file, old_sha)

    cp = json.loads(path.read_text(encoding="utf-8"))
    checks = (
        cp.get("schema_version") == SCHEMA_VERSION,
        int(cp.get("shard_id", -1)) == int(shard_id),
        int(cp.get("target_per_group", -1)) == int(target),
        int(cp.get("base_seed", -1)) == int(base_seed),
        cp.get("old_sha256") == old_sha,
    )
    if not all(checks):
        raise RuntimeError(
            "Incompatible shard checkpoint. Use a clean shard directory "
            "or the same arguments as the original run."
        )
    return cp


def sanitize(cp, old_keys):
    seen = set(old_keys)
    for method, symmetry in GROUP_ORDER:
        group = cp["groups"][gname(method, symmetry)]
        cleaned = []
        local = set()
        for row in group.get("records", []):
            if row.get("solver") != method:
                raise RuntimeError("Checkpoint solver mismatch.")
            if bool(row.get("symmetry_enabled")) != bool(symmetry):
                raise RuntimeError("Checkpoint symmetry mismatch.")
            key = pkey(row)
            if key in old_keys:
                raise RuntimeError("Checkpoint overlaps exclusion bank.")
            if key in local:
                continue
            if key in seen:
                raise RuntimeError("Checkpoint has cross-group duplicate.")
            if not pipe.validate_decoded_pattern(list(key)):
                raise RuntimeError("Checkpoint has infeasible pattern.")
            cleaned.append(row)
            local.add(key)
            seen.add(key)
        group["records"] = cleaned
        group["next_batch"] = max(0, int(group.get("next_batch", 0)))
    return seen


def generate_method(cp, checkpoint_path, method, target, workers, max_batches, global_seen):
    symmetries = [bool(x) for x in pipe.SYMMETRY_SETTINGS]

    def state(sym):
        return cp["groups"][gname(method, sym)]

    def done(sym):
        return len(state(sym)["records"]) >= target

    if all(done(sym) for sym in symmetries):
        print(f"{method.upper()} already complete.", flush=True)
        return

    wave = 0
    with Parallel(n_jobs=workers, backend="loky", batch_size=1, verbose=0) as parallel:
        while not all(done(sym) for sym in symmetries):
            wave += 1
            active = [
                sym for sym in symmetries
                if not done(sym) and state(sym)["next_batch"] < max_batches
            ]
            if not active:
                incomplete = {
                    str(sym): len(state(sym)["records"])
                    for sym in symmetries if not done(sym)
                }
                raise RuntimeError(
                    f"{method.upper()} exhausted max batch budget: {incomplete}"
                )

            base = workers // len(active)
            extra = workers % len(active)
            specs = []

            for idx, sym in enumerate(active):
                group = state(sym)
                requested = base + (1 if idx < extra else 0)
                remaining = max_batches - group["next_batch"]
                n_batches = min(requested, remaining)

                for _ in range(n_batches):
                    batch_index = int(group["next_batch"])
                    group["next_batch"] += 1
                    specs.append({
                        "method": method,
                        "use_symmetry": bool(sym),
                        "batch_index": batch_index,
                        "base_seed": int(group["base_seed"]),
                        "diversity_strength": 0.0,
                    })

            try:
                results = parallel(
                    delayed(pipe._run_solver_batch)(**spec)
                    for spec in specs
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{method.upper()} shard generation failed in wave {wave}. "
                    "Resubmit; the previous completed-wave checkpoint is intact."
                ) from exc

            added = 0
            rejected = 0

            for result in results:
                sym = bool(result["symmetry_enabled"])
                group = state(sym)
                local = {pkey(r) for r in group["records"]}

                for row in result["records"]:
                    key = pkey(row)
                    if key in global_seen:
                        rejected += 1
                        continue
                    if not pipe.validate_decoded_pattern(list(key)):
                        raise RuntimeError("Solver returned infeasible pattern.")
                    group["records"].append(row)
                    local.add(key)
                    global_seen.add(key)
                    added += 1

            print(
                f"{method.upper()} wave {wave:>3}: "
                + " | ".join(
                    f"sym={str(sym):<5} {len(state(sym)['records'])}/{target}"
                    for sym in symmetries
                )
                + f" | added={added}, rejected={rejected}",
                flush=True,
            )
            atomic_json(cp, checkpoint_path)

    # Deterministically trim local overshoot.
    for sym in symmetries:
        group = state(sym)
        group["records"].sort(
            key=lambda row, s=sym: pipe._group_sort_key(row, s)
        )
        group["records"] = group["records"][:target]

    # Rebuild global_seen because overshoot was trimmed.
    global_seen.clear()
    # Caller adds old keys again after method return.


def rebuild_seen(cp, old_keys):
    seen = set(old_keys)
    for method, symmetry in GROUP_ORDER:
        for row in cp["groups"][gname(method, symmetry)]["records"]:
            key = pkey(row)
            if key in seen:
                raise RuntimeError("Duplicate after deterministic trim.")
            seen.add(key)
    return seen


def finalize(cp, old_keys, target, output_file):
    rows = []
    for method, symmetry in GROUP_ORDER:
        group = cp["groups"][gname(method, symmetry)]["records"]
        if len(group) != target:
            raise RuntimeError(
                f"{gname(method, symmetry)} has {len(group)}, expected {target}."
            )
        for row in group:
            if not pipe.validate_decoded_pattern(row["pattern"]):
                raise RuntimeError("Final feasibility audit failed.")

    # Round-robin group order.
    for i in range(target):
        for method, symmetry in GROUP_ORDER:
            src = cp["groups"][gname(method, symmetry)]["records"][i]
            item = {k: v for k, v in src.items() if not str(k).startswith("_")}
            item["id"] = len(rows)
            item["generation_shard"] = int(cp["shard_id"])
            rows.append(item)

    keys = [pkey(r) for r in rows]
    if len(set(keys)) != len(rows):
        raise RuntimeError("Shard output contains duplicate patterns.")
    if set(keys) & old_keys:
        raise RuntimeError("Shard output overlaps exclusion bank.")

    atomic_json(rows, output_file)
    cp["complete"] = True
    atomic_json(cp, Path(output_file).with_name("checkpoint.json"))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--exclude-pattern-file", required=True)
    p.add_argument("--expected-excluded", type=int, default=7500)
    p.add_argument("--samples-per-group", type=int, default=200)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--max-batches-per-group", type=int, default=5000)
    p.add_argument("--base-seed", type=int, default=30360814)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.json"
    output_file = out / f"candidate_shard_{args.shard_id}.json"

    _, old_keys = load_old(args.exclude_pattern_file, args.expected_excluded)
    old_sha = sha256(args.exclude_pattern_file)

    cp = load_checkpoint(
        checkpoint_path,
        args.shard_id,
        args.samples_per_group,
        args.base_seed,
        args.exclude_pattern_file,
        old_sha,
        args.resume,
    )
    seen = sanitize(cp, old_keys)

    started = time.perf_counter()
    print(
        json.dumps({
            "shard_id": args.shard_id,
            "target_per_group": args.samples_per_group,
            "target_total": 4 * args.samples_per_group,
            "old_excluded": len(old_keys),
            "workers": args.workers,
            "base_seed": args.base_seed,
            "effective_shard_seed_offset": args.shard_id * 10_000_000,
        }, indent=2),
        flush=True,
    )

    for method in ("sa", "sqa"):
        generate_method(
            cp,
            checkpoint_path,
            method,
            args.samples_per_group,
            args.workers,
            args.max_batches_per_group,
            seen,
        )
        seen = rebuild_seen(cp, old_keys)
        atomic_json(cp, checkpoint_path)

    rows = finalize(cp, old_keys, args.samples_per_group, output_file)
    summary = {
        "shard_id": args.shard_id,
        "rows": len(rows),
        "unique_patterns": len({pkey(r) for r in rows}),
        "old_overlap": len({pkey(r) for r in rows} & old_keys),
        "elapsed_seconds": time.perf_counter() - started,
        "output_file": str(output_file),
    }
    atomic_json(summary, out / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
