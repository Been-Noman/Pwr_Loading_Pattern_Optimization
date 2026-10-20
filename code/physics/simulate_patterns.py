#!/usr/bin/env python3
"""Run an SA/SQA pattern-bank slice through OpenMC with restart-safe JSONL.

Ares production layout: 5 simultaneous OpenMC processes on a 48-core node.
cpu_slots() assigns three processes 10 cores and two processes 9 cores. A
25-sample chunk therefore runs in five fully occupied waves.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
import numpy as np
import openmc
from qubo_and_reactor_model import (
    GRID_SIDE, SOLVER_METHODS, SYMMETRY_SETTINGS, build_pwr_core,
    cpu_slots, require_environment, validate_decoded_pattern,
)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pattern-file', required=True)
    p.add_argument('--output-file', required=True)
    p.add_argument('--metrics-file', required=True)
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--n-samples', type=int, required=True)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--threads-per-run', type=int, default=9,
                   help='Minimum threads/process; spare allocated cores are distributed')
    p.add_argument('--batches', type=int, required=True)
    p.add_argument('--inactive', type=int, required=True)
    p.add_argument('--particles', type=int, required=True)
    p.add_argument('--tmp-root', required=True)
    p.add_argument('--group-waves', action='store_true')
    return p.parse_args()

def read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open(encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Invalid JSONL {path}:{line_no}: {exc}') from exc
    return rows

def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, separators=(',', ':')) + '\n')
        fh.flush()
        os.fsync(fh.fileno())

def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)

def simulate(record, batches, inactive, particles, cpu_ids, tmp_root):
    sid = int(record['id'])
    pattern = np.asarray(record['pattern'], dtype=int)
    total_started = time.perf_counter()
    if pattern.shape != (193,):
        return {'id': sid, 'error': f'invalid pattern shape {pattern.shape}'}
    if not validate_decoded_pattern(pattern.tolist()):
        return {'id': sid, 'error': 'hard-constraint validation failed'}
    executable, _ = require_environment()
    threads = len(cpu_ids)
    Path(tmp_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'sample_{sid:05d}_', dir=tmp_root) as tmpdir:
        try:
            t0 = time.perf_counter()
            model = build_pwr_core(pattern)
            model.settings.batches = int(batches)
            model.settings.inactive = int(inactive)
            model.settings.particles = int(particles)
            model.settings.seed = int((20260718 + sid) % (2**31 - 1))
            model.export_to_xml(tmpdir)
            build_seconds = time.perf_counter() - t0
            env = os.environ.copy()
            for _k in list(env):
                if _k.startswith(("SLURM_", "PMI_", "PMIX_", "OMPI_MCA_")):
                    env.pop(_k, None)
            env.update({
                'OMP_NUM_THREADS': str(threads), 'OMP_PROC_BIND': 'close',
                'OMP_PLACES': 'cores', 'OPENBLAS_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
            })
            cpu_list = ','.join(map(str, cpu_ids))
            command = ['taskset', '--cpu-list', cpu_list, executable, '-s', str(threads)]
            t1 = time.perf_counter()
            proc = subprocess.run(command, cwd=tmpdir, env=env,
                                  capture_output=True, text=True, check=False)
            openmc_seconds = time.perf_counter() - t1
            if proc.returncode != 0:
                return {
                    'id': sid, 'solver': record['solver'],
                    'symmetry_enabled': bool(record['symmetry_enabled']),
                    'error': f'OpenMC exit code {proc.returncode}',
                    'command': command, 'stdout_tail': proc.stdout[-3000:],
                    'stderr_tail': proc.stderr[-3000:],
                }
            statepoint = Path(tmpdir) / f'statepoint.{batches}.h5'
            if not statepoint.is_file():
                return {'id': sid, 'error': f'missing {statepoint.name}'}
            t2 = time.perf_counter()
            with openmc.StatePoint(statepoint) as sp:
                keff = float(sp.keff.n)
                tally = sp.get_tally(name='power')
                raw = tally.get_values(scores=['fission-q-recoverable'])
                pin_power = raw.reshape((GRID_SIDE * 17, GRID_SIDE * 17))
                if float(np.max(pin_power)) < 1e-10:
                    return {'id': sid, 'error': 'near-zero power tally'}
                assy = pin_power.reshape(GRID_SIDE, 17, GRID_SIDE, 17).sum(axis=(1, 3))
                assy_rates = assy[assy > 0]
                fa = float(np.max(assy_rates) / np.mean(assy_rates))
                pin_rates = pin_power[pin_power > 1e-4 * np.max(pin_power)]
                fq = float(np.max(pin_rates) / np.mean(pin_rates))
            post_seconds = time.perf_counter() - t2
            return {
                'id': sid, 'solver': record['solver'],
                'symmetry_enabled': bool(record['symmetry_enabled']),
                'pattern': pattern.tolist(), 'qubo_energy': float(record['qubo_energy']),
                'symmetry_mismatches': int(record['symmetry_mismatches']),
                'solver_seed': int(record['solver_seed']), 'keff': keff,
                'peaking_assembly': fa, 'peaking_pin': fq,
                'batches': int(batches), 'inactive': int(inactive),
                'particles': int(particles), 'openmc_threads': threads,
                'cpu_ids': list(cpu_ids), 'build_export_seconds': build_seconds,
                'openmc_seconds': openmc_seconds,
                'postprocess_seconds': post_seconds,
                'total_seconds': time.perf_counter() - total_started,
            }
        except Exception as exc:
            return {'id': sid, 'solver': record.get('solver'),
                    'symmetry_enabled': bool(record.get('symmetry_enabled')),
                    'error': f'{type(exc).__name__}: {exc}'}

def group_key(row):
    return (row['solver'], bool(row['symmetry_enabled']))

def build_waves(records, workers, group_waves):
    if not group_waves:
        return [records[i:i+workers] for i in range(0, len(records), workers)]
    waves = []
    for method in SOLVER_METHODS:
        for symmetry in SYMMETRY_SETTINGS:
            group = [r for r in records if group_key(r) == (method, bool(symmetry))]
            waves.extend(group[i:i+workers] for i in range(0, len(group), workers))
    return waves

def main():
    args = parse_args()
    if args.inactive >= args.batches:
        raise SystemExit('inactive must be smaller than batches')
    pattern_file = Path(args.pattern_file).resolve()
    output_file = Path(args.output_file).resolve()
    metrics_file = Path(args.metrics_file).resolve()
    errors_file = output_file.with_suffix(output_file.suffix + '.errors.jsonl')
    bank = sorted(json.loads(pattern_file.read_text()), key=lambda r: int(r['id']))
    if len({int(r['id']) for r in bank}) != len(bank):
        raise RuntimeError('Pattern bank contains duplicate IDs')
    end = args.start_index + args.n_samples
    selected = bank[args.start_index:end]
    if len(selected) != args.n_samples:
        raise RuntimeError(f'Requested [{args.start_index}:{end}], bank has {len(bank)} rows')
    existing = read_jsonl(output_file)
    complete = {int(r['id']): r for r in existing}
    if len(complete) != len(existing):
        raise RuntimeError('Existing output JSONL contains duplicate sample IDs')
    selected_ids = {int(r['id']) for r in selected}
    if set(complete) - selected_ids:
        raise RuntimeError('Output file contains IDs outside requested chunk')
    remaining = [r for r in selected if int(r['id']) not in complete]
    slots, allowed = cpu_slots(args.workers, args.threads_per_run)
    slot_sizes = [len(s) for s in slots]
    used = sum(slot_sizes)
    config = {
        'pattern_file': str(pattern_file), 'output_file': str(output_file),
        'slice': [args.start_index, end], 'requested': len(selected),
        'already_complete': len(complete), 'remaining': len(remaining),
        'workers': args.workers, 'slot_sizes': slot_sizes,
        'used_cpus': used, 'exposed_cpus': len(allowed),
        'batches': args.batches, 'inactive': args.inactive,
        'particles': args.particles, 'group_waves': args.group_waves,
    }
    print('[configuration]', json.dumps(config, indent=2), flush=True)
    Path(args.tmp_root).mkdir(parents=True, exist_ok=True)
    waves = build_waves(remaining, args.workers, args.group_waves)
    wave_metrics, failures = [], []
    overall = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for wave_no, wave in enumerate(waves, 1):
            started = time.perf_counter()
            pairs = [
                (pool.submit(simulate, row, args.batches, args.inactive,
                             args.particles, slots[i], args.tmp_root), row)
                for i, row in enumerate(wave)
            ]
            results = []
            for future, row in pairs:
                result = future.result()
                result.setdefault('id', int(row['id']))
                result.setdefault('solver', row['solver'])
                result.setdefault('symmetry_enabled', bool(row['symmetry_enabled']))
                results.append(result)
            success = 0
            for result in results:
                if 'error' in result:
                    append_jsonl(errors_file, result)
                    failures.append(result)
                else:
                    append_jsonl(output_file, result)
                    complete[int(result['id'])] = result
                    success += 1
            active = sum(slot_sizes[:len(wave)])
            metric = {
                'wave': wave_no, 'samples': len(wave), 'active_cores': active,
                'full_wave_cores': used, 'utilization_fraction': active/used,
                'wall_seconds': time.perf_counter()-started,
                'successful': success, 'failed': len(wave)-success,
                'groups': {f'{m}|symmetry={s}': c for (m,s),c in
                           Counter(group_key(r) for r in wave).items()},
            }
            wave_metrics.append(metric)
            print('[wave]', json.dumps(metric), flush=True)
    shutil.rmtree(args.tmp_root, ignore_errors=True)
    completed_rows = [complete[i] for i in sorted(selected_ids & set(complete))]
    metrics = {
        **config, 'successful_samples': len(completed_rows),
        'failed_attempts_this_run': len(failures),
        'waves_this_run': wave_metrics,
        'total_wall_seconds_this_run': time.perf_counter()-overall,
        'group_counts_successful': {f'{m}|symmetry={s}': c for (m,s),c in
                                    Counter(group_key(r) for r in completed_rows).items()},
    }
    atomic_json(metrics_file, metrics)
    print('[final]', json.dumps(metrics, indent=2), flush=True)
    missing = sorted(selected_ids - set(complete))
    if missing:
        raise SystemExit(f'{len(missing)} samples remain incomplete: {missing}')

if __name__ == '__main__':
    main()
