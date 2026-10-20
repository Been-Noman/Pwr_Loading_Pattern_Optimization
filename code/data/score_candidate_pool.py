#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

from cnn_architecture import CNN_TRAINING_SEEDS, CUSTOM_OBJECTS, configure_tensorflow
from geometry import FA_LIMIT, FQR_LIMIT, KEFF_HI, KEFF_LO, patterns_to_onehot_grid
from data_io import atomic_json, read_json


def metric_block(y, p):
    return {
        'MAE': float(mean_absolute_error(y, p)),
        'RMSE': float(np.sqrt(mean_squared_error(y, p))),
        'R2': float(r2_score(y, p)) if len(y) > 1 else float('nan'),
        'Spearman_rho': float(spearmanr(y, p).statistic) if len(y) > 1 else float('nan'),
    }


def inverse_predictions(model, x, scalers, batch_size=256):
    outputs = model.predict(x, batch_size=batch_size, verbose=0)
    return np.column_stack([
        scalers['keff'].inverse_transform(outputs[0]).ravel(),
        scalers['fa'].inverse_transform(outputs[1]).ravel(),
        scalers['fqr'].inverse_transform(outputs[2]).ravel(),
    ]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-dir', required=True)
    parser.add_argument('--candidate-bank', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--threads', type=int, default=None)
    args = parser.parse_args()

    configure_tensorflow(args.threads)
    round_dir = Path(args.round_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scalers = {name: joblib.load(round_dir / f'{name}_scaler.pkl') for name in ('keff', 'fa', 'fqr')}

    candidates = read_json(args.candidate_bank)
    patterns = np.asarray([r['pattern'] for r in candidates], dtype=np.int8)
    x_candidates = patterns_to_onehot_grid(patterns)
    arrays = np.load(round_dir / 'round_arrays.npz')
    x_cal = arrays['X_calibration']
    y_cal = np.column_stack([
        arrays['yk_calibration_raw'], arrays['yfa_calibration_raw'], arrays['yfqr_calibration_raw']
    ])

    candidate_predictions = []
    calibration_predictions = []
    seed_records = []
    for seed in CNN_TRAINING_SEEDS:
        model_path = round_dir / 'models' / f'cnn_seed_{seed}.keras'
        record_path = round_dir / 'histories' / f'cnn_seed_{seed}_record.json'
        if not model_path.exists() or not record_path.exists():
            raise SystemExit(f'Missing completed CNN seed artifact for {seed}.')
        model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
        candidate_predictions.append(inverse_predictions(model, x_candidates, scalers))
        calibration_predictions.append(inverse_predictions(model, x_cal, scalers))
        seed_records.append(read_json(record_path))
        del model
        tf.keras.backend.clear_session()

    cand = np.stack(candidate_predictions, axis=0)  # seeds x candidates x outputs
    cal = np.stack(calibration_predictions, axis=0)
    cand_mean = cand.mean(axis=0)
    cand_std = cand.std(axis=0, ddof=1)
    cal_mean = cal.mean(axis=0)
    cal_std = cal.std(axis=0, ddof=1)

    # Raw ensemble calibration audit. This is diagnostic for active learning, not the final
    # conformal calibration used by the research-grade final model.
    cal_metrics = {
        name: metric_block(y_cal[:, j], cal_mean[:, j])
        for j, name in enumerate(('keff', 'Fa', 'Fq_r'))
    }
    raw_pred_feasible = (
        (cal_mean[:, 0] >= KEFF_LO) & (cal_mean[:, 0] <= KEFF_HI)
        & (cal_mean[:, 1] <= FA_LIMIT) & (cal_mean[:, 2] <= FQR_LIMIT)
    )
    true_feasible = (
        (y_cal[:, 0] >= KEFF_LO) & (y_cal[:, 0] <= KEFF_HI)
        & (y_cal[:, 1] <= FA_LIMIT) & (y_cal[:, 2] <= FQR_LIMIT)
    )
    selected = int(raw_pred_feasible.sum())
    false_selected = int(np.sum(raw_pred_feasible & ~true_feasible))

    np.savez_compressed(
        out / 'candidate_ensemble_predictions.npz',
        candidate_ids=np.asarray([int(r['candidate_id']) for r in candidates], dtype=int),
        per_seed=cand,
        mean=cand_mean,
        std=cand_std,
        calibration_per_seed=cal,
        calibration_mean=cal_mean,
        calibration_std=cal_std,
    )

    csv_path = out / 'candidate_ensemble_scores.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        fieldnames = [
            'candidate_id', 'solver', 'symmetry_enabled', 'structural_cluster', 'cluster_distance',
            'mean_keff', 'std_keff', 'mean_Fa', 'std_Fa', 'mean_Fq_r', 'std_Fq_r',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, mean, std in zip(candidates, cand_mean, cand_std):
            writer.writerow({
                'candidate_id': int(row['candidate_id']),
                'solver': row.get('solver'),
                'symmetry_enabled': bool(row.get('symmetry_enabled')),
                'structural_cluster': int(row.get('structural_cluster', -1)),
                'cluster_distance': float(row.get('cluster_distance', 0.0)),
                'mean_keff': float(mean[0]), 'std_keff': float(std[0]),
                'mean_Fa': float(mean[1]), 'std_Fa': float(std[1]),
                'mean_Fq_r': float(mean[2]), 'std_Fq_r': float(std[2]),
            })

    report = {
        'cnn_seeds': list(CNN_TRAINING_SEEDS),
        'candidate_count': len(candidates),
        'calibration_count': len(y_cal),
        'calibration_ensemble_metrics': cal_metrics,
        'calibration_raw_predicted_feasible': selected,
        'calibration_raw_false_feasible': false_selected,
        'calibration_raw_false_feasible_rate': false_selected / selected if selected else None,
        'mean_ensemble_std': {
            'keff': float(np.mean(cand_std[:, 0])),
            'Fa': float(np.mean(cand_std[:, 1])),
            'Fq_r': float(np.mean(cand_std[:, 2])),
        },
        'seed_training_records': seed_records,
        'active_learning_note': 'Test split is not used for candidate acquisition. Acquisition uses ensemble predictions/disagreement only.',
    }
    atomic_json(report, out / 'ensemble_round_report.json')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
