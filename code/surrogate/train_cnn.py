#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cnn_architecture import (
    CNN_BATCH_SIZE, CNN_EPOCHS, CNN_PATIENCE, CNN_TRAINING_SEEDS,
    configure_tensorflow, create_model, set_tensorflow_seed,
)
from data_io import atomic_json


def metrics(y_true, y_pred):
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else float('nan')
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float('nan'),
        'Spearman_rho': float(rho),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-dir', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--threads', type=int, default=None)
    args = parser.parse_args()

    if args.seed not in CNN_TRAINING_SEEDS:
        raise SystemExit(f'Seed {args.seed} is not one of the notebook seeds {CNN_TRAINING_SEEDS}.')

    threads, inter = configure_tensorflow(args.threads)
    set_tensorflow_seed(args.seed)

    round_dir = Path(args.round_dir)
    model_dir = round_dir / 'models'
    history_dir = round_dir / 'histories'
    model_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'cnn_seed_{args.seed}.keras'
    history_path = history_dir / f'cnn_seed_{args.seed}_history.json'
    record_path = history_dir / f'cnn_seed_{args.seed}_record.json'

    arrays = np.load(round_dir / 'round_arrays.npz')
    X_train = arrays['X_train']
    X_val = arrays['X_val']
    X_test = arrays['X_test']
    yk_train, yk_val = arrays['yk_train'], arrays['yk_val']
    yfa_train, yfa_val = arrays['yfa_train'], arrays['yfa_val']
    yfqr_train, yfqr_val = arrays['yfqr_train'], arrays['yfqr_val']

    model = create_model(train_size=len(X_train), batch_size=CNN_BATCH_SIZE)
    callbacks = [
        __import__('tensorflow').keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=CNN_PATIENCE, restore_best_weights=True
        ),
        __import__('tensorflow').keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1
        ),
    ]

    started = time.perf_counter()
    history = model.fit(
        X_train,
        {'keff': yk_train, 'fa': yfa_train, 'fqr': yfqr_train},
        validation_data=(X_val, {'keff': yk_val, 'fa': yfa_val, 'fqr': yfqr_val}),
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )
    elapsed = time.perf_counter() - started
    history_data = {k: [float(v) for v in values] for k, values in history.history.items()}
    atomic_json(history_data, history_path)

    best_epoch_index = int(np.argmin(history_data['val_loss']))
    best_epoch = best_epoch_index + 1
    best_val_loss = float(history_data['val_loss'][best_epoch_index])

    # Independent test audit only; never used for fitting, checkpointing, or active selection weights.
    best_model = __import__('tensorflow').keras.models.load_model(model_path, compile=False)
    pred = best_model.predict(X_test, batch_size=128, verbose=0)
    scalers = {name: joblib.load(round_dir / f'{name}_scaler.pkl') for name in ('keff', 'fa', 'fqr')}
    pred_raw = {
        'keff': scalers['keff'].inverse_transform(pred[0]).ravel(),
        'fa': scalers['fa'].inverse_transform(pred[1]).ravel(),
        'fqr': scalers['fqr'].inverse_transform(pred[2]).ravel(),
    }
    truth = {
        'keff': arrays['yk_test_raw'],
        'fa': arrays['yfa_test_raw'],
        'fqr': arrays['yfqr_test_raw'],
    }
    test_metrics = {name: metrics(truth[name], pred_raw[name]) for name in truth}

    record = {
        'seed': args.seed,
        'epochs_max': CNN_EPOCHS,
        'batch_size': CNN_BATCH_SIZE,
        'patience': CNN_PATIENCE,
        'epochs_completed': len(history_data['loss']),
        'best_epoch': best_epoch,
        'best_validation_loss': best_val_loss,
        'training_wall_seconds': float(elapsed),
        'threads': threads,
        'inter_threads': inter,
        'model_path': str(model_path.resolve()),
        'test_metrics_audit_only': test_metrics,
    }
    atomic_json(record, record_path)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
