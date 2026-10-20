#!/usr/bin/env python3
from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from geometry import GRID_SIDE, POS

CNN_TRAINING_SEEDS = (42, 314, 2718, 31415, 161803)
CNN_EPOCHS = 200
CNN_BATCH_SIZE = 32
CNN_PATIENCE = 15
STATISTICAL_CONFIDENCE = 0.95
STATISTICAL_BOOTSTRAP_RESAMPLES = 5000
SURROGATE_CALIBRATION_ALPHA = 0.05
OPENMC_VERIFICATION_CONFIDENCE = 0.95

FUEL_MASK = np.zeros((GRID_SIDE, GRID_SIDE), dtype=np.float32)
for _, (col, row) in POS.items():
    FUEL_MASK[row, col] = 1.0


def configure_tensorflow(threads=None):
    threads = int(threads or os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    inter = max(1, min(4, threads))
    os.environ.setdefault('OMP_NUM_THREADS', str(threads))
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', str(threads))
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', str(inter))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(threads))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
    except RuntimeError:
        pass
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return threads, inter


def set_tensorflow_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


@tf.keras.utils.register_keras_serializable(package='PWR')
class FuelMaskLayer(layers.Layer):
    def __init__(self, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.mask_values = np.asarray(FUEL_MASK if mask is None else mask, dtype=np.float32)

    def call(self, inputs):
        mask = tf.constant(self.mask_values, dtype=inputs.dtype)
        mask = tf.expand_dims(tf.expand_dims(mask, axis=-1), axis=0)
        return tf.tile(mask, [tf.shape(inputs)[0], 1, 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({'mask': self.mask_values.tolist()})
        return config


@tf.keras.utils.register_keras_serializable(package='PWR')
class PositionalEncoding2D(layers.Layer):
    def __init__(self, height=GRID_SIDE, width=GRID_SIDE, **kwargs):
        super().__init__(**kwargs)
        self.height = int(height)
        self.width = int(width)

    def build(self, input_shape):
        x_coordinates = tf.cast(tf.linspace(-1.0, 1.0, self.width), tf.float32)
        y_coordinates = tf.cast(tf.linspace(-1.0, 1.0, self.height), tf.float32)
        xx, yy = tf.meshgrid(x_coordinates, y_coordinates)
        radius = tf.sqrt(xx ** 2 + yy ** 2)
        positional_values = tf.stack([xx, yy, radius], axis=-1)
        self.positional_encoding = self.add_weight(
            name='positional_encoding',
            shape=(self.height, self.width, 3),
            initializer=tf.keras.initializers.Constant(positional_values.numpy()),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        tiled = tf.tile(
            tf.expand_dims(self.positional_encoding, axis=0),
            [tf.shape(inputs)[0], 1, 1, 1],
        )
        return tf.concat([inputs, tiled], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'height': self.height, 'width': self.width})
        return config


def create_model(train_size, batch_size=CNN_BATCH_SIZE, input_shape=(GRID_SIDE, GRID_SIDE, 3)):
    inputs = layers.Input(shape=input_shape, name='loading_pattern')
    mask = FuelMaskLayer(name='fuel_mask')(inputs)
    x = layers.Concatenate(axis=-1)([inputs * mask, mask])
    x = PositionalEncoding2D(name='position_encoding')(x)

    for filters in (64, 128, 256):
        residual = layers.Conv2D(filters, 1, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
        x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)

    average_attention = layers.GlobalAveragePooling2D()(x)
    maximum_attention = layers.GlobalMaxPooling2D()(x)
    attention = layers.Concatenate()([average_attention, maximum_attention])
    attention = layers.Dense(32, activation='relu')(attention)
    attention = layers.Dense(256, activation='sigmoid')(attention)
    attention = layers.Reshape((1, 1, 256))(attention)
    x = layers.Multiply()([x, attention])

    average_features = layers.GlobalAveragePooling2D()(x)
    maximum_features = layers.GlobalMaxPooling2D()(x)
    shared = layers.Concatenate()([average_features, maximum_features])
    shared = layers.Dense(128, activation='relu')(shared)
    shared = layers.Dropout(0.3)(shared)

    def output_head(name):
        branch = layers.Dense(64, activation='relu')(shared)
        branch = layers.Dropout(0.3)(branch)
        return layers.Dense(1, name=name)(branch)

    outputs = [output_head('keff'), output_head('fa'), output_head('fqr')]
    model = models.Model(inputs=inputs, outputs=outputs, name='pwr_multi_output_cnn')
    steps_per_epoch = max(1, int(np.ceil(int(train_size) / int(batch_size))))
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4,
        decay_steps=CNN_EPOCHS * steps_per_epoch,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={'keff': 'mse', 'fa': 'mse', 'fqr': 'mse'},
        loss_weights={'keff': 1.0, 'fa': 1.0, 'fqr': 1.0},
        metrics={'keff': ['mae'], 'fa': ['mae'], 'fqr': ['mae']},
    )
    return model


CUSTOM_OBJECTS = {'FuelMaskLayer': FuelMaskLayer, 'PositionalEncoding2D': PositionalEncoding2D}
