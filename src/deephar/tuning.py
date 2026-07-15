"""Hyperparameter search for the LSTM.

Keras Tuner is used over Optuna because it has a native Keras `HyperModel`
API: `build(hp)` and `fit(hp, model, ...)` both receive the same `hp` object,
so architecture hyperparameters (units, dropout, layers, learning rate) and a
fit-time hyperparameter (batch size) can be tuned together in one search
without hand-rolling a training loop/objective function the way a bare
Optuna study would require.
"""
from __future__ import annotations

import copy
import json
import logging

import keras
import keras_tuner as kt
import numpy as np
import pandas as pd

from deephar.config import Config
from deephar.model import build_rnn_model

logger = logging.getLogger(__name__)

LSTM_UNITS_1_CHOICES = [32, 64, 96, 128]
LSTM_UNITS_2_CHOICES = [16, 32, 48, 64]
NUM_LSTM_LAYERS_CHOICES = [1, 2]
DROPOUT_CHOICES = [0.1, 0.2, 0.3, 0.4]
LEARNING_RATE_CHOICES = [1e-2, 1e-3, 5e-4, 1e-4]
BATCH_SIZE_CHOICES = [64, 128, 256]


class RnnHyperModel(kt.HyperModel):
    def __init__(self, input_shape: tuple[int, int], num_classes: int, base_config: Config):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_config = base_config

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        config = copy.deepcopy(self.base_config)
        config.model.lstm_units_1 = hp.Choice("lstm_units_1", LSTM_UNITS_1_CHOICES)
        config.model.lstm_units_2 = hp.Choice("lstm_units_2", LSTM_UNITS_2_CHOICES)
        config.model.num_lstm_layers = hp.Choice("num_lstm_layers", NUM_LSTM_LAYERS_CHOICES)
        config.model.dropout = hp.Choice("dropout", DROPOUT_CHOICES)
        config.model.learning_rate = hp.Choice("learning_rate", LEARNING_RATE_CHOICES)
        return build_rnn_model(self.input_shape, self.num_classes, config)

    def fit(self, hp: kt.HyperParameters, model, *args, **kwargs):
        kwargs["batch_size"] = hp.Choice("batch_size", BATCH_SIZE_CHOICES)
        return model.fit(*args, **kwargs)


def run_hyperparameter_search(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, num_classes: int, config: Config
) -> dict:
    """Random search over LSTM units/layers/dropout/learning rate/batch size.

    Runs on the subject-independent (group) train/val split so the search
    optimizes for real generalization, not for exploiting subject leakage.
    Writes every trial's hyperparameters + val_accuracy to
    outputs/metrics/tuning_trials.csv and the winning config to
    outputs/metrics/best_hyperparameters.json. Returns the best hyperparameters.
    """
    config.ensure_output_dirs()
    input_shape = (X_train.shape[1], X_train.shape[2])

    tuner = kt.RandomSearch(
        RnnHyperModel(input_shape, num_classes, config),
        objective="val_accuracy",
        max_trials=config.tuning.max_trials,
        executions_per_trial=config.tuning.executions_per_trial,
        directory=str(config.paths.outputs_dir / "tuning"),
        project_name="lstm_search",
        seed=config.seed,
        overwrite=True,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4, restore_best_weights=True, mode="max"
    )

    logger.info(
        "Starting hyperparameter search: %d trials x up to %d epochs each.",
        config.tuning.max_trials,
        config.tuning.epochs_per_trial,
    )
    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.tuning.epochs_per_trial,
        callbacks=[early_stopping],
        verbose=0,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_params = dict(best_hp.values)

    rows = []
    for trial in tuner.oracle.trials.values():
        row = dict(trial.hyperparameters.values)
        row["trial_id"] = trial.trial_id
        row["val_accuracy"] = trial.score
        rows.append(row)
    trials_df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    trials_df.to_csv(config.paths.metrics_dir / "tuning_trials.csv", index=False)

    with open(config.paths.metrics_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    logger.info(
        "Hyperparameter search complete. Best val_accuracy=%.4f, params=%s",
        trials_df["val_accuracy"].max(),
        best_params,
    )
    return best_params


def apply_best_hyperparameters(config: Config, best_params: dict) -> Config:
    """Return a copy of config with model hyperparameters overridden by a
    hyperparameter search result (e.g. loaded from best_hyperparameters.json)."""
    config = copy.deepcopy(config)
    config.model.lstm_units_1 = best_params["lstm_units_1"]
    config.model.lstm_units_2 = best_params["lstm_units_2"]
    config.model.num_lstm_layers = best_params["num_lstm_layers"]
    config.model.dropout = best_params["dropout"]
    config.model.learning_rate = best_params["learning_rate"]
    config.train.batch_size = best_params["batch_size"]
    return config
