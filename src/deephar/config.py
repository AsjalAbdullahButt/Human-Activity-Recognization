"""Centralized configuration and path resolution for the DeepHAR project.

All paths are resolved relative to the project root (the directory containing
``config.yaml``), regardless of the current working directory the code is
invoked from.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class DataConfig:
    train_csv: Path
    test_csv: Path
    raw_dir: Path  # data/UCI_HAR_Dataset -- raw Inertial Signals + subject/activity ids
    synthetic_samples_train: int
    synthetic_samples_test: int
    synthetic_n_features: int


@dataclass
class PreprocessConfig:
    val_split: float


@dataclass
class ModelConfig:
    lstm_units_1: int
    lstm_units_2: int
    num_lstm_layers: int
    dense_units: int
    dropout: float
    learning_rate: float


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    early_stopping_patience: int
    monitor: str


@dataclass
class TuningConfig:
    max_trials: int
    epochs_per_trial: int
    executions_per_trial: int


@dataclass
class PathsConfig:
    outputs_dir: Path
    models_dir: Path
    plots_dir: Path
    metrics_dir: Path


@dataclass
class Config:
    seed: int
    data: DataConfig
    preprocess: PreprocessConfig
    model: ModelConfig
    train: TrainConfig
    tuning: TuningConfig
    paths: PathsConfig
    raw: dict[str, Any] = field(default_factory=dict)

    def ensure_output_dirs(self) -> None:
        for d in (
            self.paths.outputs_dir,
            self.paths.models_dir,
            self.paths.plots_dir,
            self.paths.metrics_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path | str | None = None) -> Config:
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    raw = _load_yaml(path)

    def resolve(rel: str) -> Path:
        return PROJECT_ROOT / rel

    data_raw = raw.get("data", {})
    preprocess_raw = raw.get("preprocess", {})
    model_raw = raw.get("model", {})
    train_raw = raw.get("train", {})
    tuning_raw = raw.get("tuning", {})
    paths_raw = raw.get("paths", {})

    return Config(
        seed=raw.get("seed", 42),
        data=DataConfig(
            train_csv=resolve(data_raw.get("train_csv", "data/train.csv")),
            test_csv=resolve(data_raw.get("test_csv", "data/test.csv")),
            raw_dir=resolve(data_raw.get("raw_dir", "data/UCI_HAR_Dataset")),
            synthetic_samples_train=data_raw.get("synthetic_samples_train", 1200),
            synthetic_samples_test=data_raw.get("synthetic_samples_test", 300),
            synthetic_n_features=data_raw.get("synthetic_n_features", 561),
        ),
        preprocess=PreprocessConfig(
            val_split=preprocess_raw.get("val_split", 0.2),
        ),
        model=ModelConfig(
            lstm_units_1=model_raw.get("lstm_units_1", 64),
            lstm_units_2=model_raw.get("lstm_units_2", 32),
            num_lstm_layers=model_raw.get("num_lstm_layers", 2),
            dense_units=model_raw.get("dense_units", 50),
            dropout=model_raw.get("dropout", 0.2),
            learning_rate=model_raw.get("learning_rate", 1e-3),
        ),
        train=TrainConfig(
            epochs=train_raw.get("epochs", 100),
            batch_size=train_raw.get("batch_size", 64),
            early_stopping_patience=train_raw.get("early_stopping_patience", 15),
            monitor=train_raw.get("monitor", "val_accuracy"),
        ),
        tuning=TuningConfig(
            max_trials=tuning_raw.get("max_trials", 15),
            epochs_per_trial=tuning_raw.get("epochs_per_trial", 10),
            executions_per_trial=tuning_raw.get("executions_per_trial", 1),
        ),
        paths=PathsConfig(
            outputs_dir=resolve(paths_raw.get("outputs_dir", "outputs")),
            models_dir=resolve(paths_raw.get("models_dir", "outputs/models")),
            plots_dir=resolve(paths_raw.get("plots_dir", "outputs/plots")),
            metrics_dir=resolve(paths_raw.get("metrics_dir", "outputs/metrics")),
        ),
        raw=raw,
    )


def set_global_seed(seed: int) -> None:
    """Seed every RNG involved (Python, NumPy, and the active Keras backend)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:
        pass
