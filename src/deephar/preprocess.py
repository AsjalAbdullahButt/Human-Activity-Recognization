"""Feature scaling, label encoding, reshaping, and train/val splitting."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from deephar.config import Config

NON_FEATURE_COLUMNS = ("subject", "Activity")


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return df[feature_cols], df["Activity"]


@dataclass
class PreparedData:
    X_train: np.ndarray  # (n, 1, n_features) reshaped for LSTM input
    y_train: np.ndarray  # one-hot
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_flat: np.ndarray
    X_val_flat: np.ndarray
    X_test_flat: np.ndarray
    y_train_encoded: np.ndarray
    y_val_encoded: np.ndarray
    y_test_encoded: np.ndarray
    scaler: StandardScaler
    label_encoder: LabelEncoder
    feature_names: list[str]


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame, config: Config) -> PreparedData:
    X_train_raw, y_train_raw = split_features_labels(train_df)
    X_test_raw, y_test_raw = split_features_labels(test_df)
    feature_names = list(X_train_raw.columns)

    label_encoder = LabelEncoder()
    y_train_all_encoded = label_encoder.fit_transform(y_train_raw)
    y_test_encoded = label_encoder.transform(y_test_raw)
    num_classes = len(label_encoder.classes_)

    # Stratified, shuffled train/val split -- the original code sliced the
    # last 20% of (ordered, unshuffled) rows, which could starve validation
    # of whole classes/subjects. Stratifying on the encoded label fixes that.
    X_train_split, X_val_split, y_train_enc, y_val_enc = train_test_split(
        X_train_raw,
        y_train_all_encoded,
        test_size=config.preprocess.val_split,
        stratify=y_train_all_encoded,
        shuffle=True,
        random_state=config.seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val_split)
    X_test_scaled = scaler.transform(X_test_raw)

    y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)
    y_val_cat = to_categorical(y_val_enc, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_encoded, num_classes=num_classes)

    def reshape(x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], 1, x.shape[1])

    return PreparedData(
        X_train=reshape(X_train_scaled),
        y_train=y_train_cat,
        X_val=reshape(X_val_scaled),
        y_val=y_val_cat,
        X_test=reshape(X_test_scaled),
        y_test=y_test_cat,
        X_train_flat=X_train_scaled,
        X_val_flat=X_val_scaled,
        X_test_flat=X_test_scaled,
        y_train_encoded=y_train_enc,
        y_val_encoded=y_val_enc,
        y_test_encoded=y_test_encoded,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_names=feature_names,
    )


def save_preprocessing_artifacts(data: PreparedData, config: Config) -> None:
    config.ensure_output_dirs()
    joblib.dump(data.scaler, config.paths.models_dir / "scaler.joblib")
    joblib.dump(data.label_encoder, config.paths.models_dir / "label_encoder.joblib")


def load_preprocessing_artifacts(config: Config) -> tuple[StandardScaler, LabelEncoder]:
    scaler = joblib.load(config.paths.models_dir / "scaler.joblib")
    label_encoder = joblib.load(config.paths.models_dir / "label_encoder.joblib")
    return scaler, label_encoder
