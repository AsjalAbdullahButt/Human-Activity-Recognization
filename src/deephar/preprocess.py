"""Feature scaling, label encoding, reshaping, and train/val splitting."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from deephar.config import Config
from deephar.data import SignalWindows

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


@dataclass
class PreparedSignalData:
    X_train: np.ndarray  # (n, TIMESTEPS, N_CHANNELS), per-channel scaled
    y_train: np.ndarray  # one-hot
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_flat: np.ndarray  # (n, TIMESTEPS * N_CHANNELS), for PCA/plots only
    X_val_flat: np.ndarray
    X_test_flat: np.ndarray
    y_train_encoded: np.ndarray
    y_val_encoded: np.ndarray
    y_test_encoded: np.ndarray
    subject_train: np.ndarray
    subject_val: np.ndarray
    scaler: StandardScaler  # fit per-channel, i.e. on (n * TIMESTEPS, N_CHANNELS)
    label_encoder: LabelEncoder
    split_strategy: str  # "group" (subject-independent) or "stratified" (same-subject-leakage)


def _split_indices(y_encoded: np.ndarray, subject: np.ndarray, val_split: float, seed: int, group_split: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx).

    group_split=True uses GroupShuffleSplit keyed on subject, so no subject's
    windows appear in both train and val -- the methodologically correct
    split for HAR, since it measures generalization to unseen people.

    group_split=False uses a stratified random split that ignores subject
    identity. Because UCI HAR windows are 50%-overlapping slices of a
    continuous per-subject recording, a random shuffle puts near-duplicate,
    temporally-adjacent windows from the same subject/session into both train
    and val. The model can then partly "recognize the subject" (gait
    idiosyncrasies, sensor calibration/placement) rather than the activity,
    which inflates validation accuracy relative to true subject-independent
    generalization. This split exists only to demonstrate and quantify that
    inflation -- it is not the split a production model should be judged on.
    """
    indices = np.arange(len(y_encoded))
    if group_split:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        train_idx, val_idx = next(splitter.split(indices, y_encoded, groups=subject))
    else:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, stratify=y_encoded, shuffle=True, random_state=seed
        )
    return train_idx, val_idx


def prepare_signal_data(
    train: SignalWindows, test: SignalWindows, config: Config, group_split: bool = True
) -> PreparedSignalData:
    """Prepare raw (timesteps, channels) signal windows for the LSTM.

    Scaling is per-channel: the StandardScaler is fit on
    (n_train_windows * TIMESTEPS, N_CHANNELS) so each of the 9 sensor
    channels gets its own mean/std, not each of the 128*9 flattened
    positions -- flattening first would treat "acceleration at timestep 50"
    and "acceleration at timestep 51" as unrelated features with their own
    statistics, which makes no physical sense for a continuous signal.
    """
    label_encoder = LabelEncoder()
    y_train_all_encoded = label_encoder.fit_transform(train.y)
    y_test_encoded = label_encoder.transform(test.y)
    num_classes = len(label_encoder.classes_)

    train_idx, val_idx = _split_indices(
        y_train_all_encoded, train.subject, config.preprocess.val_split, config.seed, group_split
    )

    X_train_split, X_val_split = train.X[train_idx], train.X[val_idx]
    y_train_enc, y_val_enc = y_train_all_encoded[train_idx], y_train_all_encoded[val_idx]
    subject_train_split, subject_val_split = train.subject[train_idx], train.subject[val_idx]

    n_channels = train.X.shape[-1]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split.reshape(-1, n_channels)).reshape(X_train_split.shape)
    X_val_scaled = scaler.transform(X_val_split.reshape(-1, n_channels)).reshape(X_val_split.shape)
    X_test_scaled = scaler.transform(test.X.reshape(-1, n_channels)).reshape(test.X.shape)

    y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)
    y_val_cat = to_categorical(y_val_enc, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_encoded, num_classes=num_classes)

    def flatten(x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1)

    return PreparedSignalData(
        X_train=X_train_scaled,
        y_train=y_train_cat,
        X_val=X_val_scaled,
        y_val=y_val_cat,
        X_test=X_test_scaled,
        y_test=y_test_cat,
        X_train_flat=flatten(X_train_scaled),
        X_val_flat=flatten(X_val_scaled),
        X_test_flat=flatten(X_test_scaled),
        y_train_encoded=y_train_enc,
        y_val_encoded=y_val_enc,
        y_test_encoded=y_test_encoded,
        subject_train=subject_train_split,
        subject_val=subject_val_split,
        scaler=scaler,
        label_encoder=label_encoder,
        split_strategy="group" if group_split else "stratified",
    )


def save_preprocessing_artifacts(data: PreparedData | PreparedSignalData, config: Config) -> None:
    config.ensure_output_dirs()
    joblib.dump(data.scaler, config.paths.models_dir / "scaler.joblib")
    joblib.dump(data.label_encoder, config.paths.models_dir / "label_encoder.joblib")


def load_preprocessing_artifacts(config: Config) -> tuple[StandardScaler, LabelEncoder]:
    scaler = joblib.load(config.paths.models_dir / "scaler.joblib")
    label_encoder = joblib.load(config.paths.models_dir / "label_encoder.joblib")
    return scaler, label_encoder
