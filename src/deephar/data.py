"""Data loading for DeepHAR: real UCI HAR CSVs when present, else a synthetic
fallback so the whole pipeline is runnable without the (large, un-shippable)
real dataset.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from deephar.config import Config

logger = logging.getLogger(__name__)

ACTIVITY_LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

REQUIRED_COLUMNS = {"subject", "Activity"}


def _is_valid_har_csv(path) -> bool:
    """Reject missing files and the 404-placeholder stubs shipped in this repo."""
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    return REQUIRED_COLUMNS.issubset(df.columns)


def _class_centers(n_features: int, seed: int) -> np.ndarray:
    """Class cluster centers, derived from `seed` alone so train/test splits
    that use different per-call seeds (for different noise/subject draws)
    still share the same underlying class geometry."""
    rng = np.random.default_rng(seed)
    # Scale chosen so nearest-centroid accuracy lands around ~85-90%: small
    # enough to give the demo plots (confusion matrix, ROC) visible overlap
    # between classes, instead of the trivially-perfect separation that a
    # larger scale produces in high dimensions.
    return rng.normal(scale=0.09, size=(len(ACTIVITY_LABELS), n_features))


def generate_synthetic_har(
    n_samples: int,
    n_features: int,
    seed: int,
    n_subjects: int = 10,
    class_centers: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generate a synthetic dataset shaped like the UCI HAR feature CSVs.

    Each activity class gets its own Gaussian cluster in feature space so the
    data is separable enough for the model/tests/plots to behave meaningfully,
    without touching or fabricating the real dataset. Pass the same
    `class_centers` to train/test calls so both splits share one underlying
    distribution (otherwise a model trained on one would be meaningless on
    the other).
    """
    rng = np.random.default_rng(seed)
    if class_centers is None:
        class_centers = _class_centers(n_features, seed=0)

    labels = rng.choice(ACTIVITY_LABELS, size=n_samples)
    label_indices = np.array([ACTIVITY_LABELS.index(l) for l in labels])
    features = class_centers[label_indices] + rng.normal(scale=1.0, size=(n_samples, n_features))

    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=feature_cols)
    df["subject"] = rng.integers(1, n_subjects + 1, size=n_samples)
    df["Activity"] = labels
    return df


def load_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Load train/test data.

    Returns (train_df, test_df, used_synthetic). Falls back to synthetic data
    when the real CSVs are absent or are the 404-placeholder stubs shipped in
    this repo.
    """
    train_ok = _is_valid_har_csv(config.data.train_csv)
    test_ok = _is_valid_har_csv(config.data.test_csv)

    if train_ok and test_ok:
        logger.info("Loading real UCI HAR data from %s / %s", config.data.train_csv, config.data.test_csv)
        train_df = pd.read_csv(config.data.train_csv)
        test_df = pd.read_csv(config.data.test_csv)
        return train_df, test_df, False

    logger.warning(
        "Real dataset not found or invalid at %s / %s -- generating synthetic "
        "data instead. See scripts/download_data.py for how to get the real "
        "UCI HAR dataset.",
        config.data.train_csv,
        config.data.test_csv,
    )
    centers = _class_centers(config.data.synthetic_n_features, seed=config.seed)
    train_df = generate_synthetic_har(
        config.data.synthetic_samples_train,
        config.data.synthetic_n_features,
        seed=config.seed,
        class_centers=centers,
    )
    test_df = generate_synthetic_har(
        config.data.synthetic_samples_test,
        config.data.synthetic_n_features,
        seed=config.seed + 1,
        class_centers=centers,
    )
    return train_df, test_df, True
