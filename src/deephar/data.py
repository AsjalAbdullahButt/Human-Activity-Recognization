"""Data loading for DeepHAR.

Two independent, non-interchangeable input formats:

- Raw inertial-signal windows (128 timesteps x 9 channels per sample) -- the
  PRIMARY input for the LSTM. A real time axis, loaded from
  ``data/UCI_HAR_Dataset/{train,test}/Inertial Signals/*.txt``.
- Pre-engineered 561-feature CSVs (``data/train.csv``, ``data/test.csv``) --
  used only for a non-temporal baseline model, to compare against the LSTM.

Synthetic data is available ONLY when the caller explicitly opts in
(``allow_synthetic=True``, wired to a ``--demo``/``--synthetic`` CLI flag). It
is never a silent fallback: production training without the real dataset
present fails loudly with ``RealDataNotFoundError`` instead of quietly
producing numbers that could be mistaken for real performance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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

# Order fixes the channel axis of the (timesteps, channels) LSTM input.
SIGNAL_CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]
TIMESTEPS = 128
N_CHANNELS = len(SIGNAL_CHANNELS)

REQUIRED_COLUMNS = {"subject", "Activity"}


class RealDataNotFoundError(FileNotFoundError):
    """Raised when production training is attempted without the real dataset
    present and no explicit demo/synthetic opt-in was given."""


@dataclass
class SignalWindows:
    X: np.ndarray  # (n, TIMESTEPS, N_CHANNELS) float32
    y: np.ndarray  # (n,) activity name strings
    subject: np.ndarray  # (n,) int subject ids


# ---------------------------------------------------------------------------
# Raw inertial-signal windows (primary LSTM input)
# ---------------------------------------------------------------------------


def _signal_files_for_split(raw_dir: Path, split: str) -> list[Path]:
    signals_dir = raw_dir / split / "Inertial Signals"
    channel_files = [signals_dir / f"{channel}_{split}.txt" for channel in SIGNAL_CHANNELS]
    return channel_files + [raw_dir / split / f"subject_{split}.txt", raw_dir / split / f"y_{split}.txt"]


def _signal_data_available(raw_dir: Path) -> bool:
    return all(p.exists() for split in ("train", "test") for p in _signal_files_for_split(raw_dir, split))


def _load_split_signal_windows(raw_dir: Path, split: str) -> SignalWindows:
    signals_dir = raw_dir / split / "Inertial Signals"
    channels = [np.loadtxt(signals_dir / f"{channel}_{split}.txt") for channel in SIGNAL_CHANNELS]
    # Each channel array is (n, TIMESTEPS); stack into (n, TIMESTEPS, N_CHANNELS).
    X = np.stack(channels, axis=-1).astype("float32")

    subject = np.loadtxt(raw_dir / split / f"subject_{split}.txt", dtype=int)
    activity_id = np.loadtxt(raw_dir / split / f"y_{split}.txt", dtype=int)
    y = np.array([ACTIVITY_LABELS[i - 1] for i in activity_id])

    return SignalWindows(X=X, y=y, subject=subject)


def generate_synthetic_signal_windows(
    n_samples: int,
    seed: int,
    n_subjects: int = 10,
    class_centers: np.ndarray | None = None,
) -> SignalWindows:
    """Generate (timesteps, channels) windows shaped like the real inertial
    signals, for demo mode only. Each class gets a distinct per-channel mean
    level plus a slow sinusoidal component and per-timestep noise, so the
    data has a real (if fake) time axis rather than being constant-in-time."""
    rng = np.random.default_rng(seed)
    if class_centers is None:
        class_centers = np.random.default_rng(0).normal(scale=0.3, size=(len(ACTIVITY_LABELS), N_CHANNELS))

    label_indices = rng.integers(0, len(ACTIVITY_LABELS), size=n_samples)
    t = np.linspace(0, 2 * np.pi, TIMESTEPS)

    X = np.empty((n_samples, TIMESTEPS, N_CHANNELS), dtype="float32")
    for i, label_idx in enumerate(label_indices):
        base = class_centers[label_idx]  # (N_CHANNELS,)
        phase = rng.uniform(0, 2 * np.pi, size=N_CHANNELS)
        freq = rng.uniform(0.8, 1.5, size=N_CHANNELS)
        wave = 0.2 * np.sin(np.outer(t, freq) + phase)  # (TIMESTEPS, N_CHANNELS)
        noise = rng.normal(scale=0.15, size=(TIMESTEPS, N_CHANNELS))
        X[i] = base[None, :] + wave + noise

    y = np.array([ACTIVITY_LABELS[i] for i in label_indices])
    subject = rng.integers(1, n_subjects + 1, size=n_samples)
    return SignalWindows(X=X, y=y, subject=subject)


def load_signal_data(config: Config, allow_synthetic: bool = False) -> tuple[SignalWindows, SignalWindows, bool]:
    """Load raw (timesteps, channels) signal windows for train/test.

    Returns (train, test, used_synthetic). Raises RealDataNotFoundError if the
    real dataset is absent and allow_synthetic is False -- this must fail
    loudly rather than silently substituting synthetic data.
    """
    if _signal_data_available(config.data.raw_dir):
        logger.info("Loading real UCI HAR inertial signal windows from %s", config.data.raw_dir)
        train = _load_split_signal_windows(config.data.raw_dir, "train")
        test = _load_split_signal_windows(config.data.raw_dir, "test")
        return train, test, False

    if not allow_synthetic:
        raise RealDataNotFoundError(
            f"Real UCI HAR inertial signal data not found at {config.data.raw_dir}. "
            "Run `python scripts/download_data.py` to fetch it, or pass "
            "--demo/--synthetic to run in demo mode on synthetic data explicitly "
            "(demo mode numbers are not real performance and must not be reported as such)."
        )

    logger.warning("DEMO MODE: generating SYNTHETIC signal windows (real UCI HAR data not found).")
    centers = np.random.default_rng(config.seed).normal(scale=0.3, size=(len(ACTIVITY_LABELS), N_CHANNELS))
    train = generate_synthetic_signal_windows(config.data.synthetic_samples_train, seed=config.seed, class_centers=centers)
    test = generate_synthetic_signal_windows(config.data.synthetic_samples_test, seed=config.seed + 1, class_centers=centers)
    return train, test, True


# ---------------------------------------------------------------------------
# Pre-engineered 561-feature CSVs (non-temporal baseline only)
# ---------------------------------------------------------------------------


def _is_valid_har_csv(path: Path) -> bool:
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
    """Generate a synthetic dataset shaped like the UCI HAR feature CSVs, for
    demo mode only. Each activity class gets its own Gaussian cluster in
    feature space so the data is separable enough for the model/tests/plots
    to behave meaningfully, without touching or fabricating the real dataset.
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


def load_feature_csv_data(config: Config, allow_synthetic: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Load the pre-engineered 561-feature train/test CSVs (baseline model only).

    Returns (train_df, test_df, used_synthetic). Raises RealDataNotFoundError
    if the real CSVs are absent/invalid and allow_synthetic is False.
    """
    train_ok = _is_valid_har_csv(config.data.train_csv)
    test_ok = _is_valid_har_csv(config.data.test_csv)

    if train_ok and test_ok:
        logger.info("Loading real UCI HAR feature CSVs from %s / %s", config.data.train_csv, config.data.test_csv)
        train_df = pd.read_csv(config.data.train_csv)
        test_df = pd.read_csv(config.data.test_csv)
        return train_df, test_df, False

    if not allow_synthetic:
        raise RealDataNotFoundError(
            f"Real UCI HAR feature CSVs not found or invalid at {config.data.train_csv} / "
            f"{config.data.test_csv}. Run `python scripts/download_data.py` to fetch them, or "
            "pass --demo/--synthetic to run in demo mode on synthetic data explicitly."
        )

    logger.warning("DEMO MODE: generating SYNTHETIC feature data (real UCI HAR CSVs not found).")
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
