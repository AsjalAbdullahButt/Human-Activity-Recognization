import numpy as np
import pandas as pd
import pytest

from deephar.data import (
    ACTIVITY_LABELS,
    N_CHANNELS,
    TIMESTEPS,
    RealDataNotFoundError,
    generate_synthetic_har,
    generate_synthetic_signal_windows,
    load_feature_csv_data,
    load_signal_data,
)


def _write_signal_split(raw_dir, split, n_samples, n_subjects=5, seed=0):
    """Write a minimal real-shaped UCI HAR split (Inertial Signals + subject/y
    files) so tests can exercise the real-data-found path without the full
    dataset."""
    from deephar.data import SIGNAL_CHANNELS

    signals_dir = raw_dir / split / "Inertial Signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for channel in SIGNAL_CHANNELS:
        arr = rng.normal(size=(n_samples, TIMESTEPS))
        np.savetxt(signals_dir / f"{channel}_{split}.txt", arr)

    subject = rng.integers(1, n_subjects + 1, size=n_samples)
    np.savetxt(raw_dir / split / f"subject_{split}.txt", subject, fmt="%d")

    activity_id = rng.integers(1, len(ACTIVITY_LABELS) + 1, size=n_samples)
    np.savetxt(raw_dir / split / f"y_{split}.txt", activity_id, fmt="%d")


# ---------------------------------------------------------------------------
# Raw signal windows (primary LSTM input)
# ---------------------------------------------------------------------------


def test_generate_synthetic_signal_windows_shape():
    windows = generate_synthetic_signal_windows(n_samples=30, seed=1)
    assert windows.X.shape == (30, TIMESTEPS, N_CHANNELS)
    assert windows.y.shape == (30,)
    assert windows.subject.shape == (30,)
    assert set(np.unique(windows.y)) <= set(ACTIVITY_LABELS)


def test_load_signal_data_raises_without_real_data_or_synthetic_optin(small_config):
    """Production training must fail loudly, not silently fall back, when the
    real dataset is missing and demo mode wasn't explicitly requested."""
    with pytest.raises(RealDataNotFoundError):
        load_signal_data(small_config, allow_synthetic=False)


def test_load_signal_data_uses_synthetic_when_explicitly_allowed(small_config):
    train, test, used_synthetic = load_signal_data(small_config, allow_synthetic=True)
    assert used_synthetic is True
    assert train.X.shape == (small_config.data.synthetic_samples_train, TIMESTEPS, N_CHANNELS)
    assert test.X.shape == (small_config.data.synthetic_samples_test, TIMESTEPS, N_CHANNELS)


def test_load_signal_data_uses_real_files_when_present(small_config):
    _write_signal_split(small_config.data.raw_dir, "train", n_samples=25, seed=1)
    _write_signal_split(small_config.data.raw_dir, "test", n_samples=10, seed=2)

    train, test, used_synthetic = load_signal_data(small_config, allow_synthetic=False)
    assert used_synthetic is False
    assert train.X.shape == (25, TIMESTEPS, N_CHANNELS)
    assert test.X.shape == (10, TIMESTEPS, N_CHANNELS)


# ---------------------------------------------------------------------------
# Pre-engineered 561-feature CSVs (baseline only)
# ---------------------------------------------------------------------------


def test_generate_synthetic_har_shape_and_columns():
    df = generate_synthetic_har(n_samples=50, n_features=10, seed=1)
    assert df.shape == (50, 12)  # 10 features + subject + Activity
    assert "subject" in df.columns
    assert "Activity" in df.columns
    assert set(df["Activity"].unique()) <= set(ACTIVITY_LABELS)


def test_generate_synthetic_har_is_deterministic_given_seed():
    df1 = generate_synthetic_har(n_samples=30, n_features=5, seed=7)
    df2 = generate_synthetic_har(n_samples=30, n_features=5, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_synthetic_har_shared_centers_are_learnable():
    """Train/test splits must share class geometry, not just the same seed
    family -- otherwise a model fit on train would have no relationship to
    test and every downstream metric would be meaningless."""
    from deephar.data import _class_centers

    centers = _class_centers(n_features=30, seed=42)
    train_df = generate_synthetic_har(200, 30, seed=42, class_centers=centers)
    test_df = generate_synthetic_har(60, 30, seed=43, class_centers=centers)

    feature_cols = [c for c in train_df.columns if c not in ("subject", "Activity")]
    train_means = train_df.groupby("Activity")[feature_cols].mean()
    test_means = test_df.groupby("Activity")[feature_cols].mean()
    # Per-class means should be close between train/test since they share
    # the same underlying centers (loose tolerance -- these are noisy means).
    for label in train_means.index:
        diff = (train_means.loc[label] - test_means.loc[label]).abs().mean()
        assert diff < 0.5


def test_load_feature_csv_data_raises_without_real_data_or_synthetic_optin(small_config):
    with pytest.raises(RealDataNotFoundError):
        load_feature_csv_data(small_config, allow_synthetic=False)


def test_load_feature_csv_data_falls_back_to_synthetic_when_explicitly_allowed(small_config):
    train_df, test_df, used_synthetic = load_feature_csv_data(small_config, allow_synthetic=True)
    assert used_synthetic is True
    assert len(train_df) == small_config.data.synthetic_samples_train
    assert len(test_df) == small_config.data.synthetic_samples_test


def test_load_feature_csv_data_rejects_placeholder_stub_csv(small_config):
    """Reproduces this repo's shipped 404-placeholder CSVs -- loading must
    not try to treat "404: Not Found" text as real data."""
    small_config.data.train_csv.parent.mkdir(parents=True, exist_ok=True)
    small_config.data.train_csv.write_text("404: Not Found")
    small_config.data.test_csv.write_text("404: Not Found")

    with pytest.raises(RealDataNotFoundError):
        load_feature_csv_data(small_config, allow_synthetic=False)


def test_load_feature_csv_data_uses_real_csv_when_valid(small_config):
    small_config.data.train_csv.parent.mkdir(parents=True, exist_ok=True)
    real_train = generate_synthetic_har(20, 5, seed=1)
    real_test = generate_synthetic_har(10, 5, seed=2)
    real_train.to_csv(small_config.data.train_csv, index=False)
    real_test.to_csv(small_config.data.test_csv, index=False)

    train_df, test_df, used_synthetic = load_feature_csv_data(small_config, allow_synthetic=False)
    assert used_synthetic is False
    assert len(train_df) == 20
    assert len(test_df) == 10
