import pandas as pd

from deephar.data import ACTIVITY_LABELS, generate_synthetic_har, load_data


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


def test_load_data_falls_back_to_synthetic_when_files_missing(small_config):
    train_df, test_df, used_synthetic = load_data(small_config)
    assert used_synthetic is True
    assert len(train_df) == small_config.data.synthetic_samples_train
    assert len(test_df) == small_config.data.synthetic_samples_test


def test_load_data_rejects_placeholder_stub_csv(small_config, tmp_path):
    """Reproduces this repo's shipped 404-placeholder CSVs -- load_data must
    not try to treat "404: Not Found" text as real data."""
    small_config.data.train_csv.parent.mkdir(parents=True, exist_ok=True)
    small_config.data.train_csv.write_text("404: Not Found")
    small_config.data.test_csv.write_text("404: Not Found")

    train_df, test_df, used_synthetic = load_data(small_config)
    assert used_synthetic is True


def test_load_data_uses_real_csv_when_valid(small_config):
    small_config.data.train_csv.parent.mkdir(parents=True, exist_ok=True)
    real_train = generate_synthetic_har(20, 5, seed=1)
    real_test = generate_synthetic_har(10, 5, seed=2)
    real_train.to_csv(small_config.data.train_csv, index=False)
    real_test.to_csv(small_config.data.test_csv, index=False)

    train_df, test_df, used_synthetic = load_data(small_config)
    assert used_synthetic is False
    assert len(train_df) == 20
    assert len(test_df) == 10
