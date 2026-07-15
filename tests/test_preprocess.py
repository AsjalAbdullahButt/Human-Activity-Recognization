import numpy as np
import pandas as pd

from deephar.data import ACTIVITY_LABELS, N_CHANNELS, TIMESTEPS, generate_synthetic_har, generate_synthetic_signal_windows
from deephar.preprocess import prepare_data, prepare_signal_data, split_features_labels


def _ordered_by_class_df(n_per_class=40, n_features=8, seed=0):
    """Build a dataframe where rows are grouped in contiguous per-class
    blocks, like the real UCI HAR train.csv (sorted by subject/experiment,
    which correlates strongly with activity). This is the exact shape of
    data that breaks a "take the last 20% of rows" validation split."""
    rng = np.random.default_rng(seed)
    frames = []
    for label in ACTIVITY_LABELS:
        block = pd.DataFrame(
            rng.normal(size=(n_per_class, n_features)),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        block["subject"] = 1
        block["Activity"] = label
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def test_prepare_data_stratified_split_keeps_all_classes_in_validation():
    """Regression test for the original bug: main() sliced the last 20% of
    an unshuffled, class-ordered dataframe for validation, which could
    starve entire classes out of the validation set. A stratified, shuffled
    split must not do that."""
    from deephar.config import load_config

    config = load_config()
    config.preprocess.val_split = 0.2

    train_df = _ordered_by_class_df(n_per_class=40)
    test_df = _ordered_by_class_df(n_per_class=10, seed=1)

    prepared = prepare_data(train_df, test_df, config)

    val_classes_present = set(np.unique(prepared.y_val_encoded))
    assert val_classes_present == set(range(len(ACTIVITY_LABELS)))

    # Roughly stratified: each class should have a non-trivial share of val rows.
    counts = pd.Series(prepared.y_val_encoded).value_counts()
    assert counts.min() >= 3  # 20% of 40 == 8 per class if perfectly stratified


def test_prepare_data_scaling_and_shapes():
    from deephar.config import load_config

    config = load_config()
    train_df = generate_synthetic_har(100, 12, seed=0)
    test_df = generate_synthetic_har(30, 12, seed=1)

    prepared = prepare_data(train_df, test_df, config)

    # Reshaped for LSTM input: (n, 1, n_features)
    assert prepared.X_train.shape[1:] == (1, 12)
    assert prepared.X_val.shape[1:] == (1, 12)
    assert prepared.X_test.shape[1:] == (1, 12)

    # Scaler fit on train only -> train features approx standardized
    assert abs(prepared.X_train_flat.mean()) < 0.5
    assert abs(prepared.X_train_flat.std() - 1.0) < 0.5

    # One-hot labels match number of activity classes
    assert prepared.y_train.shape[1] == len(prepared.label_encoder.classes_)
    assert prepared.y_train.sum(axis=1).min() == 1  # each row one-hot


def test_split_features_labels_drops_non_feature_columns():
    df = generate_synthetic_har(10, 4, seed=0)
    X, y = split_features_labels(df)
    assert "subject" not in X.columns
    assert "Activity" not in X.columns
    assert len(y) == len(df)


# ---------------------------------------------------------------------------
# Raw signal window preparation: per-channel scaling + group/stratified split
# ---------------------------------------------------------------------------


def _signal_windows_with_distinct_subjects(n_per_subject=20, n_subjects=8, seed=0):
    """Windows where each subject's data all shares one activity, so a group
    split (no subject overlap) is trivially distinguishable from a
    stratified split (subject can appear on both sides)."""
    train = generate_synthetic_signal_windows(n_per_subject * n_subjects, seed=seed, n_subjects=n_subjects)
    # generate_synthetic_signal_windows already assigns subjects randomly per
    # sample; that's enough to exercise both split strategies.
    return train


def test_prepare_signal_data_shapes():
    from deephar.config import load_config

    config = load_config()
    train = generate_synthetic_signal_windows(200, seed=0)
    test = generate_synthetic_signal_windows(60, seed=1)

    prepared = prepare_signal_data(train, test, config, group_split=True)

    assert prepared.X_train.shape[1:] == (TIMESTEPS, N_CHANNELS)
    assert prepared.X_val.shape[1:] == (TIMESTEPS, N_CHANNELS)
    assert prepared.X_test.shape == test.X.shape
    assert prepared.y_train.shape[1] == len(prepared.label_encoder.classes_)
    assert prepared.X_train_flat.shape == (prepared.X_train.shape[0], TIMESTEPS * N_CHANNELS)


def test_prepare_signal_data_scales_per_channel_not_per_flattened_feature():
    """The scaler must be fit on (n*TIMESTEPS, N_CHANNELS), i.e. one
    mean/std per sensor channel -- not on the flattened (TIMESTEPS*N_CHANNELS,)
    vector, which would treat every (timestep, channel) pair as its own
    unrelated feature."""
    from deephar.config import load_config

    config = load_config()
    train = generate_synthetic_signal_windows(300, seed=0)
    test = generate_synthetic_signal_windows(60, seed=1)

    prepared = prepare_signal_data(train, test, config, group_split=True)

    assert prepared.scaler.mean_.shape == (N_CHANNELS,)
    assert prepared.scaler.scale_.shape == (N_CHANNELS,)

    # Per-channel scaled train data should be ~standardized across all
    # (samples, timesteps) for each channel independently.
    per_channel_mean = prepared.X_train.reshape(-1, N_CHANNELS).mean(axis=0)
    per_channel_std = prepared.X_train.reshape(-1, N_CHANNELS).std(axis=0)
    assert np.all(np.abs(per_channel_mean) < 0.3)
    assert np.all(np.abs(per_channel_std - 1.0) < 0.3)


def test_prepare_signal_data_group_split_has_no_subject_overlap():
    from deephar.config import load_config

    config = load_config()
    config.preprocess.val_split = 0.3
    train = generate_synthetic_signal_windows(400, seed=0, n_subjects=12)
    test = generate_synthetic_signal_windows(80, seed=1, n_subjects=12)

    prepared = prepare_signal_data(train, test, config, group_split=True)

    train_subjects = set(np.unique(prepared.subject_train))
    val_subjects = set(np.unique(prepared.subject_val))
    assert train_subjects.isdisjoint(val_subjects)
    assert prepared.split_strategy == "group"


def test_prepare_signal_data_stratified_split_can_share_subjects():
    """Contrast case for the group-split test: the stratified (leaky) split
    ignores subject identity, so with few subjects and many windows each,
    the same subject routinely lands on both sides."""
    from deephar.config import load_config

    config = load_config()
    config.preprocess.val_split = 0.3
    train = generate_synthetic_signal_windows(400, seed=0, n_subjects=3)
    test = generate_synthetic_signal_windows(80, seed=1, n_subjects=3)

    prepared = prepare_signal_data(train, test, config, group_split=False)

    train_subjects = set(np.unique(prepared.subject_train))
    val_subjects = set(np.unique(prepared.subject_val))
    assert not train_subjects.isdisjoint(val_subjects)
    assert prepared.split_strategy == "stratified"
