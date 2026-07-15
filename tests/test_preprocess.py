import numpy as np
import pandas as pd

from deephar.data import ACTIVITY_LABELS, generate_synthetic_har
from deephar.preprocess import prepare_data, split_features_labels


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
