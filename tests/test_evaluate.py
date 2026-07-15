from deephar.data import load_data
from deephar.evaluate import evaluate_model
from deephar.model import build_rnn_model
from deephar.preprocess import prepare_data


def test_evaluate_model_produces_expected_artifacts(small_config):
    train_df, test_df, _ = load_data(small_config)
    prepared = prepare_data(train_df, test_df, small_config)

    input_shape = (prepared.X_train.shape[1], prepared.X_train.shape[2])
    num_classes = prepared.y_train.shape[1]
    model = build_rnn_model(input_shape, num_classes, small_config)
    model.fit(prepared.X_train, prepared.y_train, epochs=1, batch_size=16, verbose=0)

    y_pred, accuracy, report = evaluate_model(
        model,
        prepared.X_test,
        prepared.y_test,
        prepared.label_encoder,
        prepared.X_test_flat,
        prepared.y_test_encoded,
        small_config,
    )

    assert 0.0 <= accuracy <= 1.0
    assert len(y_pred) == len(prepared.y_test_encoded)
    assert "macro avg" in report

    metrics_dir = small_config.paths.metrics_dir
    plots_dir = small_config.paths.plots_dir
    for expected in (
        "classification_metrics.csv",
        "classification_report.json",
        "predictions.csv",
        "har_report.txt",
    ):
        assert (metrics_dir / expected).exists(), f"missing {expected}"
    for expected in (
        "confusion_matrix.png",
        "normalized_confusion_matrix.png",
        "performance_metrics_by_class.png",
        "roc_curves.png",
        "pca_visualization.png",
    ):
        assert (plots_dir / expected).exists(), f"missing {expected}"


def test_evaluate_model_handles_class_with_zero_predictions(small_config):
    """Regression test: the original hand-rolled precision/sensitivity math
    divided by zero (producing NaN/warnings) whenever a class had zero
    predicted or zero true samples. classification_report(zero_division=0)
    must not raise or emit invalid-value warnings here."""
    import numpy as np

    train_df, test_df, _ = load_data(small_config)
    prepared = prepare_data(train_df, test_df, small_config)

    input_shape = (prepared.X_train.shape[1], prepared.X_train.shape[2])
    num_classes = prepared.y_train.shape[1]
    model = build_rnn_model(input_shape, num_classes, small_config)
    # Don't train at all -- an untrained model's predictions are effectively
    # random/degenerate and may never pick some classes, exercising the
    # zero-division edge case directly.

    with np.errstate(invalid="raise", divide="raise"):
        y_pred, accuracy, report = evaluate_model(
            model,
            prepared.X_test,
            prepared.y_test,
            prepared.label_encoder,
            prepared.X_test_flat,
            prepared.y_test_encoded,
            small_config,
        )
    assert 0.0 <= accuracy <= 1.0
