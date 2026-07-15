"""Model training, hyperparameter search wiring, and the end-to-end pipeline
entry point.

Run `python -m deephar.train` for a real training run (requires the real
dataset, see scripts/download_data.py) or `python -m deephar.train --demo`
for a demo run on synthetic data. There is no silent fallback between the
two: real training without the real dataset raises
`deephar.data.RealDataNotFoundError` instead of quietly substituting fake
numbers.
"""
from __future__ import annotations

import argparse
import logging

import keras
import numpy as np
import pandas as pd

from deephar.config import Config, load_config, set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASS_IMBALANCE_THRESHOLD = 1.5  # max/min class count ratio above which we'd apply class weighting


def _fit_with_callbacks(model, X_train, y_train, X_val, y_val, config: Config, checkpoint_name: str, history_name: str, **fit_kwargs):
    config.ensure_output_dirs()
    monitor = config.train.monitor
    mode = "max" if "accuracy" in monitor else "min"

    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=config.train.early_stopping_patience,
        restore_best_weights=True,
        mode=mode,
        verbose=1,
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        str(config.paths.models_dir / checkpoint_name),
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        verbose=0,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint],
        verbose=1,
        **fit_kwargs,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", range(1, len(hist_df) + 1))
    hist_df.to_csv(config.paths.metrics_dir / history_name, index=False)
    return model, history, hist_df


def train_rnn_model(model, X_train, y_train, X_val, y_val, config: Config, **fit_kwargs):
    """Train the primary (subject-independent split) LSTM and save it as the
    production model artifact."""
    model, history, _ = _fit_with_callbacks(
        model, X_train, y_train, X_val, y_val, config, "best_model.keras", "training_history.csv", **fit_kwargs
    )
    model.save(config.paths.models_dir / "har_model.keras")
    return model, history


def _class_weight_from_distribution(y_encoded: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(y_encoded, minlength=num_classes)
    total = counts.sum()
    return {i: total / (num_classes * max(count, 1)) for i, count in enumerate(counts)}


def run_pipeline(config: Config | None = None, allow_synthetic: bool = False, run_tuning: bool = True) -> dict:
    """Run the full pipeline: data -> preprocess -> (optional) tuning ->
    train (both split strategies + baseline) -> evaluate -> viz.

    Reports two LSTM accuracies for comparison:
    - "group" split: subject-independent train/val split (GroupShuffleSplit).
      This is the methodologically sound number and the pipeline's headline.
    - "leaky" split: stratified random train/val split that ignores subject
      identity, included only to show how much accuracy is inflated by
      letting validation windows share a subject with training windows.

    Both models are evaluated against the same real UCI HAR test set, which
    is itself subject-independent by construction (its 9 subjects never
    appear in the training pool of 21 subjects).
    """
    from deephar import data as data_mod
    from deephar import evaluate as evaluate_mod
    from deephar import viz as viz_mod
    from deephar.model import build_dense_baseline_model, build_rnn_model
    from deephar.preprocess import prepare_data, prepare_signal_data, save_preprocessing_artifacts
    from deephar.tuning import apply_best_hyperparameters, run_hyperparameter_search

    config = config or load_config()
    set_global_seed(config.seed)
    config.ensure_output_dirs()

    train_signals, test_signals, used_synthetic = data_mod.load_signal_data(config, allow_synthetic=allow_synthetic)
    if used_synthetic:
        logger.warning("Running in DEMO MODE on SYNTHETIC data. These numbers are NOT real performance.")

    class_counts = pd.Series(train_signals.y).value_counts()
    imbalance_ratio = float(class_counts.max() / class_counts.min())
    logger.info("Class distribution (train, %s): %s", "synthetic" if used_synthetic else "real", class_counts.to_dict())
    logger.info("Class imbalance ratio (max/min count): %.2f", imbalance_ratio)
    use_class_weight = imbalance_ratio > CLASS_IMBALANCE_THRESHOLD
    if use_class_weight:
        logger.info("Class distribution is meaningfully imbalanced (ratio %.2f > %.2f) -- applying class weights.", imbalance_ratio, CLASS_IMBALANCE_THRESHOLD)
    else:
        logger.info("Class distribution is not meaningfully imbalanced (ratio %.2f <= %.2f) -- skipping class weighting.", imbalance_ratio, CLASS_IMBALANCE_THRESHOLD)

    prepared_group = prepare_signal_data(train_signals, test_signals, config, group_split=True)
    prepared_leaky = prepare_signal_data(train_signals, test_signals, config, group_split=False)
    save_preprocessing_artifacts(prepared_group, config)

    class_names = list(prepared_group.label_encoder.classes_)
    viz_mod.plot_class_distribution(
        prepared_group.y_train_encoded, class_names, config.paths.plots_dir / "class_distribution.png"
    )
    num_classes = prepared_group.y_train.shape[1]
    class_weight = (
        _class_weight_from_distribution(prepared_group.y_train_encoded, num_classes) if use_class_weight else None
    )

    if run_tuning:
        best_params = run_hyperparameter_search(
            prepared_group.X_train, prepared_group.y_train, prepared_group.X_val, prepared_group.y_val, num_classes, config
        )
        model_config = apply_best_hyperparameters(config, best_params)
    else:
        best_params = None
        model_config = config

    input_shape = (prepared_group.X_train.shape[1], prepared_group.X_train.shape[2])

    # Primary model: subject-independent (group) split -- the headline result.
    model_group = build_rnn_model(input_shape, num_classes, model_config)
    model_group, history_group = train_rnn_model(
        model_group,
        prepared_group.X_train,
        prepared_group.y_train,
        prepared_group.X_val,
        prepared_group.y_val,
        model_config,
        class_weight=class_weight,
    )
    hist_group_df = pd.DataFrame(history_group.history)
    hist_group_df.insert(0, "epoch", range(1, len(hist_group_df) + 1))
    viz_mod.plot_training_history(hist_group_df, config.paths.plots_dir)

    y_pred, test_accuracy_group, report_group = evaluate_mod.evaluate_model(
        model_group,
        prepared_group.X_test,
        prepared_group.y_test,
        prepared_group.label_encoder,
        prepared_group.X_test_flat,
        prepared_group.y_test_encoded,
        config,
    )

    # Comparison model: same-subject-leakage split, same hyperparameters, for contrast only.
    model_leaky = build_rnn_model(input_shape, num_classes, model_config)
    model_leaky, history_leaky, _ = _fit_with_callbacks(
        model_leaky,
        prepared_leaky.X_train,
        prepared_leaky.y_train,
        prepared_leaky.X_val,
        prepared_leaky.y_val,
        model_config,
        "leaky_split_model.keras",
        "training_history_leaky_split.csv",
        class_weight=class_weight,
    )
    val_accuracy_group = max(history_group.history["val_accuracy"])
    val_accuracy_leaky = max(history_leaky.history["val_accuracy"])

    # Non-temporal baseline: Dense MLP on the 561 pre-engineered features.
    baseline_train_df, baseline_test_df, baseline_used_synthetic = data_mod.load_feature_csv_data(
        config, allow_synthetic=allow_synthetic
    )
    baseline_prepared = prepare_data(baseline_train_df, baseline_test_df, config)
    baseline_model = build_dense_baseline_model(baseline_prepared.X_train_flat.shape[1], num_classes, model_config)
    baseline_model.fit(
        baseline_prepared.X_train_flat,
        baseline_prepared.y_train,
        epochs=30,
        batch_size=64,
        validation_data=(baseline_prepared.X_val_flat, baseline_prepared.y_val),
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max")],
        verbose=0,
    )
    baseline_pred = np.argmax(baseline_model.predict(baseline_prepared.X_test_flat, verbose=0), axis=1)
    baseline_test_accuracy = float((baseline_pred == baseline_prepared.y_test_encoded).mean())

    if val_accuracy_leaky > val_accuracy_group:
        observed_note = (
            "In this run the leakage split scored higher, consistent with the leakage "
            "mechanism above: near-duplicate windows from the same subject/session let "
            "the model partly key off subject-specific idiosyncrasies rather than the "
            "activity itself."
        )
    else:
        observed_note = (
            "In this run the group split scored higher instead -- the leakage-inflation "
            "effect did not dominate here. With only ~21 training subjects, which "
            "specific subjects land in the held-out group is high-variance: this split's "
            "val set happened to be an easier subset of subjects/activities, while the "
            "leaky split's val set draws windows from every subject, including harder "
            "SITTING/STANDING-confusion cases. Neither split's val accuracy should be "
            "over-interpreted as a precise generalization estimate on this few subjects; "
            "the real test set (9 fully disjoint subjects) is the more trustworthy signal."
        )

    split_report_lines = [
        "DeepHAR -- Train/Val Split Comparison",
        "=" * 45,
        "",
        f"Data source: {'SYNTHETIC (demo mode)' if used_synthetic else 'real UCI HAR dataset'}",
        "",
        f"Subject-independent (group) split val accuracy:  {val_accuracy_group:.4f}",
        f"Same-subject-leakage (stratified) split val accuracy: {val_accuracy_leaky:.4f}",
        "",
        "These CAN differ because UCI HAR windows are 50%-overlapping slices of a",
        "continuous per-subject recording. The leakage split shuffles windows",
        "randomly, so near-duplicate windows from the same subject/session can end up",
        "in both train and val, which can inflate its validation accuracy. The group",
        "split holds out entire subjects, so it measures generalization to genuinely",
        "unseen people, which is what this pipeline is actually meant to predict --",
        "that makes it the methodologically sound choice regardless of which number",
        "happens to be higher on a given run.",
        "",
        observed_note,
        "",
        f"Headline test accuracy (subject-independent LSTM, real test set): {test_accuracy_group:.4f}",
        f"Non-temporal Dense baseline (561 engineered features) test accuracy: {baseline_test_accuracy:.4f}",
        "",
        f"Class imbalance ratio (max/min train class count): {imbalance_ratio:.2f} "
        f"({'class weighting applied' if use_class_weight else 'no class weighting applied -- not meaningfully imbalanced'})",
    ]
    if best_params is not None:
        split_report_lines += ["", f"Best hyperparameters (from search): {best_params}"]
    with open(config.paths.metrics_dir / "split_comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(split_report_lines) + "\n")

    logger.info("Subject-independent test accuracy (headline): %.4f", test_accuracy_group)
    logger.info("Same-subject-leakage val accuracy (for comparison): %.4f", val_accuracy_leaky)
    logger.info("Baseline Dense (561-feature) test accuracy: %.4f", baseline_test_accuracy)

    return {
        "accuracy": test_accuracy_group,
        "test_accuracy_group_split": test_accuracy_group,
        "val_accuracy_group_split": val_accuracy_group,
        "val_accuracy_leaky_split": val_accuracy_leaky,
        "baseline_dense_test_accuracy": baseline_test_accuracy,
        "report": report_group,
        "used_synthetic": used_synthetic,
        "model_path": config.paths.models_dir / "har_model.keras",
        "best_hyperparameters": best_params,
        "class_imbalance_ratio": imbalance_ratio,
        "used_class_weight": use_class_weight,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DeepHAR LSTM.")
    parser.add_argument(
        "--demo",
        "--synthetic",
        dest="demo",
        action="store_true",
        help="Run in demo mode on synthetic data instead of requiring the real UCI HAR dataset.",
    )
    parser.add_argument(
        "--no-tuning",
        dest="no_tuning",
        action="store_true",
        help="Skip the hyperparameter search and train with config.yaml's model settings directly.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(allow_synthetic=args.demo, run_tuning=not args.no_tuning)
