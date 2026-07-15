"""Model training, plus the end-to-end pipeline entry point."""
from __future__ import annotations

import logging

import keras
import pandas as pd

from deephar.config import Config, load_config, set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train_rnn_model(model, X_train, y_train, X_val, y_val, config: Config):
    """Train with a single, consistent "best model" criterion.

    The original script monitored val_loss for early-stopping restoration but
    val_accuracy for the checkpoint file, so the two saved artifacts could
    disagree on which epoch was "best". Both callbacks now monitor the same
    metric (config.train.monitor).
    """
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
        str(config.paths.models_dir / "best_model.keras"),
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
    )

    model.save(config.paths.models_dir / "har_model.keras")

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", range(1, len(hist_df) + 1))
    hist_df.to_csv(config.paths.metrics_dir / "training_history.csv", index=False)

    return model, history


def run_pipeline(config: Config | None = None) -> dict:
    """Run the full data -> preprocess -> train -> evaluate -> viz pipeline."""
    from deephar import data as data_mod
    from deephar import evaluate as evaluate_mod
    from deephar import viz as viz_mod
    from deephar.model import build_rnn_model
    from deephar.preprocess import prepare_data, save_preprocessing_artifacts

    config = config or load_config()
    set_global_seed(config.seed)
    config.ensure_output_dirs()

    train_df, test_df, used_synthetic = data_mod.load_data(config)
    if used_synthetic:
        logger.warning("Running on SYNTHETIC data (real UCI HAR CSVs not found).")

    prepared = prepare_data(train_df, test_df, config)
    save_preprocessing_artifacts(prepared, config)

    class_names = list(prepared.label_encoder.classes_)
    viz_mod.plot_class_distribution(
        prepared.y_train_encoded, class_names, config.paths.plots_dir / "class_distribution.png"
    )

    input_shape = (prepared.X_train.shape[1], prepared.X_train.shape[2])
    num_classes = prepared.y_train.shape[1]
    model = build_rnn_model(input_shape, num_classes, config)

    model, history = train_rnn_model(
        model, prepared.X_train, prepared.y_train, prepared.X_val, prepared.y_val, config
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", range(1, len(hist_df) + 1))
    viz_mod.plot_training_history(hist_df, config.paths.plots_dir)

    y_pred, accuracy, report = evaluate_mod.evaluate_model(
        model,
        prepared.X_test,
        prepared.y_test,
        prepared.label_encoder,
        prepared.X_test_flat,
        prepared.y_test_encoded,
        config,
    )

    logger.info("Final test accuracy: %.4f", accuracy)
    return {
        "accuracy": accuracy,
        "report": report,
        "used_synthetic": used_synthetic,
        "model_path": config.paths.models_dir / "har_model.keras",
    }


if __name__ == "__main__":
    run_pipeline()
