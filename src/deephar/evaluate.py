"""Model evaluation: metrics computation + persisted artifacts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from deephar import viz
from deephar.config import Config


def evaluate_model(model, X_test, y_test_onehot, label_encoder, X_test_flat, y_test_encoded, config: Config):
    config.ensure_output_dirs()
    class_names = list(label_encoder.classes_)

    y_true = np.argmax(y_test_onehot, axis=1)
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # zero_division=0 avoids the NaN/RuntimeWarning the original hand-rolled
    # tp/(tp+fp)-style math produced whenever a class had zero predictions.
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    accuracy = report["accuracy"]

    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
            "Value": [
                accuracy,
                report["macro avg"]["precision"],
                report["macro avg"]["recall"],
                report["macro avg"]["f1-score"],
            ],
        }
    )
    metrics_df.to_csv(config.paths.metrics_dir / "classification_metrics.csv", index=False)

    with open(config.paths.metrics_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    predictions_df = pd.DataFrame(
        {
            "True_Activity": label_encoder.inverse_transform(y_true),
            "Predicted_Activity": label_encoder.inverse_transform(y_pred),
            "Correct": y_true == y_pred,
        }
    )
    predictions_df.to_csv(config.paths.metrics_dir / "predictions.csv", index=False)

    viz.plot_confusion_matrices(y_true, y_pred, class_names, config.paths.plots_dir)
    viz.plot_metrics_by_class(report, class_names, config.paths.plots_dir / "performance_metrics_by_class.png")
    viz.plot_roc_curves(y_test_onehot, y_pred_prob, class_names, config.paths.plots_dir / "roc_curves.png")
    viz.plot_pca(X_test_flat, y_test_encoded, class_names, config.paths.plots_dir / "pca_visualization.png")

    with open(config.paths.metrics_dir / "har_report.txt", "w", encoding="utf-8") as f:
        f.write("Human Activity Recognition - Model Results\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2%}\n\n")
        f.write("Performance by Activity:\n")
        for name in class_names:
            f.write(
                f"{name}: precision={report[name]['precision']:.4f}, "
                f"recall={report[name]['recall']:.4f}, "
                f"f1={report[name]['f1-score']:.4f}\n"
            )

    return y_pred, accuracy, report
