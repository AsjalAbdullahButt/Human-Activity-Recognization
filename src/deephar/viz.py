"""Matplotlib/seaborn visualizations, saved as PNGs into config.paths.plots_dir."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.metrics import auc, confusion_matrix, roc_curve

plt.style.use("seaborn-v0_8-whitegrid")


def plot_class_distribution(y_encoded: np.ndarray, class_names: list[str], out_path) -> None:
    counts = pd.Series(y_encoded).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    # hue=class_names avoids the seaborn deprecation warning that fires when
    # `palette` is passed without `hue` on a plain (non-grouped) barplot.
    sns.barplot(
        x=counts.values,
        y=list(class_names),
        hue=list(class_names),
        palette="Blues_d",
        legend=False,
        orient="h",
        ax=ax,
    )
    for i, v in enumerate(counts.values):
        ax.text(v + max(counts.values) * 0.01, i, str(v), va="center", fontweight="bold")
    ax.set_title("Class Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Number of Samples", fontsize=13, fontweight="bold")
    ax.set_ylabel("Activity Class", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_dir) -> None:
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_percent, annot=cm, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=100, ax=ax,
    )
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title("Confusion Matrix (counts, color = row %)", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_percent, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title("Normalized Confusion Matrix (%)", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "normalized_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_by_class(report: dict, class_names: list[str], out_path) -> None:
    metrics = {
        "Precision": [report[c]["precision"] for c in class_names],
        "Recall": [report[c]["recall"] for c in class_names],
        "F1-Score": [report[c]["f1-score"] for c in class_names],
    }
    fig, ax = plt.subplots(figsize=(14, 7))
    width = 0.25
    x = np.arange(len(class_names))
    for i, (name, values) in enumerate(metrics.items()):
        ax.bar(x + i * width, values, width=width, label=name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision / Recall / F1 by Activity", fontweight="bold")
    ax.legend(loc="upper center", ncol=3)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_test_onehot: np.ndarray, y_pred_prob: np.ndarray, class_names: list[str], out_path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(class_names)))
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curves (one-vs-rest)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pca(X_flat: np.ndarray, y_encoded: np.ndarray, class_names: list[str], out_path) -> None:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(class_names):
        mask = y_encoded == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, alpha=0.7, s=40, edgecolors="w", linewidth=0.4)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontweight="bold")
    ax.set_title("PCA of Feature Space by Activity", fontweight="bold")
    ax.legend(title="Activity", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history_df: pd.DataFrame, out_dir) -> None:
    epochs = history_df["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(epochs, history_df["accuracy"], "o-", label="Train", color="#3498db")
    axes[0].plot(epochs, history_df["val_accuracy"], "o-", label="Validation", color="#e74c3c")
    axes[0].set_title("Accuracy over Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, history_df["loss"], "o-", label="Train", color="#3498db")
    axes[1].plot(epochs, history_df["val_loss"], "o-", label="Validation", color="#e74c3c")
    axes[1].set_title("Loss over Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_dir / "training_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
