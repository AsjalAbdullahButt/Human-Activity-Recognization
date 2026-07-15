"""DeepHAR Streamlit app.

Run with: streamlit run app.py

Loads precomputed artifacts from outputs/ (produced by `python -m deephar.train`
or the Training tab) when present. Otherwise every tab falls back to a small,
explicitly-labeled demo run on synthetic data so the whole app still renders.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from deephar.config import load_config, set_global_seed
from deephar.data import ACTIVITY_LABELS, SIGNAL_CHANNELS, load_feature_csv_data, load_signal_data
from deephar.preprocess import split_features_labels

st.set_page_config(page_title="DeepHAR", page_icon="\U0001f3c3", layout="wide")

CONFIG = load_config()


@st.cache_resource(show_spinner=False)
def get_demo_config():
    """A config pointing at a separate outputs/demo dir so demo runs never
    clobber a real training run's artifacts."""
    cfg = load_config()
    cfg.paths.outputs_dir = PROJECT_ROOT / "outputs" / "demo"
    cfg.paths.models_dir = cfg.paths.outputs_dir / "models"
    cfg.paths.plots_dir = cfg.paths.outputs_dir / "plots"
    cfg.paths.metrics_dir = cfg.paths.outputs_dir / "metrics"
    cfg.data.synthetic_samples_train = 400
    cfg.data.synthetic_samples_test = 120
    cfg.train.epochs = 6
    cfg.tuning.max_trials = 2
    cfg.tuning.epochs_per_trial = 3
    return cfg


def artifacts_present(cfg) -> bool:
    return (cfg.paths.models_dir / "har_model.keras").exists() and (
        cfg.paths.metrics_dir / "classification_report.json"
    ).exists()


@st.cache_resource(show_spinner="Preparing demo model (first load only)...")
def ensure_demo_artifacts():
    cfg = get_demo_config()
    if not artifacts_present(cfg):
        from deephar.train import run_pipeline

        run_pipeline(cfg, allow_synthetic=True, run_tuning=True)
    return cfg


def active_config():
    """Prefer real trained artifacts in outputs/; else use the cached demo run."""
    if artifacts_present(CONFIG):
        return CONFIG, False
    return ensure_demo_artifacts(), True


@st.cache_data(show_spinner=False)
def load_raw_signals(_cfg, is_demo: bool):
    train, test, used_synthetic = load_signal_data(_cfg, allow_synthetic=is_demo)
    return train, test, used_synthetic


@st.cache_data(show_spinner=False)
def load_baseline_features(_cfg, is_demo: bool):
    train_df, test_df, used_synthetic = load_feature_csv_data(_cfg, allow_synthetic=is_demo)
    return train_df, test_df, used_synthetic


@st.cache_resource(show_spinner=False)
def load_model_bundle(_cfg):
    import keras

    model_path = _cfg.paths.models_dir / "har_model.keras"
    model = keras.models.load_model(model_path)
    scaler = joblib.load(_cfg.paths.models_dir / "scaler.joblib")
    label_encoder = joblib.load(_cfg.paths.models_dir / "label_encoder.joblib")
    return model, scaler, label_encoder


def load_metrics(cfg):
    with open(cfg.paths.metrics_dir / "classification_report.json") as f:
        report = json.load(f)
    predictions_df = pd.read_csv(cfg.paths.metrics_dir / "predictions.csv")
    history_df = pd.read_csv(cfg.paths.metrics_dir / "training_history.csv")
    split_report_path = cfg.paths.metrics_dir / "split_comparison_report.txt"
    split_report = split_report_path.read_text(encoding="utf-8") if split_report_path.exists() else None
    return report, predictions_df, history_df, split_report


def scale_signal_windows(X: np.ndarray, scaler) -> np.ndarray:
    n_channels = X.shape[-1]
    return scaler.transform(X.reshape(-1, n_channels)).reshape(X.shape)


st.title("DeepHAR — Human Activity Recognition")
st.caption("LSTM classifier over raw 128-timestep x 9-channel smartphone accelerometer/gyroscope windows")

tabs = st.tabs(["Overview", "Data Explorer", "Training", "Performance", "Live Prediction"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    active_cfg, is_demo = active_config()
    st.subheader("Project Summary")
    st.write(
        "DeepHAR trains an LSTM classifier directly on raw inertial-signal windows "
        f"(128 timesteps x {len(SIGNAL_CHANNELS)} channels: body/total acceleration + "
        "gyroscope, x/y/z) from the UCI HAR smartphone dataset, to recognize six "
        "activities: " + ", ".join(ACTIVITY_LABELS) + ". A non-temporal Dense MLP over "
        "the dataset's 561 pre-engineered features is trained alongside it as a baseline."
    )
    if is_demo:
        st.warning(
            "**DEMO MODE** — no trained model found in `outputs/`. Showing metrics from an "
            "auto-generated run on SYNTHETIC data; these numbers are not real performance. "
            "Use the Training tab (or `python -m deephar.train`) to train on the real dataset."
        )

    if artifacts_present(active_cfg):
        report, predictions_df, history_df, split_report = load_metrics(active_cfg)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Accuracy (subject-independent)", f"{report['accuracy']:.2%}")
        col2.metric("Macro F1", f"{report['macro avg']['f1-score']:.3f}")
        col3.metric("Macro Precision", f"{report['macro avg']['precision']:.3f}")
        col4.metric("Macro Recall", f"{report['macro avg']['recall']:.3f}")
        st.caption(
            "Accuracy above uses a subject-independent (group) train/val split and the "
            "real UCI HAR test set, whose 9 subjects never appear in training."
        )
        if split_report:
            with st.expander("Split-strategy comparison (subject-independent vs. same-subject-leakage)"):
                st.text(split_report)
    else:
        st.info("No trained model found yet. Use the Training tab to train one.")

# ---------------------------------------------------------------------------
# Data Explorer
# ---------------------------------------------------------------------------
with tabs[1]:
    active_cfg, is_demo = active_config()
    train_signals, test_signals, used_synthetic = load_raw_signals(active_cfg, is_demo)
    if used_synthetic:
        st.warning("**DEMO MODE** — showing synthetic data (no real dataset found in `data/`).")

    st.subheader("Class Distribution (raw signal windows)")
    counts = pd.Series(train_signals.y).value_counts().reindex(ACTIVITY_LABELS).fillna(0)
    fig = px.bar(
        x=counts.values, y=counts.index, orientation="h",
        labels={"x": "Windows", "y": "Activity"}, color=counts.index,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample Raw Signal Window per Activity")
    channel = st.selectbox("Channel", SIGNAL_CHANNELS, index=SIGNAL_CHANNELS.index("total_acc_x"))
    channel_idx = SIGNAL_CHANNELS.index(channel)
    fig = go.Figure()
    for label in ACTIVITY_LABELS:
        idx = np.flatnonzero(train_signals.y == label)
        if len(idx) == 0:
            continue
        window = train_signals.X[idx[0], :, channel_idx]
        fig.add_scatter(y=window, name=label, mode="lines")
    fig.update_layout(xaxis_title="Timestep (128 per 2.56s window)", yaxis_title=channel, title=f"One example window per activity — {channel}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Baseline (561-feature) View")
    st.caption("Pre-engineered feature CSVs, used only for the non-temporal Dense baseline.")
    baseline_train_df, _, baseline_used_synthetic = load_baseline_features(active_cfg, is_demo)
    feature_cols = [c for c in baseline_train_df.columns if c not in ("subject", "Activity")]
    st.dataframe(baseline_train_df[feature_cols].describe().T.head(20), use_container_width=True)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(baseline_train_df[feature_cols])
    coords = PCA(n_components=2).fit_transform(X)
    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    pca_df["Activity"] = baseline_train_df["Activity"].values
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Activity", opacity=0.7, title="PCA of engineered features by activity")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Train a Model")
    st.write(
        "Trains on the real dataset in `data/` if present, otherwise fails loudly unless "
        "demo mode is selected below. Writes results to `outputs/` (overwriting any "
        "previous real run). This trains the primary subject-independent-split LSTM, a "
        "same-subject-leakage-split LSTM for comparison, and the Dense baseline."
    )
    epochs = st.slider("Epochs", min_value=1, max_value=100, value=15)
    demo_mode = st.checkbox("Demo mode (synthetic data)", value=not (CONFIG.data.raw_dir.exists()))
    run_tuning = st.checkbox("Run hyperparameter search first (slow)", value=False)
    run_clicked = st.button("Run Training", type="primary")

    if run_clicked:
        from deephar.train import run_pipeline

        run_cfg = load_config()
        run_cfg.train.epochs = epochs
        with st.spinner("Training... this can take a while, especially with tuning enabled."):
            try:
                result = run_pipeline(run_cfg, allow_synthetic=demo_mode, run_tuning=run_tuning)
                st.success(
                    f"Done. Subject-independent test accuracy: {result['accuracy']:.2%} "
                    f"(baseline Dense: {result['baseline_dense_test_accuracy']:.2%})"
                )
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Training failed: {e}")

    active_cfg, is_demo = active_config()
    if (active_cfg.paths.metrics_dir / "training_history.csv").exists():
        history_df = pd.read_csv(active_cfg.paths.metrics_dir / "training_history.csv")
        st.subheader("Learning Curves (subject-independent split)" + (" (demo run)" if is_demo else ""))
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_scatter(x=history_df["epoch"], y=history_df["accuracy"], name="Train")
            fig.add_scatter(x=history_df["epoch"], y=history_df["val_accuracy"], name="Validation")
            fig.update_layout(title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=history_df["epoch"], y=history_df["loss"], name="Train")
            fig.add_scatter(x=history_df["epoch"], y=history_df["val_loss"], name="Validation")
            fig.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No training history yet — click Run Training above.")

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
with tabs[3]:
    active_cfg, is_demo = active_config()
    report, predictions_df, history_df, split_report = load_metrics(active_cfg)
    if is_demo:
        st.warning("**DEMO MODE** — showing metrics from the auto-generated synthetic-data run.")

    class_names = [c for c in report.keys() if c not in ("accuracy", "macro avg", "weighted avg")]

    st.subheader("Confusion Matrix")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(predictions_df["True_Activity"], predictions_df["Predicted_Activity"], labels=class_names)
    normalize = st.checkbox("Normalize (row %)", value=False)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
        text = np.round(cm_display, 1)
    else:
        cm_display = cm
        text = cm

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_display, x=class_names, y=class_names, colorscale="Blues",
            text=text, texttemplate="%{text}",
        )
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="True")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-Class Precision / Recall / F1")
    metrics_df = pd.DataFrame(
        {
            "Class": class_names,
            "Precision": [report[c]["precision"] for c in class_names],
            "Recall": [report[c]["recall"] for c in class_names],
            "F1": [report[c]["f1-score"] for c in class_names],
        }
    ).melt(id_vars="Class", var_name="Metric", value_name="Score")
    fig = px.bar(metrics_df, x="Class", y="Score", color="Metric", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ROC Curves")
    if artifacts_present(active_cfg):
        try:
            model, scaler, label_encoder = load_model_bundle(active_cfg)
            _, test_signals, _ = load_raw_signals(active_cfg, is_demo)
            from sklearn.metrics import auc, roc_curve

            X_test_scaled = scale_signal_windows(test_signals.X, scaler)
            y_test_encoded = label_encoder.transform(test_signals.y)
            y_pred_prob = model.predict(X_test_scaled, verbose=0)

            fig = go.Figure()
            for i, name in enumerate(label_encoder.classes_):
                y_true_binary = (y_test_encoded == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_prob[:, i])
                fig.add_scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc(fpr, tpr):.2f})")
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False)
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute ROC curves: {e}")

# ---------------------------------------------------------------------------
# Live Prediction
# ---------------------------------------------------------------------------
with tabs[4]:
    active_cfg, is_demo = active_config()
    st.subheader("Predict on a Sample Window")
    if is_demo:
        st.warning("**DEMO MODE** — using the auto-generated demo model (no real trained model found in `outputs/`).")

    try:
        model, scaler, label_encoder = load_model_bundle(active_cfg)
    except FileNotFoundError:
        st.error("No model/scaler/encoder artifacts found. Train a model first.")
        st.stop()

    _, test_signals, _ = load_raw_signals(active_cfg, is_demo)

    st.caption(
        "Live prediction operates on full raw signal windows (128 timesteps x "
        f"{len(SIGNAL_CHANNELS)} channels), not single feature rows, since that's what the "
        "LSTM actually consumes -- pick a sample window from the test set below."
    )
    idx = st.number_input("Test window index", min_value=0, max_value=len(test_signals.X) - 1, value=0)
    window = test_signals.X[idx]
    true_label = test_signals.y[idx]

    X_scaled = scale_signal_windows(window[None, :, :], scaler)
    probs = model.predict(X_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = label_encoder.classes_[pred_idx]

    st.metric("Predicted Activity", pred_label)
    st.write(f"True label: **{true_label}** {'✅' if true_label == pred_label else '❌'}")

    prob_df = pd.DataFrame({"Activity": label_encoder.classes_, "Probability": probs}).sort_values(
        "Probability", ascending=False
    )
    fig = px.bar(prob_df, x="Probability", y="Activity", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    channel = st.selectbox("Show channel", SIGNAL_CHANNELS, index=SIGNAL_CHANNELS.index("total_acc_x"), key="live_pred_channel")
    fig = go.Figure()
    fig.add_scatter(y=window[:, SIGNAL_CHANNELS.index(channel)], mode="lines")
    fig.update_layout(title=f"Selected window — {channel}", xaxis_title="Timestep", yaxis_title=channel)
    st.plotly_chart(fig, use_container_width=True)
