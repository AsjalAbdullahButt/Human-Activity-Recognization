"""DeepHAR Streamlit app.

Run with: streamlit run app.py

Loads precomputed artifacts from outputs/ (produced by `python -m deephar.train`
or the Training tab) when present. Otherwise every tab falls back to a small,
in-memory synthetic run so the whole app still renders.
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
from deephar.data import ACTIVITY_LABELS, load_data
from deephar.model import build_rnn_model
from deephar.preprocess import prepare_data
from deephar.train import train_rnn_model

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
    cfg.data.synthetic_n_features = 80
    cfg.train.epochs = 6
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

        run_pipeline(cfg)
    return cfg


def active_config():
    """Prefer real trained artifacts in outputs/; else use the cached demo run."""
    if artifacts_present(CONFIG):
        return CONFIG, False
    return ensure_demo_artifacts(), True


@st.cache_data(show_spinner=False)
def load_raw_data(_cfg):
    train_df, test_df, used_synthetic = load_data(_cfg)
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
    return report, predictions_df, history_df


st.title("DeepHAR — Human Activity Recognition")
st.caption("LSTM-based classifier for smartphone accelerometer/gyroscope activity data")

tabs = st.tabs(["Overview", "Data Explorer", "Training", "Performance", "Live Prediction"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    active_cfg, is_demo = active_config()
    st.subheader("Project Summary")
    st.write(
        "DeepHAR trains a 2-layer LSTM classifier on the UCI HAR feature dataset "
        "(561 engineered features per sample) to recognize six activities: "
        + ", ".join(ACTIVITY_LABELS)
        + "."
    )
    if is_demo:
        st.info(
            "No trained model found in `outputs/`. Showing metrics from an "
            "auto-generated demo run on synthetic data. Use the Training tab "
            "to train on real data (or a bigger synthetic run)."
        )

    if artifacts_present(active_cfg):
        report, predictions_df, history_df = load_metrics(active_cfg)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{report['accuracy']:.2%}")
        col2.metric("Macro F1", f"{report['macro avg']['f1-score']:.3f}")
        col3.metric("Macro Precision", f"{report['macro avg']['precision']:.3f}")
        col4.metric("Macro Recall", f"{report['macro avg']['recall']:.3f}")
        st.write(f"Trained for {len(history_df)} epochs.")

    with st.expander("Known limitations"):
        st.markdown(
            "- The model sees each sample as a **single timestep** (pre-extracted "
            "feature vectors have no time axis), so the LSTM behaves like a dense "
            "layer. A temporally meaningful model would use the raw 128x9 "
            "inertial-signal windows instead.\n"
            "- No class weighting or hyperparameter search yet."
        )

# ---------------------------------------------------------------------------
# Data Explorer
# ---------------------------------------------------------------------------
with tabs[1]:
    active_cfg, is_demo = active_config()
    train_df, test_df, used_synthetic = load_raw_data(active_cfg)
    if used_synthetic:
        st.info("Showing synthetic demo data (no real dataset found in `data/`).")

    st.subheader("Class Distribution")
    counts = train_df["Activity"].value_counts().reindex(ACTIVITY_LABELS).fillna(0)
    fig = px.bar(
        x=counts.values, y=counts.index, orientation="h",
        labels={"x": "Samples", "y": "Activity"}, color=counts.index,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample Rows")
    st.dataframe(train_df.head(20), use_container_width=True)

    st.subheader("Feature Statistics")
    feature_cols = [c for c in train_df.columns if c not in ("subject", "Activity")]
    st.dataframe(train_df[feature_cols].describe().T, use_container_width=True)

    st.subheader("PCA Scatter (2D projection of features)")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(train_df[feature_cols])
    coords = PCA(n_components=2).fit_transform(X)
    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    pca_df["Activity"] = train_df["Activity"].values
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Activity", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Train a Model")
    st.write(
        "Trains on `data/train.csv`/`data/test.csv` if present, otherwise on "
        "synthetic data. Writes results to `outputs/` (overwriting any "
        "previous real run)."
    )
    epochs = st.slider("Epochs", min_value=1, max_value=100, value=15)
    run_clicked = st.button("Run Training", type="primary")

    if run_clicked:
        from deephar.train import run_pipeline

        run_cfg = load_config()
        run_cfg.train.epochs = epochs
        with st.spinner("Training..."):
            result = run_pipeline(run_cfg)
        st.success(f"Done. Test accuracy: {result['accuracy']:.2%}")
        st.cache_data.clear()
        st.cache_resource.clear()

    active_cfg, is_demo = active_config()
    if (active_cfg.paths.metrics_dir / "training_history.csv").exists():
        history_df = pd.read_csv(active_cfg.paths.metrics_dir / "training_history.csv")
        st.subheader("Learning Curves" + (" (demo run)" if is_demo else ""))
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
    report, predictions_df, history_df = load_metrics(active_cfg)
    if is_demo:
        st.info("Showing metrics from the auto-generated demo run.")

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
    if artifacts_present(active_cfg) and (active_cfg.paths.models_dir / "har_model.keras").exists():
        try:
            model, scaler, label_encoder = load_model_bundle(active_cfg)
            _, test_df, _ = load_raw_data(active_cfg)
            from deephar.preprocess import split_features_labels
            from sklearn.metrics import auc, roc_curve

            X_test_raw, y_test_raw = split_features_labels(test_df)
            X_test_scaled = scaler.transform(X_test_raw)
            X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
            y_test_encoded = label_encoder.transform(y_test_raw)
            y_pred_prob = model.predict(X_test_reshaped, verbose=0)

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
    st.subheader("Predict on a Row")
    if is_demo:
        st.info("Using the auto-generated demo model (no real trained model found in `outputs/`).")

    try:
        model, scaler, label_encoder = load_model_bundle(active_cfg)
    except FileNotFoundError:
        st.error("No model/scaler/encoder artifacts found. Train a model first.")
        st.stop()

    _, test_df, _ = load_raw_data(active_cfg)
    from deephar.preprocess import split_features_labels

    X_test_raw, y_test_raw = split_features_labels(test_df)

    source = st.radio("Row source", ["Sample from test set", "Upload CSV"], horizontal=True)

    row = None
    true_label = None
    if source == "Sample from test set":
        idx = st.number_input("Row index", min_value=0, max_value=len(X_test_raw) - 1, value=0)
        row = X_test_raw.iloc[[idx]]
        true_label = y_test_raw.iloc[idx]
    else:
        uploaded = st.file_uploader("CSV with the same feature columns as training data", type="csv")
        if uploaded is not None:
            upload_df = pd.read_csv(uploaded)
            missing = set(X_test_raw.columns) - set(upload_df.columns)
            if missing:
                st.error(f"Uploaded CSV is missing {len(missing)} expected feature column(s).")
            else:
                row = upload_df[X_test_raw.columns].iloc[[0]]

    if row is not None:
        X_scaled = scaler.transform(row)
        X_reshaped = X_scaled.reshape(1, 1, X_scaled.shape[1])
        probs = model.predict(X_reshaped, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = label_encoder.classes_[pred_idx]

        st.metric("Predicted Activity", pred_label)
        if true_label is not None:
            st.write(f"True label: **{true_label}** {'✅' if true_label == pred_label else '❌'}")

        prob_df = pd.DataFrame({"Activity": label_encoder.classes_, "Probability": probs}).sort_values(
            "Probability", ascending=False
        )
        fig = px.bar(prob_df, x="Probability", y="Activity", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
