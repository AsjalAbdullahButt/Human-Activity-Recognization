# DeepHAR – Human Activity Recognition with LSTM

DeepHAR classifies human activities (walking, sitting, standing, laying, etc.)
from the UCI HAR smartphone sensor dataset using an LSTM/RNN, with a Streamlit
app for exploring the data and model.

This repo runs out of the box on **synthetic data** — no dataset download
required. Point it at the real UCI HAR CSVs (see below) when you want real
results.

---

## Project layout

```text
src/deephar/        Core package: config, data loading, preprocessing, model,
                     training, evaluation, visualization
scripts/
  download_data.py  Helper + instructions for fetching the real UCI HAR dataset
app.py               Streamlit app (multi-tab: overview, data, training,
                      performance, live prediction)
tests/                pytest suite
legacy/HAR.py         Original monolithic script, kept as-is for reference
config.yaml           Central configuration (paths, hyperparameters, seed)
data/                 Put train.csv / test.csv here (gitignored)
outputs/              Models, plots, metrics produced by a training run (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Keras 3 is used with the `torch` backend by default (set in
`src/deephar/__init__.py`), because TensorFlow does not yet ship wheels for
every Python version. If your Python version has a TensorFlow wheel, you can
instead `pip install tensorflow` and set `KERAS_BACKEND=tensorflow`.

## Getting the real dataset (optional)

The `train.csv`/`test.csv` originally shipped in this repo were 404
placeholder stubs — the real dataset is too large for GitHub. To use it:

1. Download from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
   or the [Kaggle mirror](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones).
2. Copy `train.csv` and `test.csv` into `data/`.
3. Or run `python scripts/download_data.py` (requires `kagglehub` + Kaggle
   credentials) for an automated fetch.

Without the real data, `deephar.data.load_data()` automatically falls back to
a synthetic dataset shaped like UCI HAR (six activity classes, per-class
Gaussian clusters, `subject`/`Activity` columns) so every part of the
pipeline and app still runs.

## Training

```bash
python -m deephar.train
```

This loads data (real if present in `data/`, else synthetic), does a
stratified train/val split, trains the LSTM, and writes to `outputs/`:

- `outputs/models/har_model.keras`, `best_model.keras` — trained model
- `outputs/models/scaler.joblib`, `label_encoder.joblib` — fitted preprocessing,
  needed to run inference on new raw rows
- `outputs/plots/*.png` — class distribution, confusion matrices, ROC curves,
  PCA, training curves
- `outputs/metrics/*` — classification report, per-class metrics, predictions

Hyperparameters, paths, and the random seed all live in `config.yaml`.

## Streamlit app

```bash
streamlit run app.py
```

Tabs:

- **Overview** — project summary, headline accuracy/F1
- **Data Explorer** — class distribution, sample rows, feature stats, PCA scatter
- **Training** — trigger a training run, view learning curves
- **Performance** — confusion matrix, per-class precision/recall/F1, ROC curves
- **Live Prediction** — load the persisted model/scaler/encoder and predict on
  an uploaded or sample row

If `outputs/` doesn't have artifacts yet, the app falls back to synthetic/demo
data so every tab still renders — use the Training tab to produce real ones.

## Tests

```bash
pytest -q
```

## Known limitations / suggested next steps

- **The LSTM sees only 1 timestep.** The pre-extracted 561-feature UCI HAR
  rows have no time axis, so the LSTM layers currently behave like an
  expensive Dense layer. For a temporally meaningful model, retrain on the
  raw `Inertial Signals/*.txt` windows (128 timesteps × 9 channels) instead
  of the engineered feature CSVs.
- Class weighting / focal loss if the real dataset's class balance turns out
  to be skewed.
- Hyperparameter search (units, dropout, learning rate) via Keras Tuner or
  Optuna.
- Group (subject-wise) cross-validation in addition to the stratified split,
  since HAR generalization across unseen subjects is the harder and more
  realistic evaluation.
