# DeepHAR – Human Activity Recognition with LSTM

DeepHAR classifies human activities (walking, sitting, standing, laying, etc.)
from the UCI HAR smartphone sensor dataset. The LSTM trains directly on the
raw inertial-signal windows (128 timesteps x 9 channels per window) — a real
time axis — with a Dense MLP over the dataset's 561 pre-engineered features
trained alongside it as a non-temporal baseline. A Streamlit app is included
for exploring the data and model.

This repo requires the real UCI HAR dataset for production training (see
below); it fails loudly rather than silently substituting fake data. Pass
`--demo` to `python -m deephar.train` if you explicitly want a demo run on
synthetic data — demo-mode numbers are clearly labeled as such everywhere
they're reported and must never be read as real performance.

---

## Project layout

```text
src/deephar/        Core package: config, data loading, preprocessing, model,
                     hyperparameter tuning, training, evaluation, visualization
scripts/
  download_data.py       Fetches the real UCI HAR dataset from the UCI archive
                          and builds the baseline feature CSVs
  run_real_training.py   One-shot driver used to produce this README's real-data numbers
app.py               Streamlit app (multi-tab: overview, data, training,
                      performance, live prediction)
tests/                pytest suite
config.yaml           Central configuration (paths, hyperparameters, seed)
data/                 Real dataset lives here (gitignored, see below)
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

## Getting the real dataset

The real dataset is too large for GitHub, so it isn't bundled — fetch it with:

```bash
python scripts/download_data.py
```

This downloads the official archive directly from the UCI ML Repository (no
account needed) and lays it out as:

```text
data/UCI_HAR_Dataset/{train,test}/Inertial Signals/*.txt   128 x 9 raw windows -- primary LSTM input
data/UCI_HAR_Dataset/{train,test}/subject_*.txt, y_*.txt
data/train.csv, data/test.csv                              561-feature baseline, rebuilt from the archive
```

Manual alternative: download from the
[UCI ML Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
or the [Kaggle mirror](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones),
extract it to `data/UCI_HAR_Dataset/`, then run
`python scripts/download_data.py --skip-download` to build the baseline CSVs.

Without the real dataset, `deephar.train.run_pipeline()` raises
`RealDataNotFoundError` rather than silently generating fake numbers. Pass
`allow_synthetic=True` (or `--demo` on the CLI) to explicitly opt into a demo
run on synthetic data instead.

## Training

```bash
python -m deephar.train              # real data required; ~15-trial hyperparameter search + 3 training runs
python -m deephar.train --no-tuning  # skip the search, train with config.yaml's model settings directly
python -m deephar.train --demo       # demo mode on synthetic data (clearly labeled, not real performance)
```

Each real run:

1. Loads the raw signal windows and the 561-feature CSVs, and prints the real
   class distribution (used to decide whether class weighting is warranted —
   see [Class balance](#class-balance-and-splits) below).
2. Runs a Keras Tuner random search (LSTM units, layers, dropout, learning
   rate, batch size) on a subject-independent split, and writes every trial's
   result to `outputs/metrics/tuning_trials.csv` and the winner to
   `outputs/metrics/best_hyperparameters.json`.
3. Trains the primary LSTM on a **subject-independent (group) split** — the
   headline model — plus a second LSTM on a **same-subject-leakage
   (stratified) split** for comparison only, with identical hyperparameters.
4. Trains the Dense baseline on the 561 pre-engineered features (no time axis).
5. Evaluates the primary model against the real UCI HAR test set (itself
   subject-independent from the training pool by construction) and writes
   plots/metrics to `outputs/`.

Outputs:

- `outputs/models/har_model.keras`, `best_model.keras` — the primary
  (subject-independent) LSTM; `scaler.joblib`/`label_encoder.joblib` — fitted
  per-channel preprocessing needed for inference
- `outputs/models/leaky_split_model.keras` — the same-subject-leakage
  comparison model (not for production use)
- `outputs/plots/*.png` — class distribution, confusion matrices, ROC curves,
  PCA, training curves
- `outputs/metrics/*` — classification report, per-class metrics, predictions,
  `split_comparison_report.txt` (group vs. leaky split, explained), tuning results

Hyperparameters, paths, and the random seed all live in `config.yaml`.

## Class balance and splits

The real UCI HAR train set has 6 classes ranging from 986 to 1407 windows
(max/min ratio **1.43**) — not meaningfully imbalanced (the pipeline's
threshold is 1.5x), so no class weighting or focal loss is applied. This is
checked from the real data at the start of every run, not assumed.

Two train/val split strategies are trained and reported for the LSTM:

- **Subject-independent (group) split** — `GroupShuffleSplit` keyed on
  `subject`, so no subject's windows appear in both train and val. This is
  the headline number: it measures generalization to people the model has
  never seen, which is what a HAR system actually needs to do in practice.
- **Same-subject-leakage (stratified) split** — a plain stratified random
  split that ignores subject identity. UCI HAR windows are 50%-overlapping
  slices of a continuous per-subject recording, so a random shuffle puts
  near-duplicate, temporally-adjacent windows from the same subject/session
  into both train and val. The model can then partly key off subject-specific
  gait/sensor-placement idiosyncrasies rather than the activity itself, which
  inflates its validation accuracy relative to true subject-independent
  generalization.

Both numbers are reported below and in `outputs/metrics/split_comparison_report.txt`
precisely so the gap between them is visible, not hidden.

The official UCI HAR test set is already subject-independent from the
training pool (21 train subjects, 9 disjoint test subjects), so the
train/test evaluation is subject-independent regardless of which train/val
split strategy produced the model.

## Results (real UCI HAR dataset)

<!-- RESULTS_PLACEHOLDER -->

## Streamlit app

```bash
streamlit run app.py
```

Tabs:

- **Overview** — project summary, headline accuracy/F1, split-comparison report
- **Data Explorer** — class distribution, sample raw signal windows per
  activity, baseline (561-feature) stats/PCA
- **Training** — trigger a training run (with optional demo mode / tuning), view learning curves
- **Performance** — confusion matrix, per-class precision/recall/F1, ROC curves
- **Live Prediction** — load the persisted model/scaler/encoder and predict on
  a raw signal window from the test set

If `outputs/` doesn't have artifacts yet, the app falls back to an
explicitly-labeled synthetic demo run so every tab still renders — use the
Training tab to produce real ones.

## Tests

```bash
pytest -q
```

Covers: raw signal loading and shape (128, 9), per-channel (not
per-flattened-feature) scaling, group-split subject-disjointness vs.
stratified-split subject leakage, model construction for both the LSTM and
the Dense baseline, and that production training raises
`RealDataNotFoundError` (rather than silently falling back) when the real
dataset is missing and no `--demo` flag was passed.
