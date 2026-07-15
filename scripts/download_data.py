"""Fetch and prepare the real UCI HAR dataset.

The dataset is not bundled with this repo (it's too large for GitHub). This
script downloads the official archive directly from the UCI Machine Learning
Repository (no account/credentials required), then lays it out as:

    data/UCI_HAR_Dataset/train/Inertial Signals/*.txt   (128 x 9 raw windows)
    data/UCI_HAR_Dataset/train/subject_train.txt
    data/UCI_HAR_Dataset/train/y_train.txt
    data/UCI_HAR_Dataset/test/...                        (mirror of train/)
    data/UCI_HAR_Dataset/activity_labels.txt
    data/train.csv, data/test.csv                        (561-feature baseline,
                                                            rebuilt from X_*.txt
                                                            + subject + Activity)

`src/deephar/data.py` reads the raw `Inertial Signals/*.txt` windows as the
primary LSTM input (real time axis: 128 timesteps x 9 channels) and the
`train.csv`/`test.csv` files as a separate, pre-engineered-feature baseline
for comparison. Neither format is a substitute for the other -- keep both.

Manual alternative (no internet access required by this script):
  1. Download the dataset zip from either source:
       - UCI ML Repository: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
       - Kaggle mirror: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
  2. Unzip it (it contains a nested `UCI HAR Dataset.zip`; unzip that too).
  3. Rename/move the extracted `UCI HAR Dataset/` folder to `data/UCI_HAR_Dataset/`.
  4. Run `python scripts/download_data.py --skip-download` to build the
     baseline `train.csv`/`test.csv` from the extracted files.

Usage:
    python scripts/download_data.py --dest data/
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

UCI_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

SIGNAL_CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

ACTIVITY_ID_TO_NAME = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def download_and_extract(dest: Path) -> Path:
    """Download the UCI archive and extract it to dest/UCI_HAR_Dataset."""
    raw_dir = dest / "_download_tmp"
    raw_dir.mkdir(parents=True, exist_ok=True)
    outer_zip = raw_dir / "uci_har.zip"

    print(f"Downloading {UCI_ARCHIVE_URL} ...")
    urllib.request.urlretrieve(UCI_ARCHIVE_URL, outer_zip)

    with zipfile.ZipFile(outer_zip) as zf:
        zf.extractall(raw_dir)

    inner_zip = raw_dir / "UCI HAR Dataset.zip"
    with zipfile.ZipFile(inner_zip) as zf:
        zf.extractall(raw_dir)

    extracted = raw_dir / "UCI HAR Dataset"
    target = dest / "UCI_HAR_Dataset"
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(extracted), str(target))
    shutil.rmtree(raw_dir)
    return target


def _load_feature_names(uci_dir: Path) -> list[str]:
    features = pd.read_csv(uci_dir / "features.txt", sep=r"\s+", header=None, names=["idx", "name"])
    # The 561-feature list has duplicate names (e.g. multiple "fBodyAcc-bandsEnergy()-1,8");
    # de-duplicate by suffixing the column index so pandas doesn't collapse them.
    names = features["name"].tolist()
    seen: dict[str, int] = {}
    deduped = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        deduped.append(name if seen[name] == 1 else f"{name}__{seen[name]}")
    return deduped


def build_feature_csv(uci_dir: Path, split: str, feature_names: list[str]) -> pd.DataFrame:
    X = pd.read_csv(uci_dir / split / f"X_{split}.txt", sep=r"\s+", header=None, names=feature_names)
    subjects = pd.read_csv(uci_dir / split / f"subject_{split}.txt", header=None, names=["subject"])
    y = pd.read_csv(uci_dir / split / f"y_{split}.txt", header=None, names=["activity_id"])

    df = X.copy()
    df["subject"] = subjects["subject"].values
    df["Activity"] = y["activity_id"].map(ACTIVITY_ID_TO_NAME).values
    return df


def build_baseline_csvs(uci_dir: Path, dest: Path) -> None:
    feature_names = _load_feature_names(uci_dir)
    for split, out_name in (("train", "train.csv"), ("test", "test.csv")):
        df = build_feature_csv(uci_dir, split, feature_names)
        out_path = dest / out_name
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
        # X_{split}.txt duplicates everything now in the CSV; drop it to save space,
        # but keep Inertial Signals/, subject_*.txt, y_*.txt for the raw LSTM input.
        (uci_dir / split / f"X_{split}.txt").unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch/prepare the real UCI HAR dataset.")
    parser.add_argument(
        "--dest", default=str(PROJECT_ROOT / "data"), help="Destination directory for the dataset"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step; assume data/UCI_HAR_Dataset already exists "
        "(e.g. placed there manually) and just (re)build the baseline CSVs from it.",
    )
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    uci_dir = dest / "UCI_HAR_Dataset"
    if args.skip_download:
        if not uci_dir.exists():
            print(f"--skip-download given but {uci_dir} does not exist.")
            return 1
    else:
        try:
            uci_dir = download_and_extract(dest)
        except Exception as e:
            print(f"Automatic download failed: {e}")
            print(__doc__)
            return 1

    build_baseline_csvs(uci_dir, dest)
    print("Done. The pipeline will now use the real dataset automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
