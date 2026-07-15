"""Helper for fetching the real UCI HAR dataset.

The dataset is not bundled with this repo (it's too large for GitHub and the
placeholder CSVs here are 404 stubs). This script downloads and reshapes it
into the `data/train.csv` / `data/test.csv` layout the pipeline expects
(feature columns + `subject` + `Activity`), matching Kaggle's popular
"Human Activity Recognition with Smartphones" mirror of the dataset.

Manual alternative (no internet access required by this script):
  1. Download the dataset zip from either source:
       - UCI ML Repository: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
       - Kaggle mirror: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
  2. Unzip it and locate `train.csv` and `test.csv` (each already has the
     561 engineered features + `subject` + `Activity` columns).
  3. Copy them into this project's `data/` directory as `data/train.csv`
     and `data/test.csv`.

Usage (automatic download via curl/kagglehub, requires network + credentials
for Kaggle):
    python scripts/download_data.py --dest data/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INSTRUCTIONS = __doc__


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch/prepare the real UCI HAR dataset.")
    parser.add_argument(
        "--dest", default=str(PROJECT_ROOT / "data"), help="Destination directory for train.csv/test.csv"
    )
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError:
        print(INSTRUCTIONS)
        print(
            "\n`kagglehub` is not installed, so automatic download is unavailable in this "
            "environment. Follow the manual steps above, or run:\n"
            "    pip install kagglehub\n"
            "    python scripts/download_data.py --dest data/\n"
        )
        return 1

    print("Downloading via kagglehub (requires Kaggle credentials)...")
    path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")
    src_dir = Path(path)
    for name in ("train.csv", "test.csv"):
        src_file = next(src_dir.rglob(name), None)
        if src_file is None:
            print(f"Could not locate {name} in downloaded dataset at {src_dir}")
            return 1
        (dest / name).write_bytes(src_file.read_bytes())
        print(f"Wrote {dest / name}")

    print("Done. The pipeline will now use the real dataset automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
