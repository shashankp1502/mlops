"""
Grabs the Cleveland Heart Disease dataset straight from the UCI ML
repository and drops a clean, ready-to-use CSV into data/raw/.

Why Cleveland specifically? It's the subset of the UCI "heart-disease"
collection that almost everyone actually uses (the other 3 hospitals -
Hungary, Switzerland, VA Long Beach - have a LOT more missing values),
and it's the version the assignment's "14+ features" description maps
to directly.

Usage:
    python -m src.data.download_data
    python -m src.data.download_data --out data/raw/heart_disease.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# column order as documented by UCI for the "processed" Cleveland file
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

DEFAULT_OUT = Path("data/raw/heart_disease.csv")


def fetch_raw_text(url: str = UCI_URL, timeout: int = 20) -> str:
    """Pull the raw CSV-ish text from UCI. Raises if the server misbehaves."""
    logger.info("Downloading dataset from %s", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_cleveland(raw_text: str) -> pd.DataFrame:
    """
    UCI marks missing values with '?' instead of leaving the cell empty,
    so pandas needs to be told about that explicitly. A handful of rows
    (6 out of 303, from memory) have missing 'ca' or 'thal' values.
    """
    from io import StringIO

    df = pd.read_csv(StringIO(raw_text), header=None, names=COLUMN_NAMES, na_values="?")

    # 'num' is the original multi-class severity label (0 = no disease,
    # 1-4 = increasing severity). The assignment wants a binary target,
    # so we collapse anything > 0 into "disease present".
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    return df


def download(out_path: Path = DEFAULT_OUT, url: str = UCI_URL) -> pd.DataFrame:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw_text = fetch_raw_text(url)
        df = parse_cleveland(raw_text)
    except (requests.RequestException, ValueError) as exc:
        logger.error("Could not fetch dataset from UCI (%s).", exc)
        raise SystemExit(
            "Dataset download failed and no cached copy was found. "
            "Check your internet connection or download "
            f"'{url}' manually and save it as {out_path}."
        ) from exc

    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows x %d cols -> %s", *df.shape, out_path)
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where to save the CSV")
    parser.add_argument("--url", type=str, default=UCI_URL, help="Source URL")
    args = parser.parse_args(argv)

    df = download(args.out, args.url)
    logger.info("Target balance:\n%s", df["target"].value_counts(normalize=True).round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
