"""
Small helpers for loading the raw CSV and splitting it into X/y and
train/test. Kept deliberately thin — the actual feature transformations
(imputing, scaling, one-hot encoding) live in src/features/build_features.py
as a scikit-learn ColumnTransformer so they can be fit only on the training
fold and then reused as-is at inference time.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "target"

# column order the model (and later, the API) expects — exported so other
# modules don't have to hardcode this list in three different places
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def load_raw_data(path: str | Path = "data/raw/heart_disease.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Couldn't find {path}. Run `python -m src.data.download_data` first."
        )
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing expected columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified split — with only 303 rows total, an unlucky random split
    could easily skew the test set's class balance, so stratify is worth
    the one extra kwarg."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
