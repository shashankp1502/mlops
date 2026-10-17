"""Shared pytest fixtures.

Deliberately avoids touching the real data/raw/heart_disease.csv or
hitting the network — the unit test suite should pass on a laptop with
wifi off, and run identically in CI before the "download + train" steps
even execute. A small synthetic dataset with the right columns/dtypes
(and a couple of injected NaNs, matching the real data's `ca`/`thal`
gaps) is enough to exercise all the preprocessing/model code paths.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import FEATURE_COLUMNS, TARGET_COLUMN

N_ROWS = 40


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(seed=7)

    data = {
        "age": rng.integers(29, 77, N_ROWS).astype(float),
        "sex": rng.integers(0, 2, N_ROWS).astype(float),
        "cp": rng.integers(1, 5, N_ROWS).astype(float),
        "trestbps": rng.integers(94, 200, N_ROWS).astype(float),
        "chol": rng.integers(126, 564, N_ROWS).astype(float),
        "fbs": rng.integers(0, 2, N_ROWS).astype(float),
        "restecg": rng.integers(0, 3, N_ROWS).astype(float),
        "thalach": rng.integers(71, 202, N_ROWS).astype(float),
        "exang": rng.integers(0, 2, N_ROWS).astype(float),
        "oldpeak": np.round(rng.uniform(0, 6.2, N_ROWS), 1),
        "slope": rng.integers(1, 4, N_ROWS).astype(float),
        "ca": rng.integers(0, 4, N_ROWS).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], N_ROWS),
        TARGET_COLUMN: rng.integers(0, 2, N_ROWS),
    }
    df = pd.DataFrame(data)

    # sprinkle a few missing values the same way the real UCI file has them
    df.loc[2, "ca"] = np.nan
    df.loc[5, "thal"] = np.nan

    assert set(FEATURE_COLUMNS).issubset(df.columns)
    return df
