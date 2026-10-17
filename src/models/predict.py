"""
Loads the packaged champion pipeline (preprocessing + classifier, bundled
together by train.py) and runs predictions on new patient records.

This is the one module both the test suite and the FastAPI app import —
keeping the "load model once, predict many times" logic in a single place
means the API doesn't need to know anything about sklearn internals.
"""

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from src.data.preprocess import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"


class HeartDiseaseModel:
    """Thin wrapper around the pickled sklearn Pipeline so callers don't
    have to remember column order or how to turn a dict into a DataFrame."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No model found at {self.model_path}. Run "
                "`python -m src.models.train` to train and package one first."
            )
        self.pipeline = joblib.load(self.model_path)

    def predict_one(self, record: Dict) -> Dict:
        """record: dict with keys matching FEATURE_COLUMNS (age, sex, cp, ...).
        Returns {"prediction": 0 or 1, "confidence": float, "label": str}."""
        missing = set(FEATURE_COLUMNS) - set(record.keys())
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        row = pd.DataFrame([{col: record[col] for col in FEATURE_COLUMNS}])
        proba = float(self.pipeline.predict_proba(row)[0, 1])
        prediction = int(proba >= 0.5)

        return {
            "prediction": prediction,
            "label": "disease" if prediction == 1 else "no_disease",
            # "confidence" = how sure the model is about whichever class it picked,
            # not just "probability of disease" — reads more naturally in an API response
            "confidence": proba if prediction == 1 else 1 - proba,
            "probability_disease": proba,
        }

    def predict_batch(self, records: list[Dict]) -> list[Dict]:
        return [self.predict_one(r) for r in records]
