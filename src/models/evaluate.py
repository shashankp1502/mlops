"""Metric computation + the couple of diagnostic plots I log to MLflow
for every run (confusion matrix, ROC curve). Split out from train.py so
the tests can import compute_metrics without pulling in MLflow/sklearn
training machinery.
"""

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # headless — this runs inside CI / Docker too
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Standard binary classification metrics. zero_division=0 so a model
    that (temporarily, during a bad grid-search candidate) predicts a
    single class doesn't blow up the whole run with a warning-turned-error."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def save_confusion_matrix(y_true, y_pred, out_path: str | Path, title: str = "") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No disease", "Disease"], ax=ax, colorbar=False
    )
    ax.set_title(title or "Confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_roc_curve(y_true, y_proba, out_path: str | Path, title: str = "") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title(title or "ROC curve")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
