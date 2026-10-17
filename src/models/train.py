"""
Trains and compares two classifiers for the Heart Disease dataset —
Logistic Regression and Random Forest — with cross-validated
hyperparameter search, and logs every run (params, CV + held-out
metrics, confusion matrix, ROC curve) to MLflow.

The "champion" (whichever model wins on held-out ROC-AUC) gets its full
pipeline — preprocessing + classifier bundled together — pickled to
models/model.pkl so there's exactly one artifact the API needs to load.

Usage:
    python -m src.models.train
    python -m src.models.train --cv-folds 10
"""

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.data.preprocess import FEATURE_COLUMNS, load_raw_data, make_train_test_split, split_features_target
from src.features.build_features import build_preprocessor
from src.models.evaluate import compute_metrics, save_confusion_matrix, save_roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_OUT_PATH = PROJECT_ROOT / "models" / "model.pkl"
METADATA_OUT_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "heart-disease-classification"

RANDOM_STATE = 42


def build_candidates():
    """Returns {model_name: (pipeline, param_grid)} for every model we try."""
    preprocessor = build_preprocessor()

    logreg_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(solver="liblinear", max_iter=2000, random_state=RANDOM_STATE)),
    ])
    logreg_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l1", "l2"],
        "clf__class_weight": [None, "balanced"],
    }

    # note: need a fresh preprocessor instance per pipeline — sklearn clones
    # it during fit, but sharing the *same unfitted* object across two
    # separate Pipelines is fine since neither mutates it before cloning
    rf_pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    rf_grid = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    }

    return {
        "logistic_regression": (logreg_pipeline, logreg_grid),
        "random_forest": (rf_pipeline, rf_grid),
    }


def train_one_model(name, pipeline, param_grid, X_train, y_train, X_test, y_test, cv_folds: int):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_family", name)
        mlflow.log_param("cv_folds", cv_folds)

        logger.info("[%s] running GridSearchCV over %d candidate(s)...", name, _grid_size(param_grid))
        search = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})
        mlflow.log_metric("cv_roc_auc_mean", search.best_score_)

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        with tempfile.TemporaryDirectory() as tmp:
            cm_path = save_confusion_matrix(y_test, y_pred, Path(tmp) / "confusion_matrix.png",
                                             title=f"{name} - confusion matrix")
            roc_path = save_roc_curve(y_test, y_proba, Path(tmp) / "roc_curve.png",
                                       title=f"{name} - ROC curve")
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))

        mlflow.sklearn.log_model(best_model, artifact_path="model", input_example=X_train.head(2))

        logger.info(
            "[%s] best params=%s | cv_roc_auc=%.4f | test_roc_auc=%.4f | test_acc=%.4f",
            name, search.best_params_, search.best_score_, test_metrics["roc_auc"], test_metrics["accuracy"],
        )

        return {
            "name": name,
            "estimator": best_model,
            "best_params": search.best_params_,
            "cv_roc_auc_mean": search.best_score_,
            "test_metrics": test_metrics,
            "run_id": mlflow.active_run().info.run_id,
        }


def _grid_size(param_grid: dict) -> int:
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "data/raw/heart_disease.csv")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw_data(args.data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=args.test_size)
    logger.info("Train/test split: %d / %d rows", len(X_train), len(X_test))

    candidates = build_candidates()
    results = []
    for name, (pipeline, grid) in candidates.items():
        result = train_one_model(name, pipeline, grid, X_train, y_train, X_test, y_test, args.cv_folds)
        results.append(result)

    champion = max(results, key=lambda r: r["test_metrics"]["roc_auc"])
    logger.info(
        "Champion model: %s (test_roc_auc=%.4f)", champion["name"], champion["test_metrics"]["roc_auc"]
    )

    # log a tiny "which model won" note against the champion's own run so
    # it's visible in the MLflow UI without cross-referencing runs manually
    with mlflow.start_run(run_id=champion["run_id"]):
        mlflow.set_tag("champion", "true")

    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion["estimator"], MODEL_OUT_PATH)

    metadata = {
        "champion_model": champion["name"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "best_params": champion["best_params"],
        "cv_roc_auc_mean": champion["cv_roc_auc_mean"],
        "test_metrics": champion["test_metrics"],
        "mlflow_run_id": champion["run_id"],
        "all_candidates": [
            {"name": r["name"], "test_metrics": r["test_metrics"], "cv_roc_auc_mean": r["cv_roc_auc_mean"]}
            for r in results
        ],
    }
    with open(METADATA_OUT_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved champion pipeline -> %s", MODEL_OUT_PATH)
    logger.info("Saved metadata -> %s", METADATA_OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
