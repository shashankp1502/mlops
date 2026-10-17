import numpy as np

from src.models.evaluate import compute_metrics, save_confusion_matrix, save_roc_curve


def test_compute_metrics_perfect_predictions():
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 1, 1]
    y_proba = [0.02, 0.98, 0.1, 0.9, 0.8]

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_all_values_within_valid_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 50)
    y_pred = rng.integers(0, 2, 50)
    y_proba = rng.uniform(0, 1, 50)

    metrics = compute_metrics(y_true, y_pred, y_proba)

    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{name}={value} out of [0,1] range"


def test_compute_metrics_handles_single_class_predictions_without_crashing():
    """A bad grid-search candidate can predict only one class — this
    shouldn't raise (zero_division=0 should kick in)."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]
    y_proba = [0.1, 0.2, 0.3, 0.4]

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_save_confusion_matrix_writes_file(tmp_path):
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    out_path = tmp_path / "cm.png"

    result_path = save_confusion_matrix(y_true, y_pred, out_path)

    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_save_roc_curve_writes_file(tmp_path):
    y_true = [0, 1, 0, 1]
    y_proba = [0.2, 0.9, 0.3, 0.7]
    out_path = tmp_path / "roc.png"

    result_path = save_roc_curve(y_true, y_proba, out_path)

    assert result_path.exists()
    assert result_path.stat().st_size > 0
