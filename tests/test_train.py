"""
Structural tests for src/models/train.py.

Deliberately does NOT run the full GridSearchCV here (that's slow and is
really an integration concern, exercised by actually running
`python -m src.models.train` as its own CI step per the assignment's
"linting, unit tests, model training" pipeline). What's tested here is
that the candidate pipelines are wired correctly and that a *single*
fit/predict pass through each one works end-to-end on small data.
"""

from sklearn.pipeline import Pipeline

from src.data.preprocess import split_features_target
from src.models.train import build_candidates


def test_build_candidates_returns_both_expected_models():
    candidates = build_candidates()
    assert set(candidates.keys()) == {"logistic_regression", "random_forest"}


def test_each_candidate_is_a_pipeline_with_preprocess_and_clf_steps():
    candidates = build_candidates()
    for name, (pipeline, _grid) in candidates.items():
        assert isinstance(pipeline, Pipeline), f"{name} is not a Pipeline"
        assert [s[0] for s in pipeline.steps] == ["preprocess", "clf"]


def test_each_candidate_param_grid_uses_clf_prefix():
    """GridSearchCV needs `clf__<param>` keys to reach into the Pipeline's
    final step — a typo here would silently no-op the grid search."""
    candidates = build_candidates()
    for name, (_pipeline, grid) in candidates.items():
        assert grid, f"{name} has an empty param grid"
        assert all(key.startswith("clf__") for key in grid), name


def test_logistic_regression_candidate_fits_and_predicts(sample_df):
    X, y = split_features_target(sample_df)
    pipeline, _grid = build_candidates()["logistic_regression"]

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_random_forest_candidate_fits_and_predicts(sample_df):
    X, y = split_features_target(sample_df)
    pipeline, _grid = build_candidates()["random_forest"]

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})
