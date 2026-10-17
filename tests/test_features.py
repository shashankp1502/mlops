import numpy as np

from src.data.preprocess import FEATURE_COLUMNS, split_features_target
from src.features.build_features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)


def test_feature_groups_cover_every_column_exactly_once():
    grouped = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    assert sorted(grouped) == sorted(FEATURE_COLUMNS)
    assert len(grouped) == len(set(grouped))  # no column listed twice


def test_preprocessor_output_has_no_nan_even_with_missing_input(sample_df):
    X, _ = split_features_target(sample_df)
    assert X.isna().any().any(), "fixture should contain at least one NaN to make this test meaningful"

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert not np.isnan(transformed).any()


def test_preprocessor_output_row_count_matches_input(sample_df):
    X, _ = split_features_target(sample_df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]


def test_preprocessor_expands_categorical_columns_via_onehot(sample_df):
    X, _ = split_features_target(sample_df)
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    feature_names = preprocessor.get_feature_names_out()
    # one-hot encoding should produce more output columns than raw input
    # columns, since e.g. 'cp' with 4 categories becomes 4 binary columns
    assert len(feature_names) > len(FEATURE_COLUMNS)


def test_preprocessor_is_refittable_on_different_data(sample_df):
    """Guards against accidentally sharing mutable state between fits —
    important since train.py builds two separate pipelines and each must
    get its own independently-fit preprocessor."""
    X, _ = split_features_target(sample_df)
    half = len(X) // 2

    pre_a = build_preprocessor()
    pre_b = build_preprocessor()

    out_a = pre_a.fit_transform(X.iloc[:half])
    out_b = pre_b.fit_transform(X.iloc[half:])

    assert out_a.shape[1] == out_b.shape[1]
