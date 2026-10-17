import pytest

from src.data.preprocess import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_raw_data,
    make_train_test_split,
    split_features_target,
)


def test_load_raw_data_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_data(missing_path)


def test_load_raw_data_reads_existing_file(tmp_path, sample_df):
    csv_path = tmp_path / "heart.csv"
    sample_df.to_csv(csv_path, index=False)
    loaded = load_raw_data(csv_path)
    assert loaded.shape == sample_df.shape


def test_split_features_target_shapes(sample_df):
    X, y = split_features_target(sample_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape[0] == len(sample_df)
    assert y.name == TARGET_COLUMN


def test_split_features_target_raises_on_missing_column(sample_df):
    broken = sample_df.drop(columns=["age"])
    with pytest.raises(ValueError):
        split_features_target(broken)


def test_make_train_test_split_sizes(sample_df):
    X, y = split_features_target(sample_df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.25)

    assert len(X_train) + len(X_test) == len(X)
    assert len(X_test) == round(0.25 * len(X))


def test_make_train_test_split_is_stratified(sample_df):
    X, y = split_features_target(sample_df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.5)

    # with test_size=0.5 the class proportions in each half should match
    # the full set almost exactly (small rounding differences allowed)
    full_ratio = y.mean()
    assert abs(y_train.mean() - full_ratio) <= 0.15
    assert abs(y_test.mean() - full_ratio) <= 0.15
