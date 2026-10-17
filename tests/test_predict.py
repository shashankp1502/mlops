import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocess import split_features_target
from src.features.build_features import build_preprocessor
from src.models.predict import HeartDiseaseModel

VALID_RECORD = {
    "age": 54, "sex": 1, "cp": 3, "trestbps": 130, "chol": 246, "fbs": 0,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.2, "slope": 2,
    "ca": 0, "thal": 3,
}


@pytest.fixture
def tiny_model_path(tmp_path, sample_df):
    """Trains a throwaway (not grid-searched — this is about wiring, not
    accuracy) pipeline on the synthetic fixture data and pickles it, so
    HeartDiseaseModel can be tested without depending on the real,
    separately-trained models/model.pkl artifact."""
    X, y = split_features_target(sample_df)

    pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X, y)

    model_path = tmp_path / "tiny_model.pkl"
    joblib.dump(pipeline, model_path)
    return model_path


def test_model_file_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        HeartDiseaseModel(tmp_path / "nope.pkl")


def test_predict_one_returns_expected_keys(tiny_model_path):
    model = HeartDiseaseModel(tiny_model_path)
    result = model.predict_one(VALID_RECORD)

    assert set(result.keys()) == {"prediction", "label", "confidence", "probability_disease"}
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("disease", "no_disease")
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["probability_disease"] <= 1.0


def test_predict_one_label_matches_prediction(tiny_model_path):
    model = HeartDiseaseModel(tiny_model_path)
    result = model.predict_one(VALID_RECORD)

    expected_label = "disease" if result["prediction"] == 1 else "no_disease"
    assert result["label"] == expected_label


def test_predict_one_missing_field_raises(tiny_model_path):
    model = HeartDiseaseModel(tiny_model_path)
    incomplete = {k: v for k, v in VALID_RECORD.items() if k != "age"}

    with pytest.raises(ValueError, match="age"):
        model.predict_one(incomplete)


def test_predict_one_ignores_extra_fields(tiny_model_path):
    """The API might receive extra metadata fields — predict_one should
    only look at the columns it needs."""
    model = HeartDiseaseModel(tiny_model_path)
    record_with_extra = {**VALID_RECORD, "patient_id": "abc-123"}
    result = model.predict_one(record_with_extra)
    assert result["prediction"] in (0, 1)


def test_predict_batch_returns_one_result_per_record(tiny_model_path):
    model = HeartDiseaseModel(tiny_model_path)
    results = model.predict_batch([VALID_RECORD, VALID_RECORD, VALID_RECORD])
    assert len(results) == 3
    assert all("prediction" in r for r in results)
