"""Pydantic request/response models for the /predict endpoint.

Field bounds here mirror what's actually in the Cleveland dataset (see
notebooks/01_eda.ipynb) — e.g. `cp` only ever takes values 1-4, `thal`
only 3/6/7 — so obviously malformed requests get a clean 422 instead of
silently flowing into the model and producing a meaningless prediction.
"""

from pydantic import BaseModel, Field, field_validator

VALID_THAL_VALUES = {3, 6, 7}


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., gt=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., gt=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., gt=0, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels (0-3)")
    thal: int = Field(..., description="3 = normal, 6 = fixed defect, 7 = reversible defect")

    @field_validator("thal")
    @classmethod
    def thal_must_be_known_category(cls, v: int) -> int:
        if v not in VALID_THAL_VALUES:
            raise ValueError(f"thal must be one of {sorted(VALID_THAL_VALUES)}, got {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: int = Field(..., description="0 = no disease, 1 = disease")
    label: str
    confidence: float = Field(..., description="Model's confidence in the predicted label")
    probability_disease: float = Field(..., description="Raw predicted probability of disease")
    model_version: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    champion_model: str | None = None
