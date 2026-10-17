"""
Builds the scikit-learn ColumnTransformer used to turn the raw Heart
Disease columns into model-ready features.

Column groupings below come straight out of the EDA notebook
(notebooks/01_eda.ipynb, section 7):

- continuous/ordinal numeric  -> median impute + standard scale
- nominal categorical         -> most-frequent impute + one-hot encode
- already-binary (0/1) flags  -> most-frequent impute, left as-is

Wrapping this in a single ColumnTransformer (rather than hand-rolling the
cleaning in pandas) means it gets fit only on the training fold during
cross-validation and then serialized alongside the model — so the exact
same transformation runs at inference time with zero copy-paste risk.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# continuous + the one ordinal count feature (ca: 0-3 vessels) that behaves
# numerically even though it's technically discrete
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# nominal categories — encoding these as raw integers would imply a false
# ordering (e.g. cp=4 is not "more" than cp=1), so one-hot is the right call
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# already 0/1 in the source data, no encoding needed — just guard against
# the odd missing value showing up in a future data refresh
BINARY_FEATURES = ["sex", "fbs", "exang"]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    binary_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor
