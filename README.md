# Heart Disease Risk Prediction — End-to-End MLOps Pipeline

**Course:** AIMLCZG523 (MLOps) — Assignment 1, BITS Pilani
**Author:** Shashank Pandey (2025AE05689)

A full train -> track -> package -> test -> containerize -> deploy ->
monitor pipeline for a binary classifier that predicts heart disease
risk from patient health data, built on the
[UCI Cleveland Heart Disease dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data).

## Architecture

```
┌────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────────┐
│  UCI Repo   │ --> │ download_data │ --> │  EDA notebook   │     │             │
│ (raw CSV)   │     │     .py       │     │ (visualizations)│     │             │
└────────────┘     └──────┬───────┘     └────────────────┘     │   MLflow    │
                           │                                     │  Tracking   │
                           v                                     │  (./mlruns) │
                  ┌─────────────────┐    runs logged to -------->│             │
                  │  build_features  │                           └─────────────┘
                  │ (ColumnTransformer)
                  └────────┬────────┘
                           v
            ┌──────────────────────────┐
            │  train.py                 │
            │  - Logistic Regression    │    champion pipeline
            │  - Random Forest          │ ------------------------> models/model.pkl
            │  - GridSearchCV + CV      │                           models/model_metadata.json
            └──────────────────────────┘
                           │
                           v
            ┌──────────────────────────┐
            │  FastAPI app (src/api)    │
            │  /predict  /health        │
            │  /metrics (Prometheus)    │
            └────────────┬─────────────┘
                           │ docker build
                           v
                 ┌──────────────────┐
                 │  Docker image     │
                 │  heart-disease-api│
                 └─────────┬────────┘
                           │ kubectl apply
                           v
       ┌──────────────────────────────────────────┐
       │   Minikube (namespace: heart-disease)      │
       │  ┌────────────┐  ┌────────────┐ ┌────────┐ │
       │  │ Deployment │  │ Prometheus │ │Grafana │ │
       │  │ (2 replicas)│->│  (scrapes  │ │(dash-  │ │
       │  │  + Service  │  │  /metrics) │->│boards) │ │
       │  └────────────┘  └────────────┘ └────────┘ │
       └──────────────────────────────────────────┘
```

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR:
**lint -> unit test -> train (+ MLflow logging) -> build & smoke-test Docker image**,
failing loudly (non-zero exit + readable logs) at the first broken stage.

## Project structure

```
.
├── data/
│   ├── raw/                 # heart_disease.csv (downloaded, gitignored)
│   └── processed/
├── notebooks/
│   └── 01_eda.ipynb         # executed EDA notebook with saved outputs
├── docs/
│   ├── figures/             # PNGs exported from the EDA notebook
│   └── report.md            # full written report
├── src/
│   ├── data/                # download_data.py, preprocess.py
│   ├── features/            # build_features.py (ColumnTransformer)
│   ├── models/              # train.py, evaluate.py, predict.py
│   └── api/                 # FastAPI app (main.py, schemas.py)
├── tests/                   # pytest suite (40+ tests, no network/model dependency)
├── docker/Dockerfile        # multi-stage build for the serving API
├── k8s/
│   ├── namespace.yaml, deployment.yaml, service.yaml
│   └── monitoring/          # Prometheus + Grafana manifests
├── .github/workflows/ci.yml
├── screenshots/             # deployment/CI/Grafana screenshots for the report
├── requirements.txt         # full dev environment
└── requirements-api.txt     # minimal runtime deps baked into the Docker image
```

## 1. Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data

```bash
python -m src.data.download_data
# -> data/raw/heart_disease.csv (303 rows, 14 columns)
```

The dataset is **not** committed to the repo (only `.gitkeep` placeholders
are) — it's small and re-downloads from UCI in a couple of seconds, so
committing a copy would just be redundant bytes in git history. If UCI is
unreachable, the script exits with a clear error telling you where to get
the file manually.

## 3. Explore the data

```bash
jupyter notebook notebooks/01_eda.ipynb
```

The notebook is already executed with outputs saved (histograms, target
balance, correlation heatmap, categorical breakdowns — all exported to
`docs/figures/` too), so you can read it in GitHub without re-running
anything.

## 4. Train models + track experiments with MLflow

```bash
python -m src.models.train
```

This trains **Logistic Regression** and **Random Forest**, each tuned with
`GridSearchCV` (5-fold stratified CV, scored on ROC-AUC), logs every run's
params/metrics/confusion-matrix/ROC-curve/model to MLflow, picks the
higher-scoring model as "champion", and saves:
- `models/model.pkl` — the full pipeline (preprocessing + classifier)
- `models/model_metadata.json` — champion name, best hyperparameters, test metrics

View the MLflow UI:

```bash
mlflow ui --backend-store-uri file:./mlruns
# open http://127.0.0.1:5000
```

## 5. Run the tests

```bash
flake8 src tests          # lint
pytest                     # 40+ unit/integration tests, ~1s, no network needed
```

## 6. Run the API locally

```bash
uvicorn src.api.main:app --reload --port 8000
```

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,
       "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,
       "ca":0,"thal":6}'
# {"prediction":0,"label":"no_disease","confidence":0.7764,...}
```

Interactive docs: http://localhost:8000/docs

## 7. Containerize

```bash
docker build -t heart-disease-api:latest -f docker/Dockerfile .
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest
curl http://localhost:8000/health
docker rm -f heart-disease-api
```

> `models/model.pkl` must exist before building (step 4) — the Dockerfile
> copies it in rather than training inside the image, so the same exact
> artifact that was validated by the test suite is what ships.

## 8. Deploy to Kubernetes (Minikube)

```bash
minikube start --driver=docker --cpus=2 --memory=4000

# build the image *inside* minikube's docker daemon so it doesn't need a registry
eval $(minikube docker-env)
docker build -t heart-disease-api:latest -f docker/Dockerfile .
eval $(minikube docker-env -u)

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl -n heart-disease get pods -w          # wait for Running/Ready
minikube service heart-disease-api -n heart-disease --url
```

Hit `/health` and `/predict` on the URL `minikube service` prints to
confirm the deployment actually works end-to-end, not just that the pod
is Running.

## 9. Monitoring (Prometheus + Grafana)

```bash
kubectl apply -f k8s/monitoring/prometheus-configmap.yaml
kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
kubectl apply -f k8s/monitoring/grafana-provisioning-configmap.yaml
kubectl apply -f k8s/monitoring/grafana-deployment.yaml

minikube service prometheus -n heart-disease --url   # Prometheus UI
minikube service grafana -n heart-disease --url      # Grafana UI (admin/admin)
```

Grafana auto-provisions a Prometheus datasource and a ready-made
"Heart Disease API - Overview" dashboard (request rate, prediction
counts by label, p95 latency, status codes) — no manual dashboard
building required.

## 10. CI/CD

Every push/PR triggers `.github/workflows/ci.yml`:

| Stage | What it does | Fails the pipeline if... |
|---|---|---|
| `lint` | `flake8 src tests` | any style/syntax issue |
| `test` | `pytest` with coverage | any test fails |
| `train` | downloads data, trains both models, logs to MLflow | training errors or `model_metadata.json` missing |
| `docker` | builds the image, runs it, curls `/health` + `/predict` | container fails to start or returns a bad response |

Artifacts (test reports, coverage, trained model + MLflow run data,
Docker smoke-test logs) are uploaded on every run — including failures —
from the Actions tab.

## Results summary

| Model | CV ROC-AUC (5-fold) | Test Accuracy | Test Precision | Test Recall | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (champion) | 0.907 | 0.885 | 0.818 | 0.964 | **0.959** |
| Random Forest | 0.898 | 0.885 | 0.839 | 0.929 | 0.953 |

Both models land in a similar range; Logistic Regression edges out on
held-out ROC-AUC and is the packaged champion. Full details, confusion
matrices and ROC curves for both runs are in MLflow (`mlruns/`) and in
`docs/report.md`.

See `docs/report.md` for the full write-up (EDA discussion, modelling
choices, architecture diagram, CI/CD + deployment screenshots).
