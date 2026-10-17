# Screenshots index

Real screenshots captured from an actual end-to-end run of this project
(local Docker, Minikube deployment, and the live GitHub Actions pipeline
on `main`). Embedded directly in `docs/report.md` / `docs/report.docx`.

| File | Shows |
|---|---|
| `01_docker_build_images_ps.png` | `docker images` / `docker ps` — the built `heart-disease-api:latest` image (773MB) |
| `02_github_actions_pipeline.png` | GitHub Actions — all 4 CI jobs (Lint, Unit tests, Train, Docker) green |
| `03_k8s_pods.png` | `kubectl -n heart-disease get pods,svc,deploy` — 2/2 API replicas + Prometheus + Grafana all Running |
| `04_grafana_dashboard.png` | Grafana "Heart Disease API - Overview" dashboard with live traffic |
| `05_prometheus_targets.png` | Prometheus `/targets` page — `heart-disease-api` scrape target UP |
| `06_fastapi_swagger.png` | FastAPI Swagger UI (`/docs`) — auto-generated API documentation |
| `Train_track_with_MLflow.png` | CI `train` job logs — both models' metrics from a live pipeline run |
| `model_comparision/comparision.png` | MLflow UI — Logistic Regression vs Random Forest run comparison |
| `model_comparision/logistic_regression_confusion_matrix.png` | MLflow artifact |
| `model_comparision/logistic_regression_roc_curve.png` | MLflow artifact |
| `model_comparision/logistic_regression_model.png` | MLflow logged model artifact view |
| `model_comparision/random_forest_confusion_matrix.png` | MLflow artifact |
| `model_comparision/random_forest_roc_curve.png` | MLflow artifact |
| `model_comparision/random_forest_artifacts.png` | MLflow artifacts tab |
