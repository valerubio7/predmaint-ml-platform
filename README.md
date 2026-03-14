# PredMaint ML Platform

[![CI](https://img.shields.io/github/actions/workflow/status/valerubio7/predmaint-ml-platform/ci.yml?branch=main&label=CI&logo=github)](https://github.com/valerubio7/predmaint-ml-platform/actions/workflows/ci.yml)
[![Deploy](https://img.shields.io/github/actions/workflow/status/valerubio7/predmaint-ml-platform/deploy.yml?branch=main&label=Deploy&logo=amazonaws)](https://github.com/valerubio7/predmaint-ml-platform/actions/workflows/deploy.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)](https://codecov.io/gh/valerubio7/predmaint-ml-platform)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Production-grade MLOps platform for industrial predictive maintenance.** Ingests real sensor data, trains an XGBoost classifier with explicit class-imbalance handling, serves predictions via a REST API deployed on AWS ECS Fargate, and automatically retrains when data drift is detected — with zero manual intervention.

---

## What This Demonstrates

This project was built to reflect the full scope of an MLOps Engineer role in a production environment:

| Concern | Implementation |
|---|---|
| **ML Pipeline** | Prefect-orchestrated flow: ingest → feature engineering → stratified split → XGBoost train → MLflow log → artifact push to S3 |
| **Model Serving** | FastAPI on ECS Fargate; lazy model loading from S3; graceful degradation on startup |
| **Drift Monitoring** | Evidently AI `DataDriftPreset` vs. a held-out reference set; auto-triggers retraining |
| **Experiment Tracking** | MLflow tracking server + model registry; every run fully reproducible |
| **CI/CD** | GitHub Actions: lint → type-check → test → Docker build → ECR push → ECS deploy; OIDC auth (no static credentials) |
| **Code Quality** | ruff + mypy strict mode + 75+ pytest tests + pre-commit hooks (security, conventional commits) |
| **Reproducibility** | `uv` lockfile, pinned Python 3.12, YAML-driven config, deterministic seeds |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  GitHub Actions CI/CD                                               │
│  push to main → lint → typecheck → pytest → Docker build            │
│               → ECR push → ECS task register → blue/green deploy    │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │         AWS ECS Fargate         │
                  │    FastAPI + Uvicorn (:8000)    │
                  │   POST /predict   GET /health   │
                  │   model loaded from S3 at boot  │
                  └──────┬──────────────┬───────────┘
                         │              │
           ┌─────────────▼────┐   ┌─────▼────────────────────────┐
           │  MLflow Server   │   │  Evidently AI                │
           │  Tracking +      │   │  DataDriftPreset             │
           │  Model Registry  │   │  reference vs. production    │
           └──────────────────┘   │                              │
                                  │  drift detected?             │
                                  │       │                      │
                                  │       ▼                      │
                                  │  Prefect monitoring_pipeline │
                                  │  → triggers training_pipeline│
                                  └──────────────────────────────┘
                                               │
                                  ┌────────────▼────────────┐
                                  │  Streamlit Dashboard    │
                                  │  live predictions +     │
                                  │  dataset overview       │
                                  └─────────────────────────┘
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **ML** | XGBoost 2.1, scikit-learn 1.5, pandas 2.2 |
| **API** | FastAPI 0.115, Uvicorn 0.34, Pydantic v2 |
| **Orchestration** | Prefect 3.6 (flows + tasks) |
| **Experiment Tracking** | MLflow 3.10 (tracking server + model registry) |
| **Drift Detection** | Evidently AI 0.7 |
| **Dashboard** | Streamlit 1.55, Plotly 5.24 |
| **Cloud** | AWS S3, ECR, ECS Fargate (256 CPU / 512 MB) |
| **Containerization** | Docker multi-stage, non-root user, HEALTHCHECK |
| **CI/CD** | GitHub Actions + OIDC (no static AWS credentials) |
| **Package Manager** | uv (astral-sh) with lockfile |
| **Code Quality** | ruff, mypy (strict), pytest 9, pre-commit, bandit, detect-secrets |
| **Language** | Python 3.12 |

---

## Model Performance

The model solves a **binary classification problem with ~3.4% failure rate** on the [AI4I 2020 Predictive Maintenance dataset (UCI)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — 10,000 records with 5 sensor features (air temperature, process temperature, rotational speed, torque, tool wear) and a machine type categorical.

| Metric | Value |
|---|---|
| ROC-AUC | **0.974** |
| F1 Score | 0.711 |
| Recall | 0.779 |
| Precision | 0.654 |

**Design rationale:** In industrial maintenance, a missed failure (false negative) far outweighs an unnecessary stop (false positive). `scale_pos_weight=29` was set to mirror the 97%/3% class split, explicitly trading precision for recall. ROC-AUC of 0.974 confirms strong discriminative power regardless of threshold.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 100 | Performance/training-time balance |
| `max_depth` | 6 | Adequate capacity without overfitting on 10k rows |
| `learning_rate` | 0.1 | Standard for 100 estimators |
| `scale_pos_weight` | 29 | Mirrors ~97%/3% class imbalance |

---

## Repository Structure

```
predmaint-ml-platform/
├── src/
│   ├── core/           # Pure business logic (no I/O, no Prefect): config, loader, features
│   ├── pipelines/      # Prefect flows: data pipeline, training_pipeline, monitoring_pipeline
│   ├── monitoring/
│   │   └── drift/      # Evidently drift detection and reference dataset builder
│   ├── api/            # FastAPI app — /predict, /health, lifespan model loading
│   └── dashboard/      # Streamlit interactive dashboard
├── tests/
│   ├── core/           # Unit tests for pure business logic
│   ├── pipelines/      # Tests for Prefect tasks and flows
│   ├── monitoring/
│   │   └── drift/      # Tests for drift detection
│   └── api/            # Tests for FastAPI endpoints
├── configs/
│   └── training.yaml   # Dataset paths, split config, XGBoost hyperparameters
├── .github/
│   ├── workflows/ci.yml      # lint → typecheck → pytest
│   └── workflows/deploy.yml  # ECR push → ECS register → zero-downtime deploy
├── Dockerfile.api      # Multi-stage build, non-root appuser, HEALTHCHECK
├── ecs-task.json       # ECS Fargate task definition (256 CPU, 512 MB, awslogs)
├── Makefile            # Developer workflow: setup, train, run, test, lint, drift, monitor
└── pyproject.toml      # Project metadata, all deps, ruff/mypy/pytest config
```

---

## Running Locally

### Prerequisites

- Python 3.12
- [`uv`](https://github.com/astral-sh/uv) (`pip install uv`)
- Dataset: download `ai4i2020.csv` from [UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) and place it at `data/raw/ai4i2020.csv`

### Setup

```bash
make setup          # Install all dependencies via uv
cp .env.example .env
```

### Train the model

```bash
make train
# Runs the Prefect training pipeline:
# ingest → feature engineering → XGBoost train → MLflow log → save models/model.pkl
```

### Start the API

```bash
make run
# → http://localhost:8000/predict
# → http://localhost:8000/health
# → http://localhost:8000/docs   (Swagger UI)
```

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0,
    "type_h": 0,
    "type_l": 1,
    "type_m": 0
  }'
# → {"failure_predicted": false, "failure_probability": 0.0006}
```

### Other services

```bash
make mlflow-ui      # MLflow tracking UI → http://localhost:5000
make prefect        # Prefect orchestration UI → http://localhost:4200
make dashboard      # Streamlit dashboard → http://localhost:8501
make drift          # Run Evidently drift report → reports/drift/drift_report.html
make monitor        # Run monitoring pipeline (triggers retraining if drift detected)
```

### Port summary

| Service | Command | Port |
|---|---|---|
| FastAPI | `make run` | 8000 |
| MLflow UI | `make mlflow-ui` | 5000 |
| Prefect UI | `make prefect` | 4200 |
| Streamlit | `make dashboard` | 8501 |

---

## CI/CD Pipeline

### CI (`.github/workflows/ci.yml`)

Triggers on push to `main`/`develop` and pull requests to `main`.

1. **lint** — `ruff check src/` + `ruff format --check src/` + `mypy src/` (strict)
2. **test** (needs lint) — `pytest --cov=src --cov-report=xml` + Codecov upload

### Deploy (`.github/workflows/deploy.yml`)

Triggers on push to `main`. Targets the `production` GitHub environment.

1. AWS authentication via **OIDC** (`AWS_DEPLOY_ROLE_ARN`) — no static credentials stored anywhere
2. Build Docker image → push to ECR with `${{ github.sha }}` and `latest` tags
3. Register new ECS task definition from `ecs-task.json`
4. `aws ecs update-service --force-new-deployment` on `predmaint-cluster/predmaint-api`
5. `aws ecs wait services-stable` — blocks until zero-downtime deployment completes
6. Deregister all inactive task definitions (keeps ECS clean)

---

## Key Engineering Decisions

- **OIDC over static credentials:** The only AWS secret stored in GitHub is a role ARN. No access keys anywhere in the codebase or CI environment.
- **Graceful API degradation:** The server starts even if the model artifact is unavailable. `/health` always responds 200; `/predict` returns 503 until the model loads — so ECS health checks pass during cold starts.
- **Prefect tasks testable without a server:** All `@task` functions expose a `.fn()` attribute used in the entire test suite — no Prefect orchestration server needed to run tests.
- **XGBoost column safety:** Feature names are sanitized at transform time (strip `[`, `]`, `<`) to avoid XGBoost's strict column name validation.
- **Dual model loading:** `MODEL_PATH` accepts either a local file path (development) or an `s3://` URI (production), resolved at startup in the FastAPI `lifespan` handler.
- **Deterministic training:** Fixed `random_state=42` in the stratified split, pinned dependency lockfile, and YAML-driven hyperparameters ensure fully reproducible experiments.

---

## Author

**Valentín Rubio** — MLOps / ML Engineer

[LinkedIn](https://www.linkedin.com/in/rubiovalentin) · [GitHub](https://github.com/valerubio7)
