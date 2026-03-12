# PredMaint ML Platform

[![CI](https://img.shields.io/github/actions/workflow/status/tu-usuario/predmaint-ml-platform/ci.yml?branch=main&label=CI&logo=github)](https://github.com/tu-usuario/predmaint-ml-platform/actions)
[![Deploy](https://img.shields.io/github/actions/workflow/status/tu-usuario/predmaint-ml-platform/deploy.yml?branch=main&label=Deploy&logo=amazonaws)](https://github.com/tu-usuario/predmaint-ml-platform/actions)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)](https://codecov.io/gh/tu-usuario/predmaint-ml-platform)

> Plataforma end-to-end de mantenimiento predictivo industrial: detecta fallas de máquinas antes de que ocurran, con reentrenamiento automático ante data drift.

---

## ¿Qué hace?

- **Predice fallas industriales** en tiempo real a partir de lecturas de sensores (temperatura, torque, velocidad, desgaste de herramienta) usando XGBoost con manejo explícito del desbalance de clases.
- **Sirve predicciones vía REST API** (FastAPI) con endpoints `/predict`, `/health` y documentación OpenAPI auto-generada; desplegada en AWS ECS Fargate con zero-downtime.
- **Detecta data drift automáticamente** con Evidently AI y dispara un pipeline de reentrenamiento orquestado por Prefect — sin intervención manual.
- **Registra cada experimento** en MLflow (métricas, parámetros, artefactos) con promoción automática del mejor modelo al registry de producción.
- **Entrega CI/CD completo**: lint con ruff → type checking con mypy → tests con pytest → build Docker → push AWS ECR → deploy ECS, todo en GitHub Actions con OIDC (sin credenciales estáticas).

---

## Stack

| Categoría | Tecnologías |
|---|---|
| **ML** | scikit-learn, XGBoost, pandas |
| **API Serving** | FastAPI, Uvicorn, Pydantic |
| **Experiment Tracking** | MLflow (tracking + model registry) |
| **Orquestación** | Prefect (flows + tasks) |
| **Monitoreo ML** | Evidently AI (drift detection) |
| **Monitoreo Infra** | Prometheus, Grafana |
| **Containerización** | Docker (multi-stage), Docker Compose |
| **Cloud** | AWS S3, ECR, ECS Fargate |
| **CI/CD** | GitHub Actions (OIDC, sin credenciales estáticas) |
| **Dashboard** | Streamlit |
| **Calidad de código** | ruff, mypy, pytest, pre-commit |
| **Lenguaje** | Python 3.12 |

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                        CI/CD                                 │
│  GitHub Push → GitHub Actions                                │
│  [lint → test → build Docker → push ECR → deploy ECS]       │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │         AWS ECS Fargate          │
          │      FastAPI + Uvicorn           │
          │   /predict  /health  /metrics    │
          └──────┬──────────────┬───────────┘
                 │              │
    ┌────────────▼────┐   ┌─────▼──────────────────┐
    │  MLflow Server  │   │  Evidently AI           │
    │  (Tracking +    │   │  (Drift Detection)      │
    │   Model Reg.)   │   │                         │
    └─────────────────┘   │  drift? ──► Prefect     │
                          │             Pipeline     │
                          │          (retraining)   │
                          └─────────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Prometheus + Grafana    │
                          │  (Infrastructure Mon.)   │
                          └─────────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Streamlit Dashboard     │
                          │  (Demo / Monitoring UI)  │
                          └─────────────────────────┘
```

---

## Desarrollo Local

Cada componente corre de forma independiente. El orden recomendado es el siguiente:

### 1. Setup inicial

```bash
# Instalar dependencias
make setup

# Copiar variables de entorno
cp .env.example .env

# Colocar el dataset en data/raw/
# Fuente: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
```

### 2. Procesar datos y entrenar el modelo

```bash
make train
# Ejecuta el pipeline de features y entrena el modelo XGBoost via Prefect.
# Guarda el modelo en models/model.pkl y registra el experimento en MLflow.
```

### 3. FastAPI — API de predicciones

```bash
make run
# → http://localhost:8000/predict
# → http://localhost:8000/health
# → http://localhost:8000/docs   (Swagger UI)
```

Ejemplo de llamada:

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

### 4. MLflow — Tracking de experimentos

```bash
make mlflow-ui
# → http://localhost:5000
```

Muestra todos los runs de entrenamiento con sus métricas, y el model registry con las versiones registradas del modelo.

### 5. Prefect — Orquestación de pipelines

```bash
uv run prefect server start
# → http://localhost:4200
```

Desde la UI se pueden ver los flows ejecutados (training-pipeline, monitoring-pipeline), el historial de runs y los logs de cada task.

> Para correr el pipeline de monitoreo manualmente:
> ```bash
> uv run python -c "from src.training.train import monitoring_pipeline; monitoring_pipeline()"
> ```

### 6. Evidently — Reporte de drift

```bash
uv run python src/monitoring/drift.py
# Genera reports/drift/drift_report.html
```

Abre el archivo HTML generado en el browser para ver el reporte de drift entre los datos de referencia y produccion.

### 7. Streamlit — Dashboard

```bash
make dashboard
# → http://localhost:8501
```

Muestra el estado de la API, permite hacer predicciones interactivas con sliders y visualiza el dataset de entrenamiento.

---

### Resumen de puertos

| Servicio | Comando | Puerto |
|---|---|---|
| FastAPI | `make run` | 8000 |
| MLflow UI | `make mlflow-ui` | 5000 |
| Prefect UI | `uv run prefect server start` | 4200 |
| Streamlit | `make dashboard` | 8501 |

---

## Estructura del Repositorio

```
predmaint-ml-platform/
├── src/
│   ├── api/            # FastAPI app — endpoints /predict, /health
│   ├── data/           # Ingesta, transformación y pipeline de features
│   ├── training/       # Flows de Prefect: entrenamiento + monitoreo
│   ├── monitoring/     # Detección de drift con Evidently AI
│   └── dashboard/      # Dashboard interactivo con Streamlit
├── tests/              # Suite de tests con pytest (unit + integration)
├── configs/            # Configuración YAML del pipeline de entrenamiento
├── .github/workflows/  # CI (lint+test) y Deploy (ECR+ECS) en GitHub Actions
├── Dockerfile.api      # Imagen multi-stage para la API
└── Makefile            # Comandos de desarrollo (setup, train, lint, test)
```

---

## Resultados del Modelo

| Métrica | Valor |
|---|---|
| ROC-AUC | **0.974** |
| F1 Score | 0.711 |
| Recall | 0.779 |
| Precision | 0.654 |

**Dataset:** AI4I 2020 Predictive Maintenance (UCI) — 10.000 registros, ~3.4% tasa de falla.

**Modelo:** XGBoost con los siguientes hiperparámetros:

| Parámetro | Valor | Justificación |
|---|---|---|
| `n_estimators` | 100 | Balance entre rendimiento y tiempo de entrenamiento |
| `max_depth` | 6 | Suficiente capacidad sin sobreajuste en un dataset de 10k filas |
| `learning_rate` | 0.1 | Estándar para XGBoost con 100 árboles |
| `scale_pos_weight` | 29 | Ratio de desbalance de clases (~97% negativo / ~3% positivo) |

**Decisión de diseño:** en mantenimiento predictivo industrial, un falso negativo (no detectar una falla real) tiene un costo mucho mayor que un falso positivo (parar una máquina innecesariamente). Por eso se priorizó **Recall alto (0.779)** — el modelo detecta el 78% de las fallas reales — y se aceptó una Precision menor. El ROC-AUC de **0.974** indica excelente capacidad discriminativa del clasificador independientemente del umbral.

---

## Autor

**Valentín Rubio**
[LinkedIn](https://linkedin.com/in/tu-perfil) · [GitHub](https://github.com/tu-usuario) · [tu@email.com](mailto:tu@email.com)
