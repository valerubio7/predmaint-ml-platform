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

## Quick Start

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/predmaint-ml-platform.git
cd predmaint-ml-platform

# 2. Configurar entorno
cp .env.example .env
# Editar .env con tus valores (MLflow URI, AWS credentials, etc.)

# 3. Instalar dependencias (requiere uv)
uv sync --group dev

# 4. Descargar el dataset
# Colocar ai4i2020.csv en data/raw/
# Fuente: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

# 5. Entrenar el modelo
make train

# 6. Levantar la API localmente
make run
# → API disponible en http://localhost:8000
# → Docs en http://localhost:8000/docs
```

**Ejemplo de predicción:**
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
```

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

<!-- TODO: completar con métricas reales después de la primera ejecución completa -->

| Métrica | Valor |
|---|---|
| F1 Score | `[0.XX]` |
| Precision | `[0.XX]` |
| Recall | `[0.XX]` |
| ROC-AUC | `[0.XX]` |

> Dataset: AI4I 2020 Predictive Maintenance (UCI) — 10.000 registros, ~3.4% tasa de falla.
> Modelo: XGBoost con `scale_pos_weight=29` para corregir el desbalance de clases.

---

## Autor

**Valentín Rubio**
[LinkedIn](https://linkedin.com/in/tu-perfil) · [GitHub](https://github.com/tu-usuario) · [tu@email.com](mailto:tu@email.com)
