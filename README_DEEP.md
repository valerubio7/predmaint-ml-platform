# PredMaint ML Platform — Documentación Técnica

[![CI](https://img.shields.io/github/actions/workflow/status/tu-usuario/predmaint-ml-platform/ci.yml?branch=main&label=CI&logo=github)](https://github.com/tu-usuario/predmaint-ml-platform/actions)
[![Deploy](https://img.shields.io/github/actions/workflow/status/tu-usuario/predmaint-ml-platform/deploy.yml?branch=main&label=Deploy&logo=amazonaws)](https://github.com/tu-usuario/predmaint-ml-platform/actions)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)](https://codecov.io/gh/tu-usuario/predmaint-ml-platform)

Sistema de mantenimiento predictivo industrial end-to-end. Ingiere datos de sensores de máquinas, entrena un clasificador XGBoost, sirve predicciones vía FastAPI desplegada en AWS ECS, monitorea data drift con Evidently AI, y reentrenamiento automáticamente cuando la distribución de datos en producción se desvía del baseline — todo orquestado con Prefect y trazado con MLflow.

---

## Tabla de Contenidos

1. [Contexto de negocio](#1-contexto-de-negocio)
2. [Arquitectura técnica](#2-arquitectura-técnica)
3. [Dataset — AI4I 2020](#3-dataset--ai4i-2020)
4. [Pipeline ML detallado](#4-pipeline-ml-detallado)
5. [Loop de reentrenamiento automático](#5-loop-de-reentrenamiento-automático)
6. [Estructura del repositorio](#6-estructura-del-repositorio)
7. [Variables de entorno](#7-variables-de-entorno)
8. [Instalación y setup](#8-instalación-y-setup)
9. [Guía de desarrollo](#9-guía-de-desarrollo)
10. [CI/CD — GitHub Actions](#10-cicd--github-actions)
11. [Monitoreo y UIs](#11-monitoreo-y-uis)
12. [Decisiones de diseño (ADRs)](#12-decisiones-de-diseño-adrs)
13. [Roadmap / TODO](#13-roadmap--todo)
14. [Referencias](#14-referencias)

---

## 1. Contexto de negocio

### El problema

En industrias como manufactura, energía o aviación, el tiempo de inactividad no planificado de maquinaria cuesta entre USD 5.000 y USD 50.000 por hora dependiendo del sector. Las dos estrategias tradicionales tienen limitaciones estructurales:

| Estrategia | Descripción | Problema |
|---|---|---|
| **Mantenimiento reactivo** | Reparar cuando la máquina falla | Tiempo de inactividad no planificado, daños en cascada |
| **Mantenimiento preventivo** | Reemplazar componentes en intervalos fijos (calendario) | Reemplaza piezas que aún tienen vida útil, desperdicio de recursos |
| **Mantenimiento predictivo** | Intervenir solo cuando los datos indican degradación | Requiere datos, modelos y infraestructura |

### El approach

Este sistema implementa mantenimiento predictivo basado en datos: analiza lecturas de sensores en tiempo real (temperatura del aire, temperatura de proceso, velocidad de rotación, torque, desgaste de herramienta y tipo de máquina) para predecir si una máquina va a fallar antes de que el fallo ocurra. El modelo entregga una probabilidad de fallo (0–1) que permite priorizar intervenciones.

El diferenciador clave frente a soluciones estáticas es el **loop de reentrenamiento automático**: cuando los datos de producción comienzan a diferir del baseline histórico (data drift), el sistema detecta este cambio automáticamente y dispara un ciclo de reentrenamiento sin intervención humana.

### Sectores objetivo

- **Manufactura**: líneas CNC, tornos, fresadoras
- **Energía**: turbinas, compresores, bombas
- **Aviación**: componentes de motores, sistemas hidráulicos
- **IoT industrial**: cualquier maquinaria con sensores de proceso

---

## 2. Arquitectura técnica

### Diagrama completo del sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            DEVELOPER WORKFLOW                            │
│                                                                          │
│   git push ──► GitHub                                                    │
│                   │                                                      │
│          ┌────────▼────────────────────────────────────┐                │
│          │           GitHub Actions CI/CD               │                │
│          │                                              │                │
│          │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │                │
│          │  │  ci.yml  │  │deploy.yml│  │          │  │                │
│          │  │          │  │          │  │          │  │                │
│          │  │ 1. lint  │  │1. OIDC   │  │          │  │                │
│          │  │ 2. mypy  │──► 2. ECR   │  │          │  │                │
│          │  │ 3. pytest│  │3. Docker │  │          │  │                │
│          │  │          │  │4. ECS    │  │          │  │                │
│          │  └──────────┘  └────┬─────┘  └──────────┘  │                │
│          └───────────────────── │ ───────────────────── ┘                │
└─────────────────────────────────│───────────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────────────────────┐
                    │              AWS Infrastructure                     │
                    │                                                     │
                    │   ECR Registry ──► ECS Fargate Cluster             │
                    │                        │                           │
                    │              ┌─────────▼─────────┐                │
                    │              │   FastAPI + Uvicorn│                │
                    │              │   (2 workers)      │                │
                    │              │                    │                │
                    │              │  GET  /health      │                │
                    │              │  POST /predict     │                │
                    │              │  GET  /docs        │                │
                    │              └────┬──────┬────────┘                │
                    │                   │      │                         │
                    │              S3 Bucket   │                         │
                    │              (data +     │                         │
                    │               models)    │                         │
                    └───────────────────│──────│─────────────────────────┘
                                        │      │
              ┌─────────────────────────┘      └──────────────────────┐
              │                                                         │
   ┌──────────▼──────────┐                           ┌────────────────▼──────────┐
   │    MLflow Server     │                           │      Evidently AI          │
   │                      │                           │   (Drift Detection)        │
   │  - Experiment runs   │                           │                            │
   │  - Model registry    │                           │  reference ─► current      │
   │  - Artifacts (S3)    │                           │  data       │   data        │
   │  - Metrics history   │                           │             ▼              │
   └──────────────────────┘                           │     drift_share >= θ?      │
                                                      │             │              │
                                              ┌───────┘    yes      │   no         │
                                              │                     ▼              │
                                   ┌──────────▼────────┐   model stable           │
                                   │   Prefect          │                          │
                                   │   Orchestrator     │                          │
                                   │                    │                          │
                                   │  training-pipeline │                          │
                                   │  ├─ load-config    │                          │
                                   │  ├─ ingest-data    │                          │
                                   │  ├─ build-features │                          │
                                   │  ├─ split-data     │                          │
                                   │  ├─ train-model    │                          │
                                   │  ├─ evaluate-model │                          │
                                   │  ├─ log-to-mlflow  │                          │
                                   │  └─ save-model     │                          │
                                   └────────────────────┘                          │
                                                                                   │
              ┌────────────────────────────────────────────────────────────────────┘
              │
   ┌──────────▼──────────────────────────────────────────────────┐
   │                   Observability Layer                         │
   │                                                               │
   │  Prometheus ──► Grafana        Streamlit Dashboard            │
   │  (infra metrics)               (demo UI + predictions)        │
   └───────────────────────────────────────────────────────────────┘
```

### Componentes y justificación de elección

| Componente | Tecnología elegida | Alternativas consideradas | Justificación |
|---|---|---|---|
| ML Framework | XGBoost + scikit-learn | LightGBM, CatBoost, PyTorch Tabular | Ver ADR-002 |
| API Serving | FastAPI | Flask, Django REST, BentoML | Ver ADR-003 |
| Orquestación | Prefect | Apache Airflow, Dagster, Luigi | Ver ADR-001 |
| Experiment Tracking | MLflow | W&B, Neptune, ClearML | Open-source, auto-hosteable, integración nativa con XGBoost |
| Drift Detection | Evidently AI | NannyML, Alibi Detect, custom | API más simple, reportes HTML auto-generados |
| Containerización | Docker multi-stage | — | Imagen más pequeña, separación builder/runtime |
| Cloud | AWS (ECR + ECS) | GCP, Azure | ECS Fargate: serverless containers sin gestionar instancias EC2 |
| Package manager | uv | pip, poetry, conda | Resolución de dependencias ~100x más rápida que pip |

### Flujo de datos completo

```
Sensor raw data (CSV)
        │
        ▼
[1] loader.py — load_raw_data()
    Valida columnas esperadas contra EXPECTED_COLUMNS
        │
        ▼
[2] transformer.py — build_features()
    - Drop: UDI, Product ID, TWF, HDF, PWF, OSF, RNF
    - One-hot encoding: Type → Type_H, Type_L, Type_M
    - Sanitización de nombres: "Air temperature [K]" → "Air_temperature_K"
    - Output: X (8 features), y (Machine_failure)
        │
        ▼
[3] train.py — split_data() [stratified, 80/20]
        │
        ├──► X_train, y_train ──► train_model() ──► XGBClassifier fitted
        │                                                    │
        └──► X_test, y_test ──► evaluate_model()            │
                                    │                        │
                                    ▼                        │
                              {f1, precision,                │
                               recall, roc_auc}              │
                                    │                        │
                                    ▼                        ▼
                             log_to_mlflow()          save_model()
                             (registro + registry)    (models/model.pkl)
                                                             │
                                                             ▼
                                                      FastAPI loads
                                                      model lazily
                                                      on first request
                                                             │
                                                    POST /predict
                                                    → SensorData (Pydantic)
                                                    → DataFrame (8 cols)
                                                    → model.predict()
                                                    → PredictionResponse
                                                      {failure_predicted,
                                                       failure_probability}
```

---

## 3. Dataset — AI4I 2020

**Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

### Descripción

El AI4I 2020 Predictive Maintenance Dataset es un dataset sintético diseñado para reflejar condiciones reales de maquinaria industrial. Contiene 10.000 registros de sensores de máquinas con sus etiquetas de fallo correspondientes.

### Columnas

| Columna | Tipo | Descripción |
|---|---|---|
| `UDI` | int | Identificador único (1–10.000) — se descarta en features |
| `Product ID` | string | ID de producto con prefijo L/M/H — se descarta |
| `Type` | categorical | Calidad del producto: L (low), M (medium), H (high) |
| `Air temperature [K]` | float | Temperatura del aire en Kelvin (~300 K) |
| `Process temperature [K]` | float | Temperatura del proceso (~310 K, siempre > aire) |
| `Rotational speed [rpm]` | int | Velocidad de rotación (calculada de potencia de 2860 W + ruido) |
| `Torque [Nm]` | float | Torque aplicado (~40 Nm, distribución normal) |
| `Tool wear [min]` | int | Minutos acumulados de desgaste de herramienta (0–253) |
| `Machine failure` | int | **Target**: 0 = sin falla, 1 = falla (cualquier tipo) |
| `TWF` | int | Tool Wear Failure — falla por desgaste excesivo de herramienta |
| `HDF` | int | Heat Dissipation Failure — falla por disipación de calor insuficiente |
| `PWF` | int | Power Failure — falla por potencia fuera de rango |
| `OSF` | int | Overstrain Failure — falla por sobrecarga del producto de torque × desgaste |
| `RNF` | int | Random Failures — fallos aleatorios independientes de proceso (0.1% de prob.) |

### Tipos de falla

Los cinco modos de falla son mutuamente excluyentes en origen pero el target `Machine failure` es la unión lógica. Las columnas individuales (TWF, HDF, PWF, OSF, RNF) se descartan del feature set para predecir el fallo como evento agregado.

```
Machine failure = TWF OR HDF OR PWF OR OSF OR RNF
```

### Distribución del target

```
Machine failure = 0 (sin falla): ~9.661 registros (~96.6%)
Machine failure = 1 (falla):       ~339 registros (~3.4%)
```

El dataset es significativamente desbalanceado (~29:1). Esto se maneja con el hiperparámetro `scale_pos_weight=29` en XGBoost, que pondera la clase minoritaria inversamente proporcional a su frecuencia.

### Por qué es un buen proxy industrial

Aunque es sintético, el dataset modela comportamientos físicos reales:
- La temperatura de proceso siempre excede la temperatura del aire (física del proceso)
- La velocidad de rotación y el torque tienen una relación inversa (potencia ≈ constante)
- El desgaste de herramienta es monotónico creciente (acumulación real)
- Los umbrales de fallo están basados en condiciones físicas documentadas

---

## 4. Pipeline ML detallado

### 4.1 Ingesta y validación

**Archivo:** `src/data/loader.py`

```python
EXPECTED_COLUMNS = [
    "UDI", "Product ID", "Type",
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
]

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df
```

La validación de columnas falla rápido (fail-fast) ante datos corruptos o con schema incorrecto. Esto evita que errores de datos silenciosos contaminen el entrenamiento.

> **Nota:** La integración con Great Expectations está en el roadmap para validaciones más ricas (rangos, distribuciones, unicidad).

### 4.2 Feature engineering

**Archivo:** `src/data/transformer.py`

Transformaciones aplicadas:

1. **Drop de columnas no informativas:** `UDI`, `Product ID` (IDs sin valor predictivo), y las columnas de sub-tipo de falla (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) — se predice el fallo agregado.

2. **One-hot encoding de `Type`:** La variable categórica ordinal `Type` (L/M/H) se convierte en tres columnas binarias (`Type_H`, `Type_L`, `Type_M`) usando `pd.get_dummies` con `drop_first=False` para preservar todas las categorías.

3. **Sanitización de nombres de columna:** XGBoost no acepta caracteres especiales como `[`, `]` o `<` en los nombres de features. Se reemplazan mediante regex y los espacios se convierten a `_`.

Feature set final (8 columnas):
```
Air_temperature_K, Process_temperature_K, Rotational_speed_rpm,
Torque_Nm, Tool_wear_min, Type_H, Type_L, Type_M
```

### 4.3 Entrenamiento y tracking con MLflow

**Archivo:** `src/training/train.py`

El pipeline de entrenamiento está implementado como un **Prefect flow** con tasks individuales que permiten observabilidad granular y reintentos independientes:

```
@flow(name="training-pipeline")
def training_pipeline():
    config  = load_config_task()        # Lee configs/training.yaml
    df      = ingest_data(config)       # Carga y valida CSV
    X, y    = build_features_task(df)   # Feature engineering
    splits  = split_data(X, y, config)  # Train/test split estratificado
    model   = train_model(...)          # XGBClassifier.fit()
    metrics = evaluate_model(...)       # f1, precision, recall, roc_auc
    log_to_mlflow(model, metrics)       # Registro en MLflow
    save_model(model)                   # Persistencia en disk (joblib)
```

**Configuración del modelo** (`configs/training.yaml`):
```yaml
model:
  name: "xgboost"
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    scale_pos_weight: 29    # ~9661/339 — corrige el desbalance de clases
```

**Split estratificado:** El `train_test_split` usa `stratify=y` para garantizar que la proporción de la clase positiva (~3.4%) se preserva tanto en train como en test. Sin estratificación, es posible que el test set tenga cero ejemplos de falla.

### 4.4 Evaluación y métricas

Las métricas calculadas sobre el test set son:

| Métrica | Por qué importa en este contexto |
|---|---|
| **F1 Score** | Balance entre precision y recall — clave con clases desbalanceadas |
| **Precision** | De las fallas predichas, ¿cuántas son reales? (costo de falsas alarmas) |
| **Recall** | De las fallas reales, ¿cuántas se detectaron? (costo de fallas no detectadas) |
| **ROC-AUC** | Discriminación general del modelo independiente del threshold |

En contexto industrial, el **Recall** suele ser la métrica más crítica: una falla no detectada (falso negativo) es más costosa que una falsa alarma (falso positivo). El threshold de decisión puede ajustarse en la API según el costo relativo de cada tipo de error.

<!-- TODO: agregar tabla con métricas reales después de la primera ejecución completa -->

| Métrica | Valor |
|---|---|
| F1 Score | `[0.XX]` |
| Precision | `[0.XX]` |
| Recall | `[0.XX]` |
| ROC-AUC | `[0.XX]` |

### 4.5 Serving — FastAPI endpoints

**Archivo:** `src/api/main.py`

La API expone tres endpoints:

#### `GET /health`

```json
{"status": "ok"}
```

Usado por el HEALTHCHECK del Dockerfile y por los load balancers de ECS para verificar que el contenedor está operativo.

#### `POST /predict`

**Request body** (`SensorData`):
```json
{
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551,
  "torque": 42.8,
  "tool_wear": 0,
  "type_h": 0,
  "type_l": 1,
  "type_m": 0
}
```

**Response** (`PredictionResponse`):
```json
{
  "failure_predicted": false,
  "failure_probability": 0.0312
}
```

**Comportamiento especial:**
- El modelo se carga de forma **lazy** (al primer request): si `models/model.pkl` no existe, retorna `HTTP 503` con instrucción de entrenamiento.
- La variable global `_model` actúa como singleton en memoria para evitar recargar el modelo en cada request.
- Todos los errores inesperados del modelo se capturan y retornan como `HTTP 500` con el mensaje de la excepción.

#### `GET /docs`

Documentación OpenAPI interactiva auto-generada por FastAPI (Swagger UI).

---

## 5. Loop de reentrenamiento automático

Este es el componente diferencial del sistema. El flujo completo:

```
┌─────────────────────────────────────────────────────────────────┐
│                    monitoring-pipeline (Prefect)                  │
│                                                                   │
│  1. load_config_task()                                            │
│         │                                                         │
│  2. check_drift(config)                                           │
│     ├─ Carga dataset completo                                     │
│     ├─ Aplica feature engineering                                 │
│     ├─ Toma rows[7000:] como "production sample"                  │
│     ├─ Llama detect_drift(production_sample)                      │
│     │   ├─ Lee reference dataset (data/processed/reference.parquet) │
│     │   ├─ Ejecuta Evidently DataDriftPreset                      │
│     │   ├─ Guarda HTML report en reports/drift/drift_report.html  │
│     │   └─ Retorna drift_share >= DRIFT_THRESHOLD (default: 0.05) │
│     └─ Retorna bool: drift_detected                               │
│         │                                                         │
│  3. if drift_detected AND RETRAINING_TRIGGER == true:             │
│         │                                                         │
│         └──► training_pipeline()  [subflow completo]             │
│              (ver sección 4.3)                                    │
│                                                                   │
│     elif drift_detected AND RETRAINING_TRIGGER == false:         │
│         └──► LOG WARNING "retraining disabled"                    │
│              (útil para ambientes de staging / auditoría)         │
│                                                                   │
│     else:                                                         │
│         └──► LOG INFO "model is stable"                           │
└─────────────────────────────────────────────────────────────────┘
```

### Dataset de referencia

Antes de correr el monitoreo por primera vez, se debe construir el dataset de referencia:

```bash
uv run python src/monitoring/drift.py
```

Este script toma el 70% inicial del dataset (configurable via `REFERENCE_SPLIT_RATIO`) y lo guarda como `data/processed/reference.parquet`. El 30% restante simula datos de producción para la detección de drift.

### Configuración del threshold

El threshold de drift (`DRIFT_THRESHOLD=0.05`) representa la **fracción de features** con drift estadísticamente significativo que dispara el reentrenamiento. Con 8 features, un threshold de 0.05 significa que cualquier feature con drift dispara el pipeline (1/8 = 0.125 > 0.05).

Para ajustar el comportamiento:

```bash
# Más sensible (cualquier feature con drift → retrain)
DRIFT_THRESHOLD=0.05

# Menos sensible (más de la mitad de features → retrain)
DRIFT_THRESHOLD=0.50

# Deshabilitar reentrenamiento automático (solo alertar)
RETRAINING_TRIGGER=false
```

---

## 6. Estructura del repositorio

```
predmaint-ml-platform/
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint, mypy, pytest — se ejecuta en push a main/develop y PRs
│       └── deploy.yml          # Build Docker, push ECR, deploy ECS — solo en push a main
│
├── configs/
│   └── training.yaml           # Hiperparámetros del modelo y paths de datos
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv        # Dataset original (no commiteado en git)
│   └── processed/
│       ├── features.parquet    # Output del pipeline de features (generado)
│       └── reference.parquet   # Dataset de referencia para drift detection (generado)
│
├── models/
│   └── model.pkl               # Modelo serializado con joblib (generado por make train)
│
├── notebooks/                  # Exploración y análisis ad-hoc (no productivos)
│
├── reports/
│   └── drift/
│       └── drift_report.html   # Reporte HTML de Evidently (generado por monitoreo)
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py             # FastAPI app: endpoints /predict, /health, /docs
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Carga CSV y valida columnas esperadas
│   │   ├── transformer.py      # Feature engineering: OHE, drop, sanitización
│   │   └── pipeline.py         # Orquesta loader → transformer → save parquet
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py            # Prefect flows: training-pipeline + monitoring-pipeline
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── drift.py            # build_reference_dataset() + detect_drift() con Evidently
│   │
│   └── dashboard/
│       ├── __init__.py
│       └── app.py              # Streamlit: health check, predicción interactiva, dataset overview
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py             # Tests de endpoints FastAPI (mocking del modelo)
│   ├── test_drift.py           # Tests de detección de drift
│   ├── test_loader.py          # Tests de carga y validación de datos
│   ├── test_pipeline.py        # Tests del pipeline de features
│   ├── test_train.py           # Tests unitarios de Prefect tasks
│   └── test_transformer.py     # Tests de transformaciones de features
│
├── .env.example                # Template de variables de entorno (commiteado)
├── .pre-commit-config.yaml     # Hooks: ruff, mypy, bandit, detect-secrets, conventional commits
├── .python-version             # Versión de Python para uv/pyenv
├── Dockerfile.api              # Imagen multi-stage para la API (builder + runtime)
├── Makefile                    # Comandos de desarrollo: setup, train, lint, test, run
├── pyproject.toml              # Dependencias del proyecto y configuración de herramientas
└── uv.lock                     # Lockfile de dependencias (reproducibilidad garantizada)
```

---

## 7. Variables de entorno

Copiar `.env.example` a `.env` y completar los valores. Las variables marcadas como **Obligatoria** son necesarias para que el sistema funcione; las **Opcional** tienen valores por defecto razonables.

| Variable | Descripción | Valor por defecto | Obligatoria |
|---|---|---|---|
| `API_HOST` | Host de binding de la API | `0.0.0.0` | No |
| `API_PORT` | Puerto de la API | `8000` | No |
| `API_ENV` | Entorno de ejecución (`development`, `staging`, `production`) | `development` | No |
| `MODEL_PATH` | Path al archivo `.pkl` del modelo | `models/model.pkl` | No |
| `MODEL_VERSION` | Versión del modelo (informativo) | `latest` | No |
| `MLFLOW_TRACKING_URI` | URI del servidor MLflow | `http://localhost:5000` | Sí (para training) |
| `MLFLOW_EXPERIMENT_NAME` | Nombre del experimento en MLflow | `predmaint` | No |
| `MLFLOW_MODEL_NAME` | Nombre del modelo en MLflow Registry | `predmaint-classifier` | No |
| `PREFECT_API_URL` | URL de la API de Prefect (local o Cloud) | `http://localhost:4200/api` | Sí (para orquestación) |
| `PREFECT_API_KEY` | API key de Prefect Cloud | _(vacío)_ | Solo Prefect Cloud |
| `AWS_REGION` | Región de AWS | `us-east-1` | Sí (para deploy) |
| `AWS_ACCESS_KEY_ID` | Access key de AWS | _(vacío)_ | Solo setup local con AWS |
| `AWS_SECRET_ACCESS_KEY` | Secret key de AWS | _(vacío)_ | Solo setup local con AWS |
| `ECR_REPOSITORY` | Nombre del repositorio ECR | `predmaint-ml-platform` | Sí (para deploy) |
| `ECS_CLUSTER` | Nombre del cluster ECS | `predmaint-cluster` | Sí (para deploy) |
| `ECS_SERVICE` | Nombre del servicio ECS | `predmaint-api` | Sí (para deploy) |
| `S3_BUCKET` | Bucket S3 para datos y modelos | _(vacío)_ | Sí (para AWS) |
| `S3_RAW_DATA_PREFIX` | Prefix S3 para datos raw | `data/raw/` | No |
| `S3_PROCESSED_DATA_PREFIX` | Prefix S3 para datos procesados | `data/processed/` | No |
| `S3_MODEL_PREFIX` | Prefix S3 para modelos | `models/` | No |
| `STREAMLIT_API_URL` | URL de la API para el dashboard | `http://localhost:8000` | No |
| `DRIFT_THRESHOLD` | Fracción de features con drift que dispara reentrenamiento | `0.05` | No |
| `RETRAINING_TRIGGER` | Habilita reentrenamiento automático (`true`/`false`) | `true` | No |
| `REFERENCE_SPLIT_RATIO` | Fracción del dataset usada como referencia | `0.70` | No |
| `PRODUCTION_SAMPLE_START_ROW` | Fila desde la que empieza la muestra de producción | `7000` | No |

---

## 8. Instalación y setup

### Prerequisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gestor de dependencias y entornos virtuales)
- Docker y Docker Compose (para ejecución containerizada)
- AWS CLI configurado (solo para deploy en AWS)

Instalar `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 8.1 Setup local (desarrollo)

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/predmaint-ml-platform.git
cd predmaint-ml-platform

# 2. Instalar dependencias de desarrollo y configurar pre-commit
make setup
# Equivale a: uv sync --group dev && uv run pre-commit install

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env — mínimo: MLFLOW_TRACKING_URI

# 4. Descargar el dataset
mkdir -p data/raw
# Descargar ai4i2020.csv desde UCI y colocarlo en data/raw/
# URL: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

# 5. Iniciar MLflow localmente (en terminal separada)
make mlflow-ui
# → UI disponible en http://localhost:5000

# 6. Ejecutar el pipeline de entrenamiento
make train
# Equivale a: python src/data/pipeline.py && python src/training/train.py

# 7. Construir dataset de referencia para drift detection
uv run python src/monitoring/drift.py

# 8. Iniciar la API
make run
# → API en http://localhost:8000
# → Docs en http://localhost:8000/docs

# 9. (Opcional) Iniciar el dashboard Streamlit
make dashboard
# → Dashboard en http://localhost:8501
```

### 8.2 Setup con Docker

El `Dockerfile.api` usa un build multi-stage para minimizar el tamaño de la imagen final:

- **Stage builder**: instala dependencias con `uv` en un entorno virtual aislado
- **Stage runtime**: copia solo el entorno virtual y el código fuente; corre como usuario no-root (`appuser`)

```bash
# Build de la imagen
docker build -f Dockerfile.api -t predmaint-api:local .

# Correr el contenedor (requiere model.pkl entrenado previamente)
docker run -p 8000:8000 \
  -v ./models:/app/models \
  -e MODEL_PATH=models/model.pkl \
  predmaint-api:local

# Verificar health
curl http://localhost:8000/health
```

> **Nota:** El archivo `docker-compose.yml` con el stack completo (API + MLflow + Prefect + Grafana) está en el roadmap.

### 8.3 Setup en AWS

El deploy en AWS está automatizado vía GitHub Actions usando OIDC (sin credenciales estáticas en secrets).

**Configuración one-time en AWS (requerida una vez):**

1. Crear un OIDC Identity Provider en IAM:
   - Provider URL: `https://token.actions.githubusercontent.com`
   - Audience: `sts.amazonaws.com`

2. Crear un IAM Role (`github-actions-deploy`) con trust policy para el repositorio.

3. Adjuntar permisos al rol:
   - `AmazonEC2ContainerRegistryPowerUser`
   - `ecs:UpdateService`, `ecs:DescribeServices` sobre el cluster/servicio
   - `iam:PassRole` sobre `ecsTaskExecutionRole` y `ecsTaskRole`

4. Agregar el ARN del rol como GitHub Secret: `AWS_DEPLOY_ROLE_ARN`  <!-- pragma: allowlist secret -->

Con esta configuración, cada push a `main` dispara el workflow `deploy.yml` automáticamente.

---

## 9. Guía de desarrollo

### Correr tests

```bash
# Todos los tests
make test

# Con reporte de cobertura
uv run pytest --cov=src --cov-report=html tests/
# → Reporte en htmlcov/index.html

# Tests de un módulo específico
uv run pytest tests/test_api.py -v

# Tests con un patrón de nombre
uv run pytest -k "test_predict" -v
```

La suite de tests está diseñada para correr **sin dependencias externas** (sin MLflow, sin Prefect server, sin modelo entrenado). Los Prefect tasks se testean llamando al método `.fn()` que ejecuta la función subyacente sin el runtime de Prefect. Los modelos se mockean con `MagicMock` y `monkeypatch`.

### Correr linting y type checking

```bash
# Solo lint (sin modificar archivos — apto para CI)
make lint
# Equivale a: uv run ruff check src/

# Formatear código in-place
make format
# Equivale a: uv run ruff format src/ && uv run ruff check --fix src/

# Type checking
make typecheck
# Equivale a: uv run mypy src/

# Lint + typecheck juntos (recomendado antes de commit)
make check
```

**Configuración de ruff** (`pyproject.toml`):
- `line-length = 88` (compatible con black)
- `target-version = "py312"`
- Rules activas: `E` (pycodestyle), `F` (pyflakes), `I` (isort)

**Configuración de mypy:**
- `python_version = "3.12"`
- `namespace_packages = true` con `mypy_path = "src"`
- Módulos sin stubs (`evidently`, `prefect`, `mlflow.xgboost`) marcados con `ignore_missing_imports`

### Usar pre-commit hooks

```bash
# Instalar hooks (ya incluido en make setup)
uv run pre-commit install

# Correr manualmente sobre todos los archivos
uv run pre-commit run --all-files

# Correr un hook específico
uv run pre-commit run ruff-check --all-files
```

**Hooks configurados** (`.pre-commit-config.yaml`):

| Hook | Propósito |
|---|---|
| `trailing-whitespace`, `end-of-file-fixer` | Higiene básica de archivos |
| `check-yaml`, `check-toml`, `check-json` | Validación de sintaxis de configs |
| `check-added-large-files` | Evita commitear archivos > 500 KB (excluye notebooks: 1 MB) |
| `detect-secrets` | Detecta posibles credenciales hardcodeadas en el código |
| `bandit` | Análisis estático de seguridad en `src/` |
| `ruff-check`, `ruff-format` | Lint y formateo automático |
| `mypy` | Type checking en `src/` |
| `conventional-pre-commit` | Fuerza mensajes de commit en formato Conventional Commits |
| `uv-lock` | Verifica que `uv.lock` está sincronizado con `pyproject.toml` |

### Agregar un nuevo feature al pipeline

1. **Agregar la transformación** en `src/data/transformer.py` dentro de `build_features()`.
2. **Actualizar `FEATURE_COLUMNS`** si el feature es una columna del dataset original.
3. **Agregar tests** en `tests/test_transformer.py` para la nueva transformación.
4. **Verificar que el feature count** en `test_train.py::test_build_features_task_feature_count` sigue siendo correcto (actualmente `X.shape[1] == 8`).
5. **Re-entrenar el modelo**: `make train` — el nuevo modelo reemplaza `models/model.pkl`.
6. **Reconstruir el dataset de referencia**: `uv run python src/monitoring/drift.py` — necesario para que Evidently compare con el nuevo schema de features.

---

## 10. CI/CD — GitHub Actions

### Workflow CI (`ci.yml`)

Se activa en push a `main` o `develop`, y en pull requests contra `main`.

```
on:
  push:    branches: [main, develop]
  pull_request: branches: [main]
```

**Optimización de concurrencia:** Si se hace push mientras un workflow está corriendo en el mismo branch, el workflow anterior se cancela (`cancel-in-progress: true`).

| Job | Descripción | Dependencias |
|---|---|---|
| `lint` | Corre `ruff check`, `ruff format --check` y `mypy` sobre `src/` | — |
| `test` | Corre `pytest --cov=src --cov-report=xml` y sube coverage a Codecov | Necesita que `lint` pase |

**Instalación de dependencias:** Se usa `uv` con `setup-uv@v3` y caché habilitado (`enable-cache: true`) para que las dependencias no se resuelvan de cero en cada ejecución.

### Workflow Deploy (`deploy.yml`)

Se activa solo en push a `main`.

```
permissions:
  id-token: write   # Requerido para el exchange OIDC
  contents: read
```

**OIDC en lugar de credenciales estáticas:** No hay `AWS_ACCESS_KEY_ID` ni `AWS_SECRET_ACCESS_KEY` en los secrets del repositorio. GitHub Actions asume el rol de AWS directamente vía `aws-actions/configure-aws-credentials@v4` usando el token OIDC. Esto elimina la rotación manual de credenciales y el riesgo de exposición.

| Step | Descripción |
|---|---|
| `Checkout` | Obtiene el código fuente |
| `Configure AWS credentials (OIDC)` | Asume el rol IAM `github-actions-deploy` via OIDC |
| `Login to Amazon ECR` | Autentica Docker contra el registry privado |
| `Build, tag & push Docker image` | Construye la imagen con `Dockerfile.api`, la tagea con el SHA del commit y con `latest`, y la pushea |
| `Force new ECS deployment` | Llama a `aws ecs update-service --force-new-deployment` — ECS descarga la imagen `latest` y reemplaza los contenedores en ejecución |
| `Wait for service stability` | Espera hasta que el nuevo deployment está estable (`aws ecs wait services-stable`) — el workflow falla si el deploy no estabiliza |
| `Print deployed image` | Log del SHA de la imagen desplegada para trazabilidad |

**Tagging de imágenes:** Cada imagen se tagea con el SHA del commit (`${{ github.sha }}`), lo que permite hacer rollback a cualquier versión anterior cambiando el tag en la task definition de ECS.

**Ambiente `production`:** El job tiene `environment: production`, que puede configurarse en GitHub con reglas de aprobación manual antes de hacer deploy.

---

## 11. Monitoreo y UIs

### MLflow UI

```bash
make mlflow-ui
# → http://localhost:5000
```

Permite visualizar:
- Todos los runs de entrenamiento con sus métricas e hiperparámetros
- Comparación de runs (útil para ver la evolución tras reentrenamientos automáticos)
- Model registry con las versiones `Staging`, `Production`, `Archived`
- Artefactos del modelo (feature importance, plots, etc.)

### Prefect UI

```bash
uv run prefect server start
# → http://localhost:4200
```

Permite visualizar:
- Historial de ejecuciones de `training-pipeline` y `monitoring-pipeline`
- Estado de cada task dentro de un flow run (éxito, fallo, duración)
- Logs de cada task
- Scheduling de flows periódicos

Para ejecutar el monitoring pipeline manualmente:
```bash
uv run python -c "from training.train import monitoring_pipeline; monitoring_pipeline()"
```

### Streamlit Dashboard

```bash
make dashboard
# → http://localhost:8501
```

El dashboard incluye:
- **System Status:** Health check de la API en tiempo real
- **Real-time Prediction:** Formulario con inputs de sensores → llamada a `/predict` → resultado visual
- **Dataset Overview:** Métricas del dataset (registros, tasa de falla) y visualización de distribución de torque por tipo de falla

### Prometheus + Grafana

<!-- TODO: agregar configuración de Prometheus y dashboards de Grafana cuando se implemente el docker-compose completo -->

El stack de Prometheus + Grafana está planeado para monitorear:
- Latencia de requests a la API (p50, p95, p99)
- Throughput (requests/segundo)
- Tasa de errores (4xx, 5xx)
- Uso de CPU/memoria del contenedor ECS

### Reporte de Drift

```bash
# Generar reporte manualmente
uv run python src/monitoring/drift.py
# → Reporte en reports/drift/drift_report.html
```

El reporte HTML de Evidently muestra el análisis de drift feature por feature con distribuciones comparadas y métricas estadísticas.

---

## 12. Decisiones de diseño (ADRs)

### ADR-001: Prefect como orquestador de pipelines

**Contexto:** El pipeline de entrenamiento y el loop de monitoreo necesitan orquestación (dependencias entre steps, manejo de errores, observabilidad, reintentos).

**Decisión:** Prefect 3.x.

**Alternativas consideradas:**

| Alternativa | Por qué se descartó |
|---|---|
| **Apache Airflow** | Overhead operacional alto: requiere un scheduler, un webserver, workers, y una base de datos. Setup complejo para un equipo pequeño. |
| **Dagster** | Muy buena opción, pero más orientado a data assets con un paradigma diferente que añade complejidad conceptual para un pipeline relativamente simple. |
| **Luigi** | Maduro pero con menos features de observabilidad; el ecosistema está perdiendo tracción. |
| **Scripts Python simples** | No provee retry nativo, observabilidad, ni scheduling. |

**Justificación de Prefect:** API Python-nativa (decoradores `@flow` y `@task`), UI incluida, puede correr localmente sin servidor en modo `local`, y tiene integración con Prefect Cloud para ambientes productivos sin cambiar código.

---

### ADR-002: XGBoost sobre modelos de deep learning

**Contexto:** El target es clasificación binaria sobre datos tabulares con 10.000 registros y 8 features.

**Decisión:** XGBoost.

**Alternativas consideradas:**

| Alternativa | Por qué se descartó |
|---|---|
| **PyTorch Tabular / TabNet** | Deep learning en tabular con pocos datos y features típicamente underperforma vs. gradient boosting. Mayor complejidad de training e inferencia. |
| **Redes neuronales simples (MLP)** | Misma limitación: beneficio marginal con este tamaño de dataset. |
| **Random Forest** | Buena baseline, pero XGBoost consistentemente supera en benchmarks de clasificación tabular. |
| **Regresión logística** | No captura interacciones no-lineales importantes en datos de sensores. |

**Justificación de XGBoost:** State-of-the-art para clasificación tabular, manejo nativo del desbalance de clases via `scale_pos_weight`, feature importance interpretable, inference rápida, y tamaño de modelo pequeño (ideal para contenedor).

---

### ADR-003: FastAPI para serving

**Contexto:** El modelo necesita estar disponible como servicio HTTP para ser consumido por sistemas SCADA, dashboards y otros servicios.

**Decisión:** FastAPI + Uvicorn.

**Alternativas consideradas:**

| Alternativa | Por qué se descartó |
|---|---|
| **Flask** | Sin validación de tipos nativa, sin documentación auto-generada, async menos ergonómico. |
| **BentoML** | Excelente para serving de modelos, pero añade una abstracción adicional y dependencia pesada para un endpoint relativamente simple. |
| **Seldon / KFServing** | Overkill para este escenario; orientado a Kubernetes y clusters ML dedicados. |
| **Django REST Framework** | Overhead excesivo para una API de inference. |

**Justificación de FastAPI:** Validación automática con Pydantic, documentación OpenAPI auto-generada, soporte async nativo, typing fuerte, y excelente performance (comparado con Flask en benchmarks de async). El overhead de Pydantic es mínimo frente al tiempo de inference del modelo.

---

### ADR-004: uv como gestor de dependencias

**Contexto:** El proyecto necesita un gestor de dependencias reproducible y rápido.

**Decisión:** uv.

**Alternativas consideradas:** pip + requirements.txt, poetry, conda.

**Justificación:** uv resuelve el entorno virtual y las dependencias en milisegundos (vs. minutos con pip/poetry), tiene soporte nativo para `pyproject.toml` y lockfiles, y simplifica el Dockerfile (una sola herramienta para todo el ciclo de vida de dependencias). El `uv.lock` garantiza reproducibilidad exacta en CI y en producción.

---

### ADR-005: OIDC en lugar de credenciales estáticas en CI/CD

**Contexto:** El workflow de deploy necesita autenticarse contra AWS.

**Decisión:** OIDC (OpenID Connect) via `aws-actions/configure-aws-credentials@v4`.

**Alternativa descartada:** `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` como GitHub Secrets estáticos.

**Justificación:** Las credenciales estáticas tienen vida larga, deben rotarse manualmente, y si un secret se expone (leak de logs, fork malicioso), el impacto es alto. Con OIDC, GitHub Actions obtiene un token de corta duración en cada workflow run asumiendo un rol IAM con permisos mínimos. No hay credenciales persistentes que rotar o que puedan filtrarse.

---

## 13. Roadmap / TODO

### Corto plazo

- [ ] `docker-compose.yml` completo con API + MLflow + Prefect + Prometheus + Grafana
- [ ] Integración de Great Expectations para validación de datos rica (rangos, distribuciones, unicidad)
- [ ] Endpoint `/metrics` con exposición de métricas en formato Prometheus
- [ ] Completar métricas reales del modelo después de primera ejecución de producción
- [ ] Tests de integración end-to-end con el pipeline completo

### Mediano plazo

- [ ] Implementación de Terraform para IaC (VPC, ECR, ECS, IAM roles)
- [ ] Estrategia de A/B testing para comparar modelos en producción antes de promoverlos
- [ ] Umbral de clasificación ajustable en `/predict` (permite trade-off precision/recall según el contexto operacional)
- [ ] Alertas automáticas (Slack/email) cuando se detecta drift o cuando el reentrenamiento falla
- [ ] Versionado de features con un feature store ligero

### Largo plazo

- [ ] Soporte multi-modelo (distintos clasificadores por tipo de máquina)
- [ ] Explicabilidad de predicciones con SHAP values en el response de la API
- [ ] Pipeline de backtesting para evaluar degradación histórica del modelo
- [ ] Integración con sistemas SCADA reales vía MQTT o OPC-UA

---

## 14. Referencias

### Dataset

- Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. IEEE ICMLA 2020. [DOI: 10.1109/ICMLA51294.2020.00203](https://doi.org/10.1109/ICMLA51294.2020.00203)
- [AI4I 2020 Dataset — UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

### Herramientas y frameworks

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prefect 3.x Documentation](https://docs.prefect.io/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [uv Documentation](https://docs.astral.sh/uv/)

### MLOps y buenas prácticas

- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NIPS 2015.
- [Google MLOps Maturity Model](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [AWS Well-Architected Framework — Machine Learning Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/welcome.html)

### Seguridad en CI/CD

- [GitHub Actions: Hardening security for OIDC](https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [AWS: Configure GitHub OIDC provider](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_create_oidc.html)
