# Config
UV        := uv run
PYTHON    := $(UV) python
SRC       := src
TESTS     := tests
PORT      := 8000

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'


# Setup
.PHONY: setup
setup: ## Install dev dependencies and set up pre-commit
	uv sync --group dev
	uv run pre-commit install
	@echo "Environment ready"


# Code Quality
.PHONY: lint format typecheck check

lint: ## Check style without modifying files (CI-friendly)
	$(UV) ruff check $(SRC)

format: ## Format code in-place
	$(UV) ruff format $(SRC)
	$(UV) ruff check --fix $(SRC)

typecheck: ## Run mypy on src/
	$(UV) mypy $(SRC)

check: lint typecheck ## Run lint + typecheck (ideal before committing)


# Tests
.PHONY: test test-cov

test: ## Run all tests
	$(UV) pytest $(TESTS)

test-cov: ## Run tests with coverage report
	$(UV) pytest $(TESTS) --cov=$(SRC) --cov-report=term-missing


# Data And Training Pipeline
.PHONY: process-data train

process-data: ## Run ingestion and feature engineering pipeline
	$(PYTHON) $(SRC)/data/pipeline.py

train: process-data ## Process data and train the model
	$(PYTHON) $(SRC)/training/train.py


# Local Services
.PHONY: run mlflow-ui prefect dashboard drift monitor

run: ## Start the API in development mode (hot reload)
	$(UV) uvicorn $(SRC).api.main:app --reload --port $(PORT)

mlflow-ui: ## Open MLflow UI at http://localhost:5000
	$(UV) mlflow ui

prefect: ## Start the Prefect server at http://localhost:4200
	$(UV) prefect server start

dashboard: ## Start the Streamlit dashboard at http://localhost:8501
	$(UV) streamlit run $(SRC)/dashboard/app.py

drift: ## Generate Evidently drift report → reports/drift/drift_report.html
	$(PYTHON) $(SRC)/monitoring/drift.py

monitor: ## Run the monitoring + auto-retraining pipeline
	$(PYTHON) -c "from src.training.train import monitoring_pipeline; monitoring_pipeline()"


# Cleanup
.PHONY: clean reset

clean: ## Remove Python, linter, and notebook caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf .ipynb_checkpoints notebooks/.ipynb_checkpoints
	rm -rf reports/coverage
	@echo "Cache cleaned"

reset: clean ## Reset project to initial state (destructive)
	@echo "WARNING: This will delete models, processed data, MLflow runs and the DB. Continue? [y/N] " \
		&& read ans && [ $${ans:-N} = y ]
	rm -rf models/*.pkl
	rm -rf data/processed/
	rm -rf mlruns/ mlflow.db mlartifacts/
	rm -rf reports/
	rm -rf .venv/
	@echo "Project reset — run 'make setup' to reinitialize"
