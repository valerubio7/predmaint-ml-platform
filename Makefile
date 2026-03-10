.PHONY: setup test lint clean run process-data train deploy deploy-skip-train

setup:
	uv sync --group dev
	uv run pre-commit install

test:
	uv run pytest

lint:
	uv run ruff check src/
	uv run ruff format src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf .ipynb_checkpoints notebooks/.ipynb_checkpoints
	rm -rf models/

process-data:
	uv run python src/data/pipeline.py

train:
	uv run python src/pipelines/training_flow.py

mlflow-ui:
	uv run mlflow ui

run:
	uv run uvicorn src.api.main:app --reload

## Deploy targets (require active AWS credentials)
# Full pipeline: train → build → push to ECR → force ECS redeploy → verify /predict
deploy:
	bash scripts/deploy.sh

# Re-deploy using an existing model.pkl without retraining
deploy-skip-train:
	bash scripts/deploy.sh --skip-train
