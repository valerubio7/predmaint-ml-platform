.PHONY: setup test lint clean run process-data train

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

process-data:
	uv run python src/data/explore.py

train:
	uv run python src/models/train.py

run:
	uv run uvicorn src.api.main:app --reload
