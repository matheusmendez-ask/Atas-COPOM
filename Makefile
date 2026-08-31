.PHONY: help setup up down ingest process index run-pipeline test lint format clean

PYTHON ?= python
VENV_PYTHON ?= .venv/Scripts/python
ifeq ($(OS),Windows_NT)
    VENV_PYTHON = .venv/Scripts/python
else
    VENV_PYTHON = .venv/bin/python
endif

help:
	@echo "Enterprise COPOM RAG Lakehouse & Data Observability Pipeline"
	@echo "------------------------------------------------------------"
	@echo "Available commands:"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo "  make up           - Start Docker containers (Qdrant & Arize Phoenix)"
	@echo "  make down         - Stop Docker containers"
	@echo "  make ingest       - Run Bronze layer ingestion"
	@echo "  make process      - Run Silver layer sanitization & chunking"
	@echo "  make index        - Run Gold layer vector indexing into Qdrant"
	@echo "  make run-pipeline - Run end-to-end pipeline (Bronze -> Silver -> Gold)"
	@echo "  make test         - Run test suite with pytest & coverage"
	@echo "  make lint         - Run code linting with Ruff"
	@echo "  make format       - Format codebase with Ruff"
	@echo "  make clean        - Remove caches and temporary files"

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"

up:
	docker compose up -d

down:
	docker compose down

ingest:
	$(PYTHON) -m src.pipeline ingest

process:
	$(PYTHON) -m src.pipeline transform

index:
	$(PYTHON) -m src.pipeline index

run-pipeline:
	$(PYTHON) -m src.pipeline run-all

test:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing tests/

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/

clean:
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache build dist *.egg-info
