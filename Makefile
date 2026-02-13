# Makefile for pokemon-card-valuation project
# Usage: make <target>

.PHONY: help install install-dev clean lint format test test-coverage train-vision train-market train-fusion evaluate docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install          - Install package and dependencies"
	@echo "  install-dev      - Install with development dependencies"
	@echo "  clean            - Remove build artifacts and cache"
	@echo "  lint             - Run code quality checks"
	@echo "  format           - Format code with black and isort"
	@echo "  test             - Run unit tests"
	@echo "  test-coverage    - Run tests with coverage report"
	@echo "  train-vision     - Train vision module"
	@echo "  train-market     - Train market module"
	@echo "  train-fusion     - Train fusion network"
	@echo "  evaluate         - Run full evaluation suite"
	@echo "  docs             - Generate documentation"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,viz,tracking]"

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage

# Code Quality
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports
	pylint src/ --max-line-length=100 --disable=C0111,R0913

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black

# Testing
test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Training
train-vision:
	python scripts/training/train_vision.py --config configs/vision/cnn_config.yaml

train-market:
	python scripts/training/train_market.py --config configs/market/temporal_config.yaml

train-fusion:
	python scripts/training/train_fusion.py --config configs/fusion/fusion_config.yaml

# Evaluation
evaluate:
	python scripts/evaluation/run_evaluation.py --experiment fusion_v1

ablation:
	python scripts/evaluation/run_ablation.py --output results/ablation/

# Documentation
docs:
	cd docs && sphinx-build -b html . _build/html

# Data collection
collect-data:
	python scripts/data_collection/scrape_ebay.py --config configs/data/data_config.yaml

preprocess:
	python scripts/preprocessing/prepare_data.py --input data/raw --output data/processed

# Quick pipeline
pipeline: collect-data preprocess train-vision train-market train-fusion evaluate

# Docker (optional)
docker-build:
	docker build -t pokemon-valuation:latest .

docker-run:
	docker run -it --gpus all -v $(PWD)/data:/app/data pokemon-valuation:latest
