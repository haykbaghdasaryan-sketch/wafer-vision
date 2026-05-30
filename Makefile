.PHONY: install download preprocess train extract evaluate verify-reproducibility preflight app test lint typecheck format clean

# Python interpreter
PYTHON ?= python
PIP ?= pip

# Default configuration
CONFIG ?= configs/default.yaml
BACKBONE ?= resnet50
MODE ?= finetune

install:
	$(PIP) install -e ".[dev]"
	pre-commit install

download:
	@echo "Download the WM-811K dataset (LSWMD.pkl) and place it in data/raw/"
	@mkdir -p data/raw
	@echo "Dataset URL: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map"

preprocess:
	$(PYTHON) -m src.data.preprocessing --config-path ../$(CONFIG)

train:
	$(PYTHON) -m src.training.trainer --config-path ../$(CONFIG) backbone=$(BACKBONE) training.mode=$(MODE)

extract:
	$(PYTHON) scripts/extract_embeddings.py --backbone $(BACKBONE) --mode $(MODE)

evaluate:
	$(PYTHON) scripts/run_benchmark.py

verify-reproducibility:
	$(PYTHON) scripts/verify_reproducibility.py --backbone $(BACKBONE)

preflight:
	$(PYTHON) scripts/preflight_check.py

app:
	streamlit run app/app.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ app/ tests/ scripts/

typecheck:
	mypy src/ app/

format:
	ruff format src/ app/ tests/ scripts/
	ruff check --fix src/ app/ tests/ scripts/

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf data/processed/
	rm -rf outputs/
	rm -rf logs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
