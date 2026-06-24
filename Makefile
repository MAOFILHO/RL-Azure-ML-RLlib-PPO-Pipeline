.PHONY: setup check provision smoke-test train curriculum hpt trajectories assess deploy teardown run-all test lint clean help

PYTHON ?= python3
VENV ?= .venv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtual environment and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install -r requirements-dev.txt
	$(VENV)/bin/pip install -e .
	@echo "\n✓ Setup complete. Activate with: source $(VENV)/bin/activate"

check: ## Step 0: Verify prerequisites
	$(VENV)/bin/plato check

provision: ## Step 1: Create Azure resources
	$(VENV)/bin/plato provision

smoke-test: ## Step 3: Local training smoke test
	$(VENV)/bin/plato smoke-test

train: ## Step 4: Submit basic training to AML
	$(VENV)/bin/plato train

curriculum: ## Step 5: Submit curriculum learning to AML
	$(VENV)/bin/plato curriculum

hpt: ## Step 6: Submit hyperparameter tuning to AML
	$(VENV)/bin/plato hpt

trajectories: ## Step 7: Submit trajectory logging to AML
	$(VENV)/bin/plato trajectories

assess: ## Step 8: Submit custom assessment to AML
	$(VENV)/bin/plato assess

deploy: ## Step 9: Deploy agent to AML endpoint
	$(VENV)/bin/plato deploy

teardown: ## Step 10: Delete all Azure resources
	$(VENV)/bin/plato teardown

run-all: ## Run complete pipeline (Steps 0-9)
	$(VENV)/bin/plato run-all

test: ## Run unit tests
	$(VENV)/bin/pytest tests/ -v --tb=short

lint: ## Run linting (ruff + mypy)
	$(VENV)/bin/ruff check src/ tests/
	$(VENV)/bin/mypy src/plato_rl/ --ignore-missing-imports

clean: ## Remove local artifacts and generated files
	rm -rf .generated/ outputs/ mlruns/ ray_results/
	rm -rf __pycache__ src/plato_rl/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned local artifacts"
