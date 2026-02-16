# ═══════════════════════════════════════════════════════════════════════
#  Transformer from Scratch — Makefile
# ═══════════════════════════════════════════════════════════════════════

.PHONY: install train train-ddp test lint translate clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -r requirements.txt
	pip install -e ".[dev,eval]"

train: ## Train on single GPU
	python -m transformer.train --config configs/default.yaml

train-ddp: ## Train with DDP (auto-detect GPUs)
	bash scripts/train.sh

test: ## Run all unit tests
	python -m pytest tests/ -v

lint: ## Run ruff linter
	ruff check transformer/ tests/

translate: ## Interactive translation mode (requires trained model)
	python -m transformer.translate --checkpoint weights/tmodel_24.pt --interactive

clean: ## Remove generated files (weights, logs, tokenizers)
	rm -rf weights/ runs/ tokenizer_*.json __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
