# Contributing to Transformer from Scratch

Thank you for your interest in contributing! This project is a from-scratch
implementation of the Transformer architecture for educational purposes.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/transformer-from-scratch.git
   cd transformer-from-scratch
   ```
3. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Run tests: `python -m pytest tests/ -v`
4. Run linter: `ruff check transformer/ tests/`
5. Submit a Pull Request

## Code Style

- Follow PEP 8 (enforced by `ruff`)
- Max line length: 100 characters
- Use NumPy-style docstrings for all public functions and classes
- Add type hints to all function signatures
- Add shape annotations as comments for tensor operations

## What We Accept

- Bug fixes
- Documentation improvements
- New training features (e.g., beam search, learning rate finders)
- Performance optimizations
- Additional model architectures (decoder-only, etc.)

## What We Don't Change

The core model components (`Linear`, `Embedding`, `RMSNorm`, etc.) are
intentionally written from scratch without using `nn.Linear`, `nn.Embedding`,
etc. This is the whole point of the project. Please don't submit PRs that
replace these with PyTorch built-ins.

## Questions?

Open an issue and we'll be happy to help!
