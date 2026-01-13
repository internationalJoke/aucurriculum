# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**aucurriculum** is a Curriculum Learning Toolkit for Deep Learning Tasks built on top of [autrainer](https://github.com/autrainer/autrainer). It implements curriculum learning methodologies that progressively adjust training sample difficulty to improve model performance.

## Development Commands

```bash
# Install dependencies (Poetry required)
poetry install

# Run all tests with coverage
pytest

# Run a specific test file
pytest tests/test_pacing.py

# Run a specific test class or method
pytest tests/test_pacing.py::TestPacingFunctions::test_pacing_size

# Linting and formatting (ruff)
ruff check aucurriculum/
ruff format aucurriculum/

# Spell checking
codespell aucurriculum/ aucurriculum-configurations/ docs/source/

# Run pre-commit hooks
pre-commit run --all-files

# Build documentation
cd docs && make html
```

## CLI Commands

The main entry point is `aucurriculum`. Key commands:

```bash
aucurriculum curriculum -cn <config>  # Run scoring function computation
aucurriculum train -cn <config>       # Training with curriculum (via autrainer)
aucurriculum create curriculum        # Create curriculum configs interactively
aucurriculum list curriculum          # List available curriculum configs
aucurriculum postprocess              # Aggregate and visualize results
```

## Architecture

### Core Components

The toolkit has two main conceptual parts that work together:

1. **Scoring Functions** (`aucurriculum/curricula/scoring/`) - Compute sample difficulty
2. **Pacing Functions** (`aucurriculum/curricula/pacing/`) - Control dataset size during training

### Manager Classes

- **CurriculumScoreManager** (`curricula/curriculum_score_manager.py`) - Orchestrates scoring function lifecycle (preprocess → run → postprocess), manages multi-run scenarios
- **CurriculumPaceManager** (`curricula/curriculum_pace_manager.py`) - Integrates with autrainer's trainer, loads pre-computed scores, dynamically adjusts dataset size per iteration

### Extension Points

To implement custom scoring/pacing:

1. **Custom Scoring**: Extend `AbstractScore` and implement the `run()` method
2. **Custom Pacing**: Extend `AbstractPace` and implement `get_dataset_size(iteration)`

### Configuration System

Uses Hydra with layered config directories:
1. `conf/` (user overrides)
2. `aucurriculum-configurations/` (packaged defaults)
3. `autrainer-configurations/` (parent framework)

Key config structure in YAML:
```yaml
curriculum:
  scoring:
    type: CELoss          # Scoring function class
    run_name: training_run
  pacing:
    type: Polynomial
    initial_size: 0.1
    final_iteration: 0.5
```

### Data Flow

1. Train models with autrainer (full dataset)
2. Run `aucurriculum curriculum` to compute sample difficulty scores → outputs `scores.csv`
3. Train with curriculum config → pacing function dynamically adjusts dataset size based on scores

## Code Style

- Line length: 79 characters
- Python 3.9+ target
- Ruff for linting (rules: E4, E7, E9, F, N801, I)
- 2 blank lines after imports
- Google-style docstrings with type hints
