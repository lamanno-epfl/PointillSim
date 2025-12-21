# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PointillSim is a Python framework for simulating spatial transcriptomics data. It generates synthetic spatial datasets with ground-truth cell type annotations for benchmarking spatial analysis tools.

## Build & Development Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v --tb=short

# Run single test file
python -m pytest tests/test_pointillsim.py -v

# Run tests with coverage (Python 3.11+)
python -m pytest tests/ -v --cov=pointillsim --cov-report=term

# Run visual output tests (generates plots)
pytest tests/test_pointillsim.py -v --visual

# Lint and format
ruff check pointillsim/
ruff format pointillsim/
```

## Architecture

The simulation follows this data flow:
```
TissueCellTypes (expression) + CellTypesProperties (morphology)
    → FOVDistribution (element placement)
    → HistologicalElement(s) (polygons + cell grids + rules)
    → FOV (centroids + probabilities) → FOV.realization()
    → HybISS_Setup (Poisson observation model)
    → cells_df, dots_df, ground_truth
```

### Core Modules

**`core/`** - Foundation classes
- `TissueCellTypes`: Gene expression profiles per cell type
- `TissueSlice`, `RegionSpec`: Tissue regions with element distributions
- `FOV`, `FOVDistribution`: Field of view with stochastic element placement

**`elements/`** - Tissue structure components (Template Method pattern)
- `HistologicalElement`: Abstract base with `generate()` method
- `FrameWideElement`: Full-FOV backgrounds
- `VacuolatedStructure`: Circular structures (glands, acini)

**`rules/`** - Cell type assignment strategies (Strategy pattern)
- `CellTypeRuleBase`: Abstract base class
- Implementations: `RandomCellTypeRule`, `SingleTypeRule`, `MixOfNCellTypesRule`, `ProbabilityNodeFieldRule`, `DistanceBasedRule`, `CompositeRule`, `DeterministicNeighborAssignment`
- Rules are composable and applied sequentially

**`experiment/`** - Observation models
- `HybISS_Setup`: Poisson-based transcript sampling
- `CellTypesProperties`: Cell morphology (size, shape)
- `TechnologyPreset`: Platform presets (HybISS, Cartana, MERFISH, Visium, Xenium)

**`io/`** - Data I/O with AnnData/SpatialData support

**`viz/`** - Plotting functions (`plot_fov`, `plot_expression_matrix`)

### Key Design Patterns

1. **Template Method**: `HistologicalElement.generate()` uses deepcopy template
2. **Strategy**: `CellTypeRuleBase` with interchangeable implementations
3. **Lazy Evaluation**: `FOV.class_instance_one_hot` samples on first access
4. **Separation of concerns**: Ground truth (cell positions, probabilities) vs observations (transcript counts)

### Type Conventions

- Full type hints using `numpy.typing.NDArray`
- Dataclasses for presets and configurations
- Python 3.9+ compatible syntax

## Code Style

- Line length: 100 characters
- Linter: Ruff
- Target: Python 3.9-3.12
