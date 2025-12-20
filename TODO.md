# PointillSim Development TODO

## High Priority - Core Functionality

### Missing Histological Elements
- [ ] **LinearLumenStructure**: Tube-like structures (vessels, ducts) - mentioned in README but not implemented
- [ ] **LayeredElement**: For simulating stratified tissue layers (e.g., cortical layers, epidermis)
- [ ] **BranchingStructure**: For tree-like structures (e.g., ductal networks, vasculature)

### Cell Type Rules
- [ ] **DistanceBasedRule**: Assign cell types based on distance from element boundary or center
- [ ] **LayerRule**: Assign types based on radial position (for concentric layer patterns)
- [ ] **GradientRule**: Create smooth linear gradients across elements
- [ ] **CompositeRule**: Explicit class for combining multiple rules with configurable weights

### FOV and Expression
- [ ] **FOV.add_noise()**: Method to add realistic noise to cell positions
- [ ] **FOV.subsample()**: Method to randomly subsample cells (for sparse simulations)
- [ ] **TissueCellTypes.load_from_csv()**: Load real expression profiles from file
- [ ] **TissueCellTypes.load_from_anndata()**: Load expression profiles from AnnData objects

## Medium Priority - Usability Improvements

### Visualization
- [ ] **plot_fov()**: Convenience function to visualize a FOV with multiple panels
- [ ] **plot_tissue_slice()**: Visualization for TissueSlice with region boundaries
- [ ] **plot_expression_matrix()**: Heatmap visualization for TissueCellTypes

### Serialization & I/O
- [ ] **FOV.to_anndata()**: Export FOV as AnnData object for downstream analysis
- [ ] **FOV.to_spatialdata()**: Export to SpatialData format
- [ ] **TissueSlice.save()** / **TissueSlice.load()**: Pickle or HDF5 serialization
- [ ] **HybISS_Setup.save_config()**: Save experiment configuration for reproducibility

### Configuration
- [ ] **Config dataclass**: Centralized configuration for simulation parameters
- [ ] **from_config()** factory methods for main classes
- [ ] **YAML/JSON config loading**: Load simulation setup from config files

## Lower Priority - Advanced Features

### Noise & Realism
- [ ] **BackgroundNoise**: Add uniform background dots (false positives)
- [ ] **DropoutModel**: Model gene-specific dropout rates
- [ ] **CellSizeVariationByType**: Different size distributions per cell type
- [ ] **SpatialNoise**: Add spatial autocorrelation to cell positions

### Multi-FOV Features
- [ ] **TissueSlice.generate_multiple()**: Generate multiple realizations
- [ ] **FOVDistribution.generate_batch()**: Parallel FOV generation
- [ ] **ConsistentTiling**: Ensure cells at tile boundaries have consistent properties

### Documentation
- [ ] Jupyter notebook tutorials for common use cases
- [ ] API reference documentation (Sphinx)
- [ ] Example gallery with different tissue types

## Code Quality

### Refactoring
- [x] Split `data_simulations.py` into focused modules (see module structure below)
- [ ] Add type hints throughout codebase
- [ ] Add input validation to constructors
- [ ] Remove duplicate `SingleTypeRule` class definition (lines 339 and 1469)
- [ ] Fix `FrameWideUpdater` one-time-use limitation

### Testing
- [ ] Add property-based tests (hypothesis)
- [ ] Add integration tests for full simulation pipelines
- [ ] Add performance benchmarks
- [ ] Test edge cases (empty FOVs, single cell, etc.)

## Module Structure (After Refactoring)

```
pointillsim/
├── __init__.py           # Public API exports
├── core/
│   ├── __init__.py
│   ├── fov.py            # FOV, FOVDistribution
│   └── tissue.py         # TissueCellTypes, TissueSlice, RegionSpec
├── elements/
│   ├── __init__.py
│   ├── base.py           # HistologicalElement
│   ├── frame.py          # FrameWideElement, FrameWideUpdater
│   └── structures.py     # VacuolatedStructure, (future: LinearLumen, etc.)
├── rules/
│   ├── __init__.py
│   ├── base.py           # CellTypeRuleBase
│   ├── random.py         # RandomCellTypeRule, MixOfNCellTypesRule
│   ├── spatial.py        # ProbabilityNodeFieldRule, SingleTypeRule
│   └── neighbor.py       # DeterministicNeighborAssignment
├── experiment/
│   ├── __init__.py
│   ├── hybiss.py         # HybISS_Setup
│   ├── properties.py     # CellTypesProperties
│   └── transfer.py       # TransferFunctionBase, AffineNonNegTransfer, IdentityTransfer
├── utils/
│   ├── __init__.py
│   ├── geometry.py       # generate_uniform_points_in_circle, generate_points_asin_cell
│   ├── math.py           # lognorm_params_to_mean_std, intuitive_rand_lognormal
│   ├── interpolation.py  # LinearNDInterpolatorExt
│   └── encoding.py       # one_hot_encode_array, unfold_int_matrix
└── io/
    ├── __init__.py
    └── dataset.py        # generate_dataset, load_data
```
