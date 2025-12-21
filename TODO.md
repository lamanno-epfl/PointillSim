# PointillSim Development TODO

## High Priority - Documentation & Usability

### Documentation (Critical for Adoption)
- [x] **Update notebooks**: Make notebooks more explanatory by alternating explanations and plots of specific features, clearly describing the logic of the approach
- [x] **Sphinx documentation**: Build comprehensive API documentation using Sphinx
- [x] **Sample datasets**: Include 3-4 pre-generated sample datasets in compressed format that can be loaded directly via import
- [x] **Getting Started Notebook**: Basic usage tutorial
  - Create tissue, define FOV distribution, generate and visualize
  - Simple HybISS experiment simulation

### Tissue Descriptions & Examples
- [x] **Tissue description folder**: Create a folder with plain-text procedural descriptions of how to create FOVs for specific real tissues (e.g., colon crypts, cortical layers, skin epidermis). These descriptions will:
  - Serve as reference to verify we have all necessary primitives for detailed tissue simulation
  - Later be converted to working code showcased in notebooks
- [x] **Tissue-specific notebooks**: Create notebooks demonstrating simulations of specific real tissues
- [x] **Custom Rules Notebook**: Demonstrate all cell type assignment rules
  - Examples of each rule type with visualizations
  - Show how to combine rules for complex patterns

### Missing Histological Elements
- [x] **LayeredElement**: For simulating stratified tissue layers (e.g., cortical layers, epidermis)
- [x] **BranchingStructure**: For tree-like structures (e.g., ductal networks, vasculature)

## Medium Priority - Core Features

### Simulation Modes
- [x] **Two Modes of Operation**: Support both random FOV generation and larger tissue simulation
  - **Mode 1 (current)**: Random FOV generation with stochastic element placement
  - **Mode 2 (new)**: Larger tissue simulation where:
    - A large tissue slice is generated once with explicit placement of structures
    - Multiple FOVs are extracted as crops from the larger tissue
    - Enables spatial coherence across multiple FOVs
    - Supports reproducible tile-based simulations
- [x] **TissueSlice.extract_fov(x, y, size)**: Extract a FOV from a specific location

### Visualization
- [x] **plot_tissue_slice()**: Visualization for TissueSlice with region boundaries
- [x] **plot_probability_field()**: Visualization for ProbabilityNodeFieldRule fields
  - Render the underlying probability/logits field as a heatmap or contour plot
  - Show node locations and interpolated probabilities
  - Support for multi-class probability visualization
  - Overlay cell positions with assigned types to verify correctness

### Abstract Structures & Tissue Variety
- [ ] **More Abstract Structure Classes**: Expand the variety of histological elements
  - **FibrillarStructure**: For collagen bundles, muscle fibers, nerve tracts
  - **ClusterElement**: Groups of cells with shared properties (e.g., lymphoid aggregates)
  - **GlandularUnit**: Acinar/tubular structures common in many tissues
  - **InterfaceElement**: For modeling tissue boundaries and transition zones
  - **StromalElement**: Background connective tissue with specific properties

### Serialization & I/O
- [x] **TissueSlice.save()** / **TissueSlice.load()**: Pickle or HDF5 serialization
- [x] **HybISS_Setup.save_config()**: Save experiment configuration for reproducibility

### Configuration
- [x] **Config dataclass**: Centralized configuration for simulation parameters
- [ ] **from_config()** factory methods for main classes
- [x] **YAML/JSON config loading**: Load simulation setup from config files

## Lower Priority - Advanced Features

### Batch Effects & Technical Variation
- [ ] **BatchEffectModel**: Class for simulating technical batch effects
  - Per-FOV systematic shifts in expression levels
  - Gene-specific and global batch effects
  - Configurable effect magnitude distributions
- [ ] **TechnicalNoise**: Model sources of technical variation
  - Amplification efficiency variation
  - Optical field non-uniformity (vignetting)
  - Focus-dependent detection efficiency

### Covariates & Effect Control
- [ ] **CovariateSystem**: Framework for defining and controlling simulation covariates
  - Define covariates (continuous or categorical) that influence simulation
  - Examples: tissue region, distance from landmark, local cell density
  - Covariates can be sampled or specified explicitly
- [ ] **EffectController**: Control specific effects in the simulation
  - Enable/disable specific sources of variation
  - Set effect sizes for controlled experiments
  - Useful for generating training data with known ground truth effects
- [ ] **CovariateBasedRule**: Cell type rules that depend on covariates
  - Assign cell types based on covariate values
  - Example: cell type probabilities vary with distance from a structure
  - Enables complex spatial patterns tied to measurable covariates

### Simulation Design & Control Flow
- [ ] **SimulationDesign**: High-level class specifying simulation parameters
  - Define what covariates to vary and their ranges
  - Specify number of replicates per condition
  - Control which effects are active
- [ ] **run_simulation_design()**: Execute a design specification
  - Generate all FOVs according to design
  - Return organized dataset with metadata
  - Support for factorial and fractional factorial designs
- [ ] **DesignMatrix**: Track covariate values across generated FOVs

### Difficulty Scoring & Validation
- [ ] **DifficultyScorer**: Module for assessing simulation difficulty
  - Use XGBoost or similar to estimate how hard the classification task is
  - Metrics: class separability, spatial pattern complexity
  - Compare simulated data difficulty to real data benchmarks
- [ ] **ValidationMetrics**: Compare simulated vs. real data distributions
  - Expression profile similarity
  - Spatial statistics comparison
  - Cell type proportion calibration

### Noise & Realism
- [ ] **BackgroundNoise**: Add uniform background dots (false positives)
- [ ] **DropoutModel**: Model gene-specific dropout rates
- [ ] **CellSizeVariationByType**: Different size distributions per cell type
- [ ] **SpatialNoise**: Add spatial autocorrelation to cell positions

### Multi-FOV Features
- [ ] **TissueSlice.generate_multiple()**: Generate multiple realizations
- [ ] **FOVDistribution.generate_batch()**: Parallel FOV generation
- [ ] **ConsistentTiling**: Ensure cells at tile boundaries have consistent properties

### Showcase Notebooks (Advanced)
- [ ] **Advanced Structures Notebook**: Complex tissue simulations
  - Multiple overlapping structures
  - Layered and nested elements
- [ ] **Benchmark Notebook**: Compare simulated vs. real data statistics

## Code Quality

### Refactoring
- [x] Fix `FrameWideUpdater` one-time-use limitation

### Testing
- [x] Add property-based tests (hypothesis)
- [x] Add integration tests for full simulation pipelines
- [ ] Add performance benchmarks
- [ ] Test edge cases (empty FOVs, single cell, etc.)

### CI & Testing
- [x] **Pre-commit Hooks**: Code quality checks
  - Linting (ruff/flake8)
  - Formatting (black)
  - Type checking (mypy)

---

## Completed

### Histological Elements
- [x] **LinearLumenStructure**: Tube-like structures (vessels, ducts)

### Cell Type Rules
- [x] **DistanceBasedRule**: Assign cell types based on distance from element boundary or center
- [x] **LayerRule**: Assign types based on radial position (for concentric layer patterns)
- [x] **GradientRule**: Create smooth linear gradients across elements
- [x] **CompositeRule**: Explicit class for combining multiple rules with configurable weights

### FOV and Expression
- [x] **FOV.add_noise()**: Method to add realistic noise to cell positions
- [x] **FOV.subsample()**: Method to randomly subsample cells (for sparse simulations)
- [x] **TissueCellTypes.load_from_csv()**: Load real expression profiles from file
- [x] **TissueCellTypes.load_from_anndata()**: Load expression profiles from AnnData objects

### Technology Presets
- [x] **TechnologyPreset base class**: Abstract class representing a spatial transcriptomics technology
- [x] **HybISSPreset**, **CartanaPreset**, **MerfishPreset**, **TenXVisiumPreset**, **TenXXeniumPreset**
- [x] **Integration with HybISS_Setup**: Allow passing TechnologyPreset to configure experiment
- [x] **Technology-specific transfer functions**: Each preset defines its own detection model
- [x] **Gene panel validation**: Enforce gene count limits per technology

### Polygon Smoothing & Visual Realism
- [x] **Bevel/Node Smoothing**: Chaikin's corner cutting algorithm, configurable smoothing
- [x] **Edge Softening**: Add natural variation to polygon edges

### Visualization
- [x] **plot_fov()**: Convenience function to visualize a FOV with multiple panels
- [x] **plot_expression_matrix()**: Heatmap visualization for TissueCellTypes

### Serialization & I/O
- [x] **FOV.to_anndata()**: Export FOV as AnnData object for downstream analysis
- [x] **FOV.to_spatialdata()**: Export to SpatialData format

### CI & Testing
- [x] **GitHub Actions CI**: Automated testing pipeline (Python 3.9-3.12, coverage reporting)

### Refactoring
- [x] Split `data_simulations.py` into focused modules
- [x] Add type hints throughout codebase
- [x] Add input validation to constructors
- [x] Remove duplicate `SingleTypeRule` class definition

## Module Structure

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
│   ├── base.py           # CellTypeRuleBase, DummyRule
│   ├── random.py         # RandomCellTypeRule, MixOfNCellTypesRule
│   ├── spatial.py        # ProbabilityNodeFieldRule, SingleTypeRule
│   ├── neighbor.py       # DeterministicNeighborAssignment
│   └── composite.py      # DistanceBasedRule, CompositeRule
├── experiment/
│   ├── __init__.py
│   ├── hybiss.py         # HybISS_Setup
│   ├── properties.py     # CellTypesProperties
│   └── transfer.py       # TransferFunctionBase, AffineNonNegTransfer, IdentityTransfer
├── utils/
│   ├── __init__.py
│   ├── geometry.py       # chaikin_smooth, smooth_polygon, add_edge_noise, generate_*
│   ├── math.py           # lognorm_params_to_mean_std, intuitive_rand_lognormal
│   ├── interpolation.py  # LinearNDInterpolatorExt
│   └── encoding.py       # one_hot_encode_array, unfold_int_matrix
├── viz/
│   ├── __init__.py
│   └── plotting.py       # plot_fov, plot_expression_matrix
└── io/
    ├── __init__.py
    └── dataset.py        # generate_dataset, load_data
```
