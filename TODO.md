# PointillSim Development TODO

## Remaining Tasks

### Abstract Structures & Tissue Variety
- [ ] **More Abstract Structure Classes**: Expand the variety of histological elements
  - **FibrillarStructure**: For collagen bundles, muscle fibers, nerve tracts
  - **ClusterElement**: Groups of cells with shared properties (e.g., lymphoid aggregates)
  - **GlandularUnit**: Acinar/tubular structures common in many tissues
  - **InterfaceElement**: For modeling tissue boundaries and transition zones
  - **StromalElement**: Background connective tissue with specific properties

### Configuration
- [ ] **from_config()** factory methods for main classes

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
- [ ] **EffectController**: Control specific effects in the simulation
  - Enable/disable specific sources of variation
  - Set effect sizes for controlled experiments
- [ ] **CovariateBasedRule**: Cell type rules that depend on covariates
  - Assign cell types based on covariate values
  - Enables complex spatial patterns tied to measurable covariates

### Simulation Design & Control Flow
- [ ] **SimulationDesign**: High-level class specifying simulation parameters
  - Define what covariates to vary and their ranges
  - Specify number of replicates per condition
  - Control which effects are active
- [ ] **run_simulation_design()**: Execute a design specification
  - Generate all FOVs according to design
  - Return organized dataset with metadata
- [ ] **DesignMatrix**: Track covariate values across generated FOVs

### Difficulty Scoring & Validation
- [ ] **DifficultyScorer**: Module for assessing simulation difficulty
  - Use XGBoost or similar to estimate how hard the classification task is
  - Metrics: class separability, spatial pattern complexity
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

### Testing
- [ ] Add performance benchmarks
- [ ] Test edge cases (empty FOVs, single cell, etc.)

### Showcase Notebooks (Advanced)
- [ ] **Advanced Structures Notebook**: Complex tissue simulations
  - Multiple overlapping structures
  - Layered and nested elements
- [ ] **Benchmark Notebook**: Compare simulated vs. real data statistics

---

## Completed

### Documentation & Usability
- [x] **Update notebooks**: Make notebooks more explanatory (7 notebooks)
- [x] **Sphinx documentation**: Build comprehensive API documentation (19+ .rst files)
- [x] **Sample datasets**: Include 4 pre-generated sample datasets (simple_fov, cortex_like, gland_fov, mixed_tissue)
- [x] **Getting Started Notebook**: Basic usage tutorial (00_getting_started.ipynb)
- [x] **Tissue description folder**: Create folder with procedural descriptions (colon_crypts, cortex_layers, skin_epidermis, mammary_gland)
- [x] **Tissue-specific notebooks**: Notebooks demonstrating tissue simulations (06_tissue_simulations.ipynb)
- [x] **Custom Rules Notebook**: Demonstrate all cell type assignment rules (05_cell_type_rules.ipynb)

### Histological Elements
- [x] **LayeredElement**: For simulating stratified tissue layers
- [x] **BranchingStructure**: For tree-like structures
- [x] **LinearLumenStructure**: Tube-like structures (vessels, ducts)

### Simulation Modes
- [x] **Two Modes of Operation**: Support both random FOV and larger tissue simulation
- [x] **TissueSlice.extract_fov(x, y, size)**: Extract a FOV from a specific location

### Visualization
- [x] **plot_tissue_slice()**: Visualization for TissueSlice with region boundaries
- [x] **plot_probability_field()**: Visualization for ProbabilityNodeFieldRule fields
- [x] **plot_fov()**: Convenience function to visualize a FOV
- [x] **plot_expression_matrix()**: Heatmap visualization for TissueCellTypes

### Serialization & I/O
- [x] **TissueSlice.save()** / **TissueSlice.load()**: Pickle or HDF5 serialization
- [x] **HybISS_Setup.save_config()**: Save experiment configuration for reproducibility
- [x] **FOV.to_anndata()**: Export FOV as AnnData object
- [x] **FOV.to_spatialdata()**: Export to SpatialData format

### Configuration
- [x] **Config dataclass**: Centralized configuration for simulation parameters
- [x] **YAML/JSON config loading**: Load simulation setup from config files

### Cell Type Rules
- [x] **DistanceBasedRule**: Assign cell types based on distance from element boundary
- [x] **LayerRule**: Assign types based on radial position
- [x] **GradientRule**: Create smooth linear gradients across elements
- [x] **CompositeRule**: Combining multiple rules with configurable weights

### FOV and Expression
- [x] **FOV.add_noise()**: Add realistic noise to cell positions
- [x] **FOV.subsample()**: Randomly subsample cells
- [x] **TissueCellTypes.load_from_csv()**: Load real expression profiles from file
- [x] **TissueCellTypes.load_from_anndata()**: Load expression profiles from AnnData

### Technology Presets
- [x] **TechnologyPreset base class**: Abstract class for spatial transcriptomics technologies
- [x] **HybISSPreset**, **CartanaPreset**, **MerfishPreset**, **TenXVisiumPreset**, **TenXXeniumPreset**
- [x] **Technology-specific transfer functions**: Each preset defines its own detection model
- [x] **Gene panel validation**: Enforce gene count limits per technology

### Polygon Smoothing & Visual Realism
- [x] **Bevel/Node Smoothing**: Chaikin's corner cutting algorithm
- [x] **Edge Softening**: Add natural variation to polygon edges

### CI & Testing
- [x] **GitHub Actions CI**: Automated testing pipeline (Python 3.9-3.12)
- [x] **Pre-commit Hooks**: Code quality checks (linting, formatting)
- [x] Add property-based tests (hypothesis)
- [x] Add integration tests for full simulation pipelines

### Refactoring
- [x] Fix `FrameWideUpdater` one-time-use limitation
- [x] Split `data_simulations.py` into focused modules
- [x] Add type hints throughout codebase
- [x] Add input validation to constructors
- [x] Remove duplicate `SingleTypeRule` class definition

---

## Module Structure

```
pointillsim/
├── __init__.py           # Public API exports
├── config.py             # Configuration dataclasses
├── core/
│   ├── __init__.py
│   ├── fov.py            # FOV, FOVDistribution
│   └── tissue.py         # TissueCellTypes, TissueSlice, RegionSpec
├── data/
│   ├── __init__.py       # load_sample, list_samples
│   └── *.npz             # Pre-generated sample datasets
├── elements/
│   ├── __init__.py
│   ├── base.py           # HistologicalElement
│   ├── frame.py          # FrameWideElement, FrameWideUpdater
│   └── structures.py     # VacuolatedStructure, LayeredElement, BranchingStructure, etc.
├── rules/
│   ├── __init__.py
│   ├── base.py           # CellTypeRuleBase, DummyRule
│   ├── random.py         # RandomCellTypeRule, MixOfNCellTypesRule
│   ├── spatial.py        # ProbabilityNodeFieldRule, SingleTypeRule
│   ├── neighbor.py       # DeterministicNeighborAssignment
│   └── composite.py      # DistanceBasedRule, CompositeRule, LayerRule, GradientRule
├── experiment/
│   ├── __init__.py
│   ├── hybiss.py         # HybISS_Setup
│   ├── properties.py     # CellTypesProperties
│   ├── presets.py        # TechnologyPreset and implementations
│   └── transfer.py       # TransferFunctionBase, AffineNonNegTransfer, IdentityTransfer
├── utils/
│   ├── __init__.py
│   ├── geometry.py       # chaikin_smooth, smooth_polygon, add_edge_noise, generate_*
│   ├── math.py           # lognorm_params_to_mean_std, intuitive_rand_lognormal
│   ├── interpolation.py  # LinearNDInterpolatorExt
│   └── encoding.py       # one_hot_encode_array, unfold_int_matrix
├── viz/
│   ├── __init__.py
│   └── plotting.py       # plot_fov, plot_expression_matrix, plot_tissue_slice, plot_probability_field
└── io/
    ├── __init__.py
    └── dataset.py        # generate_dataset, load_data
```
