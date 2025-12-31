# PointillSim Development TODO

## Future Development Priorities

### 🎯 Usability & New User Access

#### High Priority
- [ ] **Quick-Start Convenience Functions**: One-liner tissue simulations
  ```python
  from pointillsim.quickstart import simulate_cortex, simulate_gland, simulate_tumor
  fov = simulate_cortex(n_layers=6, n_cells=1000)
  ```
  - Pre-configured common tissue types
  - Sensible defaults, minimal parameters
  - Returns ready-to-use FOV with observations

- [ ] **Pipeline Visualization**: Show the simulation workflow
  - `plot_pipeline()`: Diagram showing ground truth → observation → effects
  - `plot_rule_composition()`: Visualize how rules combine probabilities
  - Help users understand operation order and dependencies

- [ ] **Troubleshooting Guide**: Common issues and solutions
  - "My FOV is empty" → Check element placement, frame_size
  - "Cell types not distinct" → Adjust concentration parameter
  - "Transcripts outside cells" → Check cell properties application order
  - Add as docs/troubleshooting.rst

- [ ] **Method Chaining API**: More Pythonic interface
  ```python
  fov = (FOVDistribution(...)
         .generate_fov()
         .apply_properties(cell_props)
         .observe(hybiss)
         .add_noise(cv=0.1))
  ```

#### Medium Priority
- [ ] **Decision Tree Guide**: "What method should I use?"
  - When to use `TissueCellTypes.generate_types_and_markers()` vs `load_from_csv()` vs `load_from_anndata()`
  - When to use `FOVDistribution` vs `TissueSlice`
  - Which rule to use for different spatial patterns

- [ ] **`.summary()` Methods**: Quick inspection of objects
  ```python
  fov.summary()  # prints: 1234 cells, 5 types, 600x600 px, ...
  tissue.summary()  # prints: 50 genes, 10 types, marker genes: ...
  ```

- [ ] **`.validate()` Methods**: Check object consistency
  - `fov.validate()`: Check centroids in bounds, probabilities sum to 1
  - `tissue.validate()`: Check expression matrix is valid
  - Return list of warnings/errors

- [ ] **Interactive Jupyter Widgets**: Parameter exploration
  - Sliders for `concentration`, `tipical_cell_spacing`, etc.
  - Live preview of FOV as parameters change
  - Export final configuration

---

### 🧬 Biological Relevance & Tissue Realism

#### High Priority
- [ ] **Pathology Elements**: Disease modeling
  - `TumorElement`: Invasive growth patterns, irregular boundaries
  - `NecroticRegion`: Dead tissue zones with debris
  - `InflammationFocus`: Immune cell infiltration patterns
  - `FibrosisElement`: Scarring and fibrotic tissue

- [ ] **Vascular Network**: Realistic blood vessel patterns
  - `VascularTree`: Branching vessel network
  - Vessel caliber variation (arteries → arterioles → capillaries)
  - Perivascular cell types (pericytes, endothelial)

- [ ] **Immune Tissue Specialization**:
  - `GerminalCenter`: B cell follicles with light/dark zones
  - `TZone`: T cell zones in lymphoid tissue
  - `TertiaryLymphoidStructure`: Ectopic lymphoid aggregates

- [ ] **Cell State Gradients**: Beyond discrete types
  - Continuous differentiation trajectories
  - Pseudotime-based expression gradients
  - EMT (epithelial-mesenchymal transition) zones

#### Medium Priority
- [ ] **Additional Spatial Patterns**:
  - `PeriodicPatternRule`: Checkerboard, hexagonal, stripe patterns
  - `WavePatternRule`: Oscillating spatial patterns
  - `FractalRule`: Self-similar patterns at multiple scales

- [ ] **Extracellular Matrix (ECM) Modeling**:
  - ECM density fields affecting cell spacing
  - Basement membrane boundaries
  - Stromal density gradients

- [ ] **Stem Cell Niches**: Specialized microenvironments
  - Crypt base (intestinal stem cells)
  - Hair follicle bulge
  - Bone marrow niches

- [ ] **Innervation Patterns**:
  - Nerve fiber bundles
  - Neuroendocrine cell clustering
  - Ganglion structures

#### Lower Priority
- [ ] **3D Tissue Support**: Volumetric simulation
  - 3D cell positions and shapes
  - Multi-layer tissue sections
  - True 3D neighborhood relationships
  - Note: Major architectural change

- [ ] **Temporal/Dynamic Modeling**:
  - Cell cycle state distribution
  - Migration patterns over time
  - Differentiation dynamics

---

### ⚡ Performance & Scale

- [ ] **Parallel FOV Generation**: Speed up batch operations
  - `FOVDistribution.generate_batch_parallel(n_workers=4)`
  - Use multiprocessing for independent FOV generation
  - Expected 4-10x speedup for large batches

- [ ] **Memory-Efficient Large Tissues**:
  - Chunked TissueSlice generation
  - Lazy loading for very large datasets
  - HDF5-backed storage for out-of-core processing

---

### 🔬 Benchmarking & Validation

- [ ] **Segmentation Error Simulation**: More realistic ground truth
  - `add_segmentation_error(fov, over_seg=0.1, under_seg=0.05)`
  - Merge adjacent cells (under-segmentation)
  - Split single cells (over-segmentation)
  - Fragment cells at boundaries

- [ ] **Built-in Benchmark Classifiers**:
  - `benchmark_classifier(fov, tissue, method='logistic')`
  - Quick accuracy assessment on simulated data
  - Compare methods: logistic, SVM, random forest
  - Return classification report

- [ ] **Spatial Pattern Fidelity Metrics**:
  - Compare simulated vs real spatial statistics
  - Moran's I, Ripley's K/L functions
  - Neighborhood enrichment analysis

- [ ] **Cell Detection Metrics**:
  - Precision/recall for cell detection
  - IoU distribution for segmentation
  - Transcript assignment accuracy

---

### 📚 Documentation & Examples

- [ ] **"Common Patterns" Gallery**: Copy-paste snippets
  - Cortex with 6 layers
  - Colon crypts with stem cell niche
  - Mammary TDLU
  - Tumor microenvironment
  - Lymph node structure

- [ ] **Video Tutorials**: Screen recordings
  - "Your first simulation in 5 minutes"
  - "Creating custom tissue architectures"
  - "Adding realistic technical effects"

- [ ] **Real Data Comparison Examples**:
  - Side-by-side: simulated vs real HybISS data
  - Parameter calibration workflow
  - Validation against published datasets

---

### 🔧 API Improvements

- [ ] **Unified Export Interface**:
  - `fov.export(format='anndata')` / `'spatialdata'` / `'csv'`
  - Include all metadata automatically
  - Round-trip support (export → import)

- [ ] **Configuration Improvements**:
  - Add `.to_dict()` / `.from_dict()` to all major classes
  - Full pipeline serialization (not just config)
  - Version tracking for reproducibility

- [ ] **Plugin System**: Extensibility
  - Register custom elements: `@register_element`
  - Register custom rules: `@register_rule`
  - Auto-discovery of plugins

---

## Completed Features

<details>
<summary>Click to expand completed features (90+ items)</summary>

### Core Framework
- [x] FOV and FOVDistribution classes
- [x] TissueCellTypes with synthetic and real expression profiles
- [x] TissueSlice for large tissue regions
- [x] Consistent tiling across FOV boundaries

### Histological Elements (9 types)
- [x] HistologicalElement (base), FrameWideElement, FrameWideUpdater
- [x] VacuolatedStructure, LinearLumenStructure, LayeredElement
- [x] BranchingStructure, FibrillarStructure, ClusterElement
- [x] GlandularUnit, InterfaceElement, StromalElement

### Cell Type Rules (13 types)
- [x] RandomCellTypeRule, MixOfNCellTypesRule, SingleTypeRule
- [x] ProbabilityNodeFieldRule, DistanceBasedRule, LayerRule, GradientRule
- [x] CompositeRule, DeterministicNeighborAssignment
- [x] CovariateBasedRule, ThresholdCovariateRule, SpatialCovariateRule

### Technical Effects
- [x] BatchEffectModel, TechnicalNoise, BackgroundNoise, DropoutModel
- [x] SpatialNoise (gradient, radial, patches, perlin patterns)
- [x] Lateral2DAdmixture, ZAxisAdmixture, CompositeAdmixture
- [x] AdmixtureMetrics for quantifying misassignment

### Technology Presets (5 platforms)
- [x] HybISSPreset, MerfishPreset, CartanaPreset
- [x] TenXXeniumPreset, TenXVisiumPreset
- [x] Realistic parameters from literature

### Experimental Design
- [x] CovariateSystem, Covariate classes
- [x] EffectController for controlled experiments
- [x] SimulationDesign, DesignMatrix

### Validation & Quality
- [x] DifficultyScorer (separability, spatial complexity)
- [x] ValidationMetrics (distribution matching)

### I/O & Export
- [x] AnnData and SpatialData export
- [x] CSV loading for expression profiles
- [x] Dataset generation with metadata
- [x] Configuration save/load (YAML/JSON)

### Visualization
- [x] plot_fov, plot_expression_matrix
- [x] plot_tissue_slice, plot_probability_field

### Documentation
- [x] 14 tutorial notebooks
- [x] Sphinx documentation with API reference
- [x] Philosophy and use cases guides
- [x] Notebooks integrated into docs

### Testing & CI
- [x] 103+ tests (unit, integration, property-based)
- [x] GitHub Actions CI (Python 3.9-3.12)
- [x] Pre-commit hooks (ruff)

</details>

---

## Module Structure

```
pointillsim/
├── __init__.py           # Public API (50+ exports)
├── config.py             # Configuration dataclasses
├── core/                 # FOV, TissueCellTypes, TissueSlice
├── elements/             # 9 histological element types
├── rules/                # 13 cell type assignment rules
├── experiment/           # HybISS_Setup, presets, properties
├── effects/              # Noise, batch effects, admixture
├── design/               # Covariates, experimental design
├── validation/           # Difficulty scoring, metrics
├── viz/                  # Plotting functions
├── io/                   # Dataset generation, export
├── utils/                # Geometry, math, interpolation
└── data/                 # Sample datasets
```

---

## Contributing

Contributions welcome! Priority areas:
1. **Biological realism**: New tissue types, pathology elements
2. **Usability**: Convenience functions, better error messages
3. **Performance**: Parallelization, memory efficiency
4. **Documentation**: Examples, tutorials, troubleshooting
