# PointillSim

**Rule-based Simulation Engine for Imaging-based Spatial Transcriptomics**

[![CI](https://github.com/lamanno-epfl/PointillSim/actions/workflows/ci.yml/badge.svg)](https://github.com/lamanno-epfl/PointillSim/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

![PointillSim Example Output](tests/visual_outputs/test_complete_simulation.png)

## Overview

PointillSim creates realistic synthetic Fields of View (FOVs) for imaging-based spatial transcriptomics experiments. It generates ground truth cell-type annotations and spatial gene expression data as dots within cells, enabling benchmarking and validation of spatial transcriptomics analysis methods.

**Key capabilities:**
- Simulate diverse tissue architectures with 9+ histological structure types
- Model realistic technical artifacts (admixture, batch effects, noise)
- Support multiple technology platforms (HybISS, MERFISH, Visium, Xenium, Cartana)
- Generate controlled datasets for method benchmarking
- Integrate real expression data for realistic simulations

## Modeling Philosophy

PointillSim follows a **compositional, rule-based approach** to tissue simulation:

### Hierarchical Composition
Tissues are built from **histological elements**—discrete regions with defined boundaries and cell populations. Elements can represent:
- Background tissue (stroma, parenchyma)
- Specialized structures (glands, vessels, follicles, layers)
- Pathological features (tumors, inflammation foci)

Elements are composed within a **Field of View (FOV)**, where overlapping regions are resolved by priority, allowing foreground structures to "punch through" background tissue.

### Rule-Based Cell Type Assignment
Rather than hardcoding cell type distributions, PointillSim uses **composable rules** that determine how cell types are assigned within each element:

- **Spatial rules**: Create gradients, layers, and distance-based patterns
- **Stochastic rules**: Add biological variability via Dirichlet distributions
- **Composite rules**: Combine multiple strategies with configurable weights

This separation of *where cells are placed* from *what types they become* enables flexible modeling of diverse tissue architectures.

### Probabilistic Observation Model
The simulation pipeline separates **ground truth** from **observed data**:

1. **Ground truth**: Cell positions, type probabilities, expression levels
2. **Observation**: Poisson-sampled transcript counts, spatial dot distributions
3. **Technical effects**: Admixture, batch effects, noise modeling

This mirrors real experimental variability and enables systematic benchmarking of analysis methods.

## Installation

```bash
# Clone the repository
git clone https://github.com/lamanno-epfl/PointillSim.git
cd PointillSim

# Install with dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from pointillsim import (
    TissueCellTypes, CellTypesProperties, HybISS_Setup,
    FOVDistribution, FrameWideElement,
    RandomCellTypeRule, VacuolatedStructure, LayerRule
)

# 1. Define tissue with gene expression profiles
tissue = TissueCellTypes()
tissue.generate_types_and_markers(n_genes=50, n_cell_types=5)

# 2. Create cell type properties (morphology)
cell_props = CellTypesProperties(n_cell_types=5, sizes=12)

# 3. Define FOV with glandular structures
def create_gland():
    return VacuolatedStructure(
        frame_size=500,
        scale=80,
        hole_scale_factor=0.5,
        rules=LayerRule(
            n_cell_types=5,
            layer_types=[0, 1],  # Epithelial inner, stromal outer
            layer_boundaries=[0.6]
        )
    )

fov_dist = FOVDistribution(
    frame_size=500,
    background_element=lambda: FrameWideElement(
        frame_size=500,
        rules=RandomCellTypeRule(5)
    ),
    other_elements=[create_gland],
    elements_frequency=[0.7],
    attempts_at_elements=5
)

# 4. Generate and observe
fov = fov_dist.generate_fov()
cell_props.apply(fov)

hybiss = HybISS_Setup(tissue)
hybiss.observe_dots(fov)

# 5. Export data
dots_df = hybiss.make_pandas_df()
cells_df = fov.make_pandas_df()
```

## Package Structure

```
pointillsim/
├── core/
│   ├── fov.py            # FOV, FOVDistribution
│   └── tissue.py         # TissueCellTypes, TissueSlice, RegionSpec
├── elements/
│   ├── base.py           # HistologicalElement (base class)
│   ├── frame.py          # FrameWideElement, FrameWideUpdater
│   ├── structures.py     # VacuolatedStructure, LinearLumenStructure
│   ├── layered.py        # LayeredElement, InterfaceElement
│   ├── complex.py        # BranchingStructure, FibrillarStructure
│   └── composite.py      # ClusterElement, GlandularUnit, StromalElement
├── rules/
│   ├── base.py           # CellTypeRuleBase, DummyRule
│   ├── random.py         # RandomCellTypeRule, MixOfNCellTypesRule
│   ├── spatial.py        # ProbabilityNodeFieldRule, SingleTypeRule
│   ├── neighbor.py       # DeterministicNeighborAssignment
│   └── composite.py      # DistanceBasedRule, LayerRule, GradientRule, CompositeRule
├── experiment/
│   ├── hybiss.py         # HybISS_Setup
│   ├── properties.py     # CellTypesProperties, TechnologyPreset
│   └── transfer.py       # TransferFunctionBase, AffineNonNegTransfer
├── effects/
│   ├── admixture.py      # Lateral2DAdmixture, ZAxisAdmixture, AdmixtureMetrics
│   ├── noise.py          # Technical noise models
│   └── batch.py          # Batch effect simulation
├── validation/
│   └── difficulty.py     # Difficulty scoring for simulated data
├── utils/
│   ├── geometry.py       # Geometric operations (smoothing, point generation)
│   ├── interpolation.py  # Spatial interpolation
│   └── encoding.py       # Data encoding utilities
├── viz/
│   └── plotting.py       # plot_fov, plot_expression_matrix
└── io/
    └── dataset.py        # generate_dataset, load_data
```

## Core Components

### Histological Elements

PointillSim provides a rich library of tissue structure types:

| Element | Description | Use Cases |
|---------|-------------|-----------|
| `HistologicalElement` | Base class with convex polygonal boundaries | Generic tissue regions |
| `FrameWideElement` | Spans entire FOV | Background tissue, stroma |
| `VacuolatedStructure` | Ring-shaped with central lumen | Glands, acini, crypts, follicles |
| `LinearLumenStructure` | Tubular structures | Blood vessels, ducts |
| `LayeredElement` | Stratified horizontal/vertical layers | Epidermis, cortical layers |
| `BranchingStructure` | Tree-like branching patterns | Vasculature, nerves, ductal trees |
| `FibrillarStructure` | Parallel fiber bundles | Collagen, muscle fibers |
| `ClusterElement` | Dense cell aggregates | Lymphoid follicles, tumor nests |
| `GlandularUnit` | Complete glandular structures | Mammary glands, salivary glands |
| `InterfaceElement` | Tissue boundaries and transitions | Epithelial-stromal interfaces |
| `StromalElement` | Background connective tissue | Variable density stroma |

### Cell Type Assignment Rules

Rules determine spatial cell type distributions:

| Rule | Description | Parameters |
|------|-------------|------------|
| `RandomCellTypeRule` | Dirichlet-distributed random types | `alpha` (concentration) |
| `SingleTypeRule` | Uniform single cell type | `cell_type_ix` |
| `MixOfNCellTypesRule` | Fixed proportions of specific types | `list_N`, `proportions` |
| `ProbabilityNodeFieldRule` | Spatial interpolation from reference points | `reference_points`, `ref_probs` |
| `DistanceBasedRule` | Radial gradients from center/boundary | `inner_type`, `outer_type`, `transition_width` |
| `LayerRule` | Concentric layers with transitions | `layer_types`, `layer_boundaries` |
| `GradientRule` | Linear spatial gradients | `orientation`, `type_sequence` |
| `CompositeRule` | Weighted combination of rules | `rules`, `weights` |
| `DeterministicNeighborAssignment` | Neighbor-based patterns | `neighborhood_radius` |

### Technology Presets

Support for multiple spatial transcriptomics platforms:

| Platform | Spot/Cell Size | Resolution | Typical Usage |
|----------|----------------|------------|---------------|
| **HybISS** | Single-cell | Subcellular | High-resolution imaging |
| **MERFISH** | Single-cell | Subcellular | Multiplexed imaging |
| **Visium** | 55μm spots | Spot-based | Tissue sections (10x) |
| **Xenium** | Single-cell | Subcellular | High-plex imaging (10x) |
| **Cartana** | Single-cell | Subcellular | In situ sequencing |

### Technical Effects

Model realistic artifacts and technical variation:

| Effect | Description | Key Parameters |
|--------|-------------|----------------|
| `Lateral2DAdmixture` | Boundary-based transcript misassignment | `boundary_width`, `transfer_rate` |
| `ZAxisAdmixture` | Out-of-plane contamination | `z_contamination_rate`, `neighborhood_correlation` |
| `CompositeAdmixture` | Combined admixture effects | `models` (list of admixture models) |
| `AdmixtureMetrics` | Quantify contamination patterns | Returns per-cell and per-type metrics |
| Batch effects | Technical variation across experiments | Platform-specific parameters |
| Noise models | Poisson sampling, background noise | `genes_sensitivities`, noise levels |

## Tutorials & Notebooks

PointillSim includes 14 comprehensive tutorial notebooks:

### Getting Started
- **00_getting_started.ipynb** - Quick introduction to basic workflow
- **01_simulation_framework.ipynb** - Core concepts: FOVs, elements, rules, observation

### Building Simulations
- **02_elements_and_rules.ipynb** - Histological elements and cell type rules in detail
- **05_cell_type_rules.ipynb** - Advanced rule composition and spatial patterns
- **10_advanced_structures.ipynb** - Gallery of all 9 histological structure types
- **12_beautiful_gallery.ipynb** - Beautiful visualizations of 6 tissue types (colon, cortex, mammary, lymph node, muscle, tumor)

### Expression & Technology
- **04_real_expression_data.ipynb** - Load and integrate real expression matrices
- **11_technology_presets.ipynb** - Platform-specific simulations (HybISS, MERFISH, Visium, Xenium, Cartana)

### Complex Tissues
- **06_tissue_simulations.ipynb** - Tissue-specific examples (intestine, brain, breast, immune)

### Technical Effects
- **07_effects_and_realism.ipynb** - Noise, batch effects, technical variation
- **13_admixture_simulation.ipynb** - Model transcript misassignment (lateral 2D, z-axis, combined)

### Experimental Design
- **03_batch_generation.ipynb** - Generate large datasets with multiple FOVs
- **08_simulation_design.ipynb** - Controlled experiments with covariates
- **09_validation_and_difficulty.ipynb** - Assess simulation difficulty and validate outputs

## Data Flow

```
TissueCellTypes                    CellTypesProperties
(gene expression profiles)         (morphology per type)
         ↓                                  ↓
FOVDistribution ─────────────────────────→ FOV
(stochastic element placement)      (centroids + probabilities)
         ↓                                  ↓
HistologicalElement(s)              Apply cell properties
(polygons + cell grids + rules)     (sizes, shapes, orientations)
         ↓                                  ↓
CellTypeRule(s)                     HybISS_Setup / TechnologyPreset
(probability assignment)            (Poisson sampling + platform specifics)
         ↓                                  ↓
     Ground Truth                    Technical Effects
(positions, types, expression)      (admixture, batch, noise)
         ↓                                  ↓
                              Output DataFrames
                         (cells, dots, ground truth)
```

## Key Features

### Realistic Tissue Architecture

```python
# Colon with crypts
from pointillsim.elements import VacuolatedStructure
from pointillsim.rules.composite import LayerRule

crypt = VacuolatedStructure(
    frame_size=500,
    scale=60,
    hole_scale_factor=0.5,
    rules=LayerRule(
        n_cell_types=5,
        layer_types=[0, 1, 2],  # Stem → Transit → Mature
        layer_boundaries=[0.3, 0.65]
    )
)
```

### Admixture Simulation

```python
from pointillsim.effects import Lateral2DAdmixture, ZAxisAdmixture

# Boundary-based misassignment
lateral = Lateral2DAdmixture(
    boundary_width=8.0,
    transfer_rate=0.4,
    seed=42
)
dots_admixed = lateral.apply(dots_df, cell_centroids, cell_types, cell_radii)

# Out-of-plane contamination
z_axis = ZAxisAdmixture(
    z_contamination_rate=0.15,
    neighborhood_correlation=0.8
)
```

### Platform-Specific Simulations

```python
from pointillsim.experiment import TechnologyPreset

# Visium 10x Genomics
visium_preset = TechnologyPreset.get_preset("Visium")
fov = fov_dist.generate_fov()
visium_preset.apply(fov)
```

### Polygon Smoothing

```python
from pointillsim.utils.geometry import smooth_polygon

# Smooth element boundaries
smoothed = smooth_polygon(polygon, iterations=3, preserve_area=True)
```

### FOV Manipulation

```python
# Add realistic noise
fov.add_noise(position_std=2.0, probability_std=0.01)

# Subsample cells
fov.subsample(fraction=0.5)  # or fov.subsample(n_cells=100)
```

### Load Real Expression Data

```python
# From CSV or AnnData
tissue = TissueCellTypes.load_from_csv("expression_matrix.csv")
# tissue = TissueCellTypes.load_from_anndata("dataset.h5ad")
```

### Visualization

```python
from pointillsim import plot_fov, plot_expression_matrix

# Visualize FOV colored by cell type
fig, ax = plot_fov(fov, color_by="class")

# Visualize expression matrix
fig, ax = plot_expression_matrix(tissue, log_scale=True)
```

### Difficulty Scoring

```python
from pointillsim.validation import compute_difficulty_score

# Assess how challenging a simulation is for classification
difficulty = compute_difficulty_score(fov, tissue)
print(f"Simulation difficulty: {difficulty:.3f}")
```

## Output Files

When using `generate_dataset()`:

| File | Contents | Type |
|------|----------|------|
| `cells_FOV*.csv` | Cell positions, types, probabilities, morphology | Ground truth |
| `dots_FOV*.csv` (ground_truth/) | Transcript positions with cell assignments | Ground truth |
| `cell_centroids_FOV*.csv` | Cell centroid positions only | Observable |
| `dots_FOV*.csv` (data/) | Dot positions and gene identity | Observable |
| `cell_types.csv` | Expression matrix used for simulation | Reference |
| `metadata.json` | Simulation parameters and configuration | Metadata |

## Example Use Cases

### Benchmark Cell Segmentation Methods
Generate ground truth cell boundaries and test segmentation algorithms:
```python
# Generate FOV with known cell positions
fov = fov_dist.generate_fov()
cell_props.apply(fov)

# Export ground truth
ground_truth_cells = fov.make_pandas_df()  # True positions and types

# Generate dots for segmentation input
hybiss.observe_dots(fov)
dots_for_segmentation = hybiss.make_pandas_df()  # Observable dots
```

### Test Cell Type Deconvolution
Create spots with known cell type mixtures:
```python
# Visium spots with mixed cell types
visium = TechnologyPreset.get_preset("Visium")
fov = fov_dist.generate_fov()
visium.apply(fov)

# Ground truth: cell type proportions per spot
# Observable: aggregated gene counts per spot
```

### Evaluate Admixture Correction Methods
Generate data with controlled admixture levels:
```python
# Apply admixture with known parameters
admixture = Lateral2DAdmixture(boundary_width=8.0, transfer_rate=0.3)
dots_contaminated = admixture.apply(dots_df, cell_centroids, cell_types)

# Test correction methods against ground truth
correction_accuracy = evaluate_correction(
    observed=dots_contaminated,
    ground_truth=dots_df
)
```

## Advanced Features

### Batch Generation with Covariates
```python
from pointillsim.design import CovariateDesign

# Systematically vary parameters
design = CovariateDesign(
    n_replicates=10,
    covariates={
        "cell_density": [0.5, 1.0, 1.5],
        "admixture_rate": [0.1, 0.2, 0.3]
    }
)
datasets = design.generate_batch(fov_dist, tissue)
```

### Tissue Slice Mode
```python
from pointillsim.core import TissueSlice

# Generate large tissue, extract consistent FOVs
tissue_slice = TissueSlice(
    tissue_size=(5000, 5000),
    fov_size=500
)
fov1 = tissue_slice.extract_fov(position=(0, 0))
fov2 = tissue_slice.extract_fov(position=(500, 0))  # Adjacent FOV
```

## Dependencies

- **NumPy/SciPy** - Numerical computation and interpolation
- **Pandas** - Data handling and export
- **Shapely** - Geometric operations (polygons, containment)
- **scikit-learn** - Spatial queries (KDTree)
- **Matplotlib** - Visualization
- **Seaborn** - Statistical visualization
- **TQDM** - Progress bars

Optional:
- **AnnData** - Integration with single-cell analysis ecosystem

## Citation

If you use PointillSim in your research, please cite:

```bibtex
@software{pointillsim,
  title = {PointillSim: Rule-based Simulation for Spatial Transcriptomics},
  author = {La Manno Lab},
  year = {2024},
  url = {https://github.com/lamanno-epfl/PointillSim}
}
```

## Contributing

Contributions are welcome! Areas for contribution:
- New histological element types
- Additional cell type rules
- Platform-specific presets
- Documentation improvements
- Bug reports and feature requests

Please open an issue or pull request on GitHub.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Developed at the [La Manno Lab](https://www.epfl.ch/labs/lamanno-lab/), EPFL.
