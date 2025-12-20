# PointillSim

Rule-based Simulation Engine for Imaging-based Spatial Transcriptomics

## Overview

PointillSim creates realistic synthetic Fields of View (FOVs) for imaging-based spatial transcriptomics experiments, specifically **HybISS** (Hybridization-based In Situ Sequencing). It generates ground truth cell-type annotations and spatial gene expression data as dots within cells, enabling benchmarking and validation of spatial transcriptomics analysis methods.

## Project Structure

| File | Purpose |
|------|---------|
| `data_simulations.py` | Core simulation engine (~1,700 lines, 23 classes) |
| `generate_cortex.ipynb` | Brain cortex tissue simulation |
| `generate_skin.ipynb` | Skin tissue simulation |
| `generate_colon.ipynb` | Colon tissue simulation |
| `RuleLearning_Simulation.ipynb` | Rule learning exploration |

## Core Components

### Histological Elements (tissue building blocks)
- `HistologicalElement` - Base class for tissue regions; generates polygonal boundaries and cell centroids
- `FrameWideElement` - Element that spans the entire frame (background)
- `VacuolatedStructure` - Elements with holes (vacuoles, lumens)
- `LinearLumenStructure` - Tube-like structures (vessels, ducts)

### Cell Type Assignment Rules
- `RandomCellTypeRule` - Dirichlet-distributed cell types
- `SingleTypeRule` - All cells assigned to one type
- `MixOfNCellTypesRule` - Mixture of specific cell types with fixed proportions
- `ProbabilityNodeFieldRule` - Spatial gradient fields using reference points
- `DeterministicNeighborAssignment` - Neighbor-based rules

### FOV Management
- `FOVDistribution` - Probabilistic element placement and stochastic FOV generation
- `FOV` - Container for cell centroids, probabilities, and class assignments

### Expression Simulation
- `TissueCellTypes` - Gene expression profiles per cell type (lognormal distributions)
- `CellTypesProperties` - Maps cell types to morphological properties (size, anisotropy, RNA concentration)
- `HybISS_Setup` - Simulates experimental measurements with Poisson noise

## Data Flow

```
TissueCellTypes (gene expression profiles)
    ↓
FOVDistribution (probabilistic element placement)
    ↓
HistologicalElement(s) (generate polygons + cell centroids)
    ↓
CellTypeRule(s) (assign cell type probabilities)
    ↓
FOV (cell centroids + type probabilities)
    ↓
HybISS_Setup (Poisson sample RNA counts + spatial dot distribution)
    ↓
Output CSVs
```

## Output Files

Each FOV generates:
- `cell_centroids.csv` - Cell positions
- `dots.csv` - Transcript dot locations
- `ground_truth_cells.csv` - Cell types and expression probabilities
- `metadata.json` - Simulation parameters

## Dependencies

- **NumPy/SciPy** - Numerical computation and interpolation
- **Pandas** - Data handling
- **Shapely** - Geometric operations (polygons, containment)
- **scikit-learn** - Spatial queries (KDTree)
- **scikit-image** - Image processing utilities
- **Matplotlib** - Visualization
- **TQDM** - Progress bars
