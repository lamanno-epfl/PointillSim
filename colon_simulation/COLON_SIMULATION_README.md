# Colon Tissue Simulation with PointillSim

End-to-end Python script for generating synthetic colon tissue using the PointillSim API.

## Overview

This script generates a single field of view (FOV) of colon-like tissue with realistic spatial organization:

- **Crypt Region (Left)**: Organized glandular structures with epithelial cells
- **Sparse Middle**: Stromal, immune, and muscle cells with lower density
- **Dense Right**: High-density immune and endothelial cells

## Features

- ✅ Single FOV generation (no tiling)
- ✅ 6 cell types with realistic spatial organization
- ✅ 50 genes with controlled expression
- ✅ Gene sensitivity: 0.5
- ✅ Multiple output formats: CSV, h5ad
- ✅ Ground truth cell type annotations
- ✅ Realistic cell morphology (size, shape, rotation)

## Quick Start

### 1. Generate Tissue

```bash
python generate_colon_tissue.py
```

**Output:** Creates `colon_tissue_output/` folder with:
- `cells.csv` - Cell positions and ground truth (7,500+ cells)
- `dots.csv` - Transcript positions (200,000+ dots)
- `data.h5ad` - AnnData format for scanpy/squidpy
- `cell_types_expression.csv` - Gene expression matrix
- `visualization.png` - Overview plot

### 2. Visualize Results

```bash
python visualize_colon_output.py
```

Generates:
- Cell type distribution map
- Transcript dot plot
- Cell density heatmap
- Summary statistics

## Cell Types

| ID | Name | Description | Typical Proportion |
|----|------|-------------|-------------------|
| 0 | Epithelial_Crypt | Crypt epithelium | ~5% |
| 1 | Epithelial_Surface | Surface epithelium | ~2% |
| 2 | Stromal | Fibroblasts | ~22% |
| 3 | Immune | Immune cells | ~43% |
| 4 | Endothelial | Blood vessels | ~27% |
| 5 | Muscle | Smooth muscle | ~1% |

## Output Files

### cells.csv
Contains ground truth for each cell:
- `X`, `Y`: Cell centroid coordinates
- `Class ID`: Cell type (0-5)
- `Class 0` - `Class 5`: One-hot encoded type
- `ProbClass0` - `ProbClass5`: Soft probabilities
- `Minor Axis`, `Major Axis`, `Rotation`: Morphology
- `RNA Concentration`: Relative RNA content

### dots.csv
Contains observed transcripts:
- `x`, `y`: Transcript position
- `gene`: Gene name (Gene 0 - Gene 49)
- `cell`: Parent cell ID

### data.h5ad
AnnData object compatible with scanpy/squidpy:
- `adata.obs`: Cell metadata
- `adata.obsm['spatial']`: Spatial coordinates
- `adata.obsm['cell_type_one_hot']`: Cell type encoding
- `adata.uns['cell_type_probabilities']`: Soft labels

## Parameters

Default parameters (can be modified in script):

```python
FRAME_SIZE = 1000          # FOV size (pixels)
N_CELL_TYPES = 6           # Number of cell types
N_GENES = 50               # Number of genes
GENE_SENSITIVITY = 0.5     # Observation sensitivity
```

## Customization

### Modify Cell Type Proportions

Edit the `MixOfNCellTypesRule` proportions in each region:

```python
dense_rules = [
    MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_IMMUNE, ID_ENDOTHELIAL, ID_STROMAL],
        proportions=[0.6, 0.25, 0.15]  # <- Adjust these
    )
]
```

### Change Crypt Density

Modify the number of crypts:

```python
n_crypts_per_line = 8  # <- Increase/decrease
```

### Adjust Cell Spacing

Change cell density in each region:

```python
tipical_cell_spacing=10  # Smaller = denser
```

### Change Frame Size

```python
FRAME_SIZE = 2000  # Larger FOV
```

## Requirements

```bash
# Core dependencies (from PointillSim)
numpy
scipy
pandas
shapely
matplotlib

# For h5ad export
anndata  # Optional but recommended
```

Install with:
```bash
conda install -c conda-forge anndata
```

## Architecture

The simulation follows this pipeline:

```
1. TissueCellTypes (gene expression)
   ↓
2. CellTypesProperties (morphology)
   ↓
3. TissueSlice (spatial layout)
   ├─ Region 1: Crypts (VacuolatedStructure)
   ├─ Region 2: Sparse (FrameWideElement)
   └─ Region 3: Dense (FrameWideElement)
   ↓
4. FOV (cell positions + probabilities)
   ↓
5. HybISS_Setup (transcript observation)
   ↓
6. Output: cells.csv, dots.csv, data.h5ad
```

## Design Patterns Used

- **Template Method**: `VacuolatedStructure` for glandular units
- **Strategy Pattern**: `CellTypeRuleBase` with multiple implementations
- **Composite Pattern**: Multiple regions with different rules
- **Factory Pattern**: Lambda functions for element prototypes

## Typical Output

```
- 7,000-8,000 cells
- 200,000-250,000 transcripts
- 6 cell types with spatial organization
- 50 genes with realistic expression patterns
- ~2.2 MB cells.csv
- ~10 MB dots.csv
- ~2 MB data.h5ad
```

## Integration with Analysis Tools

### Scanpy (Python)

```python
import scanpy as sc
import anndata as ad

# Load data
adata = ad.read_h5ad('colon_tissue_output/data.h5ad')

# Standard scanpy workflow
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# Spatial plots
sc.pl.spatial(adata, color='class_id')
```

### Squidpy (Spatial analysis)

```python
import squidpy as sq

# Load data
adata = ad.read_h5ad('colon_tissue_output/data.h5ad')

# Spatial analysis
sq.gr.spatial_neighbors(adata, coord_type='generic')
sq.gr.spatial_autocorr(adata)
sq.gr.nhood_enrichment(adata, cluster_key='class_id')
```

### Seurat (R)

```r
library(Seurat)

# Load from CSV
cells <- read.csv('colon_tissue_output/cells.csv')
dots <- read.csv('colon_tissue_output/dots.csv')

# Create Seurat object
# ... (standard conversion)
```

## Troubleshooting

### AnnData not installed
```bash
conda install -c conda-forge anndata
# or
pip install anndata
```

### Memory issues
Reduce `FRAME_SIZE` or increase cell spacing

### Different cell counts each run
This is expected - simulation is stochastic. Set `np.random.seed()` for reproducibility.

## References

- PointillSim: [GitHub](https://github.com/your-repo/PointillSim)
- Documentation: See `CLAUDE.md` for architecture details
- Example notebooks: `notebooks/` folder

## Citation

If you use this simulation in your research, please cite PointillSim:

```bibtex
@software{pointillsim2024,
  title={PointillSim: Rule-based Simulation for Spatial Transcriptomics},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/PointillSim}
}
```
