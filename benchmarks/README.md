# PointillSim Benchmarks

This directory contains benchmarks for evaluating cell type mapping algorithms on simulated spatial transcriptomics data.

## Overview

The **Sparsity Benchmark** evaluates how different sparsity levels (transcript detection efficiency) affect cell type assignment accuracy across diverse tissue architectures.

## Benchmark: Cell Type Mapping with Sparsity Analysis

**Notebook**: `sparsity_benchmark.ipynb`

### Design Rationale

Real spatial transcriptomics data exhibits varying degrees of sparsity due to:
- Platform detection efficiency
- Tissue-specific gene expression levels
- Technical variation in sample preparation

This benchmark systematically tests methods across a controlled sparsity gradient while maintaining tissue architectural diversity.

### Dataset Specifications

**10 FOVs** with diverse tissue structures:
1. **Glandular Crypts** - Intestinal-like crypts with layered epithelium
2. **Cortical Layers** - Brain cortex with 5 distinct layers
3. **Vascular Network** - Blood vessels in stromal background
4. **Lymphoid Follicles** - Germinal centers with immune cell types
5. **Muscle Fibers** - Parallel fiber bundles with myocytes
6. **Tumor Microenvironment** - Tumor nests with immune infiltration
7. **Fibrillar Stroma** - Collagen-like fibrillar structure
8. **Glandular Units** - Complex multi-acinar glandular structures
9. **Branching Ducts** - Tree-like ductal networks
10. **Tissue Interface** - Epithelial-stromal boundary

### Controllable Parameters

#### Sparsity Levels (5 levels)
- **Very Low** (Dense): gene_sensitivity=1.2, expected_level=20.0
- **Low**: gene_sensitivity=1.0, expected_level=15.0
- **Medium** (Typical): gene_sensitivity=0.8, expected_level=12.0
- **High**: gene_sensitivity=0.5, expected_level=8.0
- **Very High** (Sparse): gene_sensitivity=0.3, expected_level=5.0

#### Difficulty Levels (3 levels)
- **Easy**: Highly distinct cell types, low cell size variation, well-separated spatial regions
- **Medium**: Moderate cell type distinctness, moderate cell size variation, some spatial mixing
- **Hard**: Similar cell types, high cell size variation, high spatial mixing

#### Technical Noise
- Background dots: 2%
- Dropout rate: 5%
- Position noise: 1.5 pixels

### Data Structure

```
benchmarks/
├── sparsity_benchmark.ipynb    # Main benchmark generation notebook
├── data/                        # Generated benchmark data
│   ├── ground_truth_cells_fov##.csv      # Cell positions, types, morphology
│   ├── ground_truth_dots_fov##.csv       # Dots with cell assignments
│   ├── observable_dots_fov##.csv         # Dots for methods to process
│   ├── cell_centroids_fov##.csv          # Cell locations only
│   ├── expression_matrix.csv             # Gene × cell type profiles
│   └── metadata.json                     # Complete configuration
├── figures/                     # Visualizations
│   ├── fov_##_*.png                      # Individual FOV visualizations
│   ├── gallery_overview.png              # All FOVs in one figure
│   ├── expression_matrix.png             # Heatmap of expression profiles
│   └── benchmark_statistics.png          # Summary statistics
└── results/                     # Evaluation results
    └── benchmark_summary.csv             # FOV characteristics table
```

### Generated Files

**Per FOV (40 files total for 10 FOVs):**
- `ground_truth_cells_fov##.csv` - Ground truth cell information
  - Columns: `x, y, cell_type, cell_type_name, major_axis, minor_axis, rotation, ...`
- `ground_truth_dots_fov##.csv` - Ground truth transcript assignments
  - Columns: `x, y, gene, cell`
- `observable_dots_fov##.csv` - Observable data for methods (no cell assignment)
  - Columns: `x, y, gene`
- `cell_centroids_fov##.csv` - Cell locations for segmentation-based methods
  - Columns: `x, y`

**Shared files:**
- `expression_matrix.csv` - 100 genes × 8 cell types
- `metadata.json` - Complete benchmark configuration and metrics

## Running the Benchmark

### Generate Data

```bash
# Run the benchmark notebook
jupyter notebook benchmarks/sparsity_benchmark.ipynb
```

Or run programmatically:
```python
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

with open('benchmarks/sparsity_benchmark.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': 'benchmarks/'}})

with open('benchmarks/sparsity_benchmark.ipynb', 'w') as f:
    nbformat.write(nb, f)
```

### Evaluate Your Method

```python
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load data for FOV 0
observable = pd.read_csv('benchmarks/data/observable_dots_fov00.csv')
centroids = pd.read_csv('benchmarks/data/cell_centroids_fov00.csv')
expr_matrix = pd.read_csv('benchmarks/data/expression_matrix.csv', index_col=0)
ground_truth = pd.read_csv('benchmarks/data/ground_truth_cells_fov00.csv')

# Run your method
predicted_types = your_cell_type_mapping_method(
    observable, centroids, expr_matrix
)

# Evaluate
accuracy = accuracy_score(ground_truth['cell_type'], predicted_types)
f1_macro = f1_score(ground_truth['cell_type'], predicted_types, average='macro')
f1_per_type = f1_score(ground_truth['cell_type'], predicted_types, average=None)

print(f"Accuracy: {accuracy:.3f}")
print(f"F1 (macro): {f1_macro:.3f}")
```

## Evaluation Metrics

### Primary Metrics

1. **Cell Type Accuracy**: Fraction of cells with correctly predicted type
2. **F1 Score (Macro)**: Average F1 across cell types
3. **F1 Score (Per Type)**: F1 for each individual cell type

### Secondary Metrics

4. **Transcript Assignment Accuracy**: Fraction of correctly assigned transcripts
5. **Cell Purity**: Fraction of each cell's transcripts correctly assigned
6. **Confusion Matrix**: Predicted vs true cell types

### Stratified Analysis

Evaluate performance separately by:
- **Sparsity level**: very_low, low, medium, high, very_high
- **Difficulty level**: easy, medium, hard
- **Tissue type**: 10 different architectures
- **Cell type**: 8 different cell types

## Expected Results

### Baseline Performance Expectations

**Easy + Low Sparsity** (FOV 1: Cortical Layers):
- Expected accuracy: >90%
- F1 (macro): >0.85
- Well-separated layers make classification straightforward

**Hard + Very High Sparsity** (FOV 5: Tumor Microenv):
- Expected accuracy: <60%
- F1 (macro): <0.50
- Sparse data + similar cell types = challenging

### Key Questions to Answer

1. How does accuracy degrade with increasing sparsity?
2. Which tissue architectures are most challenging?
3. Do methods perform differently on layered vs clustered structures?
4. Which cell types are most frequently confused?
5. Does spatial structure help in low-sparsity conditions?

## Benchmark Characteristics

| FOV | Name | Sparsity | Difficulty | Cells | Transcripts | Trans/Cell |
|-----|------|----------|------------|-------|-------------|------------|
| 0 | Glandular_Crypts | medium | easy | ~1500 | ~18000 | ~12 |
| 1 | Cortical_Layers | low | medium | ~1600 | ~24000 | ~15 |
| 2 | Vascular_Network | high | easy | ~1200 | ~9600 | ~8 |
| 3 | Lymphoid_Follicles | medium | medium | ~1400 | ~16800 | ~12 |
| 4 | Muscle_Fibers | low | hard | ~800 | ~12000 | ~15 |
| 5 | Tumor_Microenv | very_high | hard | ~1300 | ~6500 | ~5 |
| 6 | Fibrillar_Stroma | high | medium | ~1100 | ~8800 | ~8 |
| 7 | Glandular_Units | medium | medium | ~1500 | ~18000 | ~12 |
| 8 | Branching_Ducts | very_low | easy | ~1400 | ~28000 | ~20 |
| 9 | Tissue_Interface | high | hard | ~1300 | ~10400 | ~8 |

*Note: Exact values will vary due to stochastic generation*

## Cell Types

The benchmark uses 8 cell types:
1. **Epithelial** - Glandular/ductal lining cells
2. **Stromal** - Connective tissue fibroblasts
3. **Endothelial** - Blood vessel lining
4. **Immune_T** - T lymphocytes
5. **Immune_B** - B lymphocytes
6. **Myocyte** - Muscle cells
7. **Fibroblast** - Specialized stromal cells
8. **Neural** - Nerve cells

## Reproducibility

All random seeds are fixed for reproducibility:
- Expression matrix: seed=42
- FOV generation: seed=42+fov_id
- Technical noise: Controlled by fixed parameters

Running the notebook multiple times will produce identical results.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{pointillsim_benchmark,
  title={PointillSim Sparsity Benchmark for Cell Type Mapping},
  author={La Manno Lab},
  year={2024},
  howpublished={\url{https://github.com/lamanno-epfl/PointillSim}}
}
```

## Future Benchmarks

Planned additional benchmarks:
- **Batch Effects**: Cross-platform and cross-sample variation
- **Admixture**: Transcript misassignment due to segmentation errors
- **Spatial Effects**: Admixture from z-axis contamination
- **Multi-modal**: Integration of spatial + scRNA-seq data
- **Dynamic Range**: Varying gene expression levels

## Contact

For questions or issues with this benchmark:
- Open an issue: https://github.com/lamanno-epfl/PointillSim/issues
- Email: [contact information]

---

**Last updated**: 2024
**PointillSim version**: Latest
