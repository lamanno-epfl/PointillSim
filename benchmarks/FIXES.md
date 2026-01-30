# Benchmark Notebook Fixes

## Summary

Fixed all API errors in `sparsity_benchmark.ipynb` to ensure all cells can run without errors. The main issue was incorrect usage of the `MixOfNCellTypesRule` API.

## Main Fix: MixOfNCellTypesRule API

### Problem
The `MixOfNCellTypesRule` was being called with positional arguments:
```python
MixOfNCellTypesRule(n_cell_types, [1, 6], [0.7, 0.3])  # WRONG
```

This caused the error:
```
ValueError: N and list_N must be consistent
```

### Solution
Use keyword arguments `list_N` and `proportions`:
```python
MixOfNCellTypesRule(n_cell_types, list_N=[1, 6], proportions=[0.7, 0.3])  # CORRECT
```

## All Fixes Applied

### Cell 7: FOV Generators
Fixed all `MixOfNCellTypesRule` calls in:
1. Glandular Crypts background
2. Cortical Layers (5 layer rules)
3. Vascular Network background
4. Lymphoid Follicles background
5. Muscle Fibers background
6. Tumor Microenvironment background and cluster rules
7. Fibrillar Stroma background
8. Glandular Units background
9. Branching Ducts background
10. Tissue Interface (both sides)

### Cell 11: Visualization Function
Fixed `fov.frame_size` AttributeError:
- **Problem**: FOV objects don't have a `frame_size` attribute
- **Solution**: Pass `frame_size` as parameter and use `BENCHMARK_CONFIG['frame_size']`
- Changed function signature: `visualize_fov(fov_data, frame_size, show_transcripts=True)`

### Cell 12: Gallery Overview
Fixed `fov.frame_size` AttributeError:
- Added `frame_size = BENCHMARK_CONFIG['frame_size']` at the beginning
- Changed `fov.frame_size` to `frame_size` throughout the cell

### Dependencies Installed
Required packages installed:
- shapely
- scipy
- matplotlib
- pandas
- scikit-learn
- seaborn
- tqdm
- numpy (compatible version: >=1.23.5, <2.3)

## Verification

Successfully tested:
1. ✓ `MixOfNCellTypesRule` with multiple cell types
2. ✓ `MixOfNCellTypesRule` with single cell type
3. ✓ `LayerRule` creation
4. ✓ Full benchmark pipeline:
   - Tissue generation (100 genes, 8 cell types)
   - FOV generation (1800+ cells)
   - Cell properties application
   - Transcript generation (19000+ dots)
   - Data export

## Running the Notebook

The notebook should now run without errors. All cells are functional:

1. **Imports and setup** (Cell 1) - ✓
2. **Configuration** (Cell 3) - ✓
3. **Expression matrix** (Cell 5) - ✓
4. **FOV generators** (Cell 7) - ✓ Fixed
5. **Generate FOVs** (Cell 9) - ✓
6. **Visualization** (Cell 11) - ✓
7. **Difficulty metrics** (Cell 14) - ✓
8. **Data export** (Cell 16) - ✓
9. **Summary statistics** (Cell 18) - ✓

## Next Steps

To run the benchmark:
1. Open the notebook: `jupyter notebook benchmarks/sparsity_benchmark.ipynb`
2. Run all cells
3. Generated data will be in:
   - `benchmarks/data/` - CSV files for each FOV
   - `benchmarks/figures/` - Visualizations
   - `benchmarks/results/` - Summary tables

The benchmark will generate 10 diverse tissue FOVs with controllable sparsity and difficulty levels for evaluating cell type mapping algorithms.
