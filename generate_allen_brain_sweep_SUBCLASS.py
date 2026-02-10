#!/usr/bin/env python3
"""
Allen Brain Section Sweep — Real Spatial Layout with Controlled Degradation

Uses a real Allen Brain Atlas section (C57BL6J-638850_36.h5ad) as ground truth:
- Cell positions and cell type labels are FIXED (from the real section)
- Gene expression reference is loaded from the Yao2023 class-level reference matrix
- The gene panel defaults to the 550-gene MERFISH panel from the h5ad, but can
  optionally be sampled from the full ~32k reference genes

Three sweep cases (5 settings each = 15 datasets total):

CASE 1 — NUMBER OF GENES
  Start from the full panel, subset to top-HVG genes.
  Settings: 500, 200, 100, 50, 25 genes

CASE 2 — GENE SENSITIVITY
  Same panel, vary detection sensitivity.
  Settings: 0.9, 0.6, 0.3, 0.1, 0.05

CASE 3 — NOISE LEVELS
  Same panel, progressively add admixture + dropout + background noise.
  Settings: minimal, low, moderate, high, severe

Seeds:
  GEOMETRY_SEED is irrelevant (real positions), but EXPRESSION_SEED controls
  stochastic elements (sensitivity sampling, Poisson counts, noise).
"""

import os
import json
import numpy as np
import pandas as pd
import anndata
from scipy.spatial import cKDTree

from pointillsim import (
    TissueCellTypes,
    CellTypesProperties,
    HybISS_Setup,
    FOV,
)

from pointillsim.effects import (
    Lateral2DAdmixture,
    ZAxisAdmixture,
    CompositeAdmixture,
    BackgroundNoise,
    DropoutModel,
)
from pointillsim.experiment.transfer import IdentityTransfer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
H5AD_PATH = "allen_brain_files/C57BL6J-638850_36.h5ad"
REFERENCE_CSV_PATH = "allen_brain_files/reference_matrix_Yao2023_full_subclass.csv"
OUTPUT_FOLDER = "allen_brain_files/sweep_output_SUBCLASS"

# Column in h5ad containing class labels
CLASS_LABEL_COL = "subclass_label"
# Whether to use the full reference gene set (True) or only the h5ad panel (False)
USE_FULL_REFERENCE_GENES = False

# Random seed for stochastic experiment components
EXPRESSION_SEED = 999

# Cell morphology (uniform defaults for brain tissue)
CELL_SIZE = 10.0
CELL_SIZE_VARIATION = 2.0
CELL_ANISOTROPY = 0.80
CELL_ANISOTROPY_VARIATION = 0.10
RELATIVE_RNA_CONC = 1.0
RNA_CONC_VARIATION = 0.0

# Transfer function defaults
TRANSFER_SCALES = 1.0
TRANSFER_SCALES_STD = 0.3
TRANSFER_OFFSETS = 0.15
TRANSFER_OFFSETS_STD = 0.08


# ============================================================================
# 1. LOAD REAL SECTION DATA
# ============================================================================

print("=" * 80)
print("Allen Brain Section Sweep — Real Layout with Controlled Degradation")
print("=" * 80)

print("\n1. Loading real section data...")

adata = anndata.read_h5ad(H5AD_PATH)
print(f"   Section: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"   Cell type classes: {adata.obs[CLASS_LABEL_COL].nunique()}")

# Extract spatial coordinates (use 2D: x, y)
spatial_coords = adata.obsm["spatial"][:, :2].astype(np.float64)
print(f"   Spatial range: x=[{spatial_coords[:,0].min():.1f}, {spatial_coords[:,0].max():.1f}], "
      f"y=[{spatial_coords[:,1].min():.1f}, {spatial_coords[:,1].max():.1f}]")

# Scale coordinates to microns (the raw coords are in mm-like units)
# Multiply by 1000 to get micron-scale positions suitable for PointillSim
COORD_SCALE = 1000.0
spatial_coords_um = spatial_coords * COORD_SCALE
FRAME_WIDTH = spatial_coords_um[:, 0].max() - spatial_coords_um[:, 0].min()
FRAME_HEIGHT = spatial_coords_um[:, 1].max() - spatial_coords_um[:, 1].min()
print(f"   Scaled to µm: {FRAME_WIDTH:.0f} x {FRAME_HEIGHT:.0f} µm")

# Cell type labels
cell_labels = adata.obs[CLASS_LABEL_COL].values
unique_classes = sorted(set(cell_labels))
n_cell_types = len(unique_classes)
class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
cell_type_indices = np.array([class_to_idx[c] for c in cell_labels])

print(f"   Cell types ({n_cell_types}): {unique_classes[:5]}... ")


# ============================================================================
# 2. LOAD REFERENCE EXPRESSION MATRIX
# ============================================================================

print("\n2. Loading reference expression matrix...")

ref_df = pd.read_csv(REFERENCE_CSV_PATH, index_col=0)
# Drop NaN rows (last row)
ref_df = ref_df.dropna(axis=0, how="all")
ref_df = ref_df.loc[ref_df.index.dropna()]

# Strip numeric prefix from class names (e.g., '01 IT-ET Glut' -> 'IT-ET Glut')
ref_class_map = {}
for orig_name in ref_df.index:
    if isinstance(orig_name, str):
        parts = orig_name.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            ref_class_map[orig_name] = parts[1]
        else:
            ref_class_map[orig_name] = orig_name

ref_df.index = [ref_class_map.get(n, n) for n in ref_df.index]
print(f"   Reference: {ref_df.shape[0]} classes × {ref_df.shape[1]} genes")

# Determine gene panel
h5ad_genes = list(adata.var_names)  # 550 MERFISH panel genes

if USE_FULL_REFERENCE_GENES:
    panel_genes = list(ref_df.columns)
    print(f"   Using FULL reference panel: {len(panel_genes)} genes")
else:
    # Use h5ad panel genes that exist in the reference
    panel_genes = [g for g in h5ad_genes if g in ref_df.columns]
    print(f"   Using h5ad panel genes found in reference: {len(panel_genes)} / {len(h5ad_genes)}")

# Build expression matrix: genes × cell_types
# Rows = panel genes, columns = cell types (ordered as unique_classes)
expr_rows = []
missing_classes = []
for cls in unique_classes:
    if cls in ref_df.index:
        expr_rows.append(ref_df.loc[cls, panel_genes].values.astype(float))
    else:
        missing_classes.append(cls)
        expr_rows.append(np.zeros(len(panel_genes)))

if missing_classes:
    print(f"   WARNING: {len(missing_classes)} classes not in reference: {missing_classes}")

# expression_matrix shape: (n_genes, n_cell_types)
expression_matrix = np.array(expr_rows).T  # (n_genes, n_cell_types)
print(f"   Expression matrix: {expression_matrix.shape[0]} genes × {expression_matrix.shape[1]} cell types")
print(f"   Expression range: [{expression_matrix.min():.2f}, {expression_matrix.max():.2f}]")


# ============================================================================
# 3. BUILD POINTILLSIM OBJECTS
# ============================================================================

print("\n3. Building PointillSim objects...")

# --- TissueCellTypes from reference ---
tissue = TissueCellTypes(gene_expression_by_type=expression_matrix)
tissue._cell_type_names = list(unique_classes)
tissue._gene_names = list(panel_genes)
print(f"   TissueCellTypes: {tissue.n_genes} genes, {tissue.n_cell_types} cell types")

# --- FOV from real cell positions and one-hot labels ---
n_cells = len(cell_type_indices)
cell_probs = np.eye(n_cell_types)[cell_type_indices]  # one-hot

fov = FOV(
    cell_centroids=spatial_coords_um,
    cell_probabilities=cell_probs,
)
fov.class_instance_one_hot = cell_probs.astype(int)
print(f"   FOV: {fov.n_cells} cells, {fov.n_cell_types} types")

# --- CellTypesProperties ---
cell_props = CellTypesProperties(
    n_cell_types=n_cell_types,
    sizes=CELL_SIZE,
    size_variation=CELL_SIZE_VARIATION,
    anisotropy=CELL_ANISOTROPY,
    anisotropy_variation=CELL_ANISOTROPY_VARIATION,
    relative_rna_concentration=RELATIVE_RNA_CONC,
    rna_concentration_variation=RNA_CONC_VARIATION,
)

np.random.seed(EXPRESSION_SEED)
cell_props.apply(fov)
print(f"   Applied morphology to {fov.n_cells} cells")


# ============================================================================
# 4. SHARED OUTPUTS
# ============================================================================

print("\n4. Generating shared cells dataframe...")
cells_df = fov.make_pandas_df()
print(f"   Total cells: {len(cells_df)}")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_case(folder_name, tissue_obj, dots_df, case_params):
    """Save cells.csv, dots.csv, cell_type_expression.csv, and hyperparameters.json."""
    output_dir = f"{OUTPUT_FOLDER}/{folder_name}"
    os.makedirs(output_dir, exist_ok=True)

    cells_df.to_csv(f"{output_dir}/cells.csv", index=False)
    dots_df.to_csv(f"{output_dir}/dots.csv", index=False)

    tissue_df = tissue_obj.make_pandas_df()
    tissue_df.to_csv(f"{output_dir}/cell_type_expression.csv")

    with open(f"{output_dir}/hyperparameters.json", "w") as f:
        json.dump(case_params, f, indent=2)

    print(f"   -> {folder_name}: {len(dots_df):,} dots, {tissue_obj.n_genes} genes")


# ============================================================================
# CASE 1: NUMBER OF GENES
#
#        Use full panel, subset by HVG (variance across cell types).
# ============================================================================

print("\n" + "=" * 80)
print("CASE 1: NUMBER OF GENES")
print("=" * 80)

np.random.seed(EXPRESSION_SEED)

hybiss_c1 = HybISS_Setup(
    tissue,
    genes_sensitivities=1.0,
    genes_sensitivities_variation=0.0,
    transfer_function=IdentityTransfer(),
)
hybiss_c1.observe_dots(fov)
dots_df_full = hybiss_c1.make_pandas_df()
print(f"   Full-panel dots: {len(dots_df_full):,}")

# Compute HVG scores (variance across cell types in the reference)
hvg_scores = np.var(expression_matrix, axis=1)
hvg_ranking = np.argsort(hvg_scores)[::-1]  # descending by variance
all_gene_names = list(tissue.gene_names)

for n_genes_target in [500, 200, 100, 50, 25]:
    folder = f"CASE1_{n_genes_target}genes"

    # Select top HVG genes (or all if fewer than target)
    n_select = min(n_genes_target, len(all_gene_names))
    selected_indices = hvg_ranking[:n_select]
    selected_gene_names = [all_gene_names[i] for i in selected_indices]

    # Subset expression matrix
    tissue_sub = TissueCellTypes(
        gene_expression_by_type=expression_matrix[selected_indices, :]
    )
    tissue_sub._cell_type_names = list(unique_classes)
    tissue_sub._gene_names = selected_gene_names

    # Subset dots to selected genes only
    selected_set = set(selected_gene_names)
    dots_subset = dots_df_full[dots_df_full["gene"].isin(selected_set)].copy()

    case_params = {
        "case": "CASE1_number_of_genes",
        "n_genes": n_select,
        "gene_selection": "top_hvg_by_variance",
        "gene_sensitivity": 1.0,
        "gene_sensitivity_variation": 0.0,
        "transfer_function": "IdentityTransfer",
        "relative_rna_concentration": RELATIVE_RNA_CONC,
        "rna_concentration_variation": RNA_CONC_VARIATION,
        "expression_seed": EXPRESSION_SEED,
        "source_section": H5AD_PATH,
        "reference_matrix": REFERENCE_CSV_PATH,
        "use_full_reference_genes": USE_FULL_REFERENCE_GENES,
    }

    save_case(folder, tissue_sub, dots_subset, case_params)


# ============================================================================
# CASE 2: EFFECT OF GENE SENSITIVITY
# 
#        Same reference, vary sensitivity (variation = sensitivity / 2).
# ============================================================================

print("\n" + "=" * 80)
print("CASE 2: EFFECT OF GENE SENSITIVITY")
print("=" * 80)

print(f"   Reference: {tissue.n_genes} genes from real reference matrix")

for gs in [1.00, 0.60, 0.30, 0.10, 0.05]:
    gs_var = gs / 2.0
    folder = f"CASE2_gs{gs}"
    print(f"\n   {folder} (sensitivity={gs}, variation={gs_var})...")

    # Reset seed so TF and sensitivity sampling are reproducible per sub-case
    np.random.seed(EXPRESSION_SEED + 1000)

    hybiss_c2 = HybISS_Setup(
        tissue,
        genes_sensitivities=gs,
        genes_sensitivities_variation=gs_var,
        transfer_function=IdentityTransfer(),
    )
    hybiss_c2.observe_dots(fov)
    dots_df_c2 = hybiss_c2.make_pandas_df()

    case_params = {
        "case": "CASE2_gene_sensitivity",
        "n_genes": tissue.n_genes,
        "gene_sensitivity": gs,
        "gene_sensitivity_variation": gs_var,
        "transfer_function": "IdentityTransfer",
        "relative_rna_concentration": RELATIVE_RNA_CONC,
        "rna_concentration_variation": RNA_CONC_VARIATION,
        "expression_seed": EXPRESSION_SEED,
        "source_section": H5AD_PATH,
        "reference_matrix": REFERENCE_CSV_PATH,
    }

    save_case(folder, tissue, dots_df_c2, case_params)


# ============================================================================
# CASE 3: EFFECT OF NOISE LEVELS
# 
#        Same reference, progressively increase noise sources:
#        - Lateral 2D admixture (segmentation boundary errors)
#        - Z-axis admixture (out-of-plane contamination)
#        - Transcript dropout (detection failure)
#        - Background noise (false positive transcripts)
# ============================================================================

print("\n" + "=" * 80)
print("CASE 3: EFFECT OF NOISE LEVELS")
print("=" * 80)

# Generate clean reference dots
np.random.seed(EXPRESSION_SEED + 2000)

hybiss_c3 = HybISS_Setup(
    tissue,
    genes_sensitivities=1.0,
    genes_sensitivities_variation=0.0,
    transfer_function=IdentityTransfer(),
)
hybiss_c3.observe_dots(fov)
dots_df_c3_clean = hybiss_c3.make_pandas_df()
print(f"   Clean dots: {len(dots_df_c3_clean):,}")

# Precompute FOV properties for admixture
cell_centroids = fov.cell_centroids
cell_types = fov.class_instance
cell_radii = fov.cell_major_axis / 2
n_cells_fov = fov.n_cells
gene_names_c3 = list(tissue.gene_names)
n_genes_c3 = len(gene_names_c3)

# Precompute coordinate extent for background noise
x_min, x_max = cell_centroids[:, 0].min(), cell_centroids[:, 0].max()
y_min, y_max = cell_centroids[:, 1].min(), cell_centroids[:, 1].max()
# Use max dimension as frame_size for BackgroundNoise (square FOV API)
bg_frame_size = int(np.ceil(max(x_max - x_min, y_max - y_min))) + 1

# Build KDTree for nearest-cell assignment of background dots
cell_tree = cKDTree(cell_centroids)

# 5 noise levels: minimal -> severe
noise_configs = [
    # (name, lateral_bw, lateral_tr, lateral_dd,
    #  z_rate, z_corr, z_radius,
    #  dropout_rate, dropout_expr_dep, dropout_gene_cv,
    #  bg_rate)
    ("noise1_minimal", 3.0, 0.05, 0.5,
     0.02, 0.9, 50.0,
     0.98, 0.05, 0.05,
     0.0002),
    ("noise2_low", 6.0, 0.25, 0.3,
     0.10, 0.75, 60.0,
     0.85, 0.20, 0.15,
     0.001),
    ("noise3_moderate", 8.0, 0.35, 0.3,
     0.15, 0.70, 60.0,
     0.75, 0.30, 0.20,
     0.002),
    ("noise4_high", 10.0, 0.45, 0.2,
     0.20, 0.60, 70.0,
     0.65, 0.40, 0.30,
     0.004),
    ("noise5_severe", 12.0, 0.55, 0.15,
     0.30, 0.50, 80.0,
     0.50, 0.50, 0.40,
     0.008),
]

for (noise_name, lat_bw, lat_tr, lat_dd,
     z_rate, z_corr, z_radius,
     det_rate, det_expr_dep, det_gene_cv,
     bg_rate) in noise_configs:

    folder = f"CASE3_{noise_name}"
    print(f"\n   {folder}:")
    print(f"     Lateral admixture: bw={lat_bw}, rate={lat_tr}")
    print(f"     Z-axis admixture:  rate={z_rate}, corr={z_corr}")
    print(f"     Dropout:           detection={det_rate}")
    print(f"     Background:        rate={bg_rate}")

    # --- Step 1: Apply admixture (lateral + z-axis) ---
    lateral_admix = Lateral2DAdmixture(
        boundary_width=lat_bw,
        transfer_rate=lat_tr,
        distance_decay=lat_dd,
        seed=EXPRESSION_SEED + 3000,
    )
    z_admix = ZAxisAdmixture(
        z_contamination_rate=z_rate,
        neighborhood_correlation=z_corr,
        neighborhood_radius=z_radius,
        seed=EXPRESSION_SEED + 3001,
    )
    combined_admix = CompositeAdmixture(models=[lateral_admix, z_admix])

    dots_noisy = combined_admix.apply(
        dots_df_c3_clean.copy(),
        cell_centroids,
        cell_types,
        cell_radii,
    )
    admix_summary = combined_admix.get_admixture_summary()
    n_reassigned = admix_summary["n_reassigned"]
    print(f"     -> Admixture reassigned {n_reassigned} dots "
          f"({admix_summary['reassignment_rate']:.1%})")

    # --- Step 2: Apply dropout using DropoutModel API ---
    dropout_model = DropoutModel(
        baseline_detection_rate=det_rate,
        expression_dependence=det_expr_dep,
        gene_variation_cv=det_gene_cv,
        seed=EXPRESSION_SEED + 4000,
    )
    dropout_model.generate_gene_detection_rates(n_genes_c3)

    # Map per-gene detection rates to each dot
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names_c3)}
    dot_gene_idx = dots_noisy["gene"].map(gene_name_to_idx).values
    per_dot_factor = dropout_model.gene_detection_rates[dot_gene_idx]
    detection_prob = np.clip(det_rate * per_dot_factor, 0.0, 1.0)
    keep_mask = dropout_model.rng.random(len(dots_noisy)) < detection_prob
    n_dropped = (~keep_mask).sum()
    dots_noisy = dots_noisy[keep_mask].reset_index(drop=True)
    print(f"     -> Dropout removed {n_dropped} dots "
          f"({n_dropped / (len(dots_noisy) + n_dropped):.1%})")

    # --- Step 3: Add background noise using BackgroundNoise API ---
    # Adjust background_rate for the square frame vs actual rectangular area
    actual_area = (x_max - x_min) * (y_max - y_min)
    effective_bg_rate = bg_rate * actual_area / (bg_frame_size ** 2)

    bg_noise_model = BackgroundNoise(
        frame_size=bg_frame_size,
        background_rate=effective_bg_rate,
        spatial_pattern="uniform",
        seed=EXPRESSION_SEED + 5000,
    )
    bg_x, bg_y, bg_gene_idx = bg_noise_model.generate_background_dots(
        n_genes=n_genes_c3,
        gene_names=gene_names_c3,
    )
    n_bg = len(bg_x)

    if n_bg > 0:
        # Translate from [0, frame_size] to tissue coordinate system
        bg_x = bg_x / bg_frame_size * (x_max - x_min) + x_min
        bg_y = bg_y / bg_frame_size * (y_max - y_min) + y_min
        bg_gene_names = [gene_names_c3[gi] for gi in bg_gene_idx]
        _, bg_cell_idx = cell_tree.query(np.column_stack([bg_x, bg_y]))

        bg_df = pd.DataFrame({
            "x": bg_x,
            "y": bg_y,
            "gene": bg_gene_names,
            "cell": bg_cell_idx.astype(int),
        })
        dots_noisy = pd.concat([dots_noisy, bg_df], ignore_index=True)
        print(f"     -> Background added {n_bg} false-positive dots")
    else:
        print(f"     -> Background added 0 false-positive dots")

    print(f"     -> Final dot count: {len(dots_noisy):,} "
          f"(clean: {len(dots_df_c3_clean):,})")

    case_params = {
        "case": "CASE3_noise_level",
        "noise_level": noise_name,
        "n_genes": tissue.n_genes,
        "gene_sensitivity": 1.0,
        "gene_sensitivity_variation": 0.0,
        "transfer_function": "IdentityTransfer",
        "lateral_admixture_boundary_width": lat_bw,
        "lateral_admixture_transfer_rate": lat_tr,
        "lateral_admixture_distance_decay": lat_dd,
        "z_admixture_contamination_rate": z_rate,
        "z_admixture_neighborhood_correlation": z_corr,
        "z_admixture_neighborhood_radius": z_radius,
        "dropout_baseline_detection_rate": det_rate,
        "dropout_expression_dependence": det_expr_dep,
        "dropout_gene_variation_cv": det_gene_cv,
        "background_noise_rate": bg_rate,
        "n_dots_clean": len(dots_df_c3_clean),
        "n_dots_admixture_reassigned": int(n_reassigned),
        "n_dots_dropped": int(n_dropped),
        "n_dots_background_added": int(n_bg),
        "n_dots_final": len(dots_noisy),
        "relative_rna_concentration": RELATIVE_RNA_CONC,
        "rna_concentration_variation": RNA_CONC_VARIATION,
        "expression_seed": EXPRESSION_SEED,
        "source_section": H5AD_PATH,
        "reference_matrix": REFERENCE_CSV_PATH,
    }

    save_case(folder, tissue, dots_noisy, case_params)


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("HYPERPARAMETER SWEEP COMPLETE — 15 DATASETS GENERATED")
print("=" * 80)

print(f"\nOutput location: {OUTPUT_FOLDER}/")
print(f"\nSource section: {H5AD_PATH}")
print(f"Reference matrix: {REFERENCE_CSV_PATH}")
print(f"Spatial layout: {len(cells_df)} cells (REAL positions, fixed across all cases)")
print(f"Cell types: {n_cell_types} ({', '.join(unique_classes[:5])}...)")
print(f"Gene panel: {tissue.n_genes} genes")
print(f"Expression seed: {EXPRESSION_SEED}")

print(f"\nCASE 1 — Number of genes (5 datasets):")
print(f"  CASE1_500genes, CASE1_200genes, CASE1_100genes, "
      f"CASE1_50genes, CASE1_25genes")
print(f"  Gene selection: top HVG by variance across cell types in reference")

print(f"\nCASE 2 — Gene sensitivity (5 datasets):")
print(f"  CASE2_gs1.00, CASE2_gs0.60, CASE2_gs0.30, CASE2_gs0.10, CASE2_gs0.05")
print(f"  Full panel, same reference, variation = sensitivity/2")

print(f"\nCASE 3 — Noise levels (5 datasets):")
print(f"  CASE3_noise1_minimal, CASE3_noise2_low, CASE3_noise3_moderate, "
      f"CASE3_noise4_high, CASE3_noise5_severe")
print(f"  Full panel, same reference, increasing admixture + dropout + background")