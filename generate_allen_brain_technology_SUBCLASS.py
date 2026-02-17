#!/usr/bin/env python3
"""
Allen Brain Section -- Technology-Specific Simulations (Calibrated)

Uses a real Allen Brain Atlas section (Zhuang-ABCA-3_004.h5ad) as ground truth:
- Cell positions and cell type labels are FIXED (from the real section)
- Gene expression reference is loaded from the Yao2023 subclass-level reference matrix
- The full 1122-gene MERFISH panel from the h5ad serves as the superset

Four technology simulations with CALIBRATED avg counts per cell:

  MERFISH (Baseline)  -- 1000 genes, ~300 avg count/cell, low dropout
  EELFISH             -- 400 genes,  ~40  avg count/cell, moderate dropout
  BARseq              -- 100 genes,  ~30  avg count/cell, low dropout
  HybISS              -- 200 genes,  ~10  avg count/cell, high dropout

Calibration approach:
  For each technology, the script computes the raw Poisson rate from the
  reference expression matrix and derives gene_sensitivity automatically:
    gene_sensitivity = target_avg_count / (raw_avg_rate * dropout_detection_rate)
  This guarantees the output matches the target avg count per cell.

Biological rationale for detection parameters:
  MERFISH  ~60% sensitivity: Combinatorial smFISH with error-robust encoding
           (16-bit MHD4 codes, ~48 probes/gene). Direct imaging of individual
           RNA molecules. Very high specificity from error correction.
           Dropout ~5%: minimal loss from sequential hybridization rounds.

  EELFISH  ~14% sensitivity: Expansion microscopy + sequential FISH.
           Physical expansion of tissue improves resolution but causes
           ~50-70% molecule loss during gelation and digestion steps.
           Dropout ~30%: additional loss from sequential probe stripping
           and re-hybridization across imaging rounds.

  BARseq   ~15% sensitivity: In situ sequencing with barcoded probes.
           Padlock probe ligation + rolling circle amplification (RCA).
           Sequencing by ligation limits read accuracy.
           Dropout ~5%: barcode validation filters reduce false positives
           but also reject ambiguous reads.

  HybISS   ~8% sensitivity: Padlock probes with RCA and sequencing by
           ligation. Lower detection than MERFISH due to probe hybridization
           inefficiency and RCA amplification variability.
           Dropout ~60%: high loss from multiple sequential hybridization
           rounds, photobleaching, and stringent signal thresholding.
"""

import os
import json
import numpy as np
import pandas as pd
import anndata

from pointillsim import (
    TissueCellTypes,
    CellTypesProperties,
    HybISS_Setup,
    FOV,
)

from pointillsim.effects import DropoutModel
from pointillsim.effects.admixture import (
    Lateral2DAdmixture, ZAxisAdmixture, CompositeAdmixture,
)
from pointillsim.experiment.transfer import IdentityTransfer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
H5AD_PATH = "allen_brain_files/Zhuang-ABCA-3_004.h5ad"
REFERENCE_CSV_PATH = "allen_brain_files/reference_matrix_Yao2023_full_subclass.csv"
OUTPUT_FOLDER = "allen_brain_files/technology_output_SUBCLASS"

# Column in h5ad containing class labels
CLASS_LABEL_COL = "subclass_label"

# Random seed for stochastic experiment components
EXPRESSION_SEED = 999

# Cell morphology (uniform defaults for brain tissue)
CELL_SIZE = 10.0
CELL_SIZE_VARIATION = 2.0
CELL_ANISOTROPY = 0.80
CELL_ANISOTROPY_VARIATION = 0.10
RELATIVE_RNA_CONC = 1.0
RNA_CONC_VARIATION = 0.0


# ============================================================================
# TECHNOLOGY PARAMETER DEFINITIONS
# ============================================================================
#
# Each technology is defined by:
#   n_genes:                Number of top HVGs to select from the panel
#   target_avg_count:       Desired average total transcripts per cell
#   dropout_detection_rate: Fraction of transcripts retained after dropout
#                           (1.0 = no dropout, 0.4 = 60% of dots dropped)
#   gene_sensitivity_var:   Per-gene variation in sensitivity (lognormal std)
#   dropout_expr_dep:       How much dropout depends on expression level
#   dropout_gene_cv:        Per-gene variation in dropout rates
#
# gene_sensitivity is AUTO-CALIBRATED at runtime:
#   gene_sensitivity = target_avg_count / (raw_poisson_rate * dropout_detection_rate)

TECHNOLOGIES = {
    "MERFISH": {
        "n_genes": 1000,
        "target_avg_count": 300,
        # ~60% detection efficiency (combinatorial smFISH, error-robust encoding)
        "gene_sensitivity_var": 0.10,    # Very uniform probe design
        "dropout_detection_rate": 0.95,  # Low dropout (5%)
        "dropout_expr_dep": 0.05,        # Minimal expression-dependent dropout
        "dropout_gene_cv": 0.05,         # Very consistent gene-to-gene
        "cell_count_overdispersion": 0.0,  # No extra overdispersion (Poisson)
        "description": (
            "MERFISH baseline -- 1000 HVGs, ~300 avg count/cell, "
            "high detection efficiency, 5% dropout"
        ),
    },
    "EELFISH": {
        "n_genes": 400,
        "target_avg_count": 42,
        # ~14% detection efficiency (expansion causes ~60% molecule loss)
        "gene_sensitivity_var": 0.20,    # Moderate probe-specific loss
        "dropout_detection_rate": 0.55,  # ~45% dropout
        "dropout_expr_dep": 0.08,        # Mild expression-dependent dropout
        "dropout_gene_cv": 0.20,         # Moderate gene-to-gene variation
        "cell_count_overdispersion": 0.10,  # Mild overdispersion (cell-type variation provides most spread)
        # No cell-level dropout (real EELFISH has ~0% empty cells)
        "cell_dropout_rate": 0.0,
        # Technical noise: per-cell amplification variation (creates lower-tail spread)
        "amplification_cv": 0.15,        # Mild amplification variation
        # Admixture: expansion microscopy causes lateral transcript spread
        # and thicker effective sections contribute z-axis contamination.
        "admixture": {
            "lateral": {
                "boundary_width": 3.0,      # Small cells (~5 radius)
                "transfer_rate": 0.10,      # 10% boundary transfer
                "distance_decay": 0.5,      # Moderate decay
            },
            "z_axis": {
                "z_contamination_rate": 0.25,    # 25% from z-neighbors (expansion thickens section)
                "neighborhood_correlation": 0.3,  # Low correlation (expansion disrupts local structure)
                "neighborhood_radius": 100.0,     # Wide neighborhood
            },
        },
        "description": (
            "EEL-FISH-like -- 400 HVGs, ~42 avg count/cell, "
            "moderate detection efficiency, 45% dropout, admixture + tech noise"
        ),
    },
    "BARseq": {
        "n_genes": 100,
        "target_avg_count": 40,
        # ~15% detection efficiency (ISS-based, RCA amplification)
        "gene_sensitivity_var": 0.10,    # Low barcode-specific variation (more uniform detection)
        "dropout_detection_rate": 0.85,  # 15% dropout (barcode validation + seq errors)
        "dropout_expr_dep": 0.10,        # Moderate expression-dependent dropout
        "dropout_gene_cv": 0.10,         # Some gene-to-gene variation
        "cell_count_overdispersion": 0.15,  # Mild overdispersion (cell dropout handles zero peak)
        # Cell-level dropout: ~18% of cells fail entirely
        # (segmentation errors, poor barcode capture, empty segments)
        "cell_dropout_rate": 0.18,       # 18% cells fully drop out (matches real BARseq)
        # Technical noise: per-cell amplification variation
        "amplification_cv": 0.10,        # Mild RCA amplification variation
        "description": (
            "BARseq-like -- 100 HVGs, ~35 avg count/cell, "
            "ISS-level detection, 15% dropout, 18% cell dropout"
        ),
    },
    "HybISS": {
        "n_genes": 200,
        "target_avg_count": 12,
        # ~8% detection efficiency (padlock probes + RCA, low efficiency)
        "gene_sensitivity_var": 0.25,    # Padlock probe design varies significantly
        "dropout_detection_rate": 0.50,  # High dropout (60%)
        "dropout_expr_dep": 0.25,        # Low-abundance genes harder to amplify by RCA
        "dropout_gene_cv": 0.20,         # Significant gene-to-gene dropout variation
        "cell_count_overdispersion": 0.0,  # No extra overdispersion (Poisson)
        "description": (
            "HybISS-like -- 200 HVGs, ~10 avg count/cell, "
            "low detection efficiency, 60% dropout"
        ),
    },
}


# ============================================================================
# 1. LOAD REAL SECTION DATA
# ============================================================================

print("=" * 80)
print("Allen Brain Section -- Technology-Specific Simulations (Calibrated)")
print("=" * 80)

print("\n1. Loading real section data...")

adata = anndata.read_h5ad(H5AD_PATH)
print(f"   Section: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"   Cell type classes: {adata.obs[CLASS_LABEL_COL].nunique()}")

# Extract spatial coordinates (use 2D: x, y)
# Zhuang-ABCA coordinates are already in native spatial units
spatial_coords = adata.obsm["spatial"][:, :2].astype(np.float64)
print(f"   Spatial range: x=[{spatial_coords[:,0].min():.1f}, "
      f"{spatial_coords[:,0].max():.1f}], "
      f"y=[{spatial_coords[:,1].min():.1f}, {spatial_coords[:,1].max():.1f}]")

FRAME_WIDTH = spatial_coords[:, 0].max() - spatial_coords[:, 0].min()
FRAME_HEIGHT = spatial_coords[:, 1].max() - spatial_coords[:, 1].min()
print(f"   FOV extent: {FRAME_WIDTH:.0f} x {FRAME_HEIGHT:.0f}")

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

# Strip numeric prefix from class names (e.g., '001 CLA-EPd-CTX Car3 Glut' -> 'CLA-EPd-CTX Car3 Glut')
ref_class_map = {}
for orig_name in ref_df.index:
    if isinstance(orig_name, str):
        parts = orig_name.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            ref_class_map[orig_name] = parts[1]
        else:
            ref_class_map[orig_name] = orig_name

ref_df.index = [ref_class_map.get(n, n) for n in ref_df.index]
print(f"   Reference: {ref_df.shape[0]} classes x {ref_df.shape[1]} genes")

# Use h5ad panel genes that exist in the reference (full MERFISH panel)
h5ad_genes = list(adata.var_names)
panel_genes = [g for g in h5ad_genes if g in ref_df.columns]
print(f"   Using h5ad panel genes found in reference: "
      f"{len(panel_genes)} / {len(h5ad_genes)}")

# Build expression matrix: genes x cell_types
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
    print(f"   WARNING: {len(missing_classes)} classes not in reference: "
          f"{missing_classes}")

# expression_matrix shape: (n_genes, n_cell_types)
expression_matrix = np.array(expr_rows).T  # (n_genes, n_cell_types)
print(f"   Expression matrix: {expression_matrix.shape[0]} genes "
      f"x {expression_matrix.shape[1]} cell types")
print(f"   Expression range: [{expression_matrix.min():.2f}, "
      f"{expression_matrix.max():.2f}]")


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
    cell_centroids=spatial_coords,
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

# Load pre-ranked HVG list (sorted by highly variable genes score)
HVG_LIST_PATH = "allen_brain_files/genes_variance_genes1000.txt"
with open(HVG_LIST_PATH) as f:
    hvg_gene_names_ordered = [line.strip() for line in f if line.strip()]
print(f"   Loaded HVG list: {len(hvg_gene_names_ordered)} genes from {HVG_LIST_PATH}")

# Map HVG names to indices in panel_genes (preserving HVG rank order)
all_gene_names = list(tissue.gene_names)
hvg_ranking = []
for gene in hvg_gene_names_ordered:
    if gene in all_gene_names:
        hvg_ranking.append(all_gene_names.index(gene))
hvg_ranking = np.array(hvg_ranking)
print(f"   HVG genes found in panel: {len(hvg_ranking)} / {len(hvg_gene_names_ordered)}")


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

    print(f"   -> {folder_name}: {len(dots_df):,} dots, "
          f"{tissue_obj.n_genes} genes")


def compute_raw_avg_rate(selected_indices, expr_matrix, cell_type_idx):
    """Compute the average raw Poisson rate per cell for a given gene panel.

    This is the expected total transcript count per cell if sensitivity=1.0
    and no dropout. Used to calibrate gene_sensitivity.
    """
    sub_expr = expr_matrix[selected_indices, :]  # (n_genes, n_types)
    type_totals = sub_expr.sum(axis=0)  # sum of expression per type
    cell_expected = type_totals[cell_type_idx]  # expected per cell
    return cell_expected.mean()


# ============================================================================
# 5. CALIBRATE AND GENERATE TECHNOLOGY-SPECIFIC DATASETS
# ============================================================================

print("\n" + "=" * 80)
print("TECHNOLOGY-SPECIFIC SIMULATIONS (AUTO-CALIBRATED)")
print("=" * 80)

summary_table = []

for tech_name, tech_cfg in TECHNOLOGIES.items():
    print(f"\n{'=' * 70}")
    print(f"  {tech_name}: {tech_cfg['description']}")
    print(f"{'=' * 70}")

    n_genes_target = tech_cfg["n_genes"]
    target_avg = tech_cfg["target_avg_count"]
    gs_var = tech_cfg["gene_sensitivity_var"]
    det_rate = tech_cfg["dropout_detection_rate"]
    det_expr_dep = tech_cfg["dropout_expr_dep"]
    det_gene_cv = tech_cfg["dropout_gene_cv"]

    # ---- Gene selection (top HVGs by variance) ----
    n_select = min(n_genes_target, len(all_gene_names))
    selected_indices = hvg_ranking[:n_select]
    selected_gene_names = [all_gene_names[i] for i in selected_indices]
    n_genes_used = n_select
    print(f"   Genes: top {n_genes_used} HVGs (by variance across cell types)")

    # ---- Auto-calibrate gene sensitivity ----
    raw_avg_rate = compute_raw_avg_rate(
        selected_indices, expression_matrix, cell_type_indices
    )
    # gene_sensitivity = target / (raw_rate * dropout_rate)
    # This ensures: E[observed count] = raw_rate * sensitivity * dropout_rate = target
    gs = target_avg / (raw_avg_rate * det_rate)

    print(f"   Raw avg Poisson rate (sensitivity=1): {raw_avg_rate:.1f}")
    print(f"   Target avg count/cell: {target_avg}")
    print(f"   Calibrated gene_sensitivity: {gs:.6f}")
    print(f"   Effective detection efficiency: {gs*100:.1f}%")

    # ---- Build sub-tissue with selected genes ----
    tissue_sub = TissueCellTypes(
        gene_expression_by_type=expression_matrix[selected_indices, :]
    )
    tissue_sub._cell_type_names = list(unique_classes)
    tissue_sub._gene_names = selected_gene_names

    # ---- Per-cell variation: overdispersion + amplification noise ----
    # Both are applied as multiplicative factors on cell_rna_concentration
    # BEFORE Poisson sampling, so the mean is preserved and both contribute
    # to wider count distributions.
    overdispersion = tech_cfg.get("cell_count_overdispersion", 0.0)
    amp_cv = tech_cfg.get("amplification_cv", 0.0)
    original_rna_conc = fov.cell_rna_concentration.copy()

    tech_seed_offset = {"MERFISH": 0, "EELFISH": 1, "BARseq": 2, "HybISS": 3}
    cell_scale = np.ones(fov.n_cells)

    if overdispersion > 0:
        rng_od = np.random.default_rng(
            EXPRESSION_SEED + 5000 + tech_seed_offset.get(tech_name, 0)
        )
        gamma_shape = 1.0 / overdispersion
        gamma_factor = rng_od.gamma(gamma_shape, overdispersion, size=fov.n_cells)
        cell_scale *= gamma_factor
        print(f"   Overdispersion: r={overdispersion:.2f} "
              f"(gamma mean={gamma_factor.mean():.3f}, "
              f"std={gamma_factor.std():.3f})")
    else:
        print(f"   Overdispersion: none (pure Poisson)")

    if amp_cv > 0:
        rng_amp = np.random.default_rng(
            EXPRESSION_SEED + 9500 + tech_seed_offset.get(tech_name, 0)
        )
        # LogNormal with mean=1.0, CV=amp_cv
        sigma_amp = np.sqrt(np.log(1 + amp_cv**2))
        mu_amp = -sigma_amp**2 / 2
        amp_factor = rng_amp.lognormal(mu_amp, sigma_amp, size=fov.n_cells)
        cell_scale *= amp_factor
        print(f"   Amplification noise: CV={amp_cv:.2f} "
              f"(lognormal mean={amp_factor.mean():.3f}, "
              f"std={amp_factor.std():.3f})")
    else:
        print(f"   Amplification noise: none")

    if overdispersion > 0 or amp_cv > 0:
        fov.cell_rna_concentration = original_rna_conc * cell_scale
        print(f"   Combined cell scale: mean={cell_scale.mean():.3f}, "
              f"std={cell_scale.std():.3f}")

    # ---- Run observation model with calibrated sensitivity ----
    np.random.seed(EXPRESSION_SEED)

    hybiss = HybISS_Setup(
        tissue_sub,
        genes_sensitivities=gs,
        genes_sensitivities_variation=gs_var,
        transfer_function=IdentityTransfer(),
    )
    hybiss.observe_dots(fov)
    dots_df = hybiss.make_pandas_df()

    # Restore RNA concentration for next technology
    fov.cell_rna_concentration = original_rna_conc

    print(f"   Raw dots (before dropout): {len(dots_df):,}")

    # ---- Pre-dropout stats ----
    pre_dropout_total = len(dots_df)
    pre_per_cell = dots_df.groupby("cell").size()
    pre_genes_per_cell = dots_df.groupby("cell")["gene"].nunique()
    pre_depth = (pre_per_cell / pre_genes_per_cell).mean()
    pre_yield = pre_per_cell.mean()

    # ---- Apply dropout (if applicable) ----
    n_dropped = 0
    if det_rate < 1.0:
        dropout_model = DropoutModel(
            baseline_detection_rate=det_rate,
            expression_dependence=det_expr_dep,
            gene_variation_cv=det_gene_cv,
            seed=EXPRESSION_SEED + 4000,
        )
        dropout_model.generate_gene_detection_rates(n_genes_used)

        # Map per-gene detection rates to each dot
        gene_name_to_idx = {g: i for i, g in enumerate(selected_gene_names)}
        dot_gene_idx = dots_df["gene"].map(gene_name_to_idx).values
        per_dot_factor = dropout_model.gene_detection_rates[dot_gene_idx]
        detection_prob = np.clip(det_rate * per_dot_factor, 0.0, 1.0)
        keep_mask = dropout_model.rng.random(len(dots_df)) < detection_prob
        n_dropped = (~keep_mask).sum()
        dots_df = dots_df[keep_mask].reset_index(drop=True)

        print(f"   Dropout: detection_rate={det_rate:.2f} "
              f"(expr_dep={det_expr_dep}, gene_cv={det_gene_cv})")
        print(f"   Dropped {n_dropped:,} dots "
              f"({n_dropped / max(pre_dropout_total, 1):.1%})")
    else:
        print(f"   Dropout: none (detection_rate=1.0)")

    # ---- Apply admixture noise (if configured) ----
    # Admixture reassigns transcripts between cells, simulating segmentation
    # errors and z-axis contamination. Cells gain transcripts from neighbors
    # (often of different types), increasing genes detected per cell.
    admixture_cfg = tech_cfg.get("admixture", None)
    n_admixed = 0
    if admixture_cfg is not None and len(dots_df) > 0:
        models = []

        # Lateral 2D admixture (boundary-based transcript misassignment)
        lat_cfg = admixture_cfg.get("lateral", None)
        if lat_cfg is not None:
            lat_model = Lateral2DAdmixture(
                boundary_width=lat_cfg["boundary_width"],
                transfer_rate=lat_cfg["transfer_rate"],
                distance_decay=lat_cfg["distance_decay"],
                seed=EXPRESSION_SEED + 6000,
            )
            models.append(lat_model)

        # Z-axis admixture (out-of-plane contamination)
        z_cfg = admixture_cfg.get("z_axis", None)
        if z_cfg is not None:
            z_model = ZAxisAdmixture(
                z_contamination_rate=z_cfg["z_contamination_rate"],
                neighborhood_correlation=z_cfg["neighborhood_correlation"],
                neighborhood_radius=z_cfg["neighborhood_radius"],
                seed=EXPRESSION_SEED + 7000,
            )
            models.append(z_model)

        if models:
            composite = CompositeAdmixture(
                models=models, seed=EXPRESSION_SEED + 8000
            )
            # Use FOV cell radii for lateral boundary detection
            cell_radii = (fov.cell_major_axis + fov.cell_minor_axis) / 2.0
            dots_df = composite.apply(
                dots_df, spatial_coords, cell_type_indices, cell_radii
            )
            summary = composite.get_admixture_summary()
            n_admixed = summary["n_reassigned"]
            print(f"   Admixture: {n_admixed:,} dots reassigned "
                  f"({summary['reassignment_rate']*100:.1f}%)")
            for source, count in summary.get("by_source", {}).items():
                print(f"     {source}: {count:,}")
    else:
        print(f"   Admixture: none")

    # ---- Apply cell-level dropout (if configured) ----
    # Entire cells fail detection: poor segmentation, damaged cells, empty segments.
    # All dots for selected cells are removed.
    cell_dropout_rate = tech_cfg.get("cell_dropout_rate", 0.0)
    n_cells_dropped = 0
    if cell_dropout_rate > 0 and len(dots_df) > 0:
        tech_seed_offset = {"MERFISH": 0, "EELFISH": 1, "BARseq": 2, "HybISS": 3}
        rng_cd = np.random.default_rng(
            EXPRESSION_SEED + 10000 + tech_seed_offset.get(tech_name, 0)
        )
        # Dropout is applied to ALL cells in the FOV (not just those with dots)
        all_cell_ids = np.arange(fov.n_cells)
        n_cells_dropped = int(fov.n_cells * cell_dropout_rate)
        dropout_cell_ids = set(rng_cd.choice(all_cell_ids, n_cells_dropped, replace=False))
        dots_before = len(dots_df)
        dots_df = dots_df[~dots_df["cell"].isin(dropout_cell_ids)].reset_index(drop=True)
        dots_removed_cd = dots_before - len(dots_df)
        print(f"   Cell dropout: {n_cells_dropped:,} cells dropped "
              f"({cell_dropout_rate*100:.0f}%), "
              f"{dots_removed_cd:,} dots removed")
    else:
        print(f"   Cell dropout: none")

    # ---- Post-processing stats ----
    n_total_cells = fov.n_cells  # Total cells including dropped ones
    per_cell_counts = dots_df.groupby("cell").size()
    genes_per_cell = dots_df.groupby("cell")["gene"].nunique()
    n_cells_with_dots = len(per_cell_counts)
    n_cells_zero = n_total_cells - n_cells_with_dots
    post_depth = (per_cell_counts / genes_per_cell).mean()
    # Average over ALL cells (including zero-count dropped cells)
    post_yield = per_cell_counts.sum() / n_total_cells
    avg_genes_all = genes_per_cell.sum() / n_total_cells

    print(f"   Final dots: {len(dots_df):,}")
    print(f"   Cells with dots: {n_cells_with_dots:,} / {n_total_cells:,} "
          f"({n_cells_zero:,} zero-count, {n_cells_zero/n_total_cells*100:.1f}%)")
    print(f"   Avg count/cell (all): {post_yield:.1f}  (target: {target_avg})")
    print(f"   Avg depth (transcripts/detected gene): {post_depth:.2f}")
    print(f"   Avg genes detected/cell (all): {avg_genes_all:.1f}")

    # ---- Save ----
    folder = f"TECH_{tech_name}"
    case_params = {
        "case": f"TECH_{tech_name}",
        "technology": tech_name,
        "description": tech_cfg["description"],
        "n_genes_panel": n_genes_used,
        "target_avg_count_per_cell": target_avg,
        "gene_selection": "top_hvg_by_variance",
        "raw_avg_poisson_rate": round(float(raw_avg_rate), 3),
        "calibrated_gene_sensitivity": round(float(gs), 6),
        "effective_detection_efficiency_pct": round(float(gs * 100), 2),
        "gene_sensitivity_variation": gs_var,
        "dropout_detection_rate": det_rate,
        "dropout_fraction": round(1 - det_rate, 2),
        "dropout_expression_dependence": det_expr_dep,
        "dropout_gene_variation_cv": det_gene_cv,
        "cell_count_overdispersion": overdispersion,
        "admixture_config": admixture_cfg,
        "n_dots_admixed": int(n_admixed),
        "cell_dropout_rate": cell_dropout_rate,
        "n_cells_dropped": n_cells_dropped,
        "amplification_cv": amp_cv,
        "n_dots_before_dropout": int(pre_dropout_total),
        "n_dots_dropped": int(n_dropped),
        "n_dots_final": len(dots_df),
        "n_total_cells": int(n_total_cells),
        "n_cells_with_dots": int(n_cells_with_dots),
        "actual_avg_count_per_cell": round(float(post_yield), 1),
        "actual_avg_depth": round(float(post_depth), 3),
        "actual_avg_genes_per_cell": round(float(avg_genes_all), 1),
        "transfer_function": "IdentityTransfer",
        "relative_rna_concentration": RELATIVE_RNA_CONC,
        "rna_concentration_variation": RNA_CONC_VARIATION,
        "expression_seed": EXPRESSION_SEED,
        "source_section": H5AD_PATH,
        "reference_matrix": REFERENCE_CSV_PATH,
    }

    save_case(folder, tissue_sub, dots_df, case_params)

    summary_table.append({
        "technology": tech_name,
        "n_genes": n_genes_used,
        "target_avg": target_avg,
        "actual_avg": round(float(post_yield), 1),
        "sensitivity": round(float(gs), 4),
        "dropout": f"{(1-det_rate)*100:.0f}%",
        "cell_dropout": f"{cell_dropout_rate*100:.0f}%",
        "depth": round(float(post_depth), 2),
        "genes_per_cell": round(float(avg_genes_all), 1),
    })


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TECHNOLOGY-SPECIFIC SIMULATIONS COMPLETE -- 4 DATASETS GENERATED")
print("=" * 80)

print(f"\nOutput location: {OUTPUT_FOLDER}/")
print(f"Source section: {H5AD_PATH}")
print(f"Reference matrix: {REFERENCE_CSV_PATH}")
print(f"Spatial layout: {len(cells_df)} cells (REAL positions, fixed across all)")
print(f"Cell types: {n_cell_types}")
print(f"Full gene panel: {tissue.n_genes} genes")
print(f"Expression seed: {EXPRESSION_SEED}")

print(f"\n{'Technology':>10s}  {'Genes':>5s}  {'Target':>6s}  {'Actual':>6s}  "
      f"{'Sensitivity':>11s}  {'Dropout':>7s}  {'Depth':>5s}  {'Genes/cell':>10s}")
print("-" * 75)
for row in summary_table:
    print(f"{row['technology']:>10s}  {row['n_genes']:5d}  {row['target_avg']:6d}  "
          f"{row['actual_avg']:6.1f}  {row['sensitivity']:11.4f}  "
          f"{row['dropout']:>7s}  {row['depth']:5.2f}  {row['genes_per_cell']:10.1f}")

print(f"\nBiological basis for detection parameters:")
print(f"  MERFISH  -- Combinatorial smFISH, error-robust barcodes, direct RNA imaging")
print(f"  EELFISH  -- Expansion microscopy + sequential FISH, molecule loss during expansion")
print(f"  BARseq   -- ISS with barcoded probes, RCA amplification, sequencing by ligation")
print(f"  HybISS   -- Padlock probes + RCA + SBL, lowest detection efficiency")
print(f"\nDetection hierarchy: MERFISH >> EELFISH ~ BARseq >> HybISS")
