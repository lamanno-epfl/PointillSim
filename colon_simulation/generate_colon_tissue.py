#!/usr/bin/env python3
"""
Generate Colon Tissue Simulation with PointillSim

This script creates a single FOV of colon-like tissue with:
- 6 cell types
- 50 genes
- Gene sensitivity: 0.5
- Three regions: crypts (left), sparse middle, dense right
- Output: cells.csv, dots.csv, and data.h5ad
"""

import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from pointillsim import (
    # Core components
    TissueCellTypes,
    CellTypesProperties,
    HybISS_Setup,
    FOVDistribution,
    FOV,
    TissueSlice,
    RegionSpec,

    # Elements
    FrameWideElement,
    VacuolatedStructure,

    # Rules
    ProbabilityNodeFieldRule,
    SingleTypeRule,
    MixOfNCellTypesRule,

    # Transfer function
    AffineNonNegTransfer,
)

# Import effects module
from pointillsim.effects import (
    BackgroundNoise,
    DropoutModel,
)

# Configuration
FRAME_SIZE = 1000
N_CELL_TYPES = 6
N_GENES = 50
GENE_SENSITIVITY = 0.25
OUTPUT_FOLDER = "colon_tissue_output_v2"

# Cell type assignments (semantic naming)
ID_EPITHELIAL_A = 0  # Crypt epithelium
ID_EPITHELIAL_B = 1  # Surface epithelium
ID_STROMAL = 2       # Fibroblasts
ID_IMMUNE = 3        # Immune cells
ID_ENDOTHELIAL = 4   # Vessels
ID_MUSCLE = 5        # Smooth muscle

print("=" * 60)
print("Colon Tissue Simulation with PointillSim")
print("=" * 60)


def make_wavy_poly(x_start, x_end, y_start, y_end, waviness=30, n_points=20):
    """Create a polygon with wavy vertical edges."""
    # Left wavy edge
    y_left = np.linspace(y_start, y_end, n_points)
    x_left = np.random.normal(x_start, waviness/3, n_points)
    x_left = np.clip(x_left, x_start - waviness, x_start + waviness)
    left_edge = list(zip(x_left, y_left))

    # Right wavy edge
    y_right = np.linspace(y_end, y_start, n_points)
    x_right = np.random.normal(x_end, waviness/3, n_points)
    x_right = np.clip(x_right, x_end - waviness, x_end + waviness)
    right_edge = list(zip(x_right, y_right))

    # Combine and clean
    poly = Polygon(left_edge + right_edge)
    return poly.buffer(0)


def create_crypt_rules(n_cell_types, crypt_center, crypt_width):
    """Create rules for a single crypt structure."""
    # Epithelial cells dominate the crypt
    prob_epithelial = np.zeros(n_cell_types)
    prob_epithelial[ID_EPITHELIAL_A] = 0.8
    prob_epithelial[ID_EPITHELIAL_B] = 0.2

    # Mix for surrounding area
    prob_mixed = np.zeros(n_cell_types)
    prob_mixed[ID_EPITHELIAL_A] = 0.5
    prob_mixed[ID_STROMAL] = 0.3
    prob_mixed[ID_IMMUNE] = 0.2

    ref_points = np.array([
        crypt_center + [-crypt_width/2, 0],  # Left
        crypt_center + [crypt_width/2, 0],   # Right
        crypt_center + [0, crypt_width/2]    # Top
    ])

    ref_probs = np.array([prob_epithelial, prob_epithelial, prob_mixed])

    return [ProbabilityNodeFieldRule(
        n_cell_types,
        n_ref_points=3,
        ref_probs=ref_probs,
        reference_points=ref_points
    )]


def create_specific_crypt(center_loc):
    """Create a single crypt at a specific location."""
    scale = np.random.uniform(80, 120)
    rules = create_crypt_rules(N_CELL_TYPES, center_loc, scale)

    return VacuolatedStructure(
        frame_size=FRAME_SIZE,
        scale=scale,
        fixed_center=center_loc,
        rules=rules,
        tipical_cell_spacing=8,
        hole_scale_factor=np.random.uniform(0.5, 0.7)
    )


print("\n1. Setting up tissue gene expression...")
# Create tissue with gene expression profiles
tissue = TissueCellTypes()
tissue.generate_types_and_markers(
    n_genes=N_GENES,
    n_cell_types=N_CELL_TYPES,
    expected_level=30.0,
    expected_std_level=7.0,
    concentration=0.7
)

# Assign semantic names
tissue._cell_type_names = [
    'Epithelial_Crypt',
    'Epithelial_Surface',
    'Stromal',
    'Immune',
    'Endothelial',
    'Muscle'
]

print(f"   ✓ Created tissue with {N_GENES} genes and {N_CELL_TYPES} cell types")


print("\n2. Defining cell morphology properties...")
# Cell properties for realistic morphology
cell_props = CellTypesProperties(
    n_cell_types=N_CELL_TYPES,
    sizes=[8, 9, 10, 6, 7, 11],
    size_variation=[0.2, 0.2, 0.25, 0.15, 0.2, 0.2],
    anisotropy=[0.85, 0.8, 0.7, 0.95, 0.9, 0.6],
    anisotropy_variation=[0.1, 0.1, 0.15, 0.05, 0.1, 0.15],
    relative_rna_concentration=0.7,
    rna_concentration_variation=0.4,
)

print("   ✓ Cell morphology defined")


print("\n3. Creating colon tissue regions...")

# === REGION 1: CRYPT LAYER (Left Side) ===
print("   - Building crypt region...")

# Background gradient for crypt region
prob_band_stromal = np.zeros(N_CELL_TYPES)
prob_band_stromal[ID_STROMAL] = 0.7
prob_band_stromal[ID_ENDOTHELIAL] = 0.2
prob_band_stromal[ID_IMMUNE] = 0.1

prob_band_endo = np.zeros(N_CELL_TYPES)
prob_band_endo[ID_ENDOTHELIAL] = 0.6
prob_band_endo[ID_STROMAL] = 0.3
prob_band_endo[ID_IMMUNE] = 0.1

crypt_bg_rules = [
    ProbabilityNodeFieldRule(
        n_cell_types=N_CELL_TYPES,
        n_ref_points=9,
        reference_points=np.array([
            [60, FRAME_SIZE * 0.2],
            [60, FRAME_SIZE * 0.5],
            [60, FRAME_SIZE * 0.8],
            [250, FRAME_SIZE * 0.2],
            [250, FRAME_SIZE * 0.5],
            [250, FRAME_SIZE * 0.8],
            [380, FRAME_SIZE * 0.2],
            [380, FRAME_SIZE * 0.5],
            [380, FRAME_SIZE * 0.8],
        ]),
        ref_probs=np.array([
            prob_band_stromal, prob_band_stromal, prob_band_stromal,
            prob_band_endo, prob_band_endo, prob_band_endo,
            prob_band_endo, prob_band_endo, prob_band_endo,
        ])
    )
]

# Create crypts in three staggered lines
crypt_prototypes = []
n_crypts_per_line = 8

y_positions_line1 = np.linspace(100, FRAME_SIZE - 100, n_crypts_per_line)
y_stagger = (y_positions_line1[1] - y_positions_line1[0]) / 3
y_positions_line2 = y_positions_line1 + y_stagger
y_positions_line3 = y_positions_line1 + 2 * y_stagger

for y_pos in y_positions_line1:
    center = np.array([100 + np.random.normal(0, 10), y_pos + np.random.normal(0, 15)])
    crypt_prototypes.append(lambda c=center: create_specific_crypt(c))

for y_pos in y_positions_line2:
    center = np.array([220 + np.random.normal(0, 10), y_pos + np.random.normal(0, 15)])
    crypt_prototypes.append(lambda c=center: create_specific_crypt(c))

for y_pos in y_positions_line3:
    center = np.array([340 + np.random.normal(0, 10), y_pos + np.random.normal(0, 15)])
    crypt_prototypes.append(lambda c=center: create_specific_crypt(c))

CryptBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_SIZE,
    tipical_cell_spacing=12,
    rules=crypt_bg_rules
)

fovd_crypts = FOVDistribution(
    frame_size=FRAME_SIZE,
    background_element=CryptBGPrototype,
    other_elements=crypt_prototypes,
    elements_frequency=[1.0] * len(crypt_prototypes),
    attempts_at_elements=1
)

print(f"     ✓ Created {len(crypt_prototypes)} crypts")


# === REGION 2: SPARSE MIDDLE ===
print("   - Building sparse middle region...")

sparse_rules = [
    MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_STROMAL, ID_IMMUNE, ID_MUSCLE],
        proportions=[0.5, 0.25, 0.25]
    )
]

SparseBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_SIZE,
    tipical_cell_spacing=25,
    rules=sparse_rules
)

fovd_middle = FOVDistribution(
    frame_size=FRAME_SIZE,
    background_element=SparseBGPrototype,
    other_elements=[],
    elements_frequency=[],
    attempts_at_elements=0
)

print("     ✓ Sparse region defined")


# === REGION 3: DENSE RIGHT ===
print("   - Building dense right region...")

dense_rules = [
    MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_IMMUNE, ID_ENDOTHELIAL, ID_STROMAL],
        proportions=[0.6, 0.25, 0.15]
    )
]

DenseBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_SIZE,
    tipical_cell_spacing=10,
    rules=dense_rules
)

fovd_dense = FOVDistribution(
    frame_size=FRAME_SIZE,
    background_element=DenseBGPrototype,
    other_elements=[],
    elements_frequency=[],
    attempts_at_elements=0
)

print("     ✓ Dense region defined")


# === DEFINE REGION BOUNDARIES ===
print("\n4. Defining region boundaries...")

R1_Crypts_Poly = make_wavy_poly(
    x_start=0, x_end=450,
    y_start=0, y_end=FRAME_SIZE,
    waviness=80
)

R2_Middle_Poly = make_wavy_poly(
    x_start=450, x_end=700,
    y_start=0, y_end=FRAME_SIZE,
    waviness=100
)

R3_Dense_Poly = make_wavy_poly(
    x_start=700, x_end=FRAME_SIZE,
    y_start=0, y_end=FRAME_SIZE,
    waviness=80
)

all_regions = [
    RegionSpec(
        name="MiddleSparse",
        polygon=R2_Middle_Poly,
        fovdist=fovd_middle,
        priority=0,
        blend_band=25.0
    ),
    RegionSpec(
        name="RightDense",
        polygon=R3_Dense_Poly,
        fovdist=fovd_dense,
        priority=1,
        blend_band=25.0
    ),
    RegionSpec(
        name="Crypts",
        polygon=R1_Crypts_Poly,
        fovdist=fovd_crypts,
        priority=2,
        blend_band=15.0
    ),
]

print("   ✓ Region boundaries defined")


print("\n5. Creating tissue slice...")
tissue_slice = TissueSlice(frame_size=FRAME_SIZE, regions=all_regions)
print("   ✓ Tissue slice created")


print("\n6. Generating global tissue (this may take a minute)...")
tissue_slice.generate_global()
tissue_slice.sample_labels()
print(f"   ✓ Generated {len(tissue_slice._cells_xy)} cells")


print("\n7. Creating FOV and applying properties...")
# Create FOV from tissue slice
fov = FOV(
    cell_centroids=tissue_slice._cells_xy,
    cell_probabilities=tissue_slice._probs
)
fov.class_instance_one_hot = tissue_slice._class_onehot

# Apply cell morphology
cell_props.apply(fov)
print(f"   ✓ Applied morphology to {fov.n_cells} cells")


print("\n8. Simulating transcript observations...")

# Create affine transfer function to model detection biases
affine_tf = AffineNonNegTransfer(
    scales=0.8,       # Mean scale factor (increased for ~1 more transcript/cell)
    scales_std=0.3,   # Variation in scale (some genes detected better)
    offsets=0.2,      # Mean offset (background)
    offsets_std=0.1   # Variation in offset
)

# Create HybISS observation model with transfer function
hybiss = HybISS_Setup(
    tissue,
    genes_sensitivities=GENE_SENSITIVITY,
    genes_sensitivities_variation=0.2,
    transfer_function=affine_tf
)

# Observe transcripts
hybiss.observe_dots(fov)
dots_df = hybiss.make_pandas_df()
print(f"   ✓ Generated {len(dots_df):,} transcript dots (before effects)")

# Apply dropout model
print("\n9. Applying technical effects...")
dropout = DropoutModel(
    baseline_detection_rate=0.85,  # 85% baseline detection (reduced dropout)
    expression_dependence=0.3,     # Expression-dependent dropout
    gene_variation_cv=0.2,         # 20% gene-to-gene variation
)
dropout.generate_gene_detection_rates(N_GENES)

# Apply dropout by probabilistic removal of dots
# Create gene name to index mapping
gene_to_idx = {name: idx for idx, name in enumerate(tissue.gene_names)}
gene_indices = np.array([gene_to_idx[g] for g in dots_df['gene']])
detection_rates = dropout.gene_detection_rates[gene_indices]
keep_mask = np.random.random(len(dots_df)) < detection_rates
dots_df = dots_df[keep_mask].copy().reset_index(drop=True)
print(f"   ✓ Applied dropout model ({len(dots_df):,} dots remaining)")

# Add background noise
bg_noise = BackgroundNoise(
    frame_size=FRAME_SIZE,
    background_rate=0.003,  # dots per square pixel
    spatial_pattern='uniform',
)

bg_x, bg_y, bg_genes = bg_noise.generate_background_dots(
    n_genes=N_GENES,
    gene_names=tissue.gene_names,
)

if len(bg_x) > 0:
    bg_dots = pd.DataFrame({
        'x': bg_x,
        'y': bg_y,
        'gene': [tissue.gene_names[g] for g in bg_genes],
        'cell': -1,  # Not associated with any cell
    })
    dots_df = pd.concat([dots_df, bg_dots], ignore_index=True)
    print(f"   ✓ Added background noise ({len(bg_dots)} false positive dots)")
else:
    print(f"   ✓ Added background noise (0 false positive dots)")

print(f"   ✓ Final transcript count: {len(dots_df):,} dots")


print("\n10. Saving outputs...")
# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Save cells DataFrame
cells_df = fov.make_pandas_df()
cells_df.to_csv(f"{OUTPUT_FOLDER}/cells.csv", index=False)
print(f"   ✓ Saved cells.csv ({len(cells_df)} cells)")

# Save dots DataFrame
dots_df.to_csv(f"{OUTPUT_FOLDER}/dots.csv", index=False)
print(f"   ✓ Saved dots.csv ({len(dots_df)} transcripts)")

# Save gene expression matrix
tissue_df = tissue.make_pandas_df()
tissue_df.to_csv(f"{OUTPUT_FOLDER}/cell_types_expression.csv")
print(f"   ✓ Saved cell_types_expression.csv")

# Save AnnData h5ad file
try:
    adata = fov.to_anndata()
    adata.write_h5ad(f"{OUTPUT_FOLDER}/data.h5ad")
    print(f"   ✓ Saved data.h5ad")
    print(f"     - {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"     - Layers: {list(adata.layers.keys()) if adata.layers else 'None'}")
    print(f"     - Spatial coordinates in obsm['spatial']")
except ImportError:
    print("   ⚠ AnnData not installed - skipping h5ad export")
    print("     Install with: pip install anndata")


print("\n" + "=" * 60)
print("✓ Colon tissue simulation complete!")
print("=" * 60)
print(f"\nOutput location: {OUTPUT_FOLDER}/")
print(f"  - cells.csv: {len(cells_df)} cells with ground truth")
print(f"  - dots.csv: {len(dots_df):,} transcript positions")
print(f"  - cell_types_expression.csv: Gene expression matrix")
print(f"  - data.h5ad: AnnData format (if anndata installed)")
print("\nParameters:")
print(f"  - Cell types: {N_CELL_TYPES}")
print(f"  - Genes: {N_GENES}")
print(f"  - Gene sensitivity: {GENE_SENSITIVITY}")
print(f"  - Frame size: {FRAME_SIZE}×{FRAME_SIZE}")
print("=" * 60)
