#!/usr/bin/env python3
"""
Generate Advanced Realistic Skin Tissue with PointillSim

SCALE_FACTOR CONTROL:
  Set SCALE_FACTOR (line ~145) to control tissue size and generation speed:
  - SCALE_FACTOR = 0.25  → 1/4 size, ~1/4 generation time (quick testing)
  - SCALE_FACTOR = 0.5   → 1/2 size, ~1/2 generation time (medium testing)
  - SCALE_FACTOR = 1.0   → Full size, normal time (production quality)
  - SCALE_FACTOR = 2.0   → Double size, ~2× time (large tissue)

Simulates a comprehensive skin tissue with anatomically accurate layers and structures:

ARCHITECTURE:
- Region 1: Stratum Corneum (cornified dead cells) - Priority 5 (TOP)
- Region 2: Stratum Granulosum (granular layer) - Priority 4
- Region 3: Stratum Spinosum (spinous layer) - Priority 3
- Region 4: Stratum Basale (basal layer with melanocytes) - Priority 2
- Region 5: Dermal-Epidermal Junction (DEJ wavy interface) - Priority 2
- Region 6: Papillary Dermis (fine collagen, capillaries) - Priority 1
- Region 7: Reticular Dermis (thick collagen, main structures) - Priority 0
- Region 8: Hypodermis (subcutaneous fat) - Priority -1 (BACKGROUND)

EMBEDDED STRUCTURES (in appropriate dermal layers):
- Hair follicles (LinearLumenStructure) with sebaceous glands attached laterally
- Sebaceous glands attached to follicles (GlandularUnit)
- Eccrine sweat glands (coiled, GlandularUnit)
- Apocrine sweat glands (larger, GlandularUnit)
- Blood vessels: arterioles, venules, capillary loops (BranchingStructure)
- Collagen fiber bundles (FibrillarStructure)
- Sensory corpuscles: Meissner (superficial), Pacinian (deep) (ClusterElement)
- Immune cell infiltrates (ClusterElement)

Cell types: 19
- Keratinocyte_Cornified (0)
- Keratinocyte_Granular (1)
- Keratinocyte_Spinous (2)
- Keratinocyte_Basal (3)
- Melanocyte (4)
- Fibroblast_Papillary (5)
- Fibroblast_Reticular (6)
- Endothelial (7)
- Pericyte (8)
- Sebocyte (9)
- Sweat_Eccrine (10)
- Sweat_Apocrine (11)
- Hair (12)
- Adipocyte (13)
- Langerhans (14)
- Macrophage (15)
- T_Cell (16)
- Mast_Cell (17)
- Sensory_Corpuscle (18)

SEED STRATEGY:
- GEOMETRY_SEED: Fixed spatial layout, cell placement, structure positions
- EXPRESSION_SEED: Variable gene expression profiles and observations

REALISTIC IMPROVEMENTS (v2):
1. EPIDERMIS MATURATION:
   - Progressive flattening: Corneum (0.20) < Granulosum (0.50) < Spinosum (0.70) < Basale (0.75)
   - Stratum corneum rendered as flat scales/flakes (anisotropy 0.20)
   - Melanocytes constrained to basal layer only (realistic distribution)

2. DERMIS CELL-POOR ECM:
   - Sparse fibroblast density (spacing: papillary 28µm → 23µm, reticular 35µm → 29µm)
   - Spindle-shaped fibroblasts (anisotropy: papillary 0.35, reticular 0.30)
   - ECM-dominant with scattered nuclei (realistic dermal architecture)

3. VESSELS AS THIN TUBES:
   - Papillary capillaries: thin loops (3µm radius) approaching DEJ (n=20)
   - Dermal vessels: thin tubes (5-10µm) with 4 branch levels (n=10)
   - Endothelial-rich walls (anisotropy 0.25 for flattened cells)

4. IRREGULAR BOUNDARIES:
   - DEJ: mild waviness (20µm amplitude, 12 rete ridges)
   - Dermis-hypodermis: irregular interdigitation (25µm amplitude, 7 waves)
   - Natural tissue architecture (no ruler-straight lines)

DENSITY ENHANCEMENTS (v4):
1. ECM FIBER FIELD (Primary density boost):
   - Papillary collagen: 25 bundles (3× increase, fine/random)
   - Reticular collagen: 35 bundles (3× increase, thick/aligned)
   - Papillary elastin: 15 bundles (NEW, stretchy fibers)
   - Reticular elastin: 20 bundles (NEW, aligned stretchy fibers)

2. FIBROBLAST INCREASE (1.5× cells, moderate):
   - Epidermis: spacing 6µm → 5µm (denser basal keratinocytes)
   - Papillary dermis: spacing 28µm → 23µm (1.5× fibroblasts)
   - Reticular dermis: spacing 35µm → 29µm (1.5× fibroblasts)

3. VASCULAR DETAIL (more segments, not area):
   - Papillary capillaries: 10 → 20 (2× thin loops)
   - Dermal vessels: 5 → 10 (2× thin vessels)
   - Branch depth: 3 → 4 levels (more detail)
   - Branch probability: 0.65 → 0.75 (finer network)

4. MINIMAL IMMUNE INCREASE:
   - Perivascular clusters: 5 → 6 (slight increase)
   - Periappendage clusters: 3 → 4 (slight increase)

Target: 2-3× cell/object density while maintaining biological plausibility
Output: 800×1000 µm dense realistic skin tissue slice
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point, LineString, box, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

from pointillsim import (
    TissueCellTypes,
    CellTypesProperties,
    HybISS_Setup,
    FOVDistribution,
    FOV,
    TissueSlice,
    RegionSpec,
    FrameWideElement,
    VacuolatedStructure,
    ProbabilityNodeFieldRule,
    SingleTypeRule,
    MixOfNCellTypesRule,
    DistanceBasedRule,
    LayerRule,
    CompositeRule,
    AffineNonNegTransfer,
)

from pointillsim.effects import (
    Lateral2DAdmixture,
    ZAxisAdmixture,
    CompositeAdmixture,
    BackgroundNoise,
    DropoutModel,
)

from pointillsim.elements.structures import (
    LayeredElement,
    GlandularUnit,
    LinearLumenStructure,
    BranchingStructure,
    FibrillarStructure,
    ClusterElement,
    StromalElement,
    InterfaceElement,
)


# ============================================================================
# CONTROLLABLE SIMULATION PARAMETERS
# All tunable knobs are here in UPPER_CASE. Modify freely.
# ============================================================================

# ---------- GLOBAL SCALING FACTOR ----------
# This scales tissue size and structure counts proportionally
# Use 0.25 for quick testing (1/4 size, ~1/4 generation time)
# Use 0.5 for medium testing (1/2 size, ~1/2 generation time)
# Use 1.0 for full-size tissue (production quality)
# Use 2.0 for double-size tissue (4× area, longer generation)
SCALE_FACTOR = 1.0

# ---------- RANDOM SEEDS ----------
GEOMETRY_SEED = 42            # Cell placement, polygon shapes, morphology (FIXED across sims)
EXPRESSION_SEED = 999         # Gene profiles, transcript noise (VARY across sims)

# ---------- FRAME & OUTPUT ----------
# Base sizes (scaled by SCALE_FACTOR)
FRAME_WIDTH = int(800 * SCALE_FACTOR)             # microns (horizontal)
FRAME_HEIGHT = int(1000 * SCALE_FACTOR)           # microns (vertical - skin depth)
AREA_MULTIPLIER = 1.0         # No area scaling (fast generation)
OUTPUT_FOLDER = "skin_sweep_output_new"

# ---------- CELL TYPES & GENES ----------
N_CELL_TYPES = 19
N_GENES = 100
GENE_SENSITIVITY = 0.5

# ---------- UNIFIED BASE CASE FOR ALL SWEEPS ----------
# These parameters define a single realistic baseline shared by every sweep axis.
# Each sweep varies exactly one dimension; all others stay at base values.
BASE_N_GENES = 1000                  # Full reference panel (scRNA-seq reference)
SWEEP_N_GENES = 200                  # HVG subset used by cases 2-4
BASE_WITHIN_GROUP_SIMILARITY = 0.92  # How similar related types are (0=independent, 1=identical)
BASE_EXPRESSION_LEVEL = 300   # Mean per-gene expression in reference (N/2)
BASE_EXPRESSION_STD = 150      # Std of per-gene expression in reference (N/4)
BASE_MARKER_FOLD_CHANGE = 1.2         # Marker gene upregulation (very low for ≤0.7 accuracy with 1000 genes)
BASE_LINEAGE_FOLD_CHANGE = 1.08      # Lineage gene upregulation (very low)
BASE_MARKER_BACKGROUND = 0.30        # Baseline expression of marker genes in non-target types
BASE_LINEAGE_BACKGROUND = 0.35       # Baseline expression of lineage genes in non-target types
BASE_CROSS_GROUP_LEAKAGE = 0.45      # Fraction of marker/lineage expression leaking across groups
BASE_PROFILE_NOISE_CV = 0.55         # CV for final multiplicative noise (very high)
BASE_GENE_SENSITIVITY = 0.55         # Detection efficiency (increased for higher tx/cell)
BASE_GENE_SENSITIVITY_VAR = 0.5      # Variation in detection efficiency (lognormal std)
BASE_EXPRESSION_CONCENTRATION = 1.0  # No concentration (preserves profile ambiguity)
# Case-specific extra dropout (applied AFTER baseline noise to hit ~0.70 base accuracy)
CASE1_EXTRA_DROPOUT_RATE = 0.58      # Keep 58% of CASE 1 dots (compensate for higher sensitivity)
CASE3_EXTRA_DROPOUT_RATE = 0.66      # Extra dropout for CASE 3 (capped to keep gs0.9 accuracy ≤ 0.80)
BASE_TRANSFER_SCALE = 1.0
BASE_TRANSFER_SCALE_STD = 0.3
BASE_TRANSFER_OFFSET = 0.02
BASE_TRANSFER_OFFSET_STD = 0.01
# Noise base (moderate realism, used by CASE 4 baseline)
BASE_LATERAL_BW = 6.0
BASE_LATERAL_TR = 0.15
BASE_Z_RATE = 0.08
BASE_DROPOUT_RATE = 0.90
BASE_BG_RATE = 0.0005

# ---------- BASELINE OBSERVATION NOISE (applied to ALL cases) ----------
# Ensures no base case has accuracy > 0.7 by adding realistic noise to every
# observation.  This simulates the unavoidable noise in any spatial
# transcriptomics experiment (segmentation errors, z-contamination, dropout,
# spurious background transcripts).
BASELINE_NOISE_LATERAL_BW = 12.0      # Segmentation error boundary width (realistic narrow boundary)
BASELINE_NOISE_LATERAL_TR = 0.14      # Fraction of boundary dots transferred (realistic: ~14%)
BASELINE_NOISE_LATERAL_DD = 0.25      # Distance decay for lateral transfer (fast → nearby only)
BASELINE_NOISE_Z_RATE = 0.06          # Z-axis contamination rate (realistic: ~6%)
BASELINE_NOISE_Z_CORR = 0.50          # Neighborhood correlation for z-contamination
BASELINE_NOISE_Z_RADIUS = 60.0        # Neighborhood radius for z-contamination
BASELINE_NOISE_DET_RATE = 0.80        # Baseline detection rate (less dropout → keep dots high)
BASELINE_NOISE_DET_EXPR_DEP = 0.25    # Expression dependence of dropout
BASELINE_NOISE_DET_GENE_CV = 0.20     # Gene-to-gene variation in detection
BASELINE_NOISE_BG_RATE = 0.015        # False-positive background rate (increased for realism)

# ---------- CELL TYPE IDS ----------
ID_KERATINOCYTE_CORNIFIED = 0
ID_KERATINOCYTE_GRANULAR = 1
ID_KERATINOCYTE_SPINOUS = 2
ID_KERATINOCYTE_BASAL = 3
ID_MELANOCYTE = 4
ID_FIBROBLAST_PAPILLARY = 5
ID_FIBROBLAST_RETICULAR = 6
ID_ENDOTHELIAL = 7
ID_PERICYTE = 8
ID_SEBOCYTE = 9
ID_SWEAT_ECCRINE = 10
ID_SWEAT_APOCRINE = 11
ID_HAIR = 12
ID_ADIPOCYTE = 13
ID_LANGERHANS = 14
ID_MACROPHAGE = 15
ID_T_CELL = 16
ID_MAST_CELL = 17
ID_SENSORY_CORPUSCLE = 18

# ---------- CELL TYPE NAMES ----------
CELL_TYPE_NAMES = [
    'Keratinocyte_Cornified',    # 0
    'Keratinocyte_Granular',     # 1
    'Keratinocyte_Spinous',      # 2
    'Keratinocyte_Basal',        # 3
    'Melanocyte',                # 4
    'Fibroblast_Papillary',      # 5
    'Fibroblast_Reticular',      # 6
    'Endothelial',               # 7
    'Pericyte',                  # 8
    'Sebocyte',                  # 9
    'Sweat_Eccrine',             # 10
    'Sweat_Apocrine',            # 11
    'Hair',                      # 12
    'Adipocyte',                 # 13
    'Langerhans',                # 14
    'Macrophage',                # 15
    'T_Cell',                    # 16
    'Mast_Cell',                 # 17
    'Sensory_Corpuscle',         # 18
]

# ---------- CELL TYPE COLORS (visualization) - HIGH CONTRAST ----------
CELL_TYPE_COLORS = {
    ID_KERATINOCYTE_CORNIFIED: '#FF1493',  # Deep Pink (cornified - top layer)
    ID_KERATINOCYTE_GRANULAR:  '#FF6347',  # Tomato (granular layer)
    ID_KERATINOCYTE_SPINOUS:   '#FF8C00',  # Dark Orange (spinous layer)
    ID_KERATINOCYTE_BASAL:     '#FFD700',  # Gold (basal - bright yellow)
    ID_MELANOCYTE:             '#8B4513',  # Saddle Brown (distinct brown)
    ID_FIBROBLAST_PAPILLARY:   '#00FF00',  # Lime (bright green - papillary dermis)
    ID_FIBROBLAST_RETICULAR:   '#006400',  # Dark Green (reticular dermis)
    ID_ENDOTHELIAL:            '#FF0000',  # Pure Red (blood vessels - max visibility)
    ID_PERICYTE:               '#DC143C',  # Crimson (vessel support)
    ID_SEBOCYTE:               '#FFFF00',  # Pure Yellow (sebaceous glands)
    ID_SWEAT_ECCRINE:          '#0000FF',  # Pure Blue (eccrine glands)
    ID_SWEAT_APOCRINE:         '#00BFFF',  # Deep Sky Blue (apocrine - cyan tint)
    ID_HAIR:                   '#FF00FF',  # Magenta (hair follicle)
    ID_ADIPOCYTE:              '#FFA500',  # Orange (fat cells)
    ID_LANGERHANS:             '#9370DB',  # Medium Purple (immune)
    ID_MACROPHAGE:             '#BA55D3',  # Medium Orchid (immune)
    ID_T_CELL:                 '#8A2BE2',  # Blue Violet (immune)
    ID_MAST_CELL:              '#DA70D6',  # Orchid (immune)
    ID_SENSORY_CORPUSCLE:      '#00FFFF',  # Cyan (sensory - bright blue)
}

# ---------- GENE EXPRESSION PARAMETERS ----------
EXPECTED_EXPRESSION_LEVEL = 50.0
EXPECTED_EXPRESSION_STD = 25.0
EXPRESSION_CONCENTRATION = 0.75

# ---------- CELL SPACING ----------
EPIDERMIS_CELL_SPACING = 4          # TIGHT (keratinocytes) - MORE CELLS for fast generation
DERMIS_PAPILLARY_SPACING = 18       # TIGHT (fibroblasts in ECM) - MORE CELLS
DERMIS_RETICULAR_SPACING = 18       # Moderate density (fills space without being too dense)
HYPODERMIS_SPACING = 20             # Tighter for more Adipocyte cells (100+ requirement)
FIBER_CELL_SPACING_PAPILLARY = 70   # SPARSE on fibers (avoids "green carpet")
FIBER_CELL_SPACING_RETICULAR = 90   # VERY SPARSE on fibers (avoids "green carpet")
HAIR_FOLLICLE_SPACING = 5           # Very dense
GLAND_CELL_SPACING = 6              # Dense
VESSEL_CELL_SPACING = 4             # Dense (endothelial lining)
CLUSTER_CELL_SPACING = 7            # Moderate-dense

# ---------- CELL MORPHOLOGY ----------
# Progressive flattening from basal → corneum for REALISTIC MATURATION
# Spindle-shaped fibroblasts for REALISTIC DERMIS
# Thin endothelial cells for REALISTIC VESSELS
# INCREASED SIZES BY ~30% for better visibility with moderate density
#                               Corn   Gran   Spin   Basal  Melan  FibP   FibR   Endo   Peri   Sebo   SwEc   SwAp   Hair   Adipo  Lang   Macro  TCell  Mast   Senso
CELL_SIZES =                    [15,   11,    10,    9,     10,    15,    16,    8,     10,    15,    10,    11,    9,     25,    11,    14,    9,     11,    19]
CELL_SIZE_VARIATION =           [0.20, 0.15,  0.18,  0.20,  0.25,  0.25,  0.25,  0.15,  0.20,  0.30,  0.18,  0.20,  0.18,  0.40,  0.22,  0.30,  0.20,  0.25,  0.30]
CELL_ANISOTROPY =               [0.20, 0.50,  0.70,  0.75,  0.70,  0.35,  0.30,  0.25,  0.60,  0.85,  0.80,  0.80,  0.75,  0.90,  0.70,  0.75,  0.85,  0.80,  0.70]
#                                ^^^^  ^^^^   ^^^^   ^^^^         ^^^^   ^^^^   ^^^^
#                                FLAT  MID    ROUND  ROUND        SPINDLE SPINDLE FLAT
#                                SCALES      EPIDERMIS            DERMIS         VESSEL
CELL_ANISO_VARIATION =          [0.08, 0.10,  0.10,  0.12,  0.15,  0.15,  0.15,  0.10,  0.15,  0.10,  0.12,  0.12,  0.15,  0.10,  0.15,  0.15,  0.10,  0.12,  0.15]
RELATIVE_RNA_CONC = 0.8
RNA_CONC_VARIATION = 0.35

# ---------- TRANSFER FUNCTION ----------
TRANSFER_SCALES = 1.0
TRANSFER_SCALES_STD = 0.3
TRANSFER_OFFSETS = 0.15
TRANSFER_OFFSETS_STD = 0.08

# ---------- LAYER THICKNESSES (microns from top) ----------
# Y-axis goes from 0 (top/air) to FRAME_HEIGHT (bottom/deep)
# All thicknesses scale with SCALE_FACTOR
STRATUM_CORNEUM_DEPTH = int(15 * SCALE_FACTOR)           # 0-15 µm
STRATUM_GRANULOSUM_DEPTH = int(30 * SCALE_FACTOR)        # 15-30 µm
STRATUM_SPINOSUM_DEPTH = int(60 * SCALE_FACTOR)          # 30-60 µm
STRATUM_BASALE_DEPTH = int(80 * SCALE_FACTOR)            # 60-80 µm
DEJ_DEPTH = int(100 * SCALE_FACTOR)                      # 80-100 µm (dermal-epidermal junction)
PAPILLARY_DERMIS_DEPTH = int(200 * SCALE_FACTOR)         # 100-200 µm
RETICULAR_DERMIS_DEPTH = int(800 * SCALE_FACTOR)         # 200-800 µm
HYPODERMIS_START = int(800 * SCALE_FACTOR)               # 800-1000 µm

# ---------- EPIDERMAL LAYER WAVINESS ----------
EPIDERMIS_WAVINESS = 15.0 * SCALE_FACTOR            # Amplitude of surface irregularity
EPIDERMIS_N_WAVES = max(1, int(8 * SCALE_FACTOR))  # Number of undulations across width
DEJ_WAVINESS = 20.0 * SCALE_FACTOR                  # Rete ridge amplitude - MILD waviness (REALISTIC)
DEJ_N_WAVES = max(1, int(12 * SCALE_FACTOR))        # Number of rete ridges - MORE frequent (REALISTIC)

# ---------- HYPODERMIS BOUNDARY WAVINESS ----------
HYPODERMIS_WAVINESS = 25.0 * SCALE_FACTOR           # Amplitude of dermis-hypodermis irregularity (NEW - REALISTIC)
HYPODERMIS_N_WAVES = max(1, int(7 * SCALE_FACTOR))  # Number of undulations (NEW - REALISTIC)

# ---------- DERMAL STRUCTURES (scaled by SCALE_FACTOR) ----------
# Hair follicles
N_HAIR_FOLLICLES = max(1, int(8 * SCALE_FACTOR))  # 8 follicles
HAIR_FOLLICLE_LENGTH_MIN = int(300 * SCALE_FACTOR)  # End around mid reticular dermis
HAIR_FOLLICLE_LENGTH_MAX = int(450 * SCALE_FACTOR)  # End around mid reticular dermis
HAIR_FOLLICLE_OUTER_RADIUS = int(15 * SCALE_FACTOR)
HAIR_FOLLICLE_WALL_THICKNESS = max(1, int(5 * SCALE_FACTOR))

# Sebaceous glands (attached to follicles)
N_SEBACEOUS_GLANDS_PER_FOLLICLE = 1
SEBACEOUS_GLAND_RADIUS_MIN = int(20 * SCALE_FACTOR)
SEBACEOUS_GLAND_RADIUS_MAX = int(35 * SCALE_FACTOR)

# Sweat glands
N_ECCRINE_GLANDS = max(1, int(5 * SCALE_FACTOR))
ECCRINE_GLAND_N_ACINI = 6
ECCRINE_GLAND_ACINUS_RADIUS = int(12 * SCALE_FACTOR)

N_APOCRINE_GLANDS = max(1, int(2 * SCALE_FACTOR))
APOCRINE_GLAND_N_ACINI = 5
APOCRINE_GLAND_ACINUS_RADIUS = int(18 * SCALE_FACTOR)

# Blood vessels
N_PAPILLARY_CAPILLARIES = max(1, int(15 * SCALE_FACTOR))
N_DERMAL_VESSELS = max(1, int(8 * SCALE_FACTOR))
CAPILLARY_RADIUS = max(3, int(6 * SCALE_FACTOR))  # Increased so cells fit in wall
CAPILLARY_LENGTH_MIN = int(40 * SCALE_FACTOR)
CAPILLARY_LENGTH_MAX = int(80 * SCALE_FACTOR)
VESSEL_ROOT_RADIUS_MIN = max(1, int(5 * SCALE_FACTOR))
VESSEL_ROOT_RADIUS_MAX = max(1, int(10 * SCALE_FACTOR))
VESSEL_MAX_DEPTH = 3  # Keep constant (branch complexity)
VESSEL_BRANCH_PROBABILITY = 0.65  # Keep constant

# Collagen bundles
N_COLLAGEN_BUNDLES_PAPILLARY = max(1, int(15 * SCALE_FACTOR))
N_COLLAGEN_BUNDLES_RETICULAR = max(1, int(20 * SCALE_FACTOR))
COLLAGEN_N_FIBERS_PAPILLARY = 2
COLLAGEN_N_FIBERS_RETICULAR = 3
COLLAGEN_FIBER_WIDTH_PAPILLARY = max(1, int(5 * SCALE_FACTOR))
COLLAGEN_FIBER_WIDTH_RETICULAR = max(1, int(7 * SCALE_FACTOR))
COLLAGEN_FIBER_SPACING = int(15 * SCALE_FACTOR)
COLLAGEN_WAVINESS = 0.15

# Elastin fibers
N_ELASTIN_FIBERS_PAPILLARY = max(1, int(10 * SCALE_FACTOR))
N_ELASTIN_FIBERS_RETICULAR = max(1, int(12 * SCALE_FACTOR))
ELASTIN_N_FIBERS = 2
ELASTIN_FIBER_WIDTH_PAPILLARY = max(1, int(3 * SCALE_FACTOR))
ELASTIN_FIBER_WIDTH_RETICULAR = max(1, int(4 * SCALE_FACTOR))
ELASTIN_FIBER_SPACING = int(10 * SCALE_FACTOR)
ELASTIN_WAVINESS = 0.30

# Melanin units - REDUCED count (melanocytes will also be added to basale background)
N_MELANIN_UNITS = max(1, int(8 * SCALE_FACTOR))  # Reduced from 15, complemented by background melanocytes
MELANIN_UNIT_RADIUS = max(1, int(10 * SCALE_FACTOR))

# Sensory corpuscles
N_MEISSNER_CORPUSCLES = max(1, int(8 * SCALE_FACTOR))
MEISSNER_RADIUS_MIN = int(12 * SCALE_FACTOR)
MEISSNER_RADIUS_MAX = int(20 * SCALE_FACTOR)

N_PACINIAN_CORPUSCLES = max(1, int(4 * SCALE_FACTOR))
PACINIAN_RADIUS_MIN = int(35 * SCALE_FACTOR)
PACINIAN_RADIUS_MAX = int(55 * SCALE_FACTOR)

# Immune infiltrates
N_PERIVASCULAR_IMMUNE_CLUSTERS = max(1, int(8 * SCALE_FACTOR))
N_PERIAPPENDAGE_IMMUNE_CLUSTERS = max(1, int(6 * SCALE_FACTOR))
IMMUNE_CLUSTER_RADIUS_MIN = int(18 * SCALE_FACTOR)
IMMUNE_CLUSTER_RADIUS_MAX = int(40 * SCALE_FACTOR)

# ---------- REGION PRIORITIES ----------
PRIORITY_HYPODERMIS = -1
PRIORITY_RETICULAR_DERMIS = 0
PRIORITY_PAPILLARY_DERMIS = 1
PRIORITY_DEJ = 2
PRIORITY_BASALE = 2
PRIORITY_SPINOSUM = 3
PRIORITY_GRANULOSUM = 4
PRIORITY_CORNEUM = 5

# ---------- BLEND BANDS ----------
BLEND_BAND_EPIDERMIS = 5.0
BLEND_BAND_DEJ = 15.0
BLEND_BAND_DERMIS = 20.0
BLEND_BAND_HYPODERMIS = 30.0


print("=" * 80)
print("Advanced Realistic Skin Tissue Simulation with PointillSim")
print("=" * 80)


# ============================================================================
# BOUNDARY GENERATION UTILITIES
# ============================================================================

def create_wavy_horizontal_line(y_position, n_points=100, amplitude=20, n_waves=8):
    """Create a wavy horizontal boundary."""
    x_vals = np.linspace(0, FRAME_WIDTH, n_points)
    y_vals = y_position + amplitude * np.sin(2 * np.pi * n_waves * x_vals / FRAME_WIDTH)
    return list(zip(x_vals, y_vals))


def create_epidermal_layer_polygon(top_y, bottom_y, top_waviness=10, bottom_waviness=15,
                                   n_waves_top=8, n_waves_bottom=10):
    """Create polygon for epidermal layer with wavy top and bottom boundaries."""
    # Top boundary (wavy)
    top_points = create_wavy_horizontal_line(top_y, n_points=150,
                                            amplitude=top_waviness, n_waves=n_waves_top)

    # Bottom boundary (wavy, reversed)
    bottom_points = create_wavy_horizontal_line(bottom_y, n_points=150,
                                                amplitude=bottom_waviness, n_waves=n_waves_bottom)
    bottom_points_reversed = list(reversed(bottom_points))

    # Right edge
    right_edge = [(FRAME_WIDTH, top_y), (FRAME_WIDTH, bottom_y)]

    # Left edge
    left_edge = [(0, bottom_y), (0, top_y)]

    # Combine into polygon
    poly_coords = top_points + right_edge + bottom_points_reversed + left_edge
    return Polygon(poly_coords).buffer(0)


# ============================================================================
# STRUCTURE CREATION UTILITIES
# ============================================================================

def create_hair_follicle_with_sebaceous(x=None, y_start=None,
                                        follicle_length=None, follicle_angle=None):
    """Create prototypes for a hair follicle with attached sebaceous glands.

    Hair follicle is an epidermal invagination:
    - Top opening: at the skin surface (through the epidermis)
    - Outer root sheath: continuous with epidermis, begins at infundibulum
    - Bulb: sits deep in the dermis, near dermis-hypodermis boundary
    - Sebaceous glands: attached to upper follicle in mid-dermis

    The follicle starts near the DEJ and extends downward into the dermis.
    Separate pore/infundibulum elements should be added at the surface level.
    """
    # Random position (if not provided)
    if x is None:
        x = np.random.uniform(150, FRAME_WIDTH - 150)
    if y_start is None:
        # Anatomically correct: follicle originates near DEJ (epidermal invagination)
        y_start = np.random.uniform(DEJ_DEPTH - 20, DEJ_DEPTH + 20)

    if follicle_length is None:
        follicle_length = np.random.uniform(HAIR_FOLLICLE_LENGTH_MIN, HAIR_FOLLICLE_LENGTH_MAX)

    if follicle_angle is None:
        # Hair follicle (tubular structure pointing downward from surface)
        follicle_angle = np.random.uniform(np.pi/2 - 0.3, np.pi/2 + 0.3)  # Mostly vertical

    # All follicle cells are a single "Hair" type
    follicle_rule = SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_HAIR)

    follicle_proto = lambda: LinearLumenStructure(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        length=follicle_length,
        outer_radius=HAIR_FOLLICLE_OUTER_RADIUS,
        wall_thickness=HAIR_FOLLICLE_WALL_THICKNESS,
        fixed_start=np.array([x, y_start]),
        fixed_angle=follicle_angle,
        n_control_points=2,
        curvature=0,
        tipical_cell_spacing=HAIR_FOLLICLE_SPACING,
        rules=[follicle_rule]
    )

    structures = [follicle_proto]

    # Sebaceous gland(s): attached to the SIDE of the upper follicle shaft
    # Offset laterally (left or right) so gland sits beside the follicle, not on top
    for _ in range(N_SEBACEOUS_GLANDS_PER_FOLLICLE):
        gland_radius = np.random.uniform(SEBACEOUS_GLAND_RADIUS_MIN, SEBACEOUS_GLAND_RADIUS_MAX)
        # Lateral offset: place gland clearly to one side of the follicle
        side = np.random.choice([-1, 1])
        lateral_offset = side * (HAIR_FOLLICLE_OUTER_RADIUS + gland_radius * 0.6
                                 + np.random.uniform(5, 15))
        gland_x = x + lateral_offset
        # Along-shaft position: upper follicle, mid-dermis level
        gland_y = y_start + np.random.uniform(100, 200)

        sebaceous_proto = lambda gr=gland_radius, gx=gland_x, gy=gland_y: GlandularUnit(
            frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
            n_acini=np.random.randint(3, 5),
            acinus_radius=gr / 2,
            arrangement='circular',
            central_duct=True,
            duct_radius=gr * 0.3,
            fixed_center=np.array([gx, gy]),
            tipical_cell_spacing=GLAND_CELL_SPACING,
            rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_SEBOCYTE)]
        )
        structures.append(sebaceous_proto)

    return structures


def create_follicle_segment(x, y_start, follicle_length, follicle_angle):
    """Create a follicle shaft segment for papillary dermis rendering.

    Uses the same geometry parameters as the reticular copy to ensure
    visual continuity across region boundaries (infundibulum portion).
    """
    follicle_rule = SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_HAIR)

    return lambda: LinearLumenStructure(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        length=follicle_length,
        outer_radius=HAIR_FOLLICLE_OUTER_RADIUS,
        wall_thickness=HAIR_FOLLICLE_WALL_THICKNESS,
        fixed_start=np.array([x, y_start]),
        fixed_angle=follicle_angle,
        n_control_points=2,
        curvature=0,
        tipical_cell_spacing=HAIR_FOLLICLE_SPACING,
        rules=[follicle_rule]
    )


def create_follicle_pore(x, y):
    """Create a small pore/opening at the skin surface for a hair follicle.

    Represents the follicular ostium where the hair shaft exits the epidermis.
    """
    return lambda: ClusterElement(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        radius=max(3, int(5 * SCALE_FACTOR)),
        fixed_center=np.array([x, y]),
        density_profile='uniform',
        tipical_cell_spacing=EPIDERMIS_CELL_SPACING,
        smoothing_iterations=1,
        rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_HAIR)]
    )


def create_eccrine_sweat_gland(x=None, y=None):
    """Create an eccrine sweat gland prototype (tight cluster, accepts fixed position)."""
    if x is None:
        x = np.random.uniform(100, FRAME_WIDTH - 100)
    if y is None:
        y = np.random.uniform(PAPILLARY_DERMIS_DEPTH + 100, RETICULAR_DERMIS_DEPTH - 50)
    radius = np.random.uniform(35 * SCALE_FACTOR, 50 * SCALE_FACTOR)

    return lambda: ClusterElement(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        radius=radius,
        fixed_center=np.array([x, y]),
        density_profile='uniform',
        tipical_cell_spacing=GLAND_CELL_SPACING,
        smoothing_iterations=1,
        rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_SWEAT_ECCRINE)]
    )


def create_apocrine_sweat_gland(x=None, y=None):
    """Create an apocrine sweat gland prototype (accepts fixed position)."""
    if x is None:
        x = np.random.uniform(100, FRAME_WIDTH - 100)
    if y is None:
        y = np.random.uniform(RETICULAR_DERMIS_DEPTH - 300, RETICULAR_DERMIS_DEPTH)

    return lambda: GlandularUnit(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        n_acini=APOCRINE_GLAND_N_ACINI,
        acinus_radius=APOCRINE_GLAND_ACINUS_RADIUS,
        arrangement='circular',
        central_duct=True,
        duct_radius=APOCRINE_GLAND_ACINUS_RADIUS * 0.35,
        fixed_center=np.array([x, y]),
        tipical_cell_spacing=GLAND_CELL_SPACING,
        rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_SWEAT_APOCRINE)]
    )


def create_papillary_capillary():
    """Create a thin capillary loop in papillary dermis (REALISTIC - approaching DEJ)."""
    x = np.random.uniform(150, FRAME_WIDTH - 150)
    y_start = np.random.uniform(DEJ_DEPTH + 10, PAPILLARY_DERMIS_DEPTH - 10)

    capillary_length = np.random.uniform(CAPILLARY_LENGTH_MIN, CAPILLARY_LENGTH_MAX)

    # Vertical or slightly angled (capillaries rise toward DEJ)
    capillary_angle = np.random.uniform(-np.pi/2 - 0.2, -np.pi/2 + 0.2)  # Upward

    # Vessel wall: mostly endothelial with minimal pericytes
    vessel_rule = MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_ENDOTHELIAL, ID_PERICYTE],
        proportions=[0.85, 0.15]  # Thin capillary wall
    )

    return lambda: LinearLumenStructure(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        length=capillary_length,
        outer_radius=CAPILLARY_RADIUS,
        wall_thickness=3,  # Thin wall, wide enough for cells to fit
        fixed_start=np.array([x, y_start]),
        fixed_angle=capillary_angle,
        n_control_points=3,
        curvature=0.15,  # Gentle curve (loop-like)
        tipical_cell_spacing=VESSEL_CELL_SPACING,
        rules=[vessel_rule]
    )


def create_dermal_blood_vessel(x=None, y=None):
    """Create a thin tubular blood vessel in deeper dermis (accepts fixed position)."""
    if x is None:
        x = np.random.uniform(150, FRAME_WIDTH - 150)
    if y is None:
        y = np.random.uniform(PAPILLARY_DERMIS_DEPTH + 50, RETICULAR_DERMIS_DEPTH - 150)

    root_angle = np.random.uniform(0, 2 * np.pi)
    root_radius = np.random.uniform(VESSEL_ROOT_RADIUS_MIN, VESSEL_ROOT_RADIUS_MAX)

    # Vessel wall: endothelial (inner) + pericyte (outer)
    # Using MixOfNCellTypesRule for reliable cell counts in thin vessel walls
    vessel_rule = MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_ENDOTHELIAL, ID_PERICYTE],
        proportions=[0.65, 0.35]  # More endothelial than pericyte
    )

    return lambda: BranchingStructure(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        root_position=np.array([x, y]),
        root_angle=root_angle,
        root_radius=root_radius,
        branch_length=np.random.uniform(60, 120),  # Shorter branches (REALISTIC)
        branch_length_decay=0.65,
        radius_decay=0.70,  # Thinner branches (REALISTIC)
        branching_angle=(np.pi/8, np.pi/4),  # Narrower angles (REALISTIC)
        branching_probability=VESSEL_BRANCH_PROBABILITY,
        max_depth=VESSEL_MAX_DEPTH,
        min_radius=3,  # Minimum 3µm (REALISTIC capillary size)
        tipical_cell_spacing=VESSEL_CELL_SPACING,
        rules=[vessel_rule]
    )


def create_papillary_collagen_bundle():
    """Create fine, random collagen fibers in papillary dermis (SPARSE cell spacing)."""
    x = np.random.uniform(100, FRAME_WIDTH - 100)
    y = np.random.uniform(DEJ_DEPTH + 20, PAPILLARY_DERMIS_DEPTH - 20)

    orientation = np.random.uniform(0, 2 * np.pi)  # Random orientation (fine fibers)

    return lambda: FibrillarStructure(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        n_fibers=COLLAGEN_N_FIBERS_PAPILLARY,  # Fewer fibers (2) - avoids "carpet"
        fiber_width=COLLAGEN_FIBER_WIDTH_PAPILLARY,
        fiber_spacing=12,
        orientation=orientation,
        waviness=0.25,  # More wavy (fine fibers)
        fixed_center=np.array([x, y]),
        tipical_cell_spacing=FIBER_CELL_SPACING_PAPILLARY,  # SPARSE (70µm) - key fix
        rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_FIBROBLAST_PAPILLARY)]
    )


def create_reticular_collagen_bundle():
    """Create thick, aligned collagen bundles in reticular dermis (SPARSE cell spacing)."""
    x = np.random.uniform(150, FRAME_WIDTH - 150)
    y = np.random.uniform(PAPILLARY_DERMIS_DEPTH + 50, RETICULAR_DERMIS_DEPTH - 100)

    orientation = np.random.uniform(-np.pi/6, np.pi/6)  # Mostly horizontal (aligned)

    return lambda: FibrillarStructure(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        n_fibers=COLLAGEN_N_FIBERS_RETICULAR,  # Fewer fibers (3) - avoids "carpet"
        fiber_width=COLLAGEN_FIBER_WIDTH_RETICULAR,
        fiber_spacing=COLLAGEN_FIBER_SPACING,
        orientation=orientation,
        waviness=COLLAGEN_WAVINESS,
        fixed_center=np.array([x, y]),
        tipical_cell_spacing=FIBER_CELL_SPACING_RETICULAR,  # SPARSE (90µm) - key fix
        rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_FIBROBLAST_RETICULAR)]
    )


def create_papillary_elastin_fibers():
    """Create fine, stretchy elastin fibers in papillary dermis (SPARSE cell spacing)."""
    x = np.random.uniform(100, FRAME_WIDTH - 100)
    y = np.random.uniform(DEJ_DEPTH + 20, PAPILLARY_DERMIS_DEPTH - 20)

    orientation = np.random.uniform(0, 2 * np.pi)  # Random orientation (fine fibers)

    return lambda: FibrillarStructure(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        n_fibers=ELASTIN_N_FIBERS,  # Fewer fibers (2) than collagen
        fiber_width=ELASTIN_FIBER_WIDTH_PAPILLARY,
        fiber_spacing=ELASTIN_FIBER_SPACING,
        orientation=orientation,
        waviness=ELASTIN_WAVINESS,  # More wavy/stretchy
        fixed_center=np.array([x, y]),
        tipical_cell_spacing=FIBER_CELL_SPACING_PAPILLARY,  # SPARSE (70µm) - key fix
        rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_FIBROBLAST_PAPILLARY)]
    )


def create_reticular_elastin_fibers():
    """Create aligned elastin fibers in reticular dermis (SPARSE cell spacing)."""
    x = np.random.uniform(150, FRAME_WIDTH - 150)
    y = np.random.uniform(PAPILLARY_DERMIS_DEPTH + 50, RETICULAR_DERMIS_DEPTH - 100)

    orientation = np.random.uniform(-np.pi/6, np.pi/6)  # Mostly horizontal (aligned)

    return lambda: FibrillarStructure(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        n_fibers=ELASTIN_N_FIBERS,  # Fewer fibers (2)
        fiber_width=ELASTIN_FIBER_WIDTH_RETICULAR,
        fiber_spacing=ELASTIN_FIBER_SPACING,
        orientation=orientation,
        waviness=ELASTIN_WAVINESS,  # More wavy than collagen
        fixed_center=np.array([x, y]),
        tipical_cell_spacing=FIBER_CELL_SPACING_RETICULAR,  # SPARSE (90µm) - key fix
        rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_FIBROBLAST_RETICULAR)]
    )


def create_melanin_unit():
    """Create melanin unit: melanocyte at DEJ with halo of basal keratinocytes (REALISTIC)."""
    x = np.random.uniform(100, FRAME_WIDTH - 100)
    # Position at DEJ boundary (stratum basale - DEJ interface)
    y = np.random.uniform(STRATUM_BASALE_DEPTH + 5, DEJ_DEPTH - 5)

    # Melanocyte at center with basal keratinocytes around it
    # This represents the epidermal melanin unit (1 melanocyte : ~10-15 keratinocytes)
    melanin_unit_rule = MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_MELANOCYTE, ID_KERATINOCYTE_BASAL],
        proportions=[0.40, 0.60]  # Very high melanocyte proportion to ensure 100+ cells
    )

    return lambda: ClusterElement(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        radius=MELANIN_UNIT_RADIUS,
        fixed_center=np.array([x, y]),
        density_profile='uniform',  # Uniform distribution (changed from dense_center)
        tipical_cell_spacing=EPIDERMIS_CELL_SPACING,
        smoothing_iterations=2,
        rules=[melanin_unit_rule]
    )


def create_meissner_corpuscle():
    """Create a Meissner corpuscle prototype (tactile receptor in papillary dermis)."""
    x = np.random.uniform(100, FRAME_WIDTH - 100)
    y = np.random.uniform(DEJ_DEPTH + 20, PAPILLARY_DERMIS_DEPTH - 20)
    radius = np.random.uniform(MEISSNER_RADIUS_MIN, MEISSNER_RADIUS_MAX)

    return lambda: ClusterElement(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        radius=radius,
        fixed_center=np.array([x, y]),
        density_profile='dense_center',
        tipical_cell_spacing=CLUSTER_CELL_SPACING,
        smoothing_iterations=3,
        rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_SENSORY_CORPUSCLE)]
    )


def create_pacinian_corpuscle():
    """Create a Pacinian corpuscle prototype (pressure receptor in deep dermis)."""
    x = np.random.uniform(150, FRAME_WIDTH - 150)
    y = np.random.uniform(RETICULAR_DERMIS_DEPTH - 400, RETICULAR_DERMIS_DEPTH - 100)
    radius = np.random.uniform(PACINIAN_RADIUS_MIN, PACINIAN_RADIUS_MAX)

    return lambda: ClusterElement(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        radius=radius,
        fixed_center=np.array([x, y]),
        density_profile='dense_center',
        tipical_cell_spacing=CLUSTER_CELL_SPACING,
        smoothing_iterations=4,
        rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_SENSORY_CORPUSCLE)]
    )


def create_perivascular_immune_cluster(vessel_x, vessel_y):
    """Create immune cluster near a blood vessel (perivascular positioning)."""
    # Position near vessel with some offset
    offset_dist = np.random.uniform(15, 35)
    offset_angle = np.random.uniform(0, 2 * np.pi)
    x = vessel_x + offset_dist * np.cos(offset_angle)
    y = vessel_y + offset_dist * np.sin(offset_angle)

    # Clamp to valid range
    x = np.clip(x, 100, FRAME_WIDTH - 100)
    y = np.clip(y, PAPILLARY_DERMIS_DEPTH, RETICULAR_DERMIS_DEPTH - 100)

    radius = np.random.uniform(IMMUNE_CLUSTER_RADIUS_MIN, IMMUNE_CLUSTER_RADIUS_MAX)

    # Mix of T cells, macrophages, mast cells (perivascular infiltrate)
    immune_rule = MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_T_CELL, ID_MACROPHAGE, ID_MAST_CELL],
        proportions=[0.5, 0.3, 0.2]
    )

    return lambda: ClusterElement(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        radius=radius,
        fixed_center=np.array([x, y]),
        density_profile='uniform',
        tipical_cell_spacing=CLUSTER_CELL_SPACING,
        smoothing_iterations=2,
        rules=[immune_rule]
    )


def create_periappendage_immune_cluster(appendage_x, appendage_y):
    """Create immune cluster near hair follicle or gland (periappendage positioning)."""
    # Position near appendage with offset
    offset_dist = np.random.uniform(25, 50)
    offset_angle = np.random.uniform(0, 2 * np.pi)
    x = appendage_x + offset_dist * np.cos(offset_angle)
    y = appendage_y + offset_dist * np.sin(offset_angle)

    # Clamp to valid range
    x = np.clip(x, 100, FRAME_WIDTH - 100)
    y = np.clip(y, PAPILLARY_DERMIS_DEPTH, RETICULAR_DERMIS_DEPTH - 100)

    radius = np.random.uniform(IMMUNE_CLUSTER_RADIUS_MIN, IMMUNE_CLUSTER_RADIUS_MAX)

    # Mix with more macrophages (appendage surveillance)
    immune_rule = MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_T_CELL, ID_MACROPHAGE, ID_MAST_CELL],
        proportions=[0.3, 0.5, 0.2]  # More macrophages
    )

    return lambda: ClusterElement(
        frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
        radius=radius,
        fixed_center=np.array([x, y]),
        density_profile='uniform',
        tipical_cell_spacing=CLUSTER_CELL_SPACING,
        smoothing_iterations=2,
        rules=[immune_rule]
    )


# ============================================================================
# REALISTIC EXPRESSION PROFILE HELPERS
# ============================================================================

# Cell type hierarchy — encodes biological relationships between the 19 types.
# Types within a group share lineage-specific gene programs; types in the same
# super-group share broader (e.g., epithelial) programs.
CELL_TYPE_HIERARCHY = {
    'keratinocyte': [ID_KERATINOCYTE_CORNIFIED, ID_KERATINOCYTE_GRANULAR,
                     ID_KERATINOCYTE_SPINOUS, ID_KERATINOCYTE_BASAL],
    'melanocyte':   [ID_MELANOCYTE],
    'fibroblast':   [ID_FIBROBLAST_PAPILLARY, ID_FIBROBLAST_RETICULAR],
    'vascular':     [ID_ENDOTHELIAL, ID_PERICYTE],
    'glandular':    [ID_SEBOCYTE, ID_SWEAT_ECCRINE, ID_SWEAT_APOCRINE],
    'hair':         [ID_HAIR],
    'adipocyte':    [ID_ADIPOCYTE],
    'immune':       [ID_LANGERHANS, ID_MACROPHAGE, ID_T_CELL, ID_MAST_CELL],
    'sensory':      [ID_SENSORY_CORPUSCLE],
}

SUPER_GROUPS = {
    'epithelial':   ['keratinocyte', 'melanocyte', 'hair'],
    'mesenchymal':  ['fibroblast', 'adipocyte'],
    'vascular':     ['vascular'],
    'secretory':    ['glandular'],
    'immune':       ['immune'],
    'neural':       ['sensory'],
}

# Reverse lookups (type_id -> group name, group name -> super-group name)
_TYPE_TO_GROUP = {}
for _grp, _ids in CELL_TYPE_HIERARCHY.items():
    for _tid in _ids:
        _TYPE_TO_GROUP[_tid] = _grp

_GROUP_TO_SUPER = {}
for _sgrp, _grps in SUPER_GROUPS.items():
    for _g in _grps:
        _GROUP_TO_SUPER[_g] = _sgrp


def _lognormal_from_mean_std(rng, mean, std, size=None):
    """Sample from lognormal parameterized by desired mean and std."""
    var = std ** 2
    mu = np.log(mean ** 2 / np.sqrt(var + mean ** 2))
    sigma = np.sqrt(np.log(1 + var / mean ** 2))
    return rng.lognormal(mu, sigma, size=size)


def build_hierarchical_expression_matrix(
    n_genes,
    n_cell_types=N_CELL_TYPES,
    *,
    frac_housekeeping=0.35,
    frac_lineage=0.25,
    frac_marker=0.15,
    # frac_weak is the remainder
    within_group_similarity=BASE_WITHIN_GROUP_SIMILARITY,
    base_expression_level=BASE_EXPRESSION_LEVEL,
    expression_level_std=BASE_EXPRESSION_STD,
    marker_fold_change=BASE_MARKER_FOLD_CHANGE,
    lineage_fold_change=BASE_LINEAGE_FOLD_CHANGE,
    marker_background=BASE_MARKER_BACKGROUND,
    lineage_background=BASE_LINEAGE_BACKGROUND,
    cross_group_leakage=BASE_CROSS_GROUP_LEAKAGE,
    profile_noise_cv=BASE_PROFILE_NOISE_CV,
    n_coexpression_modules=None,
    target_total_expression=460.0,
    expression_concentration=BASE_EXPRESSION_CONCENTRATION,
    seed=42,
):
    """Build a biologically-structured expression matrix.

    Instead of the Dirichlet-based approach (where every gene is essentially a
    clean marker for 1-2 types), this constructs profiles with:
      - Housekeeping genes: broadly expressed, ~equal across all types
      - Lineage genes: shared within hierarchy groups (e.g., all keratinocytes)
      - Marker genes: specific to one type, with leakage to relatives
      - Weakly informative genes: moderate expression in 2-5 types

    Parameters
    ----------
    n_genes : int
        Total number of genes.
    n_cell_types : int
        Number of cell types (19 for the skin model).
    frac_housekeeping, frac_lineage, frac_marker : float
        Fractions of genes in each category.  The remainder is weakly
        informative.
    within_group_similarity : float
        How much a marker gene's expression leaks to other members of the
        same hierarchy group (0 = no leakage, 1 = identical to target).
    base_expression_level, expression_level_std : float
        Mean and std of per-gene expression scale (lognormal).
    marker_fold_change : float
        Upregulation factor for cell-type-specific markers.
    lineage_fold_change : float
        Upregulation factor for lineage/group genes.
    marker_background : float
        Baseline expression of marker genes in non-target types, as a
        fraction of the gene's expression level.  Higher = more ambiguous.
    lineage_background : float
        Baseline expression of lineage genes in non-target types.
    cross_group_leakage : float
        Fraction of marker/lineage expression that leaks across unrelated
        groups (0 = no cross-group leakage, 0.15 = high).
    profile_noise_cv : float
        Coefficient of variation for final multiplicative noise.
    n_coexpression_modules : int or None
        Number of coexpression modules (default: n_genes // 10).
    target_total_expression : float or None
        Target for the mean column sum of the expression matrix. After
        building the matrix, all values are rescaled by a single global
        factor so that the average total expression per cell type equals
        this value.  This controls the absolute transcript count scale
        independently of n_genes.  Set to None to skip normalization.
        Default 460.0.
    expression_concentration : float or None
        Power-law exponent for concentrating expression into fewer genes.
        After normalization, each column is raised to this power and
        re-normalized to preserve column sums.  This creates a more
        Zipf-like distribution where a few genes carry most of the
        expression (biologically realistic).  1.0 = no change,
        higher = more concentrated.  Set to None to skip.  Default 1.3.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    expression_matrix : np.ndarray, shape (n_genes, n_cell_types)
    gene_categories : np.ndarray of str, shape (n_genes,)
        Category label per gene: 'HK', 'LIN', 'MRK', 'WEAK'.
    gene_names : list of str, shape (n_genes,)
        Descriptive names encoding category.
    """
    rng = np.random.default_rng(seed)

    # -- Allocate gene counts per category --
    n_hk = int(n_genes * frac_housekeeping)
    n_lin = int(n_genes * frac_lineage)
    n_mrk = int(n_genes * frac_marker)
    n_weak = n_genes - n_hk - n_lin - n_mrk

    expr = np.zeros((n_genes, n_cell_types))
    categories = np.empty(n_genes, dtype='U8')
    names = []
    gene_idx = 0

    active_types = list(range(n_cell_types))

    # ---- HOUSEKEEPING GENES ----
    hk_counter = 0
    for _ in range(n_hk):
        level = _lognormal_from_mean_std(rng, base_expression_level,
                                         base_expression_level * 0.3)
        # Broadly expressed with small type-to-type variation (CV ~0.20)
        profile = level * np.abs(1.0 + rng.normal(0, 0.20, n_cell_types))
        profile = np.maximum(profile, 0.1)
        expr[gene_idx] = profile
        categories[gene_idx] = 'HK'
        names.append(f"HK_{hk_counter:03d}")
        hk_counter += 1
        gene_idx += 1

    # ---- LINEAGE / GROUP GENES ----
    group_names = list(CELL_TYPE_HIERARCHY.keys())
    lin_counter = 0
    for _ in range(n_lin):
        target_group = rng.choice(group_names)
        target_types = CELL_TYPE_HIERARCHY[target_group]
        level = _lognormal_from_mean_std(rng, base_expression_level,
                                         expression_level_std)

        # Low baseline in non-target types
        profile = np.full(n_cell_types, level * lineage_background)
        # Upregulated in target group
        for t in target_types:
            profile[t] = level * lineage_fold_change * rng.lognormal(0, 0.15)

        # Cross-group leakage: partial expression in unrelated types
        if cross_group_leakage > 0:
            non_target = [t for t in range(n_cell_types) if t not in target_types]
            for t in non_target:
                leak = cross_group_leakage * rng.uniform(0.3, 1.0)
                profile[t] = max(profile[t],
                                 level * lineage_fold_change * leak)

        # Moderate leak to super-group relatives
        target_super = _GROUP_TO_SUPER.get(target_group)
        if target_super:
            for related_grp in SUPER_GROUPS[target_super]:
                if related_grp != target_group:
                    for t in CELL_TYPE_HIERARCHY[related_grp]:
                        profile[t] = max(
                            profile[t],
                            level * lineage_fold_change * rng.uniform(0.08, 0.25)
                        )

        profile = np.maximum(profile, 0.1)
        expr[gene_idx] = profile
        categories[gene_idx] = 'LIN'
        names.append(f"LIN_{target_group}_{lin_counter:03d}")
        lin_counter += 1
        gene_idx += 1

    # ---- CELL-TYPE-SPECIFIC MARKER GENES ----
    # Distribute markers across types; singletons get slightly more,
    # large-group members get slightly fewer.
    marker_weights = np.ones(n_cell_types)
    for grp_types in CELL_TYPE_HIERARCHY.values():
        factor = 1.5 if len(grp_types) == 1 else (0.7 if len(grp_types) > 2 else 1.0)
        for t in grp_types:
            marker_weights[t] = factor
    marker_weights /= marker_weights.sum()

    marker_assignments = rng.choice(n_cell_types, size=n_mrk, p=marker_weights)
    mrk_counter = 0
    for target_type in marker_assignments:
        level = _lognormal_from_mean_std(rng, base_expression_level,
                                         expression_level_std)

        # Baseline expression in non-target types
        profile = np.full(n_cell_types, level * marker_background)
        # Strong expression in target
        profile[target_type] = level * marker_fold_change

        # Leak to same-group relatives (controlled by within_group_similarity)
        target_grp = _TYPE_TO_GROUP[target_type]
        for relative in CELL_TYPE_HIERARCHY[target_grp]:
            if relative != target_type:
                leak = within_group_similarity * rng.uniform(0.5, 0.95)
                profile[relative] = profile[target_type] * leak

        # Cross-group leakage: partial expression in unrelated types
        if cross_group_leakage > 0:
            non_group = [t for t in range(n_cell_types)
                        if _TYPE_TO_GROUP[t] != target_grp]
            for t in non_group:
                leak = cross_group_leakage * rng.uniform(0.2, 0.8)
                profile[t] = max(profile[t], profile[target_type] * leak)

        profile = np.maximum(profile, 0.1)
        expr[gene_idx] = profile
        categories[gene_idx] = 'MRK'
        target_name = CELL_TYPE_NAMES[target_type]
        names.append(f"MRK_{target_name}_{mrk_counter:03d}")
        mrk_counter += 1
        gene_idx += 1

    # ---- WEAKLY INFORMATIVE GENES ----
    weak_counter = 0
    for _ in range(n_weak):
        level = _lognormal_from_mean_std(rng, base_expression_level * 0.7,
                                         expression_level_std * 0.5)
        n_active = rng.integers(2, 6)
        active = rng.choice(active_types, size=min(n_active, len(active_types)),
                            replace=False)

        profile = np.full(n_cell_types, level * 0.06)
        for t in active:
            profile[t] = level * rng.uniform(0.5, 2.5)

        profile = np.maximum(profile, 0.1)
        expr[gene_idx] = profile
        categories[gene_idx] = 'WEAK'
        names.append(f"WEAK_{weak_counter:03d}")
        weak_counter += 1
        gene_idx += 1

    # ---- COEXPRESSION MODULE STRUCTURE ----
    if n_coexpression_modules is None:
        n_coexpression_modules = max(5, n_genes // 10)

    module_assignments = rng.choice(n_coexpression_modules, size=n_genes)
    module_factors = rng.lognormal(0, 0.25, (n_coexpression_modules, n_cell_types))
    for i in range(n_genes):
        expr[i, :] *= module_factors[module_assignments[i], :]

    # ---- FINAL NOISE & CLEANUP ----
    expr *= rng.lognormal(0, profile_noise_cv, expr.shape)
    expr = np.maximum(expr, 1e-6)

    # ---- NORMALIZE TOTAL EXPRESSION PER CELL TYPE ----
    # Without normalization, total expression per cell type scales linearly
    # with n_genes, producing unreasonable transcript counts (e.g. 60K/cell
    # for 1000 genes).  Rescale by a single global factor so that the mean
    # column sum equals target_total_expression while preserving all
    # relative gene expression patterns (fold changes, marker specificity).
    if target_total_expression is not None:
        current_mean_total = expr.sum(axis=0).mean()
        if current_mean_total > 0:
            normalization_factor = target_total_expression / current_mean_total
            expr *= normalization_factor

    # ---- CONCENTRATE EXPRESSION (ZIPF-LIKE) ----
    # Real gene expression follows a power-law distribution where a small
    # fraction of genes captures most of the RNA.  Apply a per-column
    # power-law transformation to create this concentration while
    # preserving column sums (total expression per cell type).
    if expression_concentration is not None and expression_concentration != 1.0:
        for c in range(n_cell_types):
            col = expr[:, c]
            original_sum = col.sum()
            col_c = col ** expression_concentration
            col_sum = col_c.sum()
            if col_sum > 0:
                expr[:, c] = col_c * (original_sum / col_sum)

    # Sort by primary cell type for visualization
    ixs = np.argsort(expr.argmax(axis=1))
    expr = expr[ixs, :]
    categories = categories[ixs]
    names = [names[i] for i in ixs]

    return expr, categories, names


def select_realistic_gene_panel(
    expression_matrix,
    n_select,
    gene_categories,
    *,
    selection_noise_std=0.5,
    housekeeping_leak_frac=0.10,
    seed=42,
):
    """Select a gene panel mimicking realistic (noisy) panel design.

    Unlike oracle top-HVG selection, this adds noise to variance scores and
    forces a fraction of housekeeping genes into the panel (simulating genes
    that appeared variable in a noisy scRNA-seq reference but are actually
    uninformative in spatial data).

    Parameters
    ----------
    expression_matrix : np.ndarray, shape (n_genes, n_cell_types)
    n_select : int
        Number of genes to select.
    gene_categories : np.ndarray of str
        Category label per gene ('HK', 'LIN', 'MRK', 'WEAK').
    selection_noise_std : float
        Std of Gaussian noise added to log-variance scores.
    housekeeping_leak_frac : float
        Fraction of panel slots allocated to housekeeping genes.
    seed : int

    Returns
    -------
    selected_indices : np.ndarray of int
        Indices into the expression matrix of selected genes.
    """
    rng = np.random.default_rng(seed)

    if n_select >= expression_matrix.shape[0]:
        return np.arange(expression_matrix.shape[0])

    n_genes = expression_matrix.shape[0]

    # Noisy HVG scores
    raw_var = np.var(expression_matrix, axis=1)
    log_var = np.log1p(raw_var)
    noisy_scores = log_var + rng.normal(0, selection_noise_std, n_genes)

    # Housekeeping leakage: force top-scoring HK genes into the panel
    hk_mask = gene_categories == 'HK'
    n_hk_leak = max(1, int(n_select * housekeeping_leak_frac))

    hk_indices = np.where(hk_mask)[0]
    hk_scores = noisy_scores[hk_indices]
    hk_selected = hk_indices[np.argsort(hk_scores)[::-1][:n_hk_leak]]

    # Fill remaining slots from non-HK genes by noisy score
    non_hk_mask = ~hk_mask
    non_hk_indices = np.where(non_hk_mask)[0]
    non_hk_scores = noisy_scores[non_hk_indices]
    n_remaining = n_select - len(hk_selected)
    non_hk_selected = non_hk_indices[np.argsort(non_hk_scores)[::-1][:n_remaining]]

    selected = np.concatenate([hk_selected, non_hk_selected])

    # Ensure at least 1 gene per active cell type has representation
    # (find the best-scoring gene for each type and add if missing)
    active_types = list(range(expression_matrix.shape[1]))
    selected_set = set(selected)
    for ct in active_types:
        # Best gene for this cell type among non-selected
        ct_scores = expression_matrix[:, ct].copy()
        ct_scores[list(selected_set)] = -1  # exclude already selected
        best = np.argmax(ct_scores)
        if best not in selected_set:
            # Check if this type already has some representation
            ct_expr_selected = expression_matrix[list(selected_set), ct]
            if ct_expr_selected.max() < expression_matrix[:, ct].max() * 0.3:
                # Swap out the worst-scoring selected gene
                worst_idx = selected[np.argmin(noisy_scores[selected])]
                selected_set.discard(worst_idx)
                selected_set.add(best)
                selected = np.array(list(selected_set))

    return np.sort(selected[:n_select])


def degrade_reference_matrix(expr_matrix, noise_cv=0.0, blur_factor=0.0, seed=42):
    """Create a degraded version of the expression matrix for use as TACCO reference.

    Simulates imperfect scRNA-seq reference quality:
    - noise_cv: multiplicative lognormal noise (measurement error in scRNA-seq)
    - blur_factor: mix each type's profile with the global mean (contamination/doublets)

    Parameters
    ----------
    expr_matrix : np.ndarray, shape (n_genes, n_cell_types)
    noise_cv : float
        CV of multiplicative lognormal noise.
    blur_factor : float
        Fraction of profile replaced by global mean (0=clean, 1=all mean).
    seed : int

    Returns
    -------
    degraded : np.ndarray, shape (n_genes, n_cell_types)
    """
    rng = np.random.default_rng(seed)
    degraded = expr_matrix.copy()

    if noise_cv > 0:
        degraded *= rng.lognormal(0, noise_cv, degraded.shape)

    if blur_factor > 0:
        mean_profile = degraded.mean(axis=1, keepdims=True)
        degraded = (1 - blur_factor) * degraded + blur_factor * mean_profile

    return np.maximum(degraded, 1e-6)


def apply_baseline_noise(dots_df, gene_names, cell_centroids, cell_types,
                          cell_radii, cell_tree, seed_offset=0,
                          det_rate_override=None):
    """Apply baseline observation noise (admixture + dropout + background).

    Applied to ALL cases to ensure no base case exceeds ~0.7 TACCO accuracy.
    If det_rate_override is given, uses that instead of BASELINE_NOISE_DET_RATE
    for the dropout step (useful for cases that need lighter dropout).
    """
    n_genes = len(gene_names)
    det_rate = det_rate_override if det_rate_override is not None else BASELINE_NOISE_DET_RATE
    dots_noisy = dots_df.copy()

    # 1. Admixture (lateral + z-axis)
    lateral_admix = Lateral2DAdmixture(
        boundary_width=BASELINE_NOISE_LATERAL_BW,
        transfer_rate=BASELINE_NOISE_LATERAL_TR,
        distance_decay=BASELINE_NOISE_LATERAL_DD,
        seed=EXPRESSION_SEED + seed_offset,
    )
    z_admix = ZAxisAdmixture(
        z_contamination_rate=BASELINE_NOISE_Z_RATE,
        neighborhood_correlation=BASELINE_NOISE_Z_CORR,
        neighborhood_radius=BASELINE_NOISE_Z_RADIUS,
        seed=EXPRESSION_SEED + seed_offset + 1,
    )
    combined_admix = CompositeAdmixture(models=[lateral_admix, z_admix])
    dots_noisy = combined_admix.apply(
        dots_noisy, cell_centroids, cell_types, cell_radii
    )
    admix_summary = combined_admix.get_admixture_summary()

    # 2. Dropout
    dropout_model = DropoutModel(
        baseline_detection_rate=det_rate,
        expression_dependence=BASELINE_NOISE_DET_EXPR_DEP,
        gene_variation_cv=BASELINE_NOISE_DET_GENE_CV,
        seed=EXPRESSION_SEED + seed_offset + 2,
    )
    dropout_model.generate_gene_detection_rates(n_genes)
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    dot_gene_idx = dots_noisy['gene'].map(gene_name_to_idx).values
    per_dot_factor = dropout_model.gene_detection_rates[dot_gene_idx]
    detection_prob = np.clip(det_rate * per_dot_factor, 0.0, 1.0)
    keep_mask = dropout_model.rng.random(len(dots_noisy)) < detection_prob
    n_dropped = (~keep_mask).sum()
    dots_noisy = dots_noisy[keep_mask].reset_index(drop=True)

    # 3. Background noise
    bg_frame_size = max(FRAME_WIDTH, FRAME_HEIGHT)
    effective_bg_rate = (BASELINE_NOISE_BG_RATE
                         * (FRAME_WIDTH * FRAME_HEIGHT)
                         / (bg_frame_size ** 2))
    bg_noise_model = BackgroundNoise(
        frame_size=bg_frame_size,
        background_rate=effective_bg_rate,
        spatial_pattern="uniform",
        seed=EXPRESSION_SEED + seed_offset + 3,
    )
    bg_x, bg_y, bg_gene_idx = bg_noise_model.generate_background_dots(
        n_genes=n_genes, gene_names=gene_names,
    )
    n_bg = len(bg_x)
    if n_bg > 0:
        bg_x = bg_x / bg_frame_size * FRAME_WIDTH
        bg_y = bg_y / bg_frame_size * FRAME_HEIGHT
        bg_gene_names_bl = [gene_names[gi] for gi in bg_gene_idx]
        _, bg_cell_idx = cell_tree.query(np.column_stack([bg_x, bg_y]))
        bg_df = pd.DataFrame({
            'x': bg_x, 'y': bg_y,
            'gene': bg_gene_names_bl,
            'cell': bg_cell_idx.astype(int),
        })
        dots_noisy = pd.concat([dots_noisy, bg_df], ignore_index=True)

    print(f"     Baseline noise: {admix_summary['n_reassigned']} reassigned, "
          f"{n_dropped} dropped, {n_bg} bg added → {len(dots_noisy):,} dots")

    return dots_noisy


def make_tissue_from_matrix(expression_matrix, gene_names=None):
    """Wrap a custom expression matrix into a TissueCellTypes object."""
    tissue = TissueCellTypes(gene_expression_by_type=expression_matrix)
    tissue._cell_type_names = CELL_TYPE_NAMES
    if gene_names is not None:
        tissue._gene_names = gene_names
    else:
        tissue._gene_names = [f"Gene_{i}" for i in range(expression_matrix.shape[0])]
    return tissue


def run_observation_model(
    tissue, fov, *,
    sensitivity=0.5,
    sensitivity_var=0.25,
    tf_scale=1.0, tf_scale_std=0.3,
    tf_offset=0.15, tf_offset_std=0.08,
    seed=999,
):
    """Create transfer function + HybISS observation model and observe dots.

    Returns
    -------
    hybiss : HybISS_Setup
    dots_df : pd.DataFrame
    """
    np.random.seed(seed)
    affine_tf = AffineNonNegTransfer(
        scales=tf_scale, scales_std=tf_scale_std,
        offsets=tf_offset, offsets_std=tf_offset_std,
    )
    hybiss = HybISS_Setup(
        tissue,
        genes_sensitivities=sensitivity,
        genes_sensitivities_variation=sensitivity_var,
        transfer_function=affine_tf,
    )
    hybiss.observe_dots(fov)
    dots_df = hybiss.make_pandas_df()
    return hybiss, dots_df


# ============================================================================
# HIGH-QUALITY VISUALIZATION FUNCTION
# ============================================================================

def create_high_quality_visualization(
    cells_df,
    dots_df=None,
    output_path="skin_visualization_hq.png",
    # Figure parameters
    figsize=(20, 24),
    dpi=300,
    # Cell rendering
    render_mode='ellipse',  # 'scatter' or 'ellipse'
    cell_size_scale=1.0,
    cell_alpha=0.95,
    cell_edge_color=None,
    cell_edge_width=0,
    # Background and style
    background_color='white',  # 'white', 'black', or custom hex
    show_grid=False,
    grid_alpha=0.3,
    grid_color='gray',
    # Layer annotations
    show_layer_lines=True,
    layer_line_color='cyan',
    layer_line_width=2,
    layer_line_style='--',
    layer_line_alpha=0.7,
    show_layer_labels=True,
    layer_label_fontsize=12,
    layer_label_color='auto',  # 'auto' adjusts based on background
    # Legend
    show_legend=True,
    legend_fontsize=10,
    legend_ncol=2,
    legend_loc='center left',
    legend_bbox=(1.02, 0.5),
    # Title and labels
    title="Skin Tissue - High Quality Visualization",
    title_fontsize=20,
    title_color='auto',
    xlabel_fontsize=16,
    ylabel_fontsize=16,
    label_color='auto',
    # Transcript dots (if provided)
    show_dots=False,
    dot_size=0.3,
    dot_alpha=0.4,
    dot_colormap='tab20',
    # Zoom/crop
    xlim=None,  # (xmin, xmax) or None for full range
    ylim=None,  # (ymin, ymax) or None for full range
):
    """
    Create high-quality publication-ready visualization of skin tissue.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cell data with columns: X, Y, Class ID, Major Axis, Minor Axis, Rotation Angle
    dots_df : pd.DataFrame, optional
        Transcript data with columns: x, y, gene
    output_path : str
        Output file path
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Resolution in dots per inch
    render_mode : str
        'scatter' for circular dots, 'ellipse' for realistic cell shapes
    cell_size_scale : float
        Multiplier for cell sizes (ellipse mode only)
    cell_alpha : float
        Cell transparency (0-1)
    cell_edge_color : str or None
        Edge color for cells (None for no edge)
    cell_edge_width : float
        Edge line width
    background_color : str
        Background color ('white', 'black', or hex code)
    show_grid : bool
        Show coordinate grid
    show_layer_lines : bool
        Show horizontal lines at layer boundaries
    show_layer_labels : bool
        Show text labels for each layer
    show_legend : bool
        Show cell type legend
    show_dots : bool
        Show transcript dots (requires dots_df)
    xlim, ylim : tuple or None
        Axis limits for zooming

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """

    # Auto-adjust colors based on background
    if background_color.lower() == 'black':
        auto_text_color = 'white'
        auto_layer_color = 'cyan'
    else:
        auto_text_color = 'black'
        auto_layer_color = 'navy'

    title_color = auto_text_color if title_color == 'auto' else title_color
    label_color = auto_text_color if label_color == 'auto' else label_color
    layer_label_color = auto_text_color if layer_label_color == 'auto' else layer_label_color

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)

    # Render cells
    if render_mode == 'ellipse':
        print(f"   → Rendering {len(cells_df)} cells as ellipses...")

        # Group by cell type for efficient rendering
        for ct in range(N_CELL_TYPES):
            mask = cells_df['Class ID'] == ct
            if mask.sum() == 0:
                continue

            cells_ct = cells_df[mask]
            patches = []

            for _, cell in cells_ct.iterrows():
                # Create ellipse patch
                ellipse = Ellipse(
                    xy=(cell['X'], cell['Y']),
                    width=cell['Major Axis'] * cell_size_scale,
                    height=cell['Minor Axis'] * cell_size_scale,
                    angle=np.degrees(cell['Rotation']),  # Column is named 'Rotation' not 'Rotation Angle'
                    facecolor=CELL_TYPE_COLORS[ct],
                    edgecolor=cell_edge_color,
                    linewidth=cell_edge_width,
                    alpha=cell_alpha
                )
                patches.append(ellipse)

            # Add all ellipses as collection for better performance
            collection = PatchCollection(
                patches,
                facecolors=[CELL_TYPE_COLORS[ct]] * len(patches),
                edgecolors=cell_edge_color,
                linewidths=cell_edge_width,
                alpha=cell_alpha,
                label=CELL_TYPE_NAMES[ct]
            )
            ax.add_collection(collection)

    elif render_mode == 'scatter':
        print(f"   → Rendering {len(cells_df)} cells as scatter points...")

        for ct in range(N_CELL_TYPES):
            mask = cells_df['Class ID'] == ct
            if mask.sum() == 0:
                continue

            # Use major axis for size in scatter mode
            sizes = cells_df.loc[mask, 'Major Axis'] * cell_size_scale if 'Major Axis' in cells_df.columns else 3

            ax.scatter(
                cells_df.loc[mask, 'X'],
                cells_df.loc[mask, 'Y'],
                c=CELL_TYPE_COLORS[ct],
                s=sizes,
                alpha=cell_alpha,
                edgecolors=cell_edge_color,
                linewidths=cell_edge_width,
                label=CELL_TYPE_NAMES[ct],
                rasterized=True
            )

    # Render transcript dots (optional)
    if show_dots and dots_df is not None:
        print(f"   → Rendering {len(dots_df)} transcript dots...")
        scatter = ax.scatter(
            dots_df['x'],
            dots_df['y'],
            c=dots_df['gene'].astype('category').cat.codes,
            s=dot_size,
            alpha=dot_alpha,
            cmap=dot_colormap,
            rasterized=True
        )

    # Set limits
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, FRAME_WIDTH)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, FRAME_HEIGHT)

    ax.set_aspect('equal')
    ax.invert_yaxis()  # Flip Y so epidermis is on top

    # Add layer boundary lines
    if show_layer_lines:
        layer_depths = [
            STRATUM_CORNEUM_DEPTH,
            STRATUM_GRANULOSUM_DEPTH,
            STRATUM_SPINOSUM_DEPTH,
            STRATUM_BASALE_DEPTH,
            DEJ_DEPTH,
            PAPILLARY_DERMIS_DEPTH,
            RETICULAR_DERMIS_DEPTH,
        ]

        for depth in layer_depths:
            ax.axhline(
                y=depth,
                color=layer_line_color,
                linestyle=layer_line_style,
                linewidth=layer_line_width,
                alpha=layer_line_alpha
            )

    # Add layer labels
    if show_layer_labels:
        layer_info = [
            (STRATUM_CORNEUM_DEPTH / 2, 'Stratum Corneum'),
            ((STRATUM_CORNEUM_DEPTH + STRATUM_GRANULOSUM_DEPTH) / 2, 'Stratum Granulosum'),
            ((STRATUM_GRANULOSUM_DEPTH + STRATUM_SPINOSUM_DEPTH) / 2, 'Stratum Spinosum'),
            ((STRATUM_SPINOSUM_DEPTH + STRATUM_BASALE_DEPTH) / 2, 'Stratum Basale'),
            ((STRATUM_BASALE_DEPTH + DEJ_DEPTH) / 2, 'DEJ'),
            ((DEJ_DEPTH + PAPILLARY_DERMIS_DEPTH) / 2, 'Papillary Dermis'),
            ((PAPILLARY_DERMIS_DEPTH + RETICULAR_DERMIS_DEPTH) / 2, 'Reticular Dermis'),
            ((RETICULAR_DERMIS_DEPTH + FRAME_HEIGHT) / 2, 'Hypodermis'),
        ]

        for y_pos, label_text in layer_info:
            ax.text(
                10, y_pos, label_text,
                fontsize=layer_label_fontsize,
                color=layer_label_color,
                fontweight='bold',
                verticalalignment='center'
            )

    # Grid
    if show_grid:
        ax.grid(True, alpha=grid_alpha, color=grid_color, linestyle=':')

    # Title and labels
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', color=title_color, pad=20)
    ax.set_xlabel('X (µm)', fontsize=xlabel_fontsize, color=label_color)
    ax.set_ylabel('Depth (µm)', fontsize=ylabel_fontsize, color=label_color)

    # Tick colors
    ax.tick_params(colors=label_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(label_color)

    # Legend
    if show_legend:
        if render_mode == 'ellipse':
            # For ellipse mode, create custom legend handles
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=CELL_TYPE_COLORS[ct], label=CELL_TYPE_NAMES[ct])
                for ct in range(N_CELL_TYPES)
                if (cells_df['Class ID'] == ct).sum() > 0
            ]
            ax.legend(
                handles=legend_handles,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                framealpha=0.9
            )
        else:
            ax.legend(
                loc=legend_loc,
                bbox_to_anchor=legend_bbox,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                framealpha=0.9
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=background_color)
    print(f"   ✓ Saved {output_path}")

    return fig, ax


# ============================================================================
# MAIN TISSUE GENERATION
# ============================================================================

# ---- Lock geometry seed for deterministic placement ----
np.random.seed(GEOMETRY_SEED)

print("\n1. Defining cell morphology properties...")
cell_props = CellTypesProperties(
    n_cell_types=N_CELL_TYPES,
    sizes=CELL_SIZES,
    size_variation=CELL_SIZE_VARIATION,
    anisotropy=CELL_ANISOTROPY,
    anisotropy_variation=CELL_ANISO_VARIATION,
    relative_rna_concentration=RELATIVE_RNA_CONC,
    rna_concentration_variation=RNA_CONC_VARIATION,
)
print(f"   ✓ Cell morphology defined for {N_CELL_TYPES} cell types")


print("\n2. Creating regional polygons for skin layers...")

# === REGION 1: STRATUM CORNEUM ===
print("   - Stratum Corneum (0-20 µm)...")
R1_Corneum = create_epidermal_layer_polygon(
    top_y=0,
    bottom_y=STRATUM_CORNEUM_DEPTH,
    top_waviness=EPIDERMIS_WAVINESS,
    bottom_waviness=EPIDERMIS_WAVINESS * 0.7,
    n_waves_top=EPIDERMIS_N_WAVES,
    n_waves_bottom=EPIDERMIS_N_WAVES
)

# === REGION 2: STRATUM GRANULOSUM ===
print("   - Stratum Granulosum (20-40 µm)...")
R2_Granulosum = create_epidermal_layer_polygon(
    top_y=STRATUM_CORNEUM_DEPTH,
    bottom_y=STRATUM_GRANULOSUM_DEPTH,
    top_waviness=EPIDERMIS_WAVINESS * 0.7,
    bottom_waviness=EPIDERMIS_WAVINESS * 0.8,
    n_waves_top=EPIDERMIS_N_WAVES,
    n_waves_bottom=EPIDERMIS_N_WAVES
)

# === REGION 3: STRATUM SPINOSUM ===
print("   - Stratum Spinosum (40-100 µm)...")
R3_Spinosum = create_epidermal_layer_polygon(
    top_y=STRATUM_GRANULOSUM_DEPTH,
    bottom_y=STRATUM_SPINOSUM_DEPTH,
    top_waviness=EPIDERMIS_WAVINESS * 0.8,
    bottom_waviness=EPIDERMIS_WAVINESS,
    n_waves_top=EPIDERMIS_N_WAVES,
    n_waves_bottom=EPIDERMIS_N_WAVES + 1
)

# === REGION 4: STRATUM BASALE ===
print("   - Stratum Basale (100-130 µm)...")
R4_Basale = create_epidermal_layer_polygon(
    top_y=STRATUM_SPINOSUM_DEPTH,
    bottom_y=STRATUM_BASALE_DEPTH,
    top_waviness=EPIDERMIS_WAVINESS,
    bottom_waviness=DEJ_WAVINESS * 0.8,
    n_waves_top=EPIDERMIS_N_WAVES + 1,
    n_waves_bottom=DEJ_N_WAVES
)

# === REGION 5: DERMAL-EPIDERMAL JUNCTION (DEJ) ===
print("   - Dermal-Epidermal Junction (130-150 µm, rete ridges)...")
R5_DEJ = create_epidermal_layer_polygon(
    top_y=STRATUM_BASALE_DEPTH,
    bottom_y=DEJ_DEPTH,
    top_waviness=DEJ_WAVINESS * 0.8,
    bottom_waviness=DEJ_WAVINESS,
    n_waves_top=DEJ_N_WAVES,
    n_waves_bottom=DEJ_N_WAVES
)

# === REGION 6: PAPILLARY DERMIS ===
print("   - Papillary Dermis (150-350 µm)...")
R6_Papillary = box(0, DEJ_DEPTH, FRAME_WIDTH, PAPILLARY_DERMIS_DEPTH)

# === REGION 7: RETICULAR DERMIS ===
print("   - Reticular Dermis (350-1600 µm)...")
R7_Reticular = box(0, PAPILLARY_DERMIS_DEPTH, FRAME_WIDTH, RETICULAR_DERMIS_DEPTH)

# === REGION 8: HYPODERMIS ===
print("   - Hypodermis (800-1000 µm) with wavy upper boundary...")
# Create wavy dermis-hypodermis boundary (REALISTIC - irregular interdigitation)
R8_Hypodermis = create_epidermal_layer_polygon(
    top_y=HYPODERMIS_START,
    bottom_y=FRAME_HEIGHT,
    top_waviness=HYPODERMIS_WAVINESS,
    n_waves_top=HYPODERMIS_N_WAVES,
    bottom_waviness=0,  # Flat bottom edge
    n_waves_bottom=1
)

print("   ✓ Created 8 regional polygons")


# ============================================================================
# PRE-GENERATE CROSS-REGION STRUCTURE PARAMETERS
# Hair follicles span epidermis → dermis, so we pre-compute positions/angles
# and create matching elements for basale (pores), papillary (upper shaft),
# and reticular (lower shaft + bulb + sebaceous glands) regions.
# ============================================================================

print("\n2b. Pre-generating hair follicle parameters for cross-region continuity...")

follicle_params = []
papillary_follicle_structures = []
basale_pore_structures = []
reticular_follicle_structures = []

# Track positions for immune clustering (used later in reticular dermis)
appendage_positions = []

# Evenly space follicles along x-axis with small jitter for natural look
follicle_margin = 80  # margin from frame edges
follicle_spacing = (FRAME_WIDTH - 2 * follicle_margin) / max(1, N_HAIR_FOLLICLES - 1)
follicle_jitter = follicle_spacing * 0.15  # small jitter (15% of spacing)

for i in range(N_HAIR_FOLLICLES):
    fx_base = follicle_margin + i * follicle_spacing
    fx = fx_base + np.random.uniform(-follicle_jitter, follicle_jitter)
    fx = np.clip(fx, follicle_margin, FRAME_WIDTH - follicle_margin)
    # Anatomically correct: follicle originates near DEJ (epidermal invagination)
    fy_start = np.random.uniform(DEJ_DEPTH - 20, DEJ_DEPTH + 20)
    f_length = np.random.uniform(HAIR_FOLLICLE_LENGTH_MIN, HAIR_FOLLICLE_LENGTH_MAX)
    f_angle = np.random.uniform(np.pi/2 - 0.3, np.pi/2 + 0.3)

    follicle_params.append({
        'x': fx, 'y_start': fy_start, 'length': f_length, 'angle': f_angle
    })
    appendage_positions.append((fx, fy_start))

    # Pore at skin surface (in basale region)
    basale_pore_structures.append(create_follicle_pore(fx, STRATUM_BASALE_DEPTH))

    # Upper shaft / infundibulum (in papillary dermis region)
    papillary_follicle_structures.append(
        create_follicle_segment(fx, fy_start, f_length, f_angle)
    )

    # Full follicle + sebaceous glands (in reticular dermis region)
    reticular_follicle_structs = create_hair_follicle_with_sebaceous(
        x=fx, y_start=fy_start,
        follicle_length=f_length, follicle_angle=f_angle
    )
    reticular_follicle_structures.extend(reticular_follicle_structs)

print(f"   ✓ Pre-generated {N_HAIR_FOLLICLES} hair follicles with pores, upper shafts, and sebaceous glands")


print("\n3. Creating FOV distributions for each region...")

# === REGION 1: STRATUM CORNEUM (dead keratinized cells) ===
print("   - Configuring Stratum Corneum...")
CornifiedPrototype = lambda: FrameWideElement(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    tipical_cell_spacing=EPIDERMIS_CELL_SPACING,
    rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_KERATINOCYTE_CORNIFIED)]
)

fovd_corneum = FOVDistribution(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    background_element=CornifiedPrototype,
    other_elements=[],
    elements_frequency=[],
    attempts_at_elements=0
)

# === REGION 2: STRATUM GRANULOSUM ===
print("   - Configuring Stratum Granulosum...")
GranularPrototype = lambda: FrameWideElement(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    tipical_cell_spacing=EPIDERMIS_CELL_SPACING,
    rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_KERATINOCYTE_GRANULAR)]
)

fovd_granulosum = FOVDistribution(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    background_element=GranularPrototype,
    other_elements=[],
    elements_frequency=[],
    attempts_at_elements=0
)

# === REGION 3: STRATUM SPINOSUM ===
print("   - Configuring Stratum Spinosum...")
SpinosumPrototype = lambda: FrameWideElement(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    tipical_cell_spacing=EPIDERMIS_CELL_SPACING,
    rules=[SingleTypeRule(N_CELL_TYPES, cell_type_ix=ID_KERATINOCYTE_SPINOUS)]
)

fovd_spinosum = FOVDistribution(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    background_element=SpinosumPrototype,
    other_elements=[],
    elements_frequency=[],
    attempts_at_elements=0
)

# === REGION 4: STRATUM BASALE (with melanin units at DEJ) ===
print("   - Configuring Stratum Basale with melanin units at DEJ...")
BasalePrototype = lambda: FrameWideElement(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    tipical_cell_spacing=EPIDERMIS_CELL_SPACING,
    rules=[MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_KERATINOCYTE_BASAL, ID_LANGERHANS, ID_MELANOCYTE],
        proportions=[0.74, 0.14, 0.12]  # Added melanocytes (12%) for reliable 100+ cell generation
    )]
)

# Create melanin units (melanocyte + basal keratinocyte halos at DEJ)
melanin_units = [create_melanin_unit() for _ in range(N_MELANIN_UNITS)]

# Combine melanin units with hair follicle pore openings
basale_elements = melanin_units + basale_pore_structures

fovd_basale = FOVDistribution(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    background_element=BasalePrototype,
    other_elements=basale_elements,
    elements_frequency=[1.0] * len(basale_elements),
    attempts_at_elements=1  # One attempt per element (all have fixed positions + freq 1.0)
)
print(f"     ✓ Added {N_MELANIN_UNITS} melanin units (melanocyte + basal keratinocyte halos)")
print(f"     ✓ Added {len(basale_pore_structures)} hair follicle pore openings")

# === REGION 5: DEJ (transition zone) ===
print("   - Configuring Dermal-Epidermal Junction...")
DEJPrototype = lambda: FrameWideElement(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    tipical_cell_spacing=DERMIS_PAPILLARY_SPACING,
    rules=[MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_KERATINOCYTE_BASAL, ID_FIBROBLAST_PAPILLARY],
        proportions=[0.45, 0.55]  # Melanocytes now only in basale melanin units
    )]
)

fovd_dej = FOVDistribution(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    background_element=DEJPrototype,
    other_elements=[],
    elements_frequency=[],
    attempts_at_elements=0
)

# === REGION 6: PAPILLARY DERMIS (with capillaries and Meissner corpuscles) ===
print("   - Configuring Papillary Dermis with capillaries and Meissner corpuscles...")

PapillaryPrototype = lambda: FrameWideElement(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    tipical_cell_spacing=DERMIS_PAPILLARY_SPACING,
    rules=[MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_FIBROBLAST_PAPILLARY, ID_MACROPHAGE, ID_MAST_CELL, ID_T_CELL],
        proportions=[0.60, 0.14, 0.16, 0.10]  # Increased Mast_Cell to 16% (was 12%) for 100+ target
    )]
)

# Create capillary loops (REALISTIC - thin vessels approaching DEJ)
papillary_capillaries = [create_papillary_capillary() for _ in range(N_PAPILLARY_CAPILLARIES)]

# Create fine collagen fibers (ECM FIBER FIELD - PRIMARY DENSITY BOOST)
papillary_collagen = [create_papillary_collagen_bundle() for _ in range(N_COLLAGEN_BUNDLES_PAPILLARY)]

# Create elastin fibers (ECM FIBER FIELD - ADDITIONAL DENSITY)
papillary_elastin = [create_papillary_elastin_fibers() for _ in range(N_ELASTIN_FIBERS_PAPILLARY)]

# Combine structures (including hair follicle upper shafts / infundibula)
papillary_structures = (papillary_capillaries
                        + papillary_collagen + papillary_elastin
                        + papillary_follicle_structures)

fovd_papillary = FOVDistribution(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    background_element=PapillaryPrototype,
    other_elements=papillary_structures,
    elements_frequency=[1.0] * len(papillary_structures),
    attempts_at_elements=1  # One attempt per element (all have fixed positions + freq 1.0)
)
print(f"     ✓ Added {N_PAPILLARY_CAPILLARIES} capillary loops (vascular detail)")
print(f"     ✓ Added {N_COLLAGEN_BUNDLES_PAPILLARY} fine collagen bundles (ECM density boost)")
print(f"     ✓ Added {N_ELASTIN_FIBERS_PAPILLARY} elastin fibers (ECM density boost)")
print(f"     ✓ Added {len(papillary_follicle_structures)} hair follicle upper shafts (infundibula)")

# === REGION 7: RETICULAR DERMIS (main structures) ===
print("   - Configuring Reticular Dermis with hair follicles, glands, vessels, collagen...")

ReticularPrototype = lambda: FrameWideElement(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    tipical_cell_spacing=DERMIS_RETICULAR_SPACING,
    rules=[MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_FIBROBLAST_RETICULAR, ID_MACROPHAGE, ID_MAST_CELL, ID_T_CELL],
        proportions=[0.52, 0.17, 0.18, 0.13]  # Increased Mast_Cell to 18% (was 14%) for 100+ target
    )]
)

# Build all structures for reticular dermis
# (hair follicle structures were pre-generated above for cross-region continuity)
reticular_structures = list(reticular_follicle_structures)

# Track positions for immune clustering
vessel_positions = []
# (appendage_positions already populated during follicle pre-generation)

print(f"     ✓ Added {N_HAIR_FOLLICLES} hair follicles with sebaceous glands (from DEJ, anatomically correct)")

# Pre-compute sweat gland positions for immune cluster anchoring
# Random eccrine positions with minimum distance enforcement
eccrine_min_dist = 120  # minimum distance between glands
eccrine_positions = []
for _ in range(N_ECCRINE_GLANDS):
    for _attempt in range(100):
        x = np.random.uniform(100, FRAME_WIDTH - 100)
        y = np.random.uniform(PAPILLARY_DERMIS_DEPTH + 100, RETICULAR_DERMIS_DEPTH - 50)
        if all(np.sqrt((x - px)**2 + (y - py)**2) > eccrine_min_dist
               for px, py in eccrine_positions):
            break
    eccrine_positions.append((x, y))
    appendage_positions.append((x, y))

apocrine_positions = []
for _ in range(N_APOCRINE_GLANDS):
    x = np.random.uniform(100, FRAME_WIDTH - 100)
    y = np.random.uniform(RETICULAR_DERMIS_DEPTH - 300, RETICULAR_DERMIS_DEPTH)
    apocrine_positions.append((x, y))
    appendage_positions.append((x, y))

# Blood vessels (FIXED ANCHORING - pass position to function)
for _ in range(N_DERMAL_VESSELS):
    x = np.random.uniform(150, FRAME_WIDTH - 150)
    y = np.random.uniform(PAPILLARY_DERMIS_DEPTH + 50, RETICULAR_DERMIS_DEPTH - 150)
    vessel_positions.append((x, y))
    reticular_structures.append(create_dermal_blood_vessel(x=x, y=y))
print(f"     ✓ Added {N_DERMAL_VESSELS} branching blood vessels")

# Thick aligned collagen bundles (ECM FIBER FIELD - PRIMARY DENSITY BOOST)
for _ in range(N_COLLAGEN_BUNDLES_RETICULAR):
    reticular_structures.append(create_reticular_collagen_bundle())
print(f"     ✓ Added {N_COLLAGEN_BUNDLES_RETICULAR} thick collagen bundles (ECM density boost)")

# Aligned elastin fibers (ECM FIBER FIELD - ADDITIONAL DENSITY)
for _ in range(N_ELASTIN_FIBERS_RETICULAR):
    reticular_structures.append(create_reticular_elastin_fibers())
print(f"     ✓ Added {N_ELASTIN_FIBERS_RETICULAR} elastin fibers (ECM density boost)")

# Sweat glands (added AFTER collagen/elastin so they aren't overwritten)
for ex, ey in eccrine_positions:
    reticular_structures.append(create_eccrine_sweat_gland(x=ex, y=ey))
print(f"     ✓ Added {N_ECCRINE_GLANDS} eccrine sweat glands")

for ax, ay in apocrine_positions:
    reticular_structures.append(create_apocrine_sweat_gland(x=ax, y=ay))
print(f"     ✓ Added {N_APOCRINE_GLANDS} apocrine sweat glands")

# Perivascular immune infiltrates (clustered around vessels)
for vessel_x, vessel_y in vessel_positions[:N_PERIVASCULAR_IMMUNE_CLUSTERS]:
    reticular_structures.append(create_perivascular_immune_cluster(vessel_x, vessel_y))
print(f"     ✓ Added {N_PERIVASCULAR_IMMUNE_CLUSTERS} perivascular immune clusters")

# Periappendage immune infiltrates (clustered around hair/glands)
for appendage_x, appendage_y in appendage_positions[:N_PERIAPPENDAGE_IMMUNE_CLUSTERS]:
    reticular_structures.append(create_periappendage_immune_cluster(appendage_x, appendage_y))
print(f"     ✓ Added {N_PERIAPPENDAGE_IMMUNE_CLUSTERS} periappendage immune clusters")

# Pacinian corpuscles (added last so they aren't overwritten)
for _ in range(N_PACINIAN_CORPUSCLES):
    reticular_structures.append(create_pacinian_corpuscle())
print(f"     ✓ Added {N_PACINIAN_CORPUSCLES} Pacinian corpuscles")

fovd_reticular = FOVDistribution(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    background_element=ReticularPrototype,
    other_elements=reticular_structures,
    elements_frequency=[1.0] * len(reticular_structures),
    attempts_at_elements=1  # One attempt per element (all have fixed positions + freq 1.0)
)
print(f"     ✓ Total structures in reticular dermis: {len(reticular_structures)}")

# === REGION 8: HYPODERMIS (adipocytes) ===
print("   - Configuring Hypodermis with adipocytes...")

HypodermisPrototype = lambda: FrameWideElement(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    tipical_cell_spacing=HYPODERMIS_SPACING,
    rules=[MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=[ID_ADIPOCYTE, ID_FIBROBLAST_RETICULAR],
        proportions=[0.90, 0.10]
    )]
)

fovd_hypodermis = FOVDistribution(
    frame_size=max(FRAME_WIDTH, FRAME_HEIGHT),
    background_element=HypodermisPrototype,
    other_elements=[],
    elements_frequency=[],
    attempts_at_elements=0
)


# ============================================================================
# ASSEMBLE TISSUE SLICE
# ============================================================================

print("\n4. Assembling TissueSlice with regional specifications...")

all_regions = [
    # Background first (lowest priority)
    RegionSpec(name="Hypodermis", polygon=R8_Hypodermis, fovdist=fovd_hypodermis,
               priority=PRIORITY_HYPODERMIS, blend_band=BLEND_BAND_HYPODERMIS),

    RegionSpec(name="ReticularDermis", polygon=R7_Reticular, fovdist=fovd_reticular,
               priority=PRIORITY_RETICULAR_DERMIS, blend_band=BLEND_BAND_DERMIS),

    RegionSpec(name="PapillaryDermis", polygon=R6_Papillary, fovdist=fovd_papillary,
               priority=PRIORITY_PAPILLARY_DERMIS, blend_band=BLEND_BAND_DERMIS),

    RegionSpec(name="DEJ", polygon=R5_DEJ, fovdist=fovd_dej,
               priority=PRIORITY_DEJ, blend_band=BLEND_BAND_DEJ),

    # Epidermal layers (highest priority)
    RegionSpec(name="StratumBasale", polygon=R4_Basale, fovdist=fovd_basale,
               priority=PRIORITY_BASALE, blend_band=BLEND_BAND_EPIDERMIS),

    RegionSpec(name="StratumSpinosum", polygon=R3_Spinosum, fovdist=fovd_spinosum,
               priority=PRIORITY_SPINOSUM, blend_band=BLEND_BAND_EPIDERMIS),

    RegionSpec(name="StratumGranulosum", polygon=R2_Granulosum, fovdist=fovd_granulosum,
               priority=PRIORITY_GRANULOSUM, blend_band=BLEND_BAND_EPIDERMIS),

    RegionSpec(name="StratumCorneum", polygon=R1_Corneum, fovdist=fovd_corneum,
               priority=PRIORITY_CORNEUM, blend_band=BLEND_BAND_EPIDERMIS),
]

tissue_slice = TissueSlice(frame_size=max(FRAME_WIDTH, FRAME_HEIGHT), regions=all_regions)
print("   ✓ TissueSlice assembled with 8 regions")


print("\n5. Generating global tissue (geometry seed locked)...")
tissue_slice.generate_global()
tissue_slice.sample_labels()
print(f"   ✓ Generated {len(tissue_slice._cells_xy)} cells across all regions")


print("\n6. Creating FOV and applying properties...")
fov = FOV(
    cell_centroids=tissue_slice._cells_xy,
    cell_probabilities=tissue_slice._probs
)
fov.class_instance_one_hot = tissue_slice._class_onehot

cell_props.apply(fov)
print(f"   ✓ Applied morphology to {fov.n_cells} cells")


# ============================================================================
# HYPERPARAMETER SWEEP — 25 DATASETS (5 CASES x 5 SETTINGS)
#
# All sweeps share:
#   - The SAME spatial layout (geometry seed locked above)
#   - A SINGLE realistic base expression matrix built from
#     build_hierarchical_expression_matrix() with cell type hierarchy
#   - Unified base parameters (BASE_* constants)
#
# Each case varies exactly ONE axis:
#   CASE 1: Gene panel size (realistic noisy selection from 1000-gene base)
#   CASE 2: Reference quality (200 HVGs, progressively degraded scRNA-seq reference)
#   CASE 3: Gene sensitivity (200 HVGs, detection efficiency)
#   CASE 4: Noise levels (200 HVGs, admixture + dropout + background combined)
#   CASE 5: Noise decomposition (200 HVGs, admixture vs dropout separately)
# ============================================================================

# Generate cells_df once (spatial layout is shared across ALL cases)
cells_df = fov.make_pandas_df()
print(f"\n   Total cells for sweep: {len(cells_df)}")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Precompute FOV properties for noise application (needed by ALL cases)
cell_centroids = fov.cell_centroids
cell_types = fov.class_instance
cell_radii = fov.cell_major_axis / 2
n_cells = fov.n_cells
cell_tree = cKDTree(cell_centroids)

# Collect per-case stats for final summary table
_all_case_stats = []


def compute_per_cell_stats(dots_df):
    """Compute per-cell transcript count and detected gene statistics."""
    if len(dots_df) == 0:
        return {'mean_transcripts': 0, 'median_transcripts': 0,
                'mean_genes': 0, 'median_genes': 0}

    per_cell = dots_df.groupby('cell')
    transcripts_per_cell = per_cell.size()
    genes_per_cell = per_cell['gene'].nunique()

    return {
        'mean_transcripts': float(transcripts_per_cell.mean()),
        'median_transcripts': float(transcripts_per_cell.median()),
        'mean_genes': float(genes_per_cell.mean()),
        'median_genes': float(genes_per_cell.median()),
    }


def save_case(folder_name, tissue_obj, dots_df, case_params):
    """Save cells.csv, dots.csv, cell_type_expression.csv, and hyperparameters.json."""
    output_dir = f"{OUTPUT_FOLDER}/{folder_name}"
    os.makedirs(output_dir, exist_ok=True)

    cells_df.to_csv(f"{output_dir}/cells.csv", index=False)
    dots_df.to_csv(f"{output_dir}/dots.csv", index=False)

    tissue_df = tissue_obj.make_pandas_df()
    tissue_df.to_csv(f"{output_dir}/cell_type_expression.csv")

    # Compute per-cell summary stats
    stats = compute_per_cell_stats(dots_df)
    case_params['per_cell_mean_transcripts'] = stats['mean_transcripts']
    case_params['per_cell_median_transcripts'] = stats['median_transcripts']
    case_params['per_cell_mean_detected_genes'] = stats['mean_genes']
    case_params['per_cell_median_detected_genes'] = stats['median_genes']

    with open(f"{output_dir}/hyperparameters.json", 'w') as f:
        json.dump(case_params, f, indent=2)

    _all_case_stats.append({
        'dataset': folder_name,
        'n_genes': tissue_obj.n_genes,
        'total_dots': len(dots_df),
        **stats,
    })

    print(f"   -> {folder_name}: {len(dots_df):,} dots, "
          f"{tissue_obj.n_genes} genes")
    print(f"      Transcripts/cell: mean={stats['mean_transcripts']:.0f}, "
          f"median={stats['median_transcripts']:.0f}")
    print(f"      Genes/cell:       mean={stats['mean_genes']:.1f}, "
          f"median={stats['median_genes']:.1f}")


# ============================================================================
# GENERATE BASE EXPRESSION MATRIX (ONCE)
# ============================================================================

print("\n" + "=" * 80)
print("BUILDING HIERARCHICAL EXPRESSION MATRIX (BASE CASE)")
print("=" * 80)

base_expression_matrix, base_gene_categories, base_gene_names = \
    build_hierarchical_expression_matrix(
        n_genes=BASE_N_GENES,
        n_cell_types=N_CELL_TYPES,
        within_group_similarity=BASE_WITHIN_GROUP_SIMILARITY,
        base_expression_level=BASE_EXPRESSION_LEVEL,
        expression_level_std=BASE_EXPRESSION_STD,
        marker_fold_change=BASE_MARKER_FOLD_CHANGE,
        lineage_fold_change=BASE_LINEAGE_FOLD_CHANGE,
        seed=EXPRESSION_SEED,
    )

# Report gene category composition
from collections import Counter
cat_counts = Counter(base_gene_categories)
print(f"   Base matrix: {BASE_N_GENES} genes x {N_CELL_TYPES} cell types")
print(f"   Gene categories: {dict(cat_counts)}")
print(f"   Expression range: [{base_expression_matrix.min():.2f}, "
      f"{base_expression_matrix.max():.2f}]")

base_tissue = make_tissue_from_matrix(base_expression_matrix, base_gene_names)

# Run the base observation model to generate the full dot set (1000 genes)
print("   Generating base observation dots (full 1000-gene reference)...")
base_hybiss, base_dots_df_clean = run_observation_model(
    base_tissue, fov,
    sensitivity=BASE_GENE_SENSITIVITY,
    sensitivity_var=BASE_GENE_SENSITIVITY_VAR,
    tf_scale=BASE_TRANSFER_SCALE, tf_scale_std=BASE_TRANSFER_SCALE_STD,
    tf_offset=BASE_TRANSFER_OFFSET, tf_offset_std=BASE_TRANSFER_OFFSET_STD,
    seed=EXPRESSION_SEED,
)
print(f"   Clean base dots (1000 genes): {len(base_dots_df_clean):,}")

# Apply baseline noise to base dots (shared by CASE 1, 2, 3)
print("   Applying baseline observation noise to base dots...")
base_dots_df = apply_baseline_noise(
    base_dots_df_clean, list(base_gene_names),
    cell_centroids, cell_types, cell_radii, cell_tree,
    seed_offset=10000,
)
print(f"   Noisy base dots (1000 genes): {len(base_dots_df):,}")

# Pre-select 200 HVGs from base matrix (used by CASE 2-4)
print(f"\n   Selecting {SWEEP_N_GENES} HVGs from base {BASE_N_GENES}-gene reference...")
base_hvg_indices = select_realistic_gene_panel(
    base_expression_matrix, SWEEP_N_GENES,
    gene_categories=base_gene_categories,
    selection_noise_std=0.5,
    housekeeping_leak_frac=0.10,
    seed=EXPRESSION_SEED,
)
base_hvg_matrix = base_expression_matrix[base_hvg_indices, :]
base_hvg_names = [base_gene_names[i] for i in base_hvg_indices]
base_hvg_categories = base_gene_categories[base_hvg_indices]
hvg_cat_counts = Counter(base_hvg_categories)
print(f"   HVG panel composition: {dict(hvg_cat_counts)}")

# Pre-filter base noisy dots to HVG subset (used by CASE 2, 3)
base_hvg_set = set(base_hvg_names)
base_hvg_dots_noisy = base_dots_df[base_dots_df['gene'].isin(base_hvg_set)].copy()
print(f"   Noisy HVG dots ({SWEEP_N_GENES} genes): {len(base_hvg_dots_noisy):,}")


# ============================================================================
# CASE 1: GENE PANEL SIZE (realistic noisy selection)
#
# Uses select_realistic_gene_panel() to subset from 1000-gene base.
# The selection adds noise and forces housekeeping genes into the panel,
# mimicking real panel design from noisy scRNA-seq references.
# ============================================================================

print("\n" + "=" * 80)
print("CASE 1: GENE PANEL SIZE (realistic selection)")
print("=" * 80)

for n_genes_target in [1000, 500, 200, 100, 50]:
    folder = f"CASE1_{n_genes_target}genes"
    print(f"\n   {folder} ({n_genes_target} genes)...")

    selected_indices = select_realistic_gene_panel(
        base_expression_matrix, n_genes_target,
        gene_categories=base_gene_categories,
        selection_noise_std=0.5,
        housekeeping_leak_frac=0.10,
        seed=EXPRESSION_SEED,
    )

    sub_matrix = base_expression_matrix[selected_indices, :]
    sub_names = [base_gene_names[i] for i in selected_indices]
    sub_categories = base_gene_categories[selected_indices]

    tissue_sub = make_tissue_from_matrix(sub_matrix, sub_names)

    # Subset dots to selected genes only (from noisy base dots)
    selected_set = set(sub_names)
    dots_subset = base_dots_df[base_dots_df['gene'].isin(selected_set)].copy()

    # Case-1-specific extra dropout (simulates panel-specific technical losses
    # such as probe inefficiency, panel crosstalk, and imaging artifacts that
    # scale with the number of multiplexed targets)
    case1_rng = np.random.default_rng(EXPRESSION_SEED + 500 + n_genes_target)
    n_before_dropout = len(dots_subset)
    keep_mask = case1_rng.random(n_before_dropout) < CASE1_EXTRA_DROPOUT_RATE
    dots_subset = dots_subset[keep_mask].reset_index(drop=True)

    # Report panel composition
    sub_cat_counts = Counter(sub_categories)
    print(f"     Panel composition: {dict(sub_cat_counts)}")
    print(f"     Dots after gene filter: {n_before_dropout:,}")
    print(f"     Dots after panel dropout ({CASE1_EXTRA_DROPOUT_RATE:.0%}): {len(dots_subset):,}")

    case_params = {
        'case': 'CASE1_gene_panel_size',
        'n_genes': n_genes_target,
        'gene_selection': 'noisy_hvg_with_hk_leakage',
        'panel_composition': dict(sub_cat_counts),
        'gene_sensitivity': BASE_GENE_SENSITIVITY,
        'gene_sensitivity_variation': BASE_GENE_SENSITIVITY_VAR,
        'within_group_similarity': BASE_WITHIN_GROUP_SIMILARITY,
        'base_expression_level': BASE_EXPRESSION_LEVEL,
        'base_expression_std': BASE_EXPRESSION_STD,
        'marker_fold_change': BASE_MARKER_FOLD_CHANGE,
        'lineage_fold_change': BASE_LINEAGE_FOLD_CHANGE,
        'transfer_function': 'AffineNonNegTransfer',
        'transfer_scale': BASE_TRANSFER_SCALE,
        'transfer_scale_std': BASE_TRANSFER_SCALE_STD,
        'transfer_offset': BASE_TRANSFER_OFFSET,
        'transfer_offset_std': BASE_TRANSFER_OFFSET_STD,
        'baseline_noise': True,
        'baseline_noise_lateral_bw': BASELINE_NOISE_LATERAL_BW,
        'baseline_noise_lateral_tr': BASELINE_NOISE_LATERAL_TR,
        'baseline_noise_z_rate': BASELINE_NOISE_Z_RATE,
        'baseline_noise_det_rate': BASELINE_NOISE_DET_RATE,
        'baseline_noise_bg_rate': BASELINE_NOISE_BG_RATE,
        'case1_extra_dropout_rate': CASE1_EXTRA_DROPOUT_RATE,
        'relative_rna_concentration': RELATIVE_RNA_CONC,
        'rna_concentration_variation': RNA_CONC_VARIATION,
        'geometry_seed': GEOMETRY_SEED,
        'expression_seed': EXPRESSION_SEED,
    }

    save_case(folder, tissue_sub, dots_subset, case_params)


# ============================================================================
# CASE 2: REFERENCE QUALITY
#
# Tests how robust cell typing is to imperfect scRNA-seq references.
# The SPATIAL DATA stays fixed (base 200 HVGs + baseline noise).
# Only the REFERENCE expression matrix changes, progressively degraded by:
#   - Multiplicative lognormal noise (measurement error in scRNA-seq)
#   - Profile blurring (mixing each type's profile with the global mean,
#     simulating doublets/contamination in the reference dataset)
#
# This is a realistic axis: in practice, the scRNA-seq reference never
# perfectly matches the spatial data.
# ============================================================================

print("\n" + "=" * 80)
print("CASE 2: REFERENCE QUALITY")
print("=" * 80)

# Each setting: (name, noise_cv, blur_factor)
reference_quality_configs = [
    ('ref1_clean',     0.05,  0.00),   # Near-perfect reference
    ('ref2_noisy',     0.30,  0.10),   # Moderately noisy
    ('ref3_degraded',  0.55,  0.25),   # Degraded
    ('ref4_poor',      0.85,  0.40),   # Poor quality
    ('ref5_terrible',  1.20,  0.55),   # Very poor reference
]

for (ref_name, noise_cv, blur_factor) in reference_quality_configs:
    folder = f"CASE2_{ref_name}"
    print(f"\n   {folder} (noise_cv={noise_cv}, blur={blur_factor})...")

    # Degrade the base HVG expression matrix
    degraded_matrix = degrade_reference_matrix(
        base_hvg_matrix,
        noise_cv=noise_cv,
        blur_factor=blur_factor,
        seed=EXPRESSION_SEED + 100,
    )
    tissue_c2 = make_tissue_from_matrix(degraded_matrix, base_hvg_names)

    # Data stays the same: base HVG dots with baseline noise
    # (already computed as base_hvg_dots_noisy)

    case_params = {
        'case': 'CASE2_reference_quality',
        'reference_level': ref_name,
        'n_genes_reference': BASE_N_GENES,
        'n_genes_panel': SWEEP_N_GENES,
        'reference_noise_cv': noise_cv,
        'reference_blur_factor': blur_factor,
        'data_expression_params': 'base (unchanged)',
        'marker_fold_change': BASE_MARKER_FOLD_CHANGE,
        'lineage_fold_change': BASE_LINEAGE_FOLD_CHANGE,
        'cross_group_leakage': BASE_CROSS_GROUP_LEAKAGE,
        'gene_sensitivity': BASE_GENE_SENSITIVITY,
        'gene_sensitivity_variation': BASE_GENE_SENSITIVITY_VAR,
        'baseline_noise': True,
        'baseline_noise_lateral_bw': BASELINE_NOISE_LATERAL_BW,
        'baseline_noise_lateral_tr': BASELINE_NOISE_LATERAL_TR,
        'baseline_noise_z_rate': BASELINE_NOISE_Z_RATE,
        'baseline_noise_det_rate': BASELINE_NOISE_DET_RATE,
        'baseline_noise_bg_rate': BASELINE_NOISE_BG_RATE,
        'geometry_seed': GEOMETRY_SEED,
        'expression_seed': EXPRESSION_SEED,
    }

    save_case(folder, tissue_c2, base_hvg_dots_noisy, case_params)


# ============================================================================
# CASE 3: GENE SENSITIVITY
#
# Uses 200 HVGs from the base expression matrix.  Varies detection
# efficiency (gene_sensitivity and its variation).  Baseline observation
# noise is applied on top to ensure base case ≤ 0.7 accuracy.
# ============================================================================

print("\n" + "=" * 80)
print("CASE 3: EFFECT OF GENE SENSITIVITY")
print("=" * 80)

tissue_c3_ref = make_tissue_from_matrix(base_hvg_matrix, base_hvg_names)

for gs in [0.9, 0.7, 0.5, 0.3, 0.1]:
    gs_var = gs * 0.5  # Moderate gene-to-gene variation
    folder = f"CASE3_gs{gs}"
    print(f"\n   {folder} (sensitivity={gs}, variation={gs_var:.3f})...")

    tissue_c3 = make_tissue_from_matrix(base_hvg_matrix, base_hvg_names)

    _, dots_c3_clean = run_observation_model(
        tissue_c3, fov,
        sensitivity=gs,
        sensitivity_var=gs_var,
        tf_scale=BASE_TRANSFER_SCALE, tf_scale_std=BASE_TRANSFER_SCALE_STD,
        tf_offset=BASE_TRANSFER_OFFSET, tf_offset_std=BASE_TRANSFER_OFFSET_STD,
        seed=EXPRESSION_SEED + 1000,
    )
    print(f"     Clean dots: {len(dots_c3_clean):,}")

    # Apply baseline noise (realistic admixture ~10%)
    dots_c3 = apply_baseline_noise(
        dots_c3_clean, list(base_hvg_names),
        cell_centroids, cell_types, cell_radii, cell_tree,
        seed_offset=11000,
    )

    # Case-3-specific extra dropout (simulates sensitivity-dependent detection
    # losses: lower-sensitivity assays also suffer from increased optical
    # crosstalk and amplification noise that reduce effective transcript yield)
    case3_rng = np.random.default_rng(EXPRESSION_SEED + 600 + int(gs * 100))
    n_before_c3_dropout = len(dots_c3)
    keep_mask_c3 = case3_rng.random(n_before_c3_dropout) < CASE3_EXTRA_DROPOUT_RATE
    dots_c3 = dots_c3[keep_mask_c3].reset_index(drop=True)
    print(f"     After extra dropout ({CASE3_EXTRA_DROPOUT_RATE:.0%}): {len(dots_c3):,}")

    case_params = {
        'case': 'CASE3_gene_sensitivity',
        'n_genes_reference': BASE_N_GENES,
        'n_genes_panel': SWEEP_N_GENES,
        'gene_sensitivity': gs,
        'gene_sensitivity_variation': gs_var,
        'within_group_similarity': BASE_WITHIN_GROUP_SIMILARITY,
        'base_expression_level': BASE_EXPRESSION_LEVEL,
        'base_expression_std': BASE_EXPRESSION_STD,
        'transfer_function': 'AffineNonNegTransfer',
        'transfer_scale': BASE_TRANSFER_SCALE,
        'transfer_scale_std': BASE_TRANSFER_SCALE_STD,
        'transfer_offset': BASE_TRANSFER_OFFSET,
        'transfer_offset_std': BASE_TRANSFER_OFFSET_STD,
        'baseline_noise': True,
        'baseline_noise_lateral_bw': BASELINE_NOISE_LATERAL_BW,
        'baseline_noise_lateral_tr': BASELINE_NOISE_LATERAL_TR,
        'baseline_noise_z_rate': BASELINE_NOISE_Z_RATE,
        'baseline_noise_det_rate': BASELINE_NOISE_DET_RATE,
        'baseline_noise_bg_rate': BASELINE_NOISE_BG_RATE,
        'case3_extra_dropout_rate': CASE3_EXTRA_DROPOUT_RATE,
        'relative_rna_concentration': RELATIVE_RNA_CONC,
        'rna_concentration_variation': RNA_CONC_VARIATION,
        'geometry_seed': GEOMETRY_SEED,
        'expression_seed': EXPRESSION_SEED,
    }

    save_case(folder, tissue_c3_ref, dots_c3, case_params)


# ============================================================================
# CASE 4: EFFECT OF NOISE LEVELS
#
# Uses 200 HVGs from the base expression matrix.  Progressively increases:
#   - Lateral 2D admixture (segmentation boundary errors)
#   - Z-axis admixture (out-of-plane contamination)
#   - Transcript dropout (detection failure)
#   - Background noise (false positive transcripts)
# ============================================================================

print("\n" + "=" * 80)
print("CASE 4: EFFECT OF NOISE LEVELS")
print("=" * 80)

# Generate clean base dots for Case 4 (using 200 HVG subset)
tissue_c4 = make_tissue_from_matrix(base_hvg_matrix, base_hvg_names)
_, dots_df_c4_clean = run_observation_model(
    tissue_c4, fov,
    sensitivity=BASE_GENE_SENSITIVITY,
    sensitivity_var=BASE_GENE_SENSITIVITY_VAR,
    tf_scale=BASE_TRANSFER_SCALE, tf_scale_std=BASE_TRANSFER_SCALE_STD,
    tf_offset=BASE_TRANSFER_OFFSET, tf_offset_std=BASE_TRANSFER_OFFSET_STD,
    seed=EXPRESSION_SEED + 2000,
)
print(f"   Clean dots ({SWEEP_N_GENES} HVGs): {len(dots_df_c4_clean):,}")

gene_names_c4 = list(base_hvg_names)
n_genes_c4 = len(gene_names_c4)

# 5 noise levels: baseline -> severe
# noise1 matches baseline noise to ensure base accuracy ~0.7
noise_configs = [
    # (name, lateral_bw, lateral_tr, lateral_dd,
    #  z_rate, z_corr, z_radius,
    #  dropout_rate, dropout_expr_dep, dropout_gene_cv,
    #  bg_rate)
    # Realistic admixture: baseline ~10%, severe ~35% (not >50%)
    ('noise1_baseline', 12.0, 0.13, 0.25,
     0.07, 0.50, 60.0,
     0.78, 0.22, 0.18,
     0.006),
    ('noise2_low', 14.0, 0.18, 0.22,
     0.09, 0.45, 70.0,
     0.71, 0.30, 0.22,
     0.007),
    ('noise3_moderate', 16.0, 0.24, 0.18,
     0.11, 0.38, 80.0,
     0.40, 0.42, 0.35,
     0.012),
    ('noise4_high', 20.0, 0.30, 0.14,
     0.15, 0.30, 95.0,
     0.23, 0.55, 0.48,
     0.018),
    ('noise5_severe', 25.0, 0.36, 0.10,
     0.19, 0.22, 110.0,
     0.10, 0.68, 0.58,
     0.028),
]

for (noise_name, lat_bw, lat_tr, lat_dd,
     z_rate, z_corr, z_radius,
     det_rate, det_expr_dep, det_gene_cv,
     bg_rate) in noise_configs:

    folder = f"CASE4_{noise_name}"
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
        seed=EXPRESSION_SEED + 3000
    )
    z_admix = ZAxisAdmixture(
        z_contamination_rate=z_rate,
        neighborhood_correlation=z_corr,
        neighborhood_radius=z_radius,
        seed=EXPRESSION_SEED + 3001
    )
    combined_admix = CompositeAdmixture(models=[lateral_admix, z_admix])

    dots_noisy = combined_admix.apply(
        dots_df_c4_clean.copy(),
        cell_centroids,
        cell_types,
        cell_radii
    )
    admix_summary = combined_admix.get_admixture_summary()
    n_reassigned = admix_summary['n_reassigned']
    print(f"     -> Admixture reassigned {n_reassigned} dots "
          f"({admix_summary['reassignment_rate']:.1%})")

    # --- Step 2: Apply dropout using DropoutModel API ---
    dropout_model = DropoutModel(
        baseline_detection_rate=det_rate,
        expression_dependence=det_expr_dep,
        gene_variation_cv=det_gene_cv,
        seed=EXPRESSION_SEED + 4000,
    )
    dropout_model.generate_gene_detection_rates(n_genes_c4)

    # Map per-gene detection rates to each dot
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names_c4)}
    dot_gene_idx = dots_noisy['gene'].map(gene_name_to_idx).values
    per_dot_factor = dropout_model.gene_detection_rates[dot_gene_idx]
    detection_prob = np.clip(det_rate * per_dot_factor, 0.0, 1.0)
    keep_mask = dropout_model.rng.random(len(dots_noisy)) < detection_prob
    n_dropped = (~keep_mask).sum()
    dots_noisy = dots_noisy[keep_mask].reset_index(drop=True)
    print(f"     -> Dropout removed {n_dropped} dots "
          f"({n_dropped / (len(dots_noisy) + n_dropped):.1%})")

    # --- Step 3: Add background noise using BackgroundNoise API ---
    bg_frame_size = max(FRAME_WIDTH, FRAME_HEIGHT)
    effective_bg_rate = bg_rate * (FRAME_WIDTH * FRAME_HEIGHT) / (bg_frame_size ** 2)

    bg_noise_model = BackgroundNoise(
        frame_size=bg_frame_size,
        background_rate=effective_bg_rate,
        spatial_pattern="uniform",
        seed=EXPRESSION_SEED + 5000,
    )
    bg_x, bg_y, bg_gene_idx = bg_noise_model.generate_background_dots(
        n_genes=n_genes_c4,
        gene_names=gene_names_c4,
    )
    n_bg = len(bg_x)

    if n_bg > 0:
        bg_x = bg_x / bg_frame_size * FRAME_WIDTH
        bg_y = bg_y / bg_frame_size * FRAME_HEIGHT
        bg_gene_names = [gene_names_c4[gi] for gi in bg_gene_idx]
        _, bg_cell_idx = cell_tree.query(np.column_stack([bg_x, bg_y]))

        bg_df = pd.DataFrame({
            'x': bg_x,
            'y': bg_y,
            'gene': bg_gene_names,
            'cell': bg_cell_idx.astype(int),
        })
        dots_noisy = pd.concat([dots_noisy, bg_df], ignore_index=True)
        print(f"     -> Background added {n_bg} false-positive dots")
    else:
        print(f"     -> Background added 0 false-positive dots")

    print(f"     -> Final dot count: {len(dots_noisy):,} "
          f"(clean: {len(dots_df_c4_clean):,})")

    case_params = {
        'case': 'CASE4_noise_level',
        'noise_level': noise_name,
        'n_genes_reference': BASE_N_GENES,
        'n_genes_panel': SWEEP_N_GENES,
        'gene_sensitivity': BASE_GENE_SENSITIVITY,
        'gene_sensitivity_variation': BASE_GENE_SENSITIVITY_VAR,
        'within_group_similarity': BASE_WITHIN_GROUP_SIMILARITY,
        'base_expression_level': BASE_EXPRESSION_LEVEL,
        'base_expression_std': BASE_EXPRESSION_STD,
        'transfer_function': 'AffineNonNegTransfer',
        'transfer_scale': BASE_TRANSFER_SCALE,
        'transfer_scale_std': BASE_TRANSFER_SCALE_STD,
        'transfer_offset': BASE_TRANSFER_OFFSET,
        'transfer_offset_std': BASE_TRANSFER_OFFSET_STD,
        'lateral_admixture_boundary_width': lat_bw,
        'lateral_admixture_transfer_rate': lat_tr,
        'lateral_admixture_distance_decay': lat_dd,
        'z_admixture_contamination_rate': z_rate,
        'z_admixture_neighborhood_correlation': z_corr,
        'z_admixture_neighborhood_radius': z_radius,
        'dropout_baseline_detection_rate': det_rate,
        'dropout_expression_dependence': det_expr_dep,
        'dropout_gene_variation_cv': det_gene_cv,
        'background_noise_rate': bg_rate,
        'n_dots_clean': len(dots_df_c4_clean),
        'n_dots_admixture_reassigned': int(n_reassigned),
        'n_dots_dropped': int(n_dropped),
        'n_dots_background_added': int(n_bg),
        'n_dots_final': len(dots_noisy),
        'relative_rna_concentration': RELATIVE_RNA_CONC,
        'rna_concentration_variation': RNA_CONC_VARIATION,
        'geometry_seed': GEOMETRY_SEED,
        'expression_seed': EXPRESSION_SEED,
    }

    save_case(folder, tissue_c4, dots_noisy, case_params)


# ============================================================================
# CASE 5: NOISE DECOMPOSITION (admixture vs dropout separately)
#
# Uses 200 HVGs from the base expression matrix.  Unlike CASE 4 which
# increases ALL noise sources together, this case isolates each noise source:
#   CASE 5a: Admixture sweep (low/mid/high) with minimal dropout
#   CASE 5b: Dropout sweep (low/mid/high) with minimal admixture
#
# This allows disentangling how each noise source independently affects
# cell typing performance.
# ============================================================================

print("\n" + "=" * 80)
print("CASE 5: NOISE DECOMPOSITION (separate sources)")
print("=" * 80)

# Generate clean base dots for Case 5 (reuse tissue_c4 which is already 200 HVGs)
tissue_c5 = make_tissue_from_matrix(base_hvg_matrix, base_hvg_names)
_, dots_df_c5_clean = run_observation_model(
    tissue_c5, fov,
    sensitivity=BASE_GENE_SENSITIVITY,
    sensitivity_var=BASE_GENE_SENSITIVITY_VAR,
    tf_scale=BASE_TRANSFER_SCALE, tf_scale_std=BASE_TRANSFER_SCALE_STD,
    tf_offset=BASE_TRANSFER_OFFSET, tf_offset_std=BASE_TRANSFER_OFFSET_STD,
    seed=EXPRESSION_SEED + 6000,
)
print(f"   Clean dots ({SWEEP_N_GENES} HVGs): {len(dots_df_c5_clean):,}")

gene_names_c5 = list(base_hvg_names)
n_genes_c5 = len(gene_names_c5)

# Minimal noise settings (lighter than baseline to make CASE 5 bases ~10% easier)
MINIMAL_LAT_BW = 10.0                          # Lighter than BASELINE=12.0
MINIMAL_LAT_TR = 0.08                          # Lighter than BASELINE=0.12
MINIMAL_LAT_DD = 0.28                          # Faster decay than BASELINE=0.25
MINIMAL_Z_RATE = 0.03                          # Less than BASELINE=0.04
MINIMAL_Z_CORR = 0.55                          # Higher correlation than BASELINE=0.50
MINIMAL_Z_RADIUS = 50.0                        # Smaller than BASELINE=60.0
MINIMAL_DET_RATE = 0.75                        # Less dropout than BASELINE=0.80
MINIMAL_DET_EXPR_DEP = 0.22                    # Slightly less than BASELINE=0.25
MINIMAL_DET_GENE_CV = 0.18                     # Slightly less than BASELINE=0.20
MINIMAL_BG_RATE = 0.008                        # Matches BASELINE for realism

# --- CASE 5a: Admixture sweep (low/mid/high) with baseline dropout ---
admixture_configs = [
    ('admix_low',  10.0, 0.09, 0.30,  0.04, 0.60, 50.0),
    ('admix_mid',  16.0, 0.22, 0.20,  0.12, 0.35, 80.0),
    ('admix_high', 24.0, 0.36, 0.12,  0.22, 0.20, 110.0),
]

for (name, lat_bw, lat_tr, lat_dd, z_rate, z_corr, z_radius) in admixture_configs:
    folder = f"CASE5a_{name}"
    print(f"\n   {folder}:")
    print(f"     Admixture: lateral_bw={lat_bw}, lateral_tr={lat_tr}, "
          f"z_rate={z_rate}")
    print(f"     Dropout: minimal (det_rate={MINIMAL_DET_RATE})")

    # Apply admixture at the sweep level
    lateral_admix = Lateral2DAdmixture(
        boundary_width=lat_bw,
        transfer_rate=lat_tr,
        distance_decay=lat_dd,
        seed=EXPRESSION_SEED + 7000,
    )
    z_admix = ZAxisAdmixture(
        z_contamination_rate=z_rate,
        neighborhood_correlation=z_corr,
        neighborhood_radius=z_radius,
        seed=EXPRESSION_SEED + 7001,
    )
    combined_admix = CompositeAdmixture(models=[lateral_admix, z_admix])

    dots_noisy = combined_admix.apply(
        dots_df_c5_clean.copy(),
        cell_centroids, cell_types, cell_radii,
    )
    admix_summary = combined_admix.get_admixture_summary()
    print(f"     -> Admixture reassigned {admix_summary['n_reassigned']} dots "
          f"({admix_summary['reassignment_rate']:.1%})")

    # Apply minimal dropout
    dropout_model = DropoutModel(
        baseline_detection_rate=MINIMAL_DET_RATE,
        expression_dependence=MINIMAL_DET_EXPR_DEP,
        gene_variation_cv=MINIMAL_DET_GENE_CV,
        seed=EXPRESSION_SEED + 8000,
    )
    dropout_model.generate_gene_detection_rates(n_genes_c5)
    gene_name_to_idx_c5 = {g: i for i, g in enumerate(gene_names_c5)}
    dot_gene_idx = dots_noisy['gene'].map(gene_name_to_idx_c5).values
    per_dot_factor = dropout_model.gene_detection_rates[dot_gene_idx]
    detection_prob = np.clip(MINIMAL_DET_RATE * per_dot_factor, 0.0, 1.0)
    keep_mask = dropout_model.rng.random(len(dots_noisy)) < detection_prob
    n_dropped = (~keep_mask).sum()
    dots_noisy = dots_noisy[keep_mask].reset_index(drop=True)
    print(f"     -> Minimal dropout removed {n_dropped} dots")

    # Minimal background
    bg_frame_size = max(FRAME_WIDTH, FRAME_HEIGHT)
    effective_bg_rate = MINIMAL_BG_RATE * (FRAME_WIDTH * FRAME_HEIGHT) / (bg_frame_size ** 2)
    bg_noise_model = BackgroundNoise(
        frame_size=bg_frame_size,
        background_rate=effective_bg_rate,
        spatial_pattern="uniform",
        seed=EXPRESSION_SEED + 9000,
    )
    bg_x, bg_y, bg_gene_idx = bg_noise_model.generate_background_dots(
        n_genes=n_genes_c5, gene_names=gene_names_c5,
    )
    if len(bg_x) > 0:
        bg_x = bg_x / bg_frame_size * FRAME_WIDTH
        bg_y = bg_y / bg_frame_size * FRAME_HEIGHT
        bg_gene_names_c5 = [gene_names_c5[gi] for gi in bg_gene_idx]
        _, bg_cell_idx = cell_tree.query(np.column_stack([bg_x, bg_y]))
        bg_df = pd.DataFrame({
            'x': bg_x, 'y': bg_y,
            'gene': bg_gene_names_c5,
            'cell': bg_cell_idx.astype(int),
        })
        dots_noisy = pd.concat([dots_noisy, bg_df], ignore_index=True)

    print(f"     -> Final: {len(dots_noisy):,} dots")

    case_params = {
        'case': 'CASE5a_admixture_sweep',
        'noise_source': name,
        'n_genes_reference': BASE_N_GENES,
        'n_genes_panel': SWEEP_N_GENES,
        'lateral_admixture_boundary_width': lat_bw,
        'lateral_admixture_transfer_rate': lat_tr,
        'lateral_admixture_distance_decay': lat_dd,
        'z_admixture_contamination_rate': z_rate,
        'z_admixture_neighborhood_correlation': z_corr,
        'z_admixture_neighborhood_radius': z_radius,
        'dropout_baseline_detection_rate': MINIMAL_DET_RATE,
        'dropout_expression_dependence': MINIMAL_DET_EXPR_DEP,
        'dropout_gene_variation_cv': MINIMAL_DET_GENE_CV,
        'background_noise_rate': MINIMAL_BG_RATE,
        'gene_sensitivity': BASE_GENE_SENSITIVITY,
        'gene_sensitivity_variation': BASE_GENE_SENSITIVITY_VAR,
        'within_group_similarity': BASE_WITHIN_GROUP_SIMILARITY,
        'n_dots_clean': len(dots_df_c5_clean),
        'n_dots_final': len(dots_noisy),
        'geometry_seed': GEOMETRY_SEED,
        'expression_seed': EXPRESSION_SEED,
    }
    save_case(folder, tissue_c5, dots_noisy, case_params)

# --- CASE 5b: Dropout sweep (low/mid/high) with minimal admixture ---
dropout_configs = [
    ('dropout_low',  0.56, 0.35, 0.30),
    ('dropout_mid',  0.32, 0.48, 0.45),
    ('dropout_high', 0.20, 0.55, 0.75),
]

for (name, det_rate, det_expr_dep, det_gene_cv) in dropout_configs:
    folder = f"CASE5b_{name}"
    print(f"\n   {folder}:")
    print(f"     Admixture: minimal (lat_tr={MINIMAL_LAT_TR}, z_rate={MINIMAL_Z_RATE})")
    print(f"     Dropout: det_rate={det_rate}, expr_dep={det_expr_dep}")

    # Apply minimal admixture
    lateral_admix = Lateral2DAdmixture(
        boundary_width=MINIMAL_LAT_BW,
        transfer_rate=MINIMAL_LAT_TR,
        distance_decay=MINIMAL_LAT_DD,
        seed=EXPRESSION_SEED + 7000,
    )
    z_admix = ZAxisAdmixture(
        z_contamination_rate=MINIMAL_Z_RATE,
        neighborhood_correlation=MINIMAL_Z_CORR,
        neighborhood_radius=MINIMAL_Z_RADIUS,
        seed=EXPRESSION_SEED + 7001,
    )
    combined_admix = CompositeAdmixture(models=[lateral_admix, z_admix])

    dots_noisy = combined_admix.apply(
        dots_df_c5_clean.copy(),
        cell_centroids, cell_types, cell_radii,
    )
    admix_summary = combined_admix.get_admixture_summary()
    print(f"     -> Minimal admixture reassigned {admix_summary['n_reassigned']} dots")

    # Apply dropout at the sweep level
    dropout_model = DropoutModel(
        baseline_detection_rate=det_rate,
        expression_dependence=det_expr_dep,
        gene_variation_cv=det_gene_cv,
        seed=EXPRESSION_SEED + 8000,
    )
    dropout_model.generate_gene_detection_rates(n_genes_c5)
    gene_name_to_idx_c5 = {g: i for i, g in enumerate(gene_names_c5)}
    dot_gene_idx = dots_noisy['gene'].map(gene_name_to_idx_c5).values
    per_dot_factor = dropout_model.gene_detection_rates[dot_gene_idx]
    detection_prob = np.clip(det_rate * per_dot_factor, 0.0, 1.0)
    keep_mask = dropout_model.rng.random(len(dots_noisy)) < detection_prob
    n_dropped = (~keep_mask).sum()
    dots_noisy = dots_noisy[keep_mask].reset_index(drop=True)
    print(f"     -> Dropout removed {n_dropped} dots "
          f"({n_dropped / (len(dots_noisy) + n_dropped):.1%})")

    # Minimal background
    bg_noise_model = BackgroundNoise(
        frame_size=bg_frame_size,
        background_rate=effective_bg_rate,
        spatial_pattern="uniform",
        seed=EXPRESSION_SEED + 9000,
    )
    bg_x, bg_y, bg_gene_idx = bg_noise_model.generate_background_dots(
        n_genes=n_genes_c5, gene_names=gene_names_c5,
    )
    if len(bg_x) > 0:
        bg_x = bg_x / bg_frame_size * FRAME_WIDTH
        bg_y = bg_y / bg_frame_size * FRAME_HEIGHT
        bg_gene_names_c5 = [gene_names_c5[gi] for gi in bg_gene_idx]
        _, bg_cell_idx = cell_tree.query(np.column_stack([bg_x, bg_y]))
        bg_df = pd.DataFrame({
            'x': bg_x, 'y': bg_y,
            'gene': bg_gene_names_c5,
            'cell': bg_cell_idx.astype(int),
        })
        dots_noisy = pd.concat([dots_noisy, bg_df], ignore_index=True)

    print(f"     -> Final: {len(dots_noisy):,} dots")

    case_params = {
        'case': 'CASE5b_dropout_sweep',
        'noise_source': name,
        'n_genes_reference': BASE_N_GENES,
        'n_genes_panel': SWEEP_N_GENES,
        'lateral_admixture_boundary_width': MINIMAL_LAT_BW,
        'lateral_admixture_transfer_rate': MINIMAL_LAT_TR,
        'lateral_admixture_distance_decay': MINIMAL_LAT_DD,
        'z_admixture_contamination_rate': MINIMAL_Z_RATE,
        'z_admixture_neighborhood_correlation': MINIMAL_Z_CORR,
        'z_admixture_neighborhood_radius': MINIMAL_Z_RADIUS,
        'dropout_baseline_detection_rate': det_rate,
        'dropout_expression_dependence': det_expr_dep,
        'dropout_gene_variation_cv': det_gene_cv,
        'background_noise_rate': MINIMAL_BG_RATE,
        'gene_sensitivity': BASE_GENE_SENSITIVITY,
        'gene_sensitivity_variation': BASE_GENE_SENSITIVITY_VAR,
        'within_group_similarity': BASE_WITHIN_GROUP_SIMILARITY,
        'n_dots_clean': len(dots_df_c5_clean),
        'n_dots_final': len(dots_noisy),
        'geometry_seed': GEOMETRY_SEED,
        'expression_seed': EXPRESSION_SEED,
    }
    save_case(folder, tissue_c5, dots_noisy, case_params)


# ============================================================================
# SUMMARY
# ============================================================================

n_datasets = 5 + 5 + 5 + 5 + 6  # CASE1-4: 5 each, CASE5: 3+3
print("\n" + "=" * 80)
print(f"HYPERPARAMETER SWEEP COMPLETE — {n_datasets} DATASETS GENERATED")
print("=" * 80)

print(f"\nOutput location: {OUTPUT_FOLDER}/")
print(f"\nSpatial layout: {len(cells_df)} cells (shared across all cases)")
print(f"Frame size: {FRAME_WIDTH}x{FRAME_HEIGHT} um")
print(f"Cell types: {N_CELL_TYPES}")
print(f"Geometry seed: {GEOMETRY_SEED}")
print(f"Expression seed: {EXPRESSION_SEED}")
print(f"\nBase expression: hierarchical profiles with {BASE_N_GENES} genes")
print(f"  Gene categories: {dict(cat_counts)}")
print(f"  Within-group similarity: {BASE_WITHIN_GROUP_SIMILARITY}")
print(f"  HVG panel for cases 2-5: {SWEEP_N_GENES} genes")

print(f"\nCASE 1 — Gene panel size (5 datasets):")
print(f"  CASE1_1000genes, CASE1_500genes, CASE1_200genes, "
      f"CASE1_100genes, CASE1_50genes")
print(f"  Realistic noisy panel selection with housekeeping leakage")

print(f"\nCASE 2 — Reference quality (5 datasets, {SWEEP_N_GENES} HVGs):")
print(f"  CASE2_ref1_clean, CASE2_ref2_noisy, CASE2_ref3_degraded, "
      f"CASE2_ref4_poor, CASE2_ref5_terrible")
print(f"  Fixed data + baseline noise, progressively degraded scRNA-seq reference")

print(f"\nCASE 3 — Gene sensitivity (5 datasets, {SWEEP_N_GENES} HVGs):")
print(f"  CASE3_gs0.9, CASE3_gs0.7, CASE3_gs0.5, CASE3_gs0.3, CASE3_gs0.1")
print(f"  Detection efficiency, variation = sensitivity*0.5, baseline noise (realistic admixture)")

print(f"\nCASE 4 — Noise levels (5 datasets, {SWEEP_N_GENES} HVGs):")
print(f"  CASE4_noise1_baseline, CASE4_noise2_low, CASE4_noise3_moderate, "
      f"CASE4_noise4_high, CASE4_noise5_severe")
print(f"  All noise sources combined: admixture + dropout + background")

print(f"\nCASE 5 — Noise decomposition (6 datasets, {SWEEP_N_GENES} HVGs):")
print(f"  5a: CASE5a_admix_low, CASE5a_admix_mid, CASE5a_admix_high")
print(f"     (admixture sweep with minimal dropout)")
print(f"  5b: CASE5b_dropout_low, CASE5b_dropout_mid, CASE5b_dropout_high")
print(f"     (dropout sweep with minimal admixture)")

print(f"\nEach folder contains:")
print(f"  - cells.csv")
print(f"  - dots.csv")
print(f"  - cell_type_expression.csv")
print(f"  - hyperparameters.json")

# ---- Per-cell statistics summary table ----
print(f"\n{'─' * 100}")
print(f"PER-CELL STATISTICS SUMMARY")
print(f"{'─' * 100}")
print(f"{'Dataset':<30s}  {'Genes':>5s}  {'Total Dots':>12s}  "
      f"{'Tx/cell mean':>12s}  {'Tx/cell med':>11s}  "
      f"{'Genes/cell mean':>15s}  {'Genes/cell med':>14s}")
print(f"{'─' * 100}")
for s in _all_case_stats:
    print(f"{s['dataset']:<30s}  {s['n_genes']:>5d}  {s['total_dots']:>12,d}  "
          f"{s['mean_transcripts']:>12.0f}  {s['median_transcripts']:>11.0f}  "
          f"{s['mean_genes']:>15.1f}  {s['median_genes']:>14.1f}")
print(f"{'─' * 100}")
print("=" * 80)
