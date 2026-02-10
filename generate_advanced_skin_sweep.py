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

Cell types: 20
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
- (reserved) (13)
- Adipocyte (14)
- Langerhans (15)
- Macrophage (16)
- T_Cell (17)
- Mast_Cell (18)
- Sensory_Corpuscle (19)

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
OUTPUT_FOLDER = "skin_sweep_output2"

# ---------- CELL TYPES & GENES ----------
N_CELL_TYPES = 20
N_GENES = 100
GENE_SENSITIVITY = 0.5

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
ID_HAIR_UNUSED = 13  # Kept for array indexing; not assigned to any cells
ID_ADIPOCYTE = 14
ID_LANGERHANS = 15
ID_MACROPHAGE = 16
ID_T_CELL = 17
ID_MAST_CELL = 18
ID_SENSORY_CORPUSCLE = 19

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
    'Hair_Unused',               # 13 (placeholder, not assigned)
    'Adipocyte',                 # 14
    'Langerhans',                # 15
    'Macrophage',                # 16
    'T_Cell',                    # 17
    'Mast_Cell',                 # 18
    'Sensory_Corpuscle',         # 19
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
    ID_HAIR_UNUSED:            '#FF1493',  # (unused placeholder)
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
#                               Corn   Gran   Spin   Basal  Melan  FibP   FibR   Endo   Peri   Sebo   SwEc   SwAp   Hair   (rsv)  Adipo  Lang   Macro  TCell  Mast   Senso
CELL_SIZES =                    [15,   11,    10,    9,     10,    15,    16,    8,     10,    15,    10,    11,    9,     10,    25,    11,    14,    9,     11,    19]
CELL_SIZE_VARIATION =           [0.20, 0.15,  0.18,  0.20,  0.25,  0.25,  0.25,  0.15,  0.20,  0.30,  0.18,  0.20,  0.18,  0.20,  0.40,  0.22,  0.30,  0.20,  0.25,  0.30]
CELL_ANISOTROPY =               [0.20, 0.50,  0.70,  0.75,  0.70,  0.35,  0.30,  0.25,  0.60,  0.85,  0.80,  0.80,  0.75,  0.75,  0.90,  0.70,  0.75,  0.85,  0.80,  0.70]
#                                ^^^^  ^^^^   ^^^^   ^^^^         ^^^^   ^^^^   ^^^^
#                                FLAT  MID    ROUND  ROUND        SPINDLE SPINDLE FLAT
#                                SCALES      EPIDERMIS            DERMIS         VESSEL
CELL_ANISO_VARIATION =          [0.08, 0.10,  0.10,  0.12,  0.15,  0.15,  0.15,  0.10,  0.15,  0.10,  0.12,  0.12,  0.15,  0.15,  0.10,  0.15,  0.15,  0.10,  0.12,  0.15]
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
print("   ✓ Cell morphology defined for 20 cell types")


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
# HYPERPARAMETER SWEEP — 20 DATASETS (4 CASES × 5 SETTINGS)
# ============================================================================

# Generate cells_df once (spatial layout is shared across ALL cases)
cells_df = fov.make_pandas_df()
print(f"\n   Total cells for sweep: {len(cells_df)}")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_case(folder_name, tissue_obj, dots_df, case_params):
    """Save cells.csv, dots.csv, cell_type_expression.csv, and hyperparameters.json."""
    output_dir = f"{OUTPUT_FOLDER}/{folder_name}"
    os.makedirs(output_dir, exist_ok=True)

    cells_df.to_csv(f"{output_dir}/cells.csv", index=False)
    dots_df.to_csv(f"{output_dir}/dots.csv", index=False)

    tissue_df = tissue_obj.make_pandas_df()
    tissue_df.to_csv(f"{output_dir}/cell_type_expression.csv")

    with open(f"{output_dir}/hyperparameters.json", 'w') as f:
        json.dump(case_params, f, indent=2)

    print(f"   -> {folder_name}: {len(dots_df):,} dots, "
          f"{tissue_obj.n_genes} genes")


# ============================================================================
# CASE 1: NUMBER OF GENES
# Fixed: TF(0.7±0.5, 0.2±0.1), sensitivity=0.5±0.5, conc=0.7,
#        expected=200±100.  Generate 1000 genes, subset by HVG.
# ============================================================================

print("\n" + "=" * 80)
print("CASE 1: NUMBER OF GENES")
print("=" * 80)

np.random.seed(EXPRESSION_SEED)

tissue_case1 = TissueCellTypes()
tissue_case1.generate_types_and_markers(
    n_genes=1000,
    n_cell_types=N_CELL_TYPES,
    expected_level=200.0,
    expected_std_level=100.0,
    concentration=0.7
)
tissue_case1._cell_type_names = CELL_TYPE_NAMES
print(f"   Reference: 1000 genes, concentration=0.7, expected=200±100")
affine_tf_c1 = AffineNonNegTransfer(
    scales=0.7, scales_std=0.5,
    offsets=0.2, offsets_std=0.1
)
hybiss_c1 = HybISS_Setup(
    tissue_case1,
    genes_sensitivities=0.5,
    genes_sensitivities_variation=0.5,
    transfer_function=affine_tf_c1
)
hybiss_c1.observe_dots(fov)
dots_df_full = hybiss_c1.make_pandas_df()
print(f"   Full 1000-gene dots: {len(dots_df_full):,}")

# Compute HVG scores (variance across cell types)
expr_matrix_c1 = tissue_case1.gene_expression_by_type  # (1000, 20)
hvg_scores = np.var(expr_matrix_c1, axis=1)
hvg_ranking = np.argsort(hvg_scores)[::-1]  # descending by variance
gene_names_all = list(tissue_case1.gene_names)

for n_genes_target in [1000, 500, 200, 100, 50]:
    folder = f"CASE1_{n_genes_target}genes"

    selected_indices = hvg_ranking[:n_genes_target]
    selected_gene_names = [gene_names_all[i] for i in selected_indices]

    # Subset expression matrix
    tissue_sub = TissueCellTypes(
        gene_expression_by_type=expr_matrix_c1[selected_indices, :]
    )
    tissue_sub._cell_type_names = CELL_TYPE_NAMES
    tissue_sub._gene_names = selected_gene_names

    # Subset dots to selected genes only
    selected_set = set(selected_gene_names)
    dots_subset = dots_df_full[dots_df_full['gene'].isin(selected_set)].copy()

    case_params = {
        'case': 'CASE1_number_of_genes',
        'n_genes': n_genes_target,
        'gene_selection': 'top_hvg_by_variance',
        'gene_sensitivity': 0.5,
        'gene_sensitivity_variation': 0.5,
        'concentration': 0.7,
        'expected_expression_level': 200.0,
        'expected_expression_std': 100.0,
        'transfer_function': 'AffineNonNegTransfer',
        'transfer_scale': 0.7,
        'transfer_scale_std': 0.5,
        'transfer_offset': 0.2,
        'transfer_offset_std': 0.1,
        'relative_rna_concentration': RELATIVE_RNA_CONC,
        'rna_concentration_variation': RNA_CONC_VARIATION,
        'geometry_seed': GEOMETRY_SEED,
        'expression_seed': EXPRESSION_SEED,
    }

    save_case(folder, tissue_sub, dots_subset, case_params)


# ============================================================================
# CASE 2: EFFECT OF CONCENTRATION
# Fixed: N_genes=100, TF(1.0±0.3, 0.15±0.08), sensitivity=0.5±0.25,
#        expected=40±20.  Regenerate reference for each concentration.
# ============================================================================

print("\n" + "=" * 80)
print("CASE 2: EFFECT OF CONCENTRATION")
print("=" * 80)

for conc in [0.90, 0.70, 0.50, 0.30, 0.10]:
    folder = f"CASE2_conc{conc:.2f}"
    print(f"\n   {folder} (concentration={conc})...")

    np.random.seed(EXPRESSION_SEED)

    tissue_c2 = TissueCellTypes()
    tissue_c2.generate_types_and_markers(
        n_genes=100,
        n_cell_types=N_CELL_TYPES,
        expected_level=40.0,
        expected_std_level=20.0,
        concentration=conc
    )
    tissue_c2._cell_type_names = CELL_TYPE_NAMES

    affine_tf_c2 = AffineNonNegTransfer(
        scales=1.0, scales_std=0.3,
        offsets=0.15, offsets_std=0.08
    )
    hybiss_c2 = HybISS_Setup(
        tissue_c2,
        genes_sensitivities=0.5,
        genes_sensitivities_variation=0.25,
        transfer_function=affine_tf_c2
    )
    hybiss_c2.observe_dots(fov)
    dots_df_c2 = hybiss_c2.make_pandas_df()

    case_params = {
        'case': 'CASE2_concentration',
        'n_genes': 100,
        'gene_sensitivity': 0.5,
        'gene_sensitivity_variation': 0.25,
        'concentration': conc,
        'expected_expression_level': 30.0,
        'expected_expression_std': 10.0,
        'transfer_function': 'AffineNonNegTransfer',
        'transfer_scale': 1.0,
        'transfer_scale_std': 0.3,
        'transfer_offset': 0.15,
        'transfer_offset_std': 0.08,
        'relative_rna_concentration': RELATIVE_RNA_CONC,
        'rna_concentration_variation': RNA_CONC_VARIATION,
        'geometry_seed': GEOMETRY_SEED,
        'expression_seed': EXPRESSION_SEED,
    }

    save_case(folder, tissue_c2, dots_df_c2, case_params)


# ============================================================================
# CASE 3: EFFECT OF GENE SENSITIVITY
# Fixed: N_genes=100, conc=0.75, TF(1.0±0.3, 0.15±0.08), expected=40±20.
#        Same reference, vary sensitivity (variation = sensitivity / 2).
# ============================================================================

print("\n" + "=" * 80)
print("CASE 3: EFFECT OF GENE SENSITIVITY")
print("=" * 80)

# Generate reference ONCE
np.random.seed(EXPRESSION_SEED)

tissue_c3 = TissueCellTypes()
tissue_c3.generate_types_and_markers(
    n_genes=100,
    n_cell_types=N_CELL_TYPES,
    expected_level=40.0,
    expected_std_level=20.0,
    concentration=0.75
)
tissue_c3._cell_type_names = CELL_TYPE_NAMES
print(f"   Reference: 100 genes, concentration=0.75, expected=50±25")

for gs in [0.9, 0.6, 0.3, 0.1, 0.05]:
    gs_var = gs / 2.0
    folder = f"CASE3_gs{gs}"
    print(f"\n   {folder} (sensitivity={gs}, variation={gs_var})...")

    # Reset seed so TF and sensitivity sampling are reproducible per sub-case
    np.random.seed(EXPRESSION_SEED + 1000)

    affine_tf_c3 = AffineNonNegTransfer(
        scales=1.0, scales_std=0.3,
        offsets=0.15, offsets_std=0.08
    )
    hybiss_c3 = HybISS_Setup(
        tissue_c3,
        genes_sensitivities=gs,
        genes_sensitivities_variation=gs_var,
        transfer_function=affine_tf_c3
    )
    hybiss_c3.observe_dots(fov)
    dots_df_c3 = hybiss_c3.make_pandas_df()

    case_params = {
        'case': 'CASE3_gene_sensitivity',
        'n_genes': 100,
        'gene_sensitivity': gs,
        'gene_sensitivity_variation': gs_var,
        'concentration': 0.75,
        'expected_expression_level': 40.0,
        'expected_expression_std': 20.0,
        'transfer_function': 'AffineNonNegTransfer',
        'transfer_scale': 1.0,
        'transfer_scale_std': 0.3,
        'transfer_offset': 0.15,
        'transfer_offset_std': 0.08,
        'relative_rna_concentration': RELATIVE_RNA_CONC,
        'rna_concentration_variation': RNA_CONC_VARIATION,
        'geometry_seed': GEOMETRY_SEED,
        'expression_seed': EXPRESSION_SEED,
    }

    save_case(folder, tissue_c3, dots_df_c3, case_params)


# ============================================================================
# CASE 4: EFFECT OF NOISE LEVELS
# Fixed: N_genes=100, conc=0.7, sensitivity=0.5±0.25, expected=40±20.
#        Same reference, progressively increase all noise sources:
#        - Lateral 2D admixture (segmentation boundary errors)
#        - Z-axis admixture (out-of-plane contamination)
#        - Transcript dropout (detection failure)
#        - Background noise (false positive transcripts)
# ============================================================================

print("\n" + "=" * 80)
print("CASE 4: EFFECT OF NOISE LEVELS")
print("=" * 80)

# Generate clean reference tissue and dots for Case 4
np.random.seed(EXPRESSION_SEED)

tissue_c4 = TissueCellTypes()
tissue_c4.generate_types_and_markers(
    n_genes=100,
    n_cell_types=N_CELL_TYPES,
    expected_level=40.0,
    expected_std_level=20.0,
    concentration=0.7
)
tissue_c4._cell_type_names = CELL_TYPE_NAMES
print(f"   Reference: 100 genes, concentration=0.7, expected=40±20")

np.random.seed(EXPRESSION_SEED + 2000)

affine_tf_c4 = AffineNonNegTransfer(
    scales=TRANSFER_SCALES, scales_std=TRANSFER_SCALES_STD,
    offsets=TRANSFER_OFFSETS, offsets_std=TRANSFER_OFFSETS_STD
)
hybiss_c4 = HybISS_Setup(
    tissue_c4,
    genes_sensitivities=0.5,
    genes_sensitivities_variation=0.25,
    transfer_function=affine_tf_c4
)
hybiss_c4.observe_dots(fov)
dots_df_c4_clean = hybiss_c4.make_pandas_df()
print(f"   Clean dots: {len(dots_df_c4_clean):,}")

# Precompute FOV properties for admixture
cell_centroids = fov.cell_centroids
cell_types = fov.class_instance
cell_radii = fov.cell_major_axis / 2
n_cells = fov.n_cells
gene_names_c4 = list(tissue_c4.gene_names)
n_genes_c4 = len(gene_names_c4)

# Build KDTree for nearest-cell assignment of background dots
cell_tree = cKDTree(cell_centroids)

# 5 noise levels: minimal → severe
# Each level increases ALL noise sources together
noise_configs = [
    # (name, lateral_bw, lateral_tr, lateral_dd,
    #  z_rate, z_corr, z_radius,
    #  dropout_rate, dropout_expr_dep, dropout_gene_cv,
    #  bg_rate)
    ('noise1_minimal', 3.0, 0.05, 0.5,
     0.02, 0.9, 50.0,
     0.98, 0.05, 0.05,
     0.0002),
    ('noise2_low', 6.0, 0.25, 0.3,
     0.10, 0.75, 60.0,
     0.85, 0.20, 0.15,
     0.001),
    ('noise3_moderate', 8.0, 0.35, 0.3,
     0.15, 0.70, 60.0,
     0.75, 0.30, 0.20,
     0.002),
    ('noise4_high', 10.0, 0.45, 0.2,
     0.20, 0.60, 70.0,
     0.65, 0.40, 0.30,
     0.004),
    ('noise5_severe', 12.0, 0.55, 0.15,
     0.30, 0.50, 80.0,
     0.50, 0.50, 0.40,
     0.008),
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
    # Adjust background_rate for the square frame vs actual rectangular area
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
        # Remap from [0, frame_size] square to [0, FRAME_WIDTH] x [0, FRAME_HEIGHT]
        bg_x = bg_x / bg_frame_size * FRAME_WIDTH
        bg_y = bg_y / bg_frame_size * FRAME_HEIGHT
        bg_gene_names = [gene_names_c4[gi] for gi in bg_gene_idx]
        # Assign background dots to nearest cell
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
        'n_genes': 100,
        'gene_sensitivity': 0.5,
        'gene_sensitivity_variation': 0.25,
        'concentration': 0.7,
        'expected_expression_level': 40.0,
        'expected_expression_std': 20.0,
        'transfer_function': 'AffineNonNegTransfer',
        'transfer_scale': TRANSFER_SCALES,
        'transfer_scale_std': TRANSFER_SCALES_STD,
        'transfer_offset': TRANSFER_OFFSETS,
        'transfer_offset_std': TRANSFER_OFFSETS_STD,
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
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("HYPERPARAMETER SWEEP COMPLETE — 20 DATASETS GENERATED")
print("=" * 80)

print(f"\nOutput location: {OUTPUT_FOLDER}/")
print(f"\nSpatial layout: {len(cells_df)} cells (shared across all cases)")
print(f"Frame size: {FRAME_WIDTH}x{FRAME_HEIGHT} um")
print(f"Cell types: {N_CELL_TYPES}")
print(f"Geometry seed: {GEOMETRY_SEED}")
print(f"Expression seed: {EXPRESSION_SEED}")

print(f"\nCASE 1 — Number of genes (5 datasets):")
print(f"  CASE1_1000genes, CASE1_500genes, CASE1_200genes, "
      f"CASE1_100genes, CASE1_50genes")
print(f"  Reference: 1000 genes, conc=0.75, expected=500±250")
print(f"  Gene selection: top HVG by variance across cell types")

print(f"\nCASE 2 — Concentration (5 datasets):")
print(f"  CASE2_conc0.95, CASE2_conc0.80, CASE2_conc0.65, "
      f"CASE2_conc0.50, CASE2_conc0.35")
print(f"  N_genes=100, expected=50±25, reference regenerated per concentration")

print(f"\nCASE 3 — Gene sensitivity (5 datasets):")
print(f"  CASE3_gs0.9, CASE3_gs0.6, CASE3_gs0.3, CASE3_gs0.1, CASE3_gs0.05")
print(f"  N_genes=100, conc=0.75, same reference, variation = sensitivity/2")

print(f"\nCASE 4 — Noise levels (5 datasets):")
print(f"  CASE4_noise1_minimal, CASE4_noise2_low, CASE4_noise3_high, "
      f"CASE4_noise4_severe, CASE4_noise5_extreme")
print(f"  N_genes=100, conc=0.7, expected=40±20, sensitivity=0.5±0.25")
print(f"  Noise: lateral admixture + z-axis admixture + dropout + background")

print(f"\nEach folder contains:")
print(f"  - cells.csv")
print(f"  - dots.csv")
print(f"  - cell_type_expression.csv")
print(f"  - hyperparameters.json")
print("=" * 80)
