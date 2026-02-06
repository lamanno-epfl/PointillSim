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
import numpy as np
import pandas as pd
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
OUTPUT_FOLDER = "advanced_skin_output"

# ---------- CELL TYPES & GENES ----------
N_CELL_TYPES = 20
N_GENES = 150
GENE_SENSITIVITY = 0.85

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
EXPECTED_EXPRESSION_LEVEL = 100.0
EXPECTED_EXPRESSION_STD = 15.0
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
TRANSFER_SCALES = 0.75
TRANSFER_SCALES_STD = 0.25
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
# SWITCH TO EXPRESSION SEED
# ============================================================================
np.random.seed(EXPRESSION_SEED)

print("\n7. Setting up tissue gene expression (expression seed)...")
tissue = TissueCellTypes()
tissue.generate_types_and_markers(
    n_genes=N_GENES,
    n_cell_types=N_CELL_TYPES,
    expected_level=EXPECTED_EXPRESSION_LEVEL,
    expected_std_level=EXPECTED_EXPRESSION_STD,
    concentration=EXPRESSION_CONCENTRATION
)
tissue._cell_type_names = CELL_TYPE_NAMES
print(f"   ✓ Created tissue with {N_GENES} genes and {N_CELL_TYPES} cell types")


print("\n8. Simulating transcript observations...")

affine_tf = AffineNonNegTransfer(
    scales=TRANSFER_SCALES,
    scales_std=TRANSFER_SCALES_STD,
    offsets=TRANSFER_OFFSETS,
    offsets_std=TRANSFER_OFFSETS_STD
)

hybiss = HybISS_Setup(
    tissue,
    genes_sensitivities=GENE_SENSITIVITY,
    genes_sensitivities_variation=0.2,
    transfer_function=affine_tf
)

hybiss.observe_dots(fov)
dots_df = hybiss.make_pandas_df()
print(f"   ✓ Generated {len(dots_df):,} transcript dots")


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n9. Saving outputs...")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cells_df = fov.make_pandas_df()
cells_df.to_csv(f"{OUTPUT_FOLDER}/cells.csv", index=False)
print(f"   ✓ Saved cells.csv ({len(cells_df)} cells)")

dots_df.to_csv(f"{OUTPUT_FOLDER}/dots.csv", index=False)
print(f"   ✓ Saved dots.csv ({len(dots_df)} transcripts)")

tissue_df = tissue.make_pandas_df()
tissue_df.to_csv(f"{OUTPUT_FOLDER}/cell_types_expression.csv")
print(f"   ✓ Saved cell_types_expression.csv")

try:
    adata = fov.to_anndata()
    adata.write_h5ad(f"{OUTPUT_FOLDER}/data.h5ad")
    print(f"   ✓ Saved data.h5ad ({adata.n_obs} cells × {adata.n_vars} genes)")
except ImportError:
    print("   ⚠ AnnData not installed — skipping h5ad export")


# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n10. Creating visualizations...")

# === VISUALIZATION 1: Ellipse rendering (white background) ===
print("\n   Creating ellipse-based visualization (white background)...")
fig1, ax1 = create_high_quality_visualization(
    cells_df=cells_df,
    output_path=f"{OUTPUT_FOLDER}/skin_tissue_ellipses_white.png",
    figsize=(22, 28),
    dpi=300,
    render_mode='ellipse',
    cell_size_scale=1.0,
    cell_alpha=0.90,
    cell_edge_color='black',
    cell_edge_width=0.3,
    background_color='white',
    show_grid=False,
    show_layer_lines=True,
    layer_line_color='darkblue',
    layer_line_width=2.5,
    layer_line_alpha=0.6,
    show_layer_labels=True,
    layer_label_fontsize=13,
    show_legend=True,
    legend_fontsize=11,
    legend_ncol=2,
    title="Skin Tissue - Realistic Cell Morphology (Ellipse Mode)",
    title_fontsize=22
)
plt.close(fig1)

# === VISUALIZATION 2: Ellipse rendering (black background) ===
print("\n   Creating ellipse-based visualization (black background)...")
fig2, ax2 = create_high_quality_visualization(
    cells_df=cells_df,
    output_path=f"{OUTPUT_FOLDER}/skin_tissue_ellipses_black.png",
    figsize=(22, 28),
    dpi=300,
    render_mode='ellipse',
    cell_size_scale=1.0,
    cell_alpha=0.95,
    cell_edge_color=None,
    cell_edge_width=0,
    background_color='black',
    show_grid=False,
    show_layer_lines=True,
    layer_line_color='cyan',
    layer_line_width=2.5,
    layer_line_alpha=0.8,
    show_layer_labels=True,
    layer_label_fontsize=13,
    show_legend=True,
    legend_fontsize=11,
    legend_ncol=2,
    title="Skin Tissue - Realistic Cell Morphology (Ellipse Mode)",
    title_fontsize=22
)
plt.close(fig2)

# === VISUALIZATION 3: Scatter mode (for comparison) ===
print("\n   Creating scatter-based visualization...")
fig3, ax3 = create_high_quality_visualization(
    cells_df=cells_df,
    output_path=f"{OUTPUT_FOLDER}/skin_tissue_scatter.png",
    figsize=(20, 26),
    dpi=300,
    render_mode='scatter',
    cell_size_scale=8.0,
    cell_alpha=0.85,
    background_color='white',
    show_layer_lines=True,
    show_layer_labels=True,
    show_legend=True,
    title="Skin Tissue - Cell Type Distribution (Scatter Mode)",
    title_fontsize=22
)
plt.close(fig3)

# === VISUALIZATION 4: Dual-panel (Cells + Transcripts) ===
print("\n   Creating dual-panel visualization (cells + transcripts)...")
fig4, axes = plt.subplots(1, 2, figsize=(32, 20), dpi=300)

# Panel 1: Cells (scatter mode for speed)
ax_cells = axes[0]
for ct in range(N_CELL_TYPES):
    mask = cells_df['Class ID'] == ct
    if mask.sum() > 0:
        ax_cells.scatter(
            cells_df.loc[mask, 'X'], cells_df.loc[mask, 'Y'],
            c=CELL_TYPE_COLORS[ct], s=5, alpha=0.85,
            label=CELL_TYPE_NAMES[ct], rasterized=True
        )

ax_cells.set_xlim(0, FRAME_WIDTH)
ax_cells.set_ylim(0, FRAME_HEIGHT)
ax_cells.set_aspect('equal')
ax_cells.invert_yaxis()
ax_cells.set_title('Cell Type Distribution', fontsize=20, fontweight='bold', pad=15)
ax_cells.set_xlabel('X (µm)', fontsize=16)
ax_cells.set_ylabel('Depth (µm)', fontsize=16)
ax_cells.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, ncol=1)
ax_cells.grid(True, alpha=0.3)

# Add layer lines
for depth in [STRATUM_CORNEUM_DEPTH, STRATUM_GRANULOSUM_DEPTH, STRATUM_SPINOSUM_DEPTH,
              STRATUM_BASALE_DEPTH, DEJ_DEPTH, PAPILLARY_DERMIS_DEPTH, RETICULAR_DERMIS_DEPTH]:
    ax_cells.axhline(y=depth, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
ax_cells.axhline(y=DEJ_DEPTH, color='cyan', linestyle='--', alpha=0.7, linewidth=2.5)

# Panel 2: Transcripts (sampled)
ax_dots = axes[1]
dots_sample = dots_df.iloc[::20]  # Sample for visibility

scatter = ax_dots.scatter(
    dots_sample['x'], dots_sample['y'],
    c=dots_sample['gene'].astype('category').cat.codes,
    s=0.8, alpha=0.5, cmap='tab20', rasterized=True
)

ax_dots.set_xlim(0, FRAME_WIDTH)
ax_dots.set_ylim(0, FRAME_HEIGHT)
ax_dots.set_aspect('equal')
ax_dots.invert_yaxis()
ax_dots.set_title(f'Transcript Dots (sampled 1/20)', fontsize=20, fontweight='bold', pad=15)
ax_dots.set_xlabel('X (µm)', fontsize=16)
ax_dots.set_ylabel('Depth (µm)', fontsize=16)
ax_dots.grid(True, alpha=0.3)

# Add layer lines
ax_dots.axhline(y=DEJ_DEPTH, color='cyan', linestyle='--', alpha=0.7, linewidth=2, label='DEJ')
ax_dots.axhline(y=PAPILLARY_DERMIS_DEPTH, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Papillary/Reticular')
ax_dots.axhline(y=RETICULAR_DERMIS_DEPTH, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Reticular/Hypodermis')
ax_dots.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/skin_tissue_dual_panel.png", dpi=300, bbox_inches='tight')
print(f"   ✓ Saved skin_tissue_dual_panel.png")
plt.close(fig4)

# === VISUALIZATION 5: Zoomed epidermis (ellipse mode) ===
print("\n   Creating zoomed epidermis view (ellipse mode)...")
fig5, ax5 = create_high_quality_visualization(
    cells_df=cells_df,
    output_path=f"{OUTPUT_FOLDER}/skin_epidermis_zoom_ellipses.png",
    figsize=(24, 12),
    dpi=350,
    render_mode='ellipse',
    cell_size_scale=1.2,
    cell_alpha=0.90,
    cell_edge_color='black',
    cell_edge_width=0.5,
    background_color='white',
    show_layer_lines=True,
    layer_line_color='darkblue',
    layer_line_width=3,
    show_layer_labels=True,
    layer_label_fontsize=14,
    show_legend=True,
    legend_fontsize=12,
    legend_ncol=3,
    title="Epidermis - Stratified Layers (Zoomed View)",
    title_fontsize=24,
    xlim=(150, 650),  # Zoom to central region
    ylim=(0, 150)     # Only epidermis and upper DEJ
)
plt.close(fig5)

# === VISUALIZATION 6: Zoomed dermis structures (ellipse mode) ===
print("\n   Creating zoomed dermis view showing structures (ellipse mode)...")
fig6, ax6 = create_high_quality_visualization(
    cells_df=cells_df,
    output_path=f"{OUTPUT_FOLDER}/skin_dermis_zoom_ellipses.png",
    figsize=(24, 18),
    dpi=350,
    render_mode='ellipse',
    cell_size_scale=1.1,
    cell_alpha=0.92,
    cell_edge_color='darkgray',
    cell_edge_width=0.4,
    background_color='#f8f8f8',  # Light gray background
    show_layer_lines=True,
    layer_line_color='navy',
    layer_line_width=2.5,
    show_layer_labels=True,
    layer_label_fontsize=13,
    show_legend=True,
    legend_fontsize=11,
    legend_ncol=2,
    title="Dermis - Hair Follicles, Glands, and Vessels (Zoomed View)",
    title_fontsize=22,
    xlim=(100, 700),   # Central region
    ylim=(150, 700)    # Papillary + upper reticular dermis
)
plt.close(fig6)

print("\n   ✓ All high-quality visualizations created!")
print(f"\n   Summary of visualizations:")
print(f"   - skin_tissue_ellipses_white.png: Full tissue, ellipse mode, white bg")
print(f"   - skin_tissue_ellipses_black.png: Full tissue, ellipse mode, black bg")
print(f"   - skin_tissue_scatter.png: Full tissue, scatter mode")
print(f"   - skin_tissue_dual_panel.png: Cells + transcripts side-by-side")
print(f"   - skin_epidermis_zoom_ellipses.png: Zoomed epidermis with layers")
print(f"   - skin_dermis_zoom_ellipses.png: Zoomed dermis showing structures")

plt.close('all')


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✓ ADVANCED REALISTIC SKIN TISSUE SIMULATION COMPLETE!")
print("=" * 80)
print(f"\nOutput location: {OUTPUT_FOLDER}/")
print(f"\n📁 Data Files:")
print(f"  - cells.csv: {len(cells_df)} cells with ground truth (X, Y, Class ID, morphology)")
print(f"  - dots.csv: {len(dots_df):,} transcript positions (x, y, gene)")
print(f"  - cell_types_expression.csv: Gene expression matrix ({N_GENES} genes × {N_CELL_TYPES} types)")
print(f"  - data.h5ad: AnnData format (if anndata installed)")
print(f"\n🎨 Visualizations:")
print(f"  1. skin_tissue_ellipses_white.png - Full tissue, realistic cell shapes (white bg)")
print(f"  2. skin_tissue_ellipses_black.png - Full tissue, realistic cell shapes (black bg)")
print(f"  3. skin_tissue_scatter.png - Full tissue, scatter plot mode")
print(f"  4. skin_tissue_dual_panel.png - Cells + transcripts side-by-side")
print(f"  5. skin_epidermis_zoom_ellipses.png - Zoomed epidermis showing stratification")
print(f"  6. skin_dermis_zoom_ellipses.png - Zoomed dermis showing embedded structures")
print(f"\n📊 Tissue Architecture:")
print(f"  - Total regions: 8 (Corneum → Granulosum → Spinosum → Basale → DEJ → Papillary → Reticular → Hypodermis)")
print(f"  - Cell types: {N_CELL_TYPES}")
print(f"  - Genes: {N_GENES}")
print(f"  - Frame size: {FRAME_WIDTH}×{FRAME_HEIGHT} µm")
print(f"\n🔬 Embedded Structures:")
print(f"  - Hair follicles: {N_HAIR_FOLLICLES} (with sebaceous glands)")
print(f"  - Eccrine sweat glands: {N_ECCRINE_GLANDS}")
print(f"  - Apocrine sweat glands: {N_APOCRINE_GLANDS}")
print(f"  - Melanin units: {N_MELANIN_UNITS} (melanocyte + basal keratinocyte halos at DEJ)")
print(f"  - Papillary capillaries: {N_PAPILLARY_CAPILLARIES} (vascular detail - 2× increase)")
print(f"  - Dermal vessels: {N_DERMAL_VESSELS} (vascular detail - 2× increase, 4 branch levels)")
print(f"  - Papillary collagen: {N_COLLAGEN_BUNDLES_PAPILLARY} (ECM density boost - 3× increase)")
print(f"  - Reticular collagen: {N_COLLAGEN_BUNDLES_RETICULAR} (ECM density boost - 3× increase)")
print(f"  - Papillary elastin: {N_ELASTIN_FIBERS_PAPILLARY} (ECM density boost - NEW)")
print(f"  - Reticular elastin: {N_ELASTIN_FIBERS_RETICULAR} (ECM density boost - NEW)")
print(f"  - Pacinian corpuscles: {N_PACINIAN_CORPUSCLES}")
print(f"  - Perivascular immune: {N_PERIVASCULAR_IMMUNE_CLUSTERS} (minimal increase)")
print(f"  - Periappendage immune: {N_PERIAPPENDAGE_IMMUNE_CLUSTERS} (minimal increase)")
print(f"\n🧬 Reproducibility:")
print(f"  - Geometry seed: {GEOMETRY_SEED} (cell placement — fixed across sims)")
print(f"  - Expression seed: {EXPRESSION_SEED} (gene profiles — vary across sims)")
print(f"\n💡 Visualization Modes:")
print(f"  - Ellipse mode: Realistic cell shapes using major/minor axes and rotation angle")
print(f"  - Scatter mode: Fast rendering with circular points")
print(f"  - Both modes support: white/black backgrounds, layer annotations, zoom regions")
print("=" * 80)
