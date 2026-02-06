#!/usr/bin/env python3
"""
Generate Region-Based Hierarchical Tissue with PointillSim - CLONAL HETEROGENEITY

Uses TissueSlice architecture with organic polygon boundaries and
distance-based cell type rules for continuous, dense, intermixed tissue masses.

Architecture (18 Regions with Clonal Expansion, Blue Satellites & Lobule Cluster):
- Region 1 (Left): Organic circular mass with ASYMMETRICAL rainbow gradient (Priority 1 - HIGHEST DOMINANCE)
- Region 2 (Right): Buffered dendritic skeleton with purple-to-edge gradient (Priority 2 - SECONDARY DOMINANCE)
- Region 3 (Background): Sparse stromal interface (Priority 0)
- Region 4 (Top-Right): Satellite lobule with rainbow gradient + voids (Priority 3)
- Region 5 (Top-Left): Small satellite dendrite with purple gradient (Priority 3)
- Region 6 (Invasive Root): Deep integration from right mass (blend_band=60µm) (Priority 3)
- Region 7 (Hybrid Nodule): Mixed morphology at bottom-left (circle + branches) (Priority 3)
- Region 8 (Hybrid Clone - Top-Left): Clonal infiltration of left mass (Priority 3)
- Region 9 (Dendritic Connector): Aligned with right mass starting point (Priority 3)
- Region 10 (Hybrid Clone - Central): Deep clonal infiltration of left mass (Priority 3)
- Region 11 (Blue Satellite West-1): Blue/cyan satellite on west side of left mass (Priority 3)
- Region 12 (Blue Satellite West-2): Blue/cyan satellite on west side of left mass (Priority 3)
- Region 13 (Blue Satellite West-3): Blue/cyan satellite on west side of left mass (Priority 3)
- Region 14 (Lobule Northeast): Small lobule cluster near right mass with 3 voids (Priority 3)
- Region 15 (Lobule East): Small lobule cluster near right mass with 2 voids (Priority 3)
- Region 16 (Lobule Southeast): Small lobule cluster near right mass with 5 voids (Priority 3)
- Region 17 (Lobule Far East): Rightmost small lobule with 4 voids (Priority 3)
- Region 18 (Connector East): Dendritic connector on rightmost lobule (Priority 3)

Output: 2000x1500 µm tissue slice

Cell types: 10 (split from original 6: Core→Blue+Cyan, Green→Dark+Light, Red→Red+Deep, Purple→Dark+Light)
Genes: 100

SEED STRATEGY:
- GEOMETRY_SEED: Controls polygon shapes, cell placement, cell type assignment, morphology.
  Fixed across simulations to keep spatial layout identical.
- EXPRESSION_SEED: Controls gene expression profiles and transcript observations.
  Vary this to change molecular readouts while keeping spatial layout.
"""

import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, LineString, box, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

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
    ClusterElement,
    GlandularUnit,
)


# ============================================================================
# CONTROLLABLE SIMULATION PARAMETERS
# All tunable knobs are here in UPPER_CASE. Modify freely.
# ============================================================================

# ---------- RANDOM SEEDS ----------
GEOMETRY_SEED = 42            # Cell placement, polygon shapes, morphology (FIXED across sims)
EXPRESSION_SEED = 123         # Gene profiles, transcript noise (VARY across sims)

# ---------- FRAME & OUTPUT ----------
FRAME_WIDTH = 2000            # microns
FRAME_HEIGHT = 1500           # microns
OUTPUT_FOLDER = "region_based_tissue_output"

# ---------- CELL TYPES & GENES ----------
N_CELL_TYPES = 10
N_GENES = 100
GENE_SENSITIVITY = 0.25

# ---------- CELL TYPE IDS ----------
ID_CORE_BLUE = 0
ID_CORE_CYAN = 1
ID_GREEN_DARK = 2
ID_GREEN_LIGHT = 3
ID_YELLOW = 4
ID_ORANGE = 5
ID_RED = 6
ID_RED_DEEP = 7
ID_PURPLE_DARK = 8
ID_PURPLE_LIGHT = 9

# ---------- CELL TYPE NAMES ----------
CELL_TYPE_NAMES = [
    'Core_Blue',          # 0
    'Core_Cyan',          # 1
    'Zone_GreenDark',     # 2
    'Zone_GreenLight',    # 3
    'Zone_Yellow',        # 4
    'Zone_Orange',        # 5
    'Outer_Red',          # 6
    'Outer_RedDeep',      # 7
    'Branch_PurpleDark',  # 8
    'Branch_PurpleLight', # 9
]

# ---------- CELL TYPE COLORS (visualisation) ----------
CELL_TYPE_COLORS = {
    ID_CORE_BLUE:    '#0000FF',
    ID_CORE_CYAN:    '#00CED1',
    ID_GREEN_DARK:   '#228B22',
    ID_GREEN_LIGHT:  '#32CD32',
    ID_YELLOW:       '#FFD700',
    ID_ORANGE:       '#FF8C00',
    ID_RED:          '#DC143C',
    ID_RED_DEEP:     '#8B0000',
    ID_PURPLE_DARK:  '#6A0DAD',
    ID_PURPLE_LIGHT: '#9370DB',
}

# ---------- CELL TYPE SPLIT PROPORTIONS ----------
# When a parent type is split into two sub-types, these control the ratio.
CORE_BLUE_FRACTION = 0.55       # Blue vs Cyan in core regions
GREEN_DARK_FRACTION = 0.50      # Dark vs Light in green regions
RED_DEEP_FRACTION = 0.40        # RedDeep fraction (Red = 1 - this)
PURPLE_DARK_FRACTION = 0.55     # Dark vs Light in purple regions

# ---------- GENE EXPRESSION PARAMETERS ----------
EXPECTED_EXPRESSION_LEVEL = 30.0
EXPECTED_EXPRESSION_STD = 7.0
EXPRESSION_CONCENTRATION = 0.7

# ---------- CELL SPACING ----------
FOREGROUND_CELL_SPACING = 8     # Dense tumor regions (µm)
BACKGROUND_CELL_SPACING = 25    # Sparse stroma (µm) — increased to reduce background cells
VOID_CELL_SPACING = 50          # Sparse necrotic debris (µm)
GLAND_CELL_SPACING = 6          # Glandular units (µm)
CLUSTER_CELL_SPACING = 6        # Cluster elements (µm)
VACUOLE_CELL_SPACING = 6        # Vacuolated structures (µm)

# ---------- CELL MORPHOLOGY ----------
#                       BlueCore CyanCore GrnDk  GrnLt  Yel    Org    Red    RedDp  PurDk  PurLt
CELL_SIZES =            [8,      8,       9,     9,     9,     9,     8,     8,     10,    10]
CELL_SIZE_VARIATION =   [0.20,   0.20,    0.20,  0.20,  0.20,  0.20,  0.20,  0.20,  0.25,  0.25]
CELL_ANISOTROPY =       [0.85,   0.85,    0.80,  0.80,  0.80,  0.80,  0.90,  0.90,  0.75,  0.75]
CELL_ANISO_VARIATION =  [0.10,   0.10,    0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.15,  0.15]
RELATIVE_RNA_CONC = 0.7
RNA_CONC_VARIATION = 0.4

# ---------- TRANSFER FUNCTION ----------
TRANSFER_SCALES = 0.8
TRANSFER_SCALES_STD = 0.3
TRANSFER_OFFSETS = 0.2
TRANSFER_OFFSETS_STD = 0.1

# ---------- BACKGROUND COMPOSITION ----------
BACKGROUND_TYPES = [ID_GREEN_DARK, ID_GREEN_LIGHT, ID_YELLOW]
BACKGROUND_PROPORTIONS = [0.25, 0.25, 0.50]

# ---------- HYBRID CLONE PROPORTIONS (Regions 7, 8, 10) ----------
HYBRID_CORE_PROP = 0.40
HYBRID_PURPLE_PROP = 0.40
HYBRID_GREEN_PROP = 0.20

# ---------- REGION 1: LEFT MASS (Cribriform / Cystic) ----------
R1_CENTER_X = 600
R1_RADIUS = 300
R1_NOISE_STRENGTH = 0.28
R1_N_BOUNDARY_POINTS = 150
R1_BLUE_CORE_OFFSET_X = -150    # Asymmetrical core offset (westward)
R1_N_VOIDS = 6
R1_VOID_RADIUS_MIN = 30
R1_VOID_RADIUS_MAX = 70
R1_VOID_FREQUENCY = 0.8
R1_VOID_ATTEMPTS = 10
R1_PRIORITY = 1                 # HIGHEST DOMINANCE
R1_BLEND_BAND = 40.0

# ---------- REGION 2: RIGHT MASS (Papillary / Dendritic) ----------
R2_START_X = 1400
R2_START_ANGLE = np.pi           # Points left
R2_BUFFER_RADIUS = 95
R2_N_GLANDS = 2
R2_GLAND_RADIUS_MIN = 20
R2_GLAND_RADIUS_MAX = 35
R2_N_CLUSTERS = 2
R2_CLUSTER_RADIUS_MIN = 25
R2_CLUSTER_RADIUS_MAX = 45
R2_STRUCTURE_FREQUENCY = 0.7
R2_STRUCTURE_ATTEMPTS = 5
R2_PRIORITY = 2                  # SECONDARY DOMINANCE
R2_BLEND_BAND = 40.0

# ---------- REGION 3: BACKGROUND ----------
R3_PRIORITY = 0
R3_BLEND_BAND = 40.0

# ---------- REGION 4: SATELLITE LOBULE (Top-Right) ----------
R4_CENTER = [1700, 1200]
R4_RADIUS = 200
R4_NOISE_STRENGTH = 0.35
R4_N_BOUNDARY_POINTS = 150
R4_N_VOIDS = 4
R4_VOID_RADIUS_MIN = 25
R4_VOID_RADIUS_MAX = 50
R4_VOID_FREQUENCY = 0.75
R4_VOID_ATTEMPTS = 8
R4_PRIORITY = 3
R4_BLEND_BAND = 35.0

# ---------- REGION 5: SATELLITE DENDRITE (Top-Left) ----------
R5_START = [300, 1300]
R5_START_ANGLE = -np.pi / 4
R5_BUFFER_RADIUS = 60
R5_INITIAL_LENGTH = 150
R5_PRIORITY = 3
R5_BLEND_BAND = 35.0

# ---------- REGION 6: INVASIVE ROOT ----------
R6_START = [1550, 500]
R6_START_ANGLE = 5 * np.pi / 4
R6_BUFFER_RADIUS = 75
R6_INITIAL_LENGTH = 150
R6_PRIORITY = 3
R6_BLEND_BAND = 60.0            # Deep fuzzy integration

# ---------- REGION 7: HYBRID NODULE (Bottom-Left) ----------
R7_CENTER = [500, 400]
R7_CIRCLE_RADIUS = 180
R7_CIRCLE_NOISE = 0.32
R7_CIRCLE_N_POINTS = 150
R7_BRANCH_ANGLE = np.pi / 2
R7_BRANCH_BUFFER = 55
R7_BRANCH_LENGTH = 120
R7_PRIORITY = 3
R7_BLEND_BAND = 35.0

# ---------- REGION 8: HYBRID CLONE (Top-Left) ----------
R8_CENTER = [400, 1100]
R8_CIRCLE_RADIUS = 150
R8_CIRCLE_NOISE = 0.32
R8_CIRCLE_N_POINTS = 150
R8_BRANCH_ANGLE = np.pi / 3
R8_BRANCH_BUFFER = 45
R8_BRANCH_LENGTH = 100
R8_PRIORITY = 3
R8_BLEND_BAND = 35.0

# ---------- REGION 9: DENDRITIC CONNECTOR (Bottom) ----------
R9_BUFFER_RADIUS = 50
R9_INITIAL_LENGTH = 130
R9_START_ANGLE = 5 * np.pi / 4
R9_PRIORITY = 3
R9_BLEND_BAND = 35.0

# ---------- REGION 10: HYBRID CLONE (Central) ----------
R10_CENTER = [800, 700]
R10_CIRCLE_RADIUS = 140
R10_CIRCLE_NOISE = 0.30
R10_CIRCLE_N_POINTS = 150
R10_BRANCH_ANGLE = -np.pi / 6
R10_BRANCH_BUFFER = 40
R10_BRANCH_LENGTH = 95
R10_PRIORITY = 3
R10_BLEND_BAND = 35.0

# ---------- REGION 11: BLUE SATELLITE (West-1) ----------
R11_CENTER = [320, 900]
R11_RADIUS = 120
R11_NOISE_STRENGTH = 0.30
R11_N_POINTS = 150
R11_PRIORITY = 3
R11_BLEND_BAND = 30.0

# ---------- REGION 12: BLUE SATELLITE (West-2) ----------
R12_CENTER = [300, 550]
R12_RADIUS = 100
R12_NOISE_STRENGTH = 0.32
R12_N_POINTS = 150
R12_PRIORITY = 3
R12_BLEND_BAND = 30.0

# ---------- REGION 13: BLUE SATELLITE (West-3) ----------
R13_CENTER = [270, 1100]
R13_RADIUS = 90
R13_NOISE_STRENGTH = 0.28
R13_N_POINTS = 150
R13_PRIORITY = 3
R13_BLEND_BAND = 30.0

# ---------- REGION 14: LOBULE NORTHEAST ----------
R14_CENTER = [1650, 900]
R14_RADIUS = 130
R14_NOISE_STRENGTH = 0.38
R14_N_POINTS = 150
R14_N_VOIDS = 3
R14_VOID_RADIUS_MIN = 20
R14_VOID_RADIUS_MAX = 40
R14_VOID_FREQUENCY = 0.75
R14_VOID_ATTEMPTS = 6
R14_PRIORITY = 3
R14_BLEND_BAND = 35.0

# ---------- REGION 15: LOBULE EAST ----------
R15_CENTER = [1730, 730]
R15_RADIUS = 110
R15_NOISE_STRENGTH = 0.35
R15_N_POINTS = 150
R15_N_VOIDS = 2
R15_VOID_RADIUS_MIN = 18
R15_VOID_RADIUS_MAX = 35
R15_VOID_FREQUENCY = 0.70
R15_VOID_ATTEMPTS = 5
R15_PRIORITY = 3
R15_BLEND_BAND = 35.0

# ---------- REGION 16: LOBULE SOUTHEAST ----------
R16_CENTER = [1680, 580]
R16_RADIUS = 120
R16_NOISE_STRENGTH = 0.40
R16_N_POINTS = 150
R16_N_VOIDS = 5
R16_VOID_RADIUS_MIN = 15
R16_VOID_RADIUS_MAX = 38
R16_VOID_FREQUENCY = 0.80
R16_VOID_ATTEMPTS = 7
R16_PRIORITY = 3
R16_BLEND_BAND = 35.0

# ---------- REGION 17: LOBULE FAR EAST (Rightmost) ----------
R17_CENTER = [1820, 780]
R17_RADIUS = 105
R17_NOISE_STRENGTH = 0.33
R17_N_POINTS = 150
R17_N_VOIDS = 4
R17_VOID_RADIUS_MIN = 17
R17_VOID_RADIUS_MAX = 35
R17_VOID_FREQUENCY = 0.75
R17_VOID_ATTEMPTS = 6
R17_PRIORITY = 3
R17_BLEND_BAND = 35.0

# ---------- REGION 18: CONNECTOR EAST ----------
R18_START_ANGLE = 0              # Points east
R18_BUFFER_RADIUS = 40
R18_INITIAL_LENGTH = 120
R18_PRIORITY = 3
R18_BLEND_BAND = 35.0

# ---------- BRANCHING SKELETON ----------
BRANCHING_MAX_DEPTH = 5
BRANCHING_MIN_LENGTH = 30
BRANCHING_PROBABILITY = 0.75
BRANCHING_LENGTH_DECAY = 0.73
BRANCHING_STRAIGHT_DECAY = 0.75
BRANCHING_ANGLE_MIN = np.pi / 8
BRANCHING_ANGLE_MAX = np.pi / 4
BRANCHING_CURVATURE = 0.15

# ---------- FOURIER BOUNDARY SYNTHESIS ----------
FOURIER_N_HARMONICS = 8


# ============================================================================
# DERIVED CONSTANTS (computed from config — do not edit directly)
# ============================================================================

# Split fraction shorthands for probability assignments
_CB = CORE_BLUE_FRACTION                  # Core-Blue fraction
_CC = 1.0 - CORE_BLUE_FRACTION           # Core-Cyan fraction
_GD = GREEN_DARK_FRACTION                 # Green-Dark fraction
_GL = 1.0 - GREEN_DARK_FRACTION          # Green-Light fraction
_RR = 1.0 - RED_DEEP_FRACTION            # Red fraction
_RD = RED_DEEP_FRACTION                   # Red-Deep fraction
_PD = PURPLE_DARK_FRACTION               # Purple-Dark fraction
_PL = 1.0 - PURPLE_DARK_FRACTION         # Purple-Light fraction


print("=" * 70)
print("Region-Based Hierarchical Tissue with TissueSlice")
print("=" * 70)


# ============================================================================
# BOUNDARY GENERATION UTILITIES
# ============================================================================

def create_organic_polygon_from_circle(center, radius, noise_strength=0.25, n_points=200):
    """Create an organic polygon by warping a circle with Fourier synthesis."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    perturbations = np.zeros(n_points)
    for k in range(1, FOURIER_N_HARMONICS + 1):
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = 1.0 / k
        perturbations += amplitude * np.sin(k * angles + phase)

    pmax = np.abs(perturbations).max()
    if pmax > 0:
        perturbations /= pmax

    points = []
    for angle, perturb in zip(angles, perturbations):
        r_mod = radius * (1.0 + noise_strength * perturb)
        x = center[0] + r_mod * np.cos(angle)
        y = center[1] + r_mod * np.sin(angle)
        points.append([x, y])

    return Polygon(points).buffer(0)


def create_branching_skeleton(start_pos, start_angle, initial_length, max_depth=None):
    """Create a branching line skeleton for dendritic structures."""
    if max_depth is None:
        max_depth = BRANCHING_MAX_DEPTH
    branches = []

    def grow_branch(pos, angle, length, depth):
        if depth > max_depth or length < BRANCHING_MIN_LENGTH:
            return
        angle += np.random.uniform(-BRANCHING_CURVATURE, BRANCHING_CURVATURE)
        end_x = pos[0] + length * np.cos(angle)
        end_y = pos[1] + length * np.sin(angle)
        if not (50 < end_x < FRAME_WIDTH - 50 and 50 < end_y < FRAME_HEIGHT - 50):
            return
        end_pos = np.array([end_x, end_y])
        branches.append(LineString([pos, end_pos]))
        if np.random.random() < BRANCHING_PROBABILITY:
            new_length = length * BRANCHING_LENGTH_DECAY
            left_angle = angle + np.random.uniform(BRANCHING_ANGLE_MIN, BRANCHING_ANGLE_MAX)
            right_angle = angle - np.random.uniform(BRANCHING_ANGLE_MIN, BRANCHING_ANGLE_MAX)
            grow_branch(end_pos, left_angle, new_length, depth + 1)
            grow_branch(end_pos, right_angle, new_length, depth + 1)
        else:
            new_length = length * BRANCHING_STRAIGHT_DECAY
            grow_branch(end_pos, angle, new_length, depth + 1)

    grow_branch(start_pos, start_angle, initial_length, 0)
    return branches


def create_organic_branching_polygon(start_pos, start_angle, buffer_radius=140, initial_length=250):
    """Create an organic branching polygon by buffering a skeleton."""
    branches = create_branching_skeleton(start_pos, start_angle, initial_length=initial_length)
    if not branches:
        end_pos = start_pos + 200 * np.array([np.cos(start_angle), np.sin(start_angle)])
        branches = [LineString([start_pos, end_pos])]

    buffered_branches = []
    for i, branch in enumerate(branches):
        depth_factor = 1.0 - (i / len(branches)) * 0.4
        random_factor = np.random.uniform(0.85, 1.15)
        local_buffer = buffer_radius * depth_factor * random_factor
        buffered = branch.buffer(local_buffer, cap_style=1, resolution=8)
        buffered_branches.append(buffered)

    return unary_union(buffered_branches)


# ============================================================================
# STRUCTURE CREATION UTILITIES
# ============================================================================

def create_glandular_structures(region_polygon, n_glands=5, acinus_radius_range=(40, 80)):
    """Create glandular units with layered epithelial structure."""
    minx, miny, maxx, maxy = region_polygon.bounds
    gland_prototypes = []
    attempts = 0

    while len(gland_prototypes) < n_glands and attempts < n_glands * 20:
        attempts += 1
        x = np.random.uniform(minx + 150, maxx - 150)
        y = np.random.uniform(miny + 150, maxy - 150)
        if region_polygon.contains(Point(x, y)):
            acinus_rad = np.random.uniform(*acinus_radius_range)
            n_acini = np.random.randint(4, 8)
            center = np.array([x, y])
            layer_rule = LayerRule(
                n_cell_types=N_CELL_TYPES,
                layer_types=[ID_CORE_BLUE, ID_YELLOW, ID_ORANGE],
                layer_boundaries=[0.5, 0.8],
                transition_width=6.0
            )
            gland_prototypes.append(lambda c=center, r=acinus_rad, n=n_acini: GlandularUnit(
                frame_size=FRAME_WIDTH,
                n_acini=n,
                acinus_radius=r,
                arrangement='circular',
                central_duct=True,
                duct_radius=r * 0.4,
                fixed_center=c,
                tipical_cell_spacing=GLAND_CELL_SPACING,
                rules=[layer_rule]
            ))
    return gland_prototypes


def create_vacuolated_structures(region_polygon, n_vacuoles=4, scale_range=(70, 150)):
    """Create vacuolated structures with layered boundaries."""
    minx, miny, maxx, maxy = region_polygon.bounds
    vacuole_prototypes = []
    attempts = 0

    while len(vacuole_prototypes) < n_vacuoles and attempts < n_vacuoles * 20:
        attempts += 1
        x = np.random.uniform(minx + 120, maxx - 120)
        y = np.random.uniform(miny + 120, maxy - 120)
        if region_polygon.contains(Point(x, y)):
            scale = np.random.uniform(*scale_range)
            center = np.array([x, y])
            layer_rule = LayerRule(
                n_cell_types=N_CELL_TYPES,
                layer_types=[ID_GREEN_DARK, ID_RED],
                layer_boundaries=[0.70],
                transition_width=10.0
            )
            vacuole_prototypes.append(lambda c=center, s=scale: VacuolatedStructure(
                frame_size=FRAME_WIDTH,
                scale=s,
                fixed_center=c,
                hole_scale_factor=0.55,
                tipical_cell_spacing=VACUOLE_CELL_SPACING,
                smoothing_iterations=4,
                rules=[layer_rule]
            ))
    return vacuole_prototypes


def create_cluster_attachments(region_polygon, n_clusters=6, radius_range=(40, 80)):
    """Create cluster elements to attach to dendritic structure."""
    minx, miny, maxx, maxy = region_polygon.bounds
    cluster_prototypes = []
    attempts = 0

    while len(cluster_prototypes) < n_clusters and attempts < n_clusters * 20:
        attempts += 1
        x = np.random.uniform(minx + 100, maxx - 100)
        y = np.random.uniform(miny + 100, maxy - 100)
        if region_polygon.contains(Point(x, y)):
            radius = np.random.uniform(*radius_range)
            center = np.array([x, y])
            cluster_rule = MixOfNCellTypesRule(
                N_CELL_TYPES,
                list_N=[ID_GREEN_DARK, ID_GREEN_LIGHT, ID_YELLOW, ID_ORANGE],
                proportions=[0.25, 0.25, 0.30, 0.20]
            )
            cluster_prototypes.append(lambda c=center, r=radius: ClusterElement(
                frame_size=FRAME_WIDTH,
                radius=r,
                fixed_center=c,
                density_profile='dense_center',
                tipical_cell_spacing=CLUSTER_CELL_SPACING,
                smoothing_iterations=3,
                rules=[cluster_rule]
            ))
    return cluster_prototypes


def create_void_elements(region_polygon, n_voids=3, void_radius_range=(50, 120)):
    """Create void / necrotic elements within a region."""
    minx, miny, maxx, maxy = region_polygon.bounds
    void_prototypes = []
    attempts = 0

    while len(void_prototypes) < n_voids and attempts < n_voids * 10:
        attempts += 1
        x = np.random.uniform(minx + 100, maxx - 100)
        y = np.random.uniform(miny + 100, maxy - 100)
        if region_polygon.contains(Point(x, y)):
            radius = np.random.uniform(*void_radius_range)
            center = np.array([x, y])
            void_prototypes.append(lambda c=center, r=radius: VacuolatedStructure(
                frame_size=FRAME_WIDTH,
                scale=r,
                fixed_center=c,
                tipical_cell_spacing=VOID_CELL_SPACING,
                hole_scale_factor=0.9,
                rules=[MixOfNCellTypesRule(
                    N_CELL_TYPES,
                    list_N=[ID_CORE_BLUE, ID_CORE_CYAN],
                    proportions=[_CB, _CC]
                )]
            ))
    return void_prototypes


# ============================================================================
# GRADIENT REFERENCE POINT HELPERS
# ============================================================================

def make_rainbow_gradient_refs(centroid, core_r, green_r, yellow_r, red_r,
                               core_n=6, green_n=8, yellow_n=8, red_n=8):
    """Build reference points + probability arrays for a rainbow gradient.

    Rings from centroid: core (blue/cyan) → green → yellow → red/orange.
    """
    pts, probs = [], []

    for angle in np.linspace(0, 2 * np.pi, core_n, endpoint=False):
        pts.append(centroid + core_r * np.array([np.cos(angle), np.sin(angle)]))
        p = np.zeros(N_CELL_TYPES)
        p[ID_CORE_BLUE] = _CB
        p[ID_CORE_CYAN] = _CC
        probs.append(p)

    for angle in np.linspace(0, 2 * np.pi, green_n, endpoint=False):
        pts.append(centroid + green_r * np.array([np.cos(angle), np.sin(angle)]))
        p = np.zeros(N_CELL_TYPES)
        p[ID_GREEN_DARK] = 0.7 * _GD
        p[ID_GREEN_LIGHT] = 0.7 * _GL
        p[ID_CORE_BLUE] = 0.3 * _CB
        p[ID_CORE_CYAN] = 0.3 * _CC
        probs.append(p)

    for angle in np.linspace(0, 2 * np.pi, yellow_n, endpoint=False):
        pts.append(centroid + yellow_r * np.array([np.cos(angle), np.sin(angle)]))
        p = np.zeros(N_CELL_TYPES)
        p[ID_YELLOW] = 0.7
        p[ID_GREEN_DARK] = 0.3 * _GD
        p[ID_GREEN_LIGHT] = 0.3 * _GL
        probs.append(p)

    for angle in np.linspace(0, 2 * np.pi, red_n, endpoint=False):
        pts.append(centroid + red_r * np.array([np.cos(angle), np.sin(angle)]))
        p = np.zeros(N_CELL_TYPES)
        p[ID_RED] = 0.6 * _RR
        p[ID_RED_DEEP] = 0.6 * _RD
        p[ID_ORANGE] = 0.3
        p[ID_YELLOW] = 0.1
        probs.append(p)

    return pts, probs


def make_purple_gradient_refs(centroid, core_r, mix_r, edge_r,
                              core_n=6, mix_n=8, edge_n=8):
    """Build reference points + probability arrays for a purple-to-edge gradient.

    Rings from centroid: purple core → purple/green mix → green/yellow edge.
    """
    pts, probs = [], []

    for angle in np.linspace(0, 2 * np.pi, core_n, endpoint=False):
        pts.append(centroid + core_r * np.array([np.cos(angle), np.sin(angle)]))
        p = np.zeros(N_CELL_TYPES)
        p[ID_PURPLE_DARK] = _PD
        p[ID_PURPLE_LIGHT] = _PL
        probs.append(p)

    for angle in np.linspace(0, 2 * np.pi, mix_n, endpoint=False):
        pts.append(centroid + mix_r * np.array([np.cos(angle), np.sin(angle)]))
        p = np.zeros(N_CELL_TYPES)
        p[ID_PURPLE_DARK] = 0.5 * _PD
        p[ID_PURPLE_LIGHT] = 0.5 * _PL
        p[ID_GREEN_DARK] = 0.3 * _GD
        p[ID_GREEN_LIGHT] = 0.3 * _GL
        p[ID_YELLOW] = 0.2
        probs.append(p)

    for angle in np.linspace(0, 2 * np.pi, edge_n, endpoint=False):
        pts.append(centroid + edge_r * np.array([np.cos(angle), np.sin(angle)]))
        p = np.zeros(N_CELL_TYPES)
        p[ID_GREEN_DARK] = 0.6 * _GD
        p[ID_GREEN_LIGHT] = 0.6 * _GL
        p[ID_YELLOW] = 0.3
        p[ID_ORANGE] = 0.1
        probs.append(p)

    return pts, probs


def make_blue_satellite_refs(centroid, core_r, edge_r, core_n=6, edge_n=8):
    """Build reference points for a blue/cyan satellite gradient."""
    pts, probs = [], []

    for angle in np.linspace(0, 2 * np.pi, core_n, endpoint=False):
        pts.append(centroid + core_r * np.array([np.cos(angle), np.sin(angle)]))
        p = np.zeros(N_CELL_TYPES)
        p[ID_CORE_BLUE] = _CB
        p[ID_CORE_CYAN] = _CC
        probs.append(p)

    for angle in np.linspace(0, 2 * np.pi, edge_n, endpoint=False):
        pts.append(centroid + edge_r * np.array([np.cos(angle), np.sin(angle)]))
        p = np.zeros(N_CELL_TYPES)
        p[ID_CORE_BLUE] = 0.8 * _CB
        p[ID_CORE_CYAN] = 0.8 * _CC
        p[ID_GREEN_DARK] = 0.2 * _GD
        p[ID_GREEN_LIGHT] = 0.2 * _GL
        probs.append(p)

    return pts, probs


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
print("   ✓ Cell morphology defined")


print("\n2. Creating organic regional polygons...")

# === REGION 1: LEFT MASS ===
print("   - Generating left lobular mass polygon...")
left_center = np.array([R1_CENTER_X, FRAME_HEIGHT / 2])

R1_Left_Mass = create_organic_polygon_from_circle(
    center=left_center,
    radius=R1_RADIUS,
    noise_strength=R1_NOISE_STRENGTH,
    n_points=R1_N_BOUNDARY_POINTS
)
print(f"     ✓ Created organic left mass (area: {R1_Left_Mass.area:.0f} µm²)")


# === REGION 2: RIGHT MASS ===
print("   - Generating right dendritic mass polygon...")
right_start = np.array([R2_START_X, FRAME_HEIGHT / 2])

R2_Right_Mass = create_organic_branching_polygon(
    start_pos=right_start,
    start_angle=R2_START_ANGLE,
    buffer_radius=R2_BUFFER_RADIUS
)
print(f"     ✓ Created organic dendritic mass")


# === REGION 4: TOP-RIGHT SATELLITE LOBULE ===
print("   - Generating top-right satellite lobule...")
R4_Satellite_TopRight = create_organic_polygon_from_circle(
    center=np.array(R4_CENTER),
    radius=R4_RADIUS,
    noise_strength=R4_NOISE_STRENGTH,
    n_points=R4_N_BOUNDARY_POINTS
)
print(f"     ✓ Created satellite lobule (area: {R4_Satellite_TopRight.area:.0f} µm²)")


# === REGION 5: TOP-LEFT SATELLITE DENDRITE ===
print("   - Generating top-left satellite dendrite...")
R5_Satellite_TopLeft = create_organic_branching_polygon(
    start_pos=np.array(R5_START),
    start_angle=R5_START_ANGLE,
    buffer_radius=R5_BUFFER_RADIUS,
    initial_length=R5_INITIAL_LENGTH
)
print(f"     ✓ Created top-left satellite dendrite")


# === REGION 6: INVASIVE ROOT ===
print("   - Generating deeply integrated invasive root...")
R6_Invasive_Root = create_organic_branching_polygon(
    start_pos=np.array(R6_START),
    start_angle=R6_START_ANGLE,
    buffer_radius=R6_BUFFER_RADIUS,
    initial_length=R6_INITIAL_LENGTH
)
print(f"     ✓ Created deeply integrated invasive root")


# === REGION 7: HYBRID NODULE ===
print("   - Generating hybrid nodule with mixed morphology...")
hybrid_center = np.array(R7_CENTER)
hybrid_circle = create_organic_polygon_from_circle(
    center=hybrid_center, radius=R7_CIRCLE_RADIUS,
    noise_strength=R7_CIRCLE_NOISE, n_points=R7_CIRCLE_N_POINTS
)
hybrid_branches = create_organic_branching_polygon(
    start_pos=hybrid_center, start_angle=R7_BRANCH_ANGLE,
    buffer_radius=R7_BRANCH_BUFFER, initial_length=R7_BRANCH_LENGTH
)
R7_Hybrid_Nodule = unary_union([hybrid_circle, hybrid_branches])
print(f"     ✓ Created hybrid nodule (mixed phenotype)")


# === REGION 8: HYBRID CLONE (Top-Left) ===
print("   - Generating hybrid clone infiltrating left mass (top-left)...")
clone_tl_center = np.array(R8_CENTER)
clone_tl_circle = create_organic_polygon_from_circle(
    center=clone_tl_center, radius=R8_CIRCLE_RADIUS,
    noise_strength=R8_CIRCLE_NOISE, n_points=R8_CIRCLE_N_POINTS
)
clone_tl_branches = create_organic_branching_polygon(
    start_pos=clone_tl_center, start_angle=R8_BRANCH_ANGLE,
    buffer_radius=R8_BRANCH_BUFFER, initial_length=R8_BRANCH_LENGTH
)
R8_Clone_TopLeft = unary_union([clone_tl_circle, clone_tl_branches])
print(f"     ✓ Created hybrid clone (infiltrating left mass)")


# === REGION 9: CENTER-BOTTOM BRIDGE ===
print("   - Generating center-bottom dendritic bridge...")
R9_Bridge_Bottom = create_organic_branching_polygon(
    start_pos=right_start,
    start_angle=R9_START_ANGLE,
    buffer_radius=R9_BUFFER_RADIUS,
    initial_length=R9_INITIAL_LENGTH
)
print(f"     ✓ Created dendritic connector bridge")


# === REGION 10: HYBRID CLONE (Central) ===
print("   - Generating hybrid clone deep inside left mass (central)...")
clone_c_center = np.array(R10_CENTER)
clone_c_circle = create_organic_polygon_from_circle(
    center=clone_c_center, radius=R10_CIRCLE_RADIUS,
    noise_strength=R10_CIRCLE_NOISE, n_points=R10_CIRCLE_N_POINTS
)
clone_c_branches = create_organic_branching_polygon(
    start_pos=clone_c_center, start_angle=R10_BRANCH_ANGLE,
    buffer_radius=R10_BRANCH_BUFFER, initial_length=R10_BRANCH_LENGTH
)
R10_Clone_Central = unary_union([clone_c_circle, clone_c_branches])
print(f"     ✓ Created central hybrid clone (deep infiltration)")


# === REGION 11-13: BLUE SATELLITES ===
print("   - Generating blue/cyan satellites...")
R11_Satellite_West1 = create_organic_polygon_from_circle(
    center=np.array(R11_CENTER), radius=R11_RADIUS,
    noise_strength=R11_NOISE_STRENGTH, n_points=R11_N_POINTS
)
R12_Satellite_West2 = create_organic_polygon_from_circle(
    center=np.array(R12_CENTER), radius=R12_RADIUS,
    noise_strength=R12_NOISE_STRENGTH, n_points=R12_N_POINTS
)
R13_Satellite_West3 = create_organic_polygon_from_circle(
    center=np.array(R13_CENTER), radius=R13_RADIUS,
    noise_strength=R13_NOISE_STRENGTH, n_points=R13_N_POINTS
)
print(f"     ✓ Created 3 blue/cyan satellites")


# === REGIONS 14-17: LOBULE CLUSTER ===
print("   - Generating lobule cluster near right mass...")
lobule1_center = np.array(R14_CENTER)
R14_Lobule_NE = create_organic_polygon_from_circle(
    center=lobule1_center, radius=R14_RADIUS,
    noise_strength=R14_NOISE_STRENGTH, n_points=R14_N_POINTS
)
lobule2_center = np.array(R15_CENTER)
R15_Lobule_E = create_organic_polygon_from_circle(
    center=lobule2_center, radius=R15_RADIUS,
    noise_strength=R15_NOISE_STRENGTH, n_points=R15_N_POINTS
)
lobule3_center = np.array(R16_CENTER)
R16_Lobule_SE = create_organic_polygon_from_circle(
    center=lobule3_center, radius=R16_RADIUS,
    noise_strength=R16_NOISE_STRENGTH, n_points=R16_N_POINTS
)
lobule4_center = np.array(R17_CENTER)
R17_Lobule_FarE = create_organic_polygon_from_circle(
    center=lobule4_center, radius=R17_RADIUS,
    noise_strength=R17_NOISE_STRENGTH, n_points=R17_N_POINTS
)
print(f"     ✓ Created 4 lobules (NE, E, SE, FarE)")


# === REGION 18: CONNECTOR EAST ===
print("   - Generating dendritic connector on rightmost lobule...")
R18_Connector_East = create_organic_branching_polygon(
    start_pos=lobule4_center,
    start_angle=R18_START_ANGLE,
    buffer_radius=R18_BUFFER_RADIUS,
    initial_length=R18_INITIAL_LENGTH
)
print(f"     ✓ Created dendritic connector on rightmost lobule")


# === BOOLEAN GEOMETRY: COOKIE CUTTER (REVERSED) ===
print("\n   - Applying cookie cutter logic to prevent cell overlap...")

main_masses_union = unary_union([R1_Left_Mass, R2_Right_Mass])

# Main masses DOMINATE invaders — subtract main mass footprints from all Priority-3 regions
R4_Satellite_TopRight = R4_Satellite_TopRight.difference(main_masses_union)
R5_Satellite_TopLeft = R5_Satellite_TopLeft.difference(main_masses_union)
R6_Invasive_Root = R6_Invasive_Root.difference(main_masses_union)
R7_Hybrid_Nodule = R7_Hybrid_Nodule.difference(main_masses_union)
R8_Clone_TopLeft = R8_Clone_TopLeft.difference(main_masses_union)
R9_Bridge_Bottom = R9_Bridge_Bottom.difference(main_masses_union)
R10_Clone_Central = R10_Clone_Central.difference(main_masses_union)
R11_Satellite_West1 = R11_Satellite_West1.difference(main_masses_union)
R12_Satellite_West2 = R12_Satellite_West2.difference(main_masses_union)
R13_Satellite_West3 = R13_Satellite_West3.difference(main_masses_union)
R14_Lobule_NE = R14_Lobule_NE.difference(main_masses_union)
R15_Lobule_E = R15_Lobule_E.difference(main_masses_union)
R16_Lobule_SE = R16_Lobule_SE.difference(main_masses_union)
R17_Lobule_FarE = R17_Lobule_FarE.difference(main_masses_union)
R18_Connector_East = R18_Connector_East.difference(main_masses_union)

print("     ✓ Main masses dominate — removed footprints from invaders (no cell overlap)")


# === REGION 3: BACKGROUND ===
print("   - Defining background region...")
frame_box = box(0, 0, FRAME_WIDTH, FRAME_HEIGHT)

all_foreground = unary_union([
    R1_Left_Mass, R2_Right_Mass,
    R4_Satellite_TopRight, R5_Satellite_TopLeft,
    R6_Invasive_Root, R7_Hybrid_Nodule,
    R8_Clone_TopLeft, R9_Bridge_Bottom, R10_Clone_Central,
    R11_Satellite_West1, R12_Satellite_West2, R13_Satellite_West3,
    R14_Lobule_NE, R15_Lobule_E, R16_Lobule_SE,
    R17_Lobule_FarE, R18_Connector_East
])

R3_Background = frame_box.difference(all_foreground)
print("     ✓ Background region defined (cookie cutter complete)")


# ============================================================================
# FOV DISTRIBUTIONS
# ============================================================================

print("\n3. Creating FOV distributions for each region...")

# === REGION 1 (Left Mass — Rainbow with asymmetric blue core) ===
print("   - Configuring left mass rainbow gradient (asymmetrical blue core)...")

left_centroid = np.array([R1_Left_Mass.centroid.x, R1_Left_Mass.centroid.y])
blue_core_offset = np.array([left_centroid[0] + R1_BLUE_CORE_OFFSET_X, left_centroid[1]])

ref_points_left = []
ref_probs_left = []

# Core ring centred on OFFSET position (8 points for RBF stability)
for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    ref_points_left.append(blue_core_offset + 30 * np.array([np.cos(angle), np.sin(angle)]))
    p = np.zeros(N_CELL_TYPES);  p[ID_CORE_BLUE] = _CB;  p[ID_CORE_CYAN] = _CC
    ref_probs_left.append(p)

# Extra dense blue on western edge
for y_off in [-40, 0, 40]:
    ref_points_left.append(blue_core_offset + np.array([-50, y_off]))
    p = np.zeros(N_CELL_TYPES);  p[ID_CORE_BLUE] = _CB;  p[ID_CORE_CYAN] = _CC
    ref_probs_left.append(p)

# Green ring (centred on real centroid)
for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    ref_points_left.append(left_centroid + 100 * np.array([np.cos(angle), np.sin(angle)]))
    p = np.zeros(N_CELL_TYPES)
    p[ID_GREEN_DARK] = 0.7 * _GD;  p[ID_GREEN_LIGHT] = 0.7 * _GL
    p[ID_CORE_BLUE] = 0.3 * _CB;   p[ID_CORE_CYAN] = 0.3 * _CC
    ref_probs_left.append(p)

# Yellow ring
for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
    ref_points_left.append(left_centroid + 200 * np.array([np.cos(angle), np.sin(angle)]))
    p = np.zeros(N_CELL_TYPES)
    p[ID_YELLOW] = 0.7;  p[ID_GREEN_DARK] = 0.3 * _GD;  p[ID_GREEN_LIGHT] = 0.3 * _GL
    ref_probs_left.append(p)

# Red / orange outer ring
for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
    ref_points_left.append(left_centroid + 280 * np.array([np.cos(angle), np.sin(angle)]))
    p = np.zeros(N_CELL_TYPES)
    p[ID_RED] = 0.6 * _RR;  p[ID_RED_DEEP] = 0.6 * _RD;  p[ID_ORANGE] = 0.3;  p[ID_YELLOW] = 0.1
    ref_probs_left.append(p)

rainbow_rule = ProbabilityNodeFieldRule(
    n_cell_types=N_CELL_TYPES,
    n_ref_points=len(ref_points_left),
    reference_points=np.array(ref_points_left),
    ref_probs=np.array(ref_probs_left)
)

LeftMassBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_WIDTH,
    tipical_cell_spacing=FOREGROUND_CELL_SPACING,
    rules=[rainbow_rule]
)

print("   - Creating cystic/cribriform architecture...")
void_prototypes = create_void_elements(R1_Left_Mass, n_voids=R1_N_VOIDS,
                                       void_radius_range=(R1_VOID_RADIUS_MIN, R1_VOID_RADIUS_MAX))

fovd_left = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=LeftMassBGPrototype,
    other_elements=void_prototypes,
    elements_frequency=[R1_VOID_FREQUENCY] * len(void_prototypes),
    attempts_at_elements=R1_VOID_ATTEMPTS
)
print(f"     ✓ Cribriform pattern with {len(void_prototypes)} necrotic voids")


# === REGION 2 (Right Mass — Purple gradient) ===
print("   - Configuring right mass with purple-to-edge gradient...")
right_centroid = np.array([R2_START_X, FRAME_HEIGHT / 2])

rp2, rpr2 = make_purple_gradient_refs(right_centroid, core_r=50, mix_r=150, edge_r=250,
                                      core_n=8, mix_n=10, edge_n=12)

dendrite_rule = ProbabilityNodeFieldRule(
    n_cell_types=N_CELL_TYPES,
    n_ref_points=len(rp2),
    reference_points=np.array(rp2),
    ref_probs=np.array(rpr2)
)

RightMassBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_WIDTH,
    tipical_cell_spacing=FOREGROUND_CELL_SPACING,
    rules=[dendrite_rule]
)

print("   - Adding internal structures to papillary mass...")
gland_prototypes_right = create_glandular_structures(
    R2_Right_Mass, n_glands=R2_N_GLANDS,
    acinus_radius_range=(R2_GLAND_RADIUS_MIN, R2_GLAND_RADIUS_MAX)
)
cluster_prototypes = create_cluster_attachments(
    R2_Right_Mass, n_clusters=R2_N_CLUSTERS,
    radius_range=(R2_CLUSTER_RADIUS_MIN, R2_CLUSTER_RADIUS_MAX)
)
all_right_structures = gland_prototypes_right + cluster_prototypes

fovd_right = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=RightMassBGPrototype,
    other_elements=all_right_structures,
    elements_frequency=[R2_STRUCTURE_FREQUENCY] * len(all_right_structures),
    attempts_at_elements=R2_STRUCTURE_ATTEMPTS
)
print(f"     ✓ Papillary architecture with {len(gland_prototypes_right)} glands and {len(cluster_prototypes)} clusters")


# === REGION 3 (Background — Sparse stroma) ===
print("   - Configuring stromal interface background...")

BackgroundPrototype = lambda: FrameWideElement(
    frame_size=FRAME_WIDTH,
    tipical_cell_spacing=BACKGROUND_CELL_SPACING,
    rules=[MixOfNCellTypesRule(
        N_CELL_TYPES,
        list_N=BACKGROUND_TYPES,
        proportions=BACKGROUND_PROPORTIONS
    )]
)

fovd_background = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=BackgroundPrototype,
    other_elements=[],
    elements_frequency=[],
    attempts_at_elements=0
)
print("     ✓ Sparse background defined")


# === REGION 4 (Satellite Lobule Top-Right — Rainbow) ===
print("   - Configuring top-right satellite lobule with rainbow gradient...")
sat_tr_centroid = np.array([R4_Satellite_TopRight.centroid.x, R4_Satellite_TopRight.centroid.y])
rp4, rpr4 = make_rainbow_gradient_refs(sat_tr_centroid, core_r=15, green_r=60, yellow_r=120, red_r=180,
                                       core_n=6, green_n=8, yellow_n=8, red_n=8)

SatTopRightBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_WIDTH,
    tipical_cell_spacing=FOREGROUND_CELL_SPACING,
    rules=[ProbabilityNodeFieldRule(
        n_cell_types=N_CELL_TYPES, n_ref_points=len(rp4),
        reference_points=np.array(rp4), ref_probs=np.array(rpr4)
    )]
)

void_prototypes_sat_tr = create_void_elements(R4_Satellite_TopRight, n_voids=R4_N_VOIDS,
                                              void_radius_range=(R4_VOID_RADIUS_MIN, R4_VOID_RADIUS_MAX))

fovd_sat_tr = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=SatTopRightBGPrototype,
    other_elements=void_prototypes_sat_tr,
    elements_frequency=[R4_VOID_FREQUENCY] * len(void_prototypes_sat_tr),
    attempts_at_elements=R4_VOID_ATTEMPTS
)
print(f"     ✓ Satellite lobule with {len(void_prototypes_sat_tr)} voids")


# === REGION 5 (Satellite Dendrite Top-Left — Purple gradient) ===
print("   - Configuring top-left satellite dendrite...")
sat_tl_centroid = np.array(R5_START)
rp5, rpr5 = make_purple_gradient_refs(sat_tl_centroid, core_r=30, mix_r=80, edge_r=130,
                                      core_n=6, mix_n=8, edge_n=8)

SatTopLeftBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_WIDTH,
    tipical_cell_spacing=FOREGROUND_CELL_SPACING,
    rules=[ProbabilityNodeFieldRule(
        n_cell_types=N_CELL_TYPES, n_ref_points=len(rp5),
        reference_points=np.array(rp5), ref_probs=np.array(rpr5)
    )]
)

fovd_sat_tl = FOVDistribution(
    frame_size=FRAME_WIDTH, background_element=SatTopLeftBGPrototype,
    other_elements=[], elements_frequency=[], attempts_at_elements=0
)
print("     ✓ Top-left satellite dendrite configured")


# === REGION 6 (Invasive Root — Purple gradient) ===
print("   - Configuring invasive root...")
inv_centroid = np.array([1600, 600])
rp6, rpr6 = make_purple_gradient_refs(inv_centroid, core_r=30, mix_r=80, edge_r=130,
                                      core_n=6, mix_n=8, edge_n=8)

InvasiveRootBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_WIDTH,
    tipical_cell_spacing=FOREGROUND_CELL_SPACING,
    rules=[ProbabilityNodeFieldRule(
        n_cell_types=N_CELL_TYPES, n_ref_points=len(rp6),
        reference_points=np.array(rp6), ref_probs=np.array(rpr6)
    )]
)

fovd_root = FOVDistribution(
    frame_size=FRAME_WIDTH, background_element=InvasiveRootBGPrototype,
    other_elements=[], elements_frequency=[], attempts_at_elements=0
)
print("     ✓ Invasive root configured")


# === REGIONS 7, 8, 10 (Hybrid clones — Mixed phenotype) ===
print("   - Configuring hybrid nodules with mixed phenotype...")

# Build hybrid proportions from config, splitting core/purple/green into sub-types
_hybrid_types = [ID_CORE_BLUE, ID_CORE_CYAN,
                 ID_PURPLE_DARK, ID_PURPLE_LIGHT,
                 ID_GREEN_DARK, ID_GREEN_LIGHT]
_hybrid_props = [HYBRID_CORE_PROP * _CB, HYBRID_CORE_PROP * _CC,
                 HYBRID_PURPLE_PROP * _PD, HYBRID_PURPLE_PROP * _PL,
                 HYBRID_GREEN_PROP * _GD, HYBRID_GREEN_PROP * _GL]

hybrid_rule = MixOfNCellTypesRule(N_CELL_TYPES, list_N=_hybrid_types, proportions=_hybrid_props)
hybrid_rule_tl = MixOfNCellTypesRule(N_CELL_TYPES, list_N=_hybrid_types, proportions=_hybrid_props)
hybrid_rule_c = MixOfNCellTypesRule(N_CELL_TYPES, list_N=_hybrid_types, proportions=_hybrid_props)

HybridNoduleBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_WIDTH, tipical_cell_spacing=FOREGROUND_CELL_SPACING, rules=[hybrid_rule])
CloneTopLeftBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_WIDTH, tipical_cell_spacing=FOREGROUND_CELL_SPACING, rules=[hybrid_rule_tl])
CloneCentralBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_WIDTH, tipical_cell_spacing=FOREGROUND_CELL_SPACING, rules=[hybrid_rule_c])

fovd_hybrid = FOVDistribution(frame_size=FRAME_WIDTH, background_element=HybridNoduleBGPrototype,
                              other_elements=[], elements_frequency=[], attempts_at_elements=0)
fovd_clone_tl = FOVDistribution(frame_size=FRAME_WIDTH, background_element=CloneTopLeftBGPrototype,
                                other_elements=[], elements_frequency=[], attempts_at_elements=0)
fovd_clone_c = FOVDistribution(frame_size=FRAME_WIDTH, background_element=CloneCentralBGPrototype,
                               other_elements=[], elements_frequency=[], attempts_at_elements=0)
print("     ✓ Hybrid nodules (R7, R8, R10) configured")


# === REGION 9 (Dendritic Bridge — Purple gradient) ===
print("   - Configuring center-bottom dendritic connector...")
rp9, rpr9 = make_purple_gradient_refs(right_start, core_r=30, mix_r=70, edge_r=110,
                                      core_n=6, mix_n=8, edge_n=8)

BridgeBottomBGPrototype = lambda: FrameWideElement(
    frame_size=FRAME_WIDTH,
    tipical_cell_spacing=FOREGROUND_CELL_SPACING,
    rules=[ProbabilityNodeFieldRule(
        n_cell_types=N_CELL_TYPES, n_ref_points=len(rp9),
        reference_points=np.array(rp9), ref_probs=np.array(rpr9)
    )]
)

fovd_bridge_bottom = FOVDistribution(frame_size=FRAME_WIDTH, background_element=BridgeBottomBGPrototype,
                                     other_elements=[], elements_frequency=[], attempts_at_elements=0)
print("     ✓ Dendritic connector configured")


# === REGIONS 11-13 (Blue Satellites) ===
print("   - Configuring blue/cyan satellites...")

sat_w1_centroid = np.array([R11_Satellite_West1.centroid.x, R11_Satellite_West1.centroid.y])
rp11, rpr11 = make_blue_satellite_refs(sat_w1_centroid, core_r=20, edge_r=80)
fovd_sat_w1 = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=lambda: FrameWideElement(
        frame_size=FRAME_WIDTH, tipical_cell_spacing=FOREGROUND_CELL_SPACING,
        rules=[ProbabilityNodeFieldRule(n_cell_types=N_CELL_TYPES, n_ref_points=len(rp11),
               reference_points=np.array(rp11), ref_probs=np.array(rpr11))]),
    other_elements=[], elements_frequency=[], attempts_at_elements=0)

sat_w2_centroid = np.array([R12_Satellite_West2.centroid.x, R12_Satellite_West2.centroid.y])
rp12, rpr12 = make_blue_satellite_refs(sat_w2_centroid, core_r=18, edge_r=70)
fovd_sat_w2 = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=lambda: FrameWideElement(
        frame_size=FRAME_WIDTH, tipical_cell_spacing=FOREGROUND_CELL_SPACING,
        rules=[ProbabilityNodeFieldRule(n_cell_types=N_CELL_TYPES, n_ref_points=len(rp12),
               reference_points=np.array(rp12), ref_probs=np.array(rpr12))]),
    other_elements=[], elements_frequency=[], attempts_at_elements=0)

sat_w3_centroid = np.array([R13_Satellite_West3.centroid.x, R13_Satellite_West3.centroid.y])
rp13, rpr13 = make_blue_satellite_refs(sat_w3_centroid, core_r=15, edge_r=65)
fovd_sat_w3 = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=lambda: FrameWideElement(
        frame_size=FRAME_WIDTH, tipical_cell_spacing=FOREGROUND_CELL_SPACING,
        rules=[ProbabilityNodeFieldRule(n_cell_types=N_CELL_TYPES, n_ref_points=len(rp13),
               reference_points=np.array(rp13), ref_probs=np.array(rpr13))]),
    other_elements=[], elements_frequency=[], attempts_at_elements=0)

print("     ✓ Blue/cyan satellites (R11, R12, R13) configured")


# === REGIONS 14-17 (Lobule cluster — Rainbow gradients) ===
print("   - Configuring lobule cluster with rainbow gradients...")

# R14 Northeast
lob1_centroid = np.array([R14_Lobule_NE.centroid.x, R14_Lobule_NE.centroid.y])
rp14, rpr14 = make_rainbow_gradient_refs(lob1_centroid, core_r=15, green_r=50, yellow_r=90, red_r=120)
void_prototypes_lob1 = create_void_elements(R14_Lobule_NE, n_voids=R14_N_VOIDS,
                                            void_radius_range=(R14_VOID_RADIUS_MIN, R14_VOID_RADIUS_MAX))
fovd_lob1 = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=lambda: FrameWideElement(
        frame_size=FRAME_WIDTH, tipical_cell_spacing=FOREGROUND_CELL_SPACING,
        rules=[ProbabilityNodeFieldRule(n_cell_types=N_CELL_TYPES, n_ref_points=len(rp14),
               reference_points=np.array(rp14), ref_probs=np.array(rpr14))]),
    other_elements=void_prototypes_lob1,
    elements_frequency=[R14_VOID_FREQUENCY] * len(void_prototypes_lob1),
    attempts_at_elements=R14_VOID_ATTEMPTS)
print(f"     ✓ Northeast lobule with {len(void_prototypes_lob1)} voids")

# R15 East
lob2_centroid = np.array([R15_Lobule_E.centroid.x, R15_Lobule_E.centroid.y])
rp15, rpr15 = make_rainbow_gradient_refs(lob2_centroid, core_r=12, green_r=45, yellow_r=75, red_r=100)
void_prototypes_lob2 = create_void_elements(R15_Lobule_E, n_voids=R15_N_VOIDS,
                                            void_radius_range=(R15_VOID_RADIUS_MIN, R15_VOID_RADIUS_MAX))
fovd_lob2 = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=lambda: FrameWideElement(
        frame_size=FRAME_WIDTH, tipical_cell_spacing=FOREGROUND_CELL_SPACING,
        rules=[ProbabilityNodeFieldRule(n_cell_types=N_CELL_TYPES, n_ref_points=len(rp15),
               reference_points=np.array(rp15), ref_probs=np.array(rpr15))]),
    other_elements=void_prototypes_lob2,
    elements_frequency=[R15_VOID_FREQUENCY] * len(void_prototypes_lob2),
    attempts_at_elements=R15_VOID_ATTEMPTS)
print(f"     ✓ East lobule with {len(void_prototypes_lob2)} voids")

# R16 Southeast
lob3_centroid = np.array([R16_Lobule_SE.centroid.x, R16_Lobule_SE.centroid.y])
rp16, rpr16 = make_rainbow_gradient_refs(lob3_centroid, core_r=14, green_r=48, yellow_r=85, red_r=110)
void_prototypes_lob3 = create_void_elements(R16_Lobule_SE, n_voids=R16_N_VOIDS,
                                            void_radius_range=(R16_VOID_RADIUS_MIN, R16_VOID_RADIUS_MAX))
fovd_lob3 = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=lambda: FrameWideElement(
        frame_size=FRAME_WIDTH, tipical_cell_spacing=FOREGROUND_CELL_SPACING,
        rules=[ProbabilityNodeFieldRule(n_cell_types=N_CELL_TYPES, n_ref_points=len(rp16),
               reference_points=np.array(rp16), ref_probs=np.array(rpr16))]),
    other_elements=void_prototypes_lob3,
    elements_frequency=[R16_VOID_FREQUENCY] * len(void_prototypes_lob3),
    attempts_at_elements=R16_VOID_ATTEMPTS)
print(f"     ✓ Southeast lobule with {len(void_prototypes_lob3)} voids")

# R17 Far East
lob4_centroid = np.array([R17_Lobule_FarE.centroid.x, R17_Lobule_FarE.centroid.y])
rp17, rpr17 = make_rainbow_gradient_refs(lob4_centroid, core_r=13, green_r=43, yellow_r=72, red_r=95)
void_prototypes_lob4 = create_void_elements(R17_Lobule_FarE, n_voids=R17_N_VOIDS,
                                            void_radius_range=(R17_VOID_RADIUS_MIN, R17_VOID_RADIUS_MAX))
fovd_lob4 = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=lambda: FrameWideElement(
        frame_size=FRAME_WIDTH, tipical_cell_spacing=FOREGROUND_CELL_SPACING,
        rules=[ProbabilityNodeFieldRule(n_cell_types=N_CELL_TYPES, n_ref_points=len(rp17),
               reference_points=np.array(rp17), ref_probs=np.array(rpr17))]),
    other_elements=void_prototypes_lob4,
    elements_frequency=[R17_VOID_FREQUENCY] * len(void_prototypes_lob4),
    attempts_at_elements=R17_VOID_ATTEMPTS)
print(f"     ✓ Far east lobule with {len(void_prototypes_lob4)} voids")


# === REGION 18 (Connector East — Purple gradient) ===
print("   - Configuring dendritic connector on rightmost lobule...")
rp18, rpr18 = make_purple_gradient_refs(lobule4_center, core_r=25, mix_r=60, edge_r=95,
                                        core_n=6, mix_n=8, edge_n=8)

fovd_conn = FOVDistribution(
    frame_size=FRAME_WIDTH,
    background_element=lambda: FrameWideElement(
        frame_size=FRAME_WIDTH, tipical_cell_spacing=FOREGROUND_CELL_SPACING,
        rules=[ProbabilityNodeFieldRule(n_cell_types=N_CELL_TYPES, n_ref_points=len(rp18),
               reference_points=np.array(rp18), ref_probs=np.array(rpr18))]),
    other_elements=[], elements_frequency=[], attempts_at_elements=0)
print("     ✓ Dendritic connector configured")


# ============================================================================
# ASSEMBLE TISSUE SLICE
# ============================================================================

print("\n4. Assembling TissueSlice with regional specifications...")

all_regions = [
    RegionSpec(name="Background",                polygon=R3_Background,       fovdist=fovd_background,    priority=R3_PRIORITY,  blend_band=R3_BLEND_BAND),
    RegionSpec(name="RightPapillaryMass",         polygon=R2_Right_Mass,       fovdist=fovd_right,         priority=R2_PRIORITY,  blend_band=R2_BLEND_BAND),
    RegionSpec(name="LeftCribriformMass",          polygon=R1_Left_Mass,        fovdist=fovd_left,          priority=R1_PRIORITY,  blend_band=R1_BLEND_BAND),
    RegionSpec(name="SatelliteLobule_TopRight",    polygon=R4_Satellite_TopRight, fovdist=fovd_sat_tr,      priority=R4_PRIORITY,  blend_band=R4_BLEND_BAND),
    RegionSpec(name="SatelliteDendrite_TopLeft",   polygon=R5_Satellite_TopLeft,  fovdist=fovd_sat_tl,      priority=R5_PRIORITY,  blend_band=R5_BLEND_BAND),
    RegionSpec(name="InvasiveRoot",                polygon=R6_Invasive_Root,    fovdist=fovd_root,          priority=R6_PRIORITY,  blend_band=R6_BLEND_BAND),
    RegionSpec(name="HybridNodule_BottomLeft",     polygon=R7_Hybrid_Nodule,    fovdist=fovd_hybrid,        priority=R7_PRIORITY,  blend_band=R7_BLEND_BAND),
    RegionSpec(name="HybridClone_TopLeft",         polygon=R8_Clone_TopLeft,    fovdist=fovd_clone_tl,      priority=R8_PRIORITY,  blend_band=R8_BLEND_BAND),
    RegionSpec(name="DendriticConnector_Bottom",   polygon=R9_Bridge_Bottom,    fovdist=fovd_bridge_bottom, priority=R9_PRIORITY,  blend_band=R9_BLEND_BAND),
    RegionSpec(name="HybridClone_Central",         polygon=R10_Clone_Central,   fovdist=fovd_clone_c,       priority=R10_PRIORITY, blend_band=R10_BLEND_BAND),
    RegionSpec(name="BlueSatellite_West1",         polygon=R11_Satellite_West1, fovdist=fovd_sat_w1,        priority=R11_PRIORITY, blend_band=R11_BLEND_BAND),
    RegionSpec(name="BlueSatellite_West2",         polygon=R12_Satellite_West2, fovdist=fovd_sat_w2,        priority=R12_PRIORITY, blend_band=R12_BLEND_BAND),
    RegionSpec(name="BlueSatellite_West3",         polygon=R13_Satellite_West3, fovdist=fovd_sat_w3,        priority=R13_PRIORITY, blend_band=R13_BLEND_BAND),
    RegionSpec(name="Lobule_Northeast",            polygon=R14_Lobule_NE,       fovdist=fovd_lob1,          priority=R14_PRIORITY, blend_band=R14_BLEND_BAND),
    RegionSpec(name="Lobule_East",                 polygon=R15_Lobule_E,        fovdist=fovd_lob2,          priority=R15_PRIORITY, blend_band=R15_BLEND_BAND),
    RegionSpec(name="Lobule_Southeast",            polygon=R16_Lobule_SE,       fovdist=fovd_lob3,          priority=R16_PRIORITY, blend_band=R16_BLEND_BAND),
    RegionSpec(name="Lobule_FarEast",              polygon=R17_Lobule_FarE,     fovdist=fovd_lob4,          priority=R17_PRIORITY, blend_band=R17_BLEND_BAND),
    RegionSpec(name="Connector_East",              polygon=R18_Connector_East,  fovdist=fovd_conn,          priority=R18_PRIORITY, blend_band=R18_BLEND_BAND),
]

tissue_slice = TissueSlice(frame_size=FRAME_WIDTH, regions=all_regions)
print("   ✓ TissueSlice assembled with 18 regions")


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
# SWITCH TO EXPRESSION SEED — everything below here changes with EXPRESSION_SEED
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
# VISUALISATIONS
# ============================================================================

print("\n10. Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(20, 12))

# Plot 1: Cell types
ax1 = axes[0]
for ct in range(N_CELL_TYPES):
    mask = cells_df['Class ID'] == ct
    if mask.sum() > 0:
        ax1.scatter(
            cells_df.loc[mask, 'X'], cells_df.loc[mask, 'Y'],
            c=CELL_TYPE_COLORS[ct], s=4, alpha=0.8,
            label=CELL_TYPE_NAMES[ct]
        )

ax1.set_xlim(0, FRAME_WIDTH);  ax1.set_ylim(0, FRAME_HEIGHT)
ax1.set_aspect('equal')
ax1.set_title('Cell Type Distribution (Region-Based)', fontsize=16, fontweight='bold')
ax1.set_xlabel('X (µm)', fontsize=12);  ax1.set_ylabel('Y (µm)', fontsize=12)
ax1.legend(loc='upper right', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Plot 2: Regional boundaries
ax2 = axes[1]

if isinstance(R1_Left_Mass, Polygon):
    x, y = R1_Left_Mass.exterior.xy
    ax2.plot(x, y, 'b-', linewidth=2, label='Left mass boundary')
    ax2.fill(x, y, color='blue', alpha=0.1)

if isinstance(R2_Right_Mass, Polygon):
    x, y = R2_Right_Mass.exterior.xy
    ax2.plot(x, y, 'purple', linewidth=2, label='Right mass boundary')
    ax2.fill(x, y, color='purple', alpha=0.1)
elif isinstance(R2_Right_Mass, MultiPolygon):
    for geom in R2_Right_Mass.geoms:
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax2.plot(x, y, 'purple', linewidth=2)
            ax2.fill(x, y, color='purple', alpha=0.1)

for ct in range(N_CELL_TYPES):
    mask = cells_df['Class ID'] == ct
    if mask.sum() > 0:
        ax2.scatter(cells_df.loc[mask, 'X'], cells_df.loc[mask, 'Y'],
                    c=CELL_TYPE_COLORS[ct], s=1, alpha=0.3)

ax2.set_xlim(0, FRAME_WIDTH);  ax2.set_ylim(0, FRAME_HEIGHT)
ax2.set_aspect('equal')
ax2.set_title('Regional Architecture', fontsize=16, fontweight='bold')
ax2.set_xlabel('X (µm)', fontsize=12);  ax2.set_ylabel('Y (µm)', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/tissue_visualization.png", dpi=300, bbox_inches='tight')
print(f"   ✓ Saved tissue_visualization.png")


# High-res cell-only plot
fig2, ax = plt.subplots(figsize=(16, 12))
for ct in range(N_CELL_TYPES):
    mask = cells_df['Class ID'] == ct
    if mask.sum() > 0:
        ax.scatter(cells_df.loc[mask, 'X'], cells_df.loc[mask, 'Y'],
                   c=CELL_TYPE_COLORS[ct], s=2, alpha=0.9, rasterized=True)

ax.set_xlim(0, FRAME_WIDTH);  ax.set_ylim(0, FRAME_HEIGHT)
ax.set_aspect('equal')
ax.set_facecolor('black')
ax.set_title('Region-Based Hierarchical Tissue', fontsize=18, fontweight='bold', color='white')
ax.set_xlabel('X (µm)', fontsize=14, color='white')
ax.set_ylabel('Y (µm)', fontsize=14, color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/tissue_cells_highres.png", dpi=300, bbox_inches='tight', facecolor='black')
print(f"   ✓ Saved tissue_cells_highres.png")

plt.close('all')


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("✓ CLONAL HETEROGENEITY TISSUE SIMULATION COMPLETE!")
print("=" * 70)
print(f"\nOutput location: {OUTPUT_FOLDER}/")
print(f"  - cells.csv: {len(cells_df)} cells with ground truth")
print(f"  - dots.csv: {len(dots_df):,} transcript positions")
print(f"  - tissue_visualization.png: Dual-panel visualization")
print(f"  - tissue_cells_highres.png: High-resolution cell plot")
print(f"  - data.h5ad: AnnData format (if anndata installed)")
print(f"\n📊 Parameters:")
print(f"  - Total regions: 18")
print(f"  - Cell types: {N_CELL_TYPES}")
print(f"  - Genes: {N_GENES}")
print(f"  - Frame size: {FRAME_WIDTH}×{FRAME_HEIGHT} µm")
print(f"  - Geometry seed: {GEOMETRY_SEED} (cell placement — fixed across sims)")
print(f"  - Expression seed: {EXPRESSION_SEED} (gene profiles — vary across sims)")
print(f"  - Background cell spacing: {BACKGROUND_CELL_SPACING} µm")
print(f"  - Foreground cell spacing: {FOREGROUND_CELL_SPACING} µm")
print("=" * 70)
