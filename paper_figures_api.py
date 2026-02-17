"""
Paper-quality visualizations of PointillSim structures and rules.

Output figures:
  1. structures_gray_1x6    – 6 structures in dark gray
  2. rules_inferno_1x5      – 5 rules on same circular structure, inferno
  3. structures_types_1x6   – 6 structures realized with ellipses
  4. composite_tissue       – compact tissue with all structures on stroma
  5. zoom_transcripts       – close-up: dark gray cells + gene-colored dots
  6. pgm_generative_model   – probabilistic graphical model of observation
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from pathlib import Path

np.random.seed(42)

# ── PointillSim imports ──────────────────────────────────────────────────────
from pointillsim import (
    TissueCellTypes, CellTypesProperties, HybISS_Setup,
    HistologicalElement, FrameWideElement, VacuolatedStructure,
    FOVDistribution,
    RandomCellTypeRule, SingleTypeRule, MixOfNCellTypesRule,
    ProbabilityNodeFieldRule,
    DistanceBasedRule, CompositeRule, LayerRule, GradientRule,
    IdentityTransfer,
)
from pointillsim.elements.structures import (
    LinearLumenStructure, BranchingStructure, LayeredElement,
    ClusterElement, FibrillarStructure,
)

# ── Output ───────────────────────────────────────────────────────────────────
OUT = Path("paper_figures")
OUT.mkdir(exist_ok=True)

# ── High-contrast palette ────────────────────────────────────────────────────
PALETTE = [
    "#D62728",  # 0  Red        – background stroma
    "#1F77B4",  # 1  Blue       – lumen
    "#2CA02C",  # 2  Green      – branching A
    "#9467BD",  # 3  Purple     – branching B
    "#FF7F0E",  # 4  Orange     – fibrillar A
    "#17BECF",  # 5  Cyan       – fibrillar B
    "#E377C2",  # 6  Pink       – cluster A
    "#BCBD22",  # 7  Olive      – cluster B
    "#8C564B",  # 8  Brown      – vacuolated A
    "#FFD92F",  # 9  Gold       – vacuolated B
]

DARK_GRAY      = "#4a4a4a"
DARK_GRAY_RGB  = (0.29, 0.29, 0.29)
DARK_GRAY_EDGE = (0.22, 0.22, 0.22, 0.35)
BOUNDARY_COLOR = "#888888"
LAYER_LINE_COL = "#666666"
DPI = 300
N = 10  # cell types
FS = 500
SP = 6

INFERNO = plt.cm.inferno

# ── Per-type cell morphology: (major_axis, minor_axis, base_angle_degrees) ──
# Sized to avoid overlap at spacing ~6
CELL_MORPHOLOGY = [
    (5.5, 3.5, 0),     # 0  Red:    medium ellipse, horizontal
    (4.0, 4.0, 0),     # 1  Blue:   circular
    (6.5, 2.8, 40),    # 2  Green:  elongated, tilted
    (5.0, 5.0, 0),     # 3  Purple: large circular
    (7.0, 2.5, -25),   # 4  Orange: very elongated, tilted
    (3.5, 3.5, 0),     # 5  Cyan:   small circular
    (6.0, 3.0, 65),    # 6  Pink:   elongated, steep tilt
    (4.8, 3.0, 15),    # 7  Olive:  medium ellipse, slight tilt
    (5.2, 3.5, -40),   # 8  Brown:  medium ellipse, reverse tilt
    (4.5, 4.5, 0),     # 9  Gold:   circular medium
]
ANGLE_JITTER = 20


# ═════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _clean(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")


def _save(fig, name):
    kw = dict(dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", **kw)
    fig.savefig(OUT / f"{name}.pdf", **kw)
    plt.close(fig)
    print(f"  saved {name}")


def _lims(centroids, frac=0.08):
    xmn, xmx = centroids[:, 0].min(), centroids[:, 0].max()
    ymn, ymx = centroids[:, 1].min(), centroids[:, 1].max()
    dx = (xmx - xmn) * frac
    dy = (ymx - ymn) * frac
    return (xmn - dx, xmx + dx), (ymn - dy, ymx + dy)


def _apply_lims(ax, centroids, frac=0.08):
    xl, yl = _lims(centroids, frac)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)


def _draw_boundary(ax, polygon, hole=None, lumen=None, dashed=False):
    ls = "--" if dashed else "-"
    if polygon is not None:
        try:
            x, y = polygon.exterior.xy
            ax.plot(x, y, color=BOUNDARY_COLOR, linewidth=0.8, alpha=0.5, linestyle=ls)
        except Exception:
            pass
    inner = hole if hole is not None else lumen
    if inner is not None:
        try:
            x, y = inner.exterior.xy
            ax.plot(x, y, color=BOUNDARY_COLOR, linewidth=0.8, alpha=0.5, linestyle=ls)
        except Exception:
            pass


def _draw_layer_lines(ax, layer_boundaries, centroids, frac=0.02):
    xl, _ = _lims(centroids, frac)
    for yb in layer_boundaries:
        ax.plot(xl, [yb, yb], color=LAYER_LINE_COL, linewidth=0.7,
                alpha=0.55, linestyle="--")


def _circles(centroids, r):
    return [Circle((x, y), radius=r) for x, y in centroids]


def _ellipses(centroids, class_instance, scale=1.0):
    patches = []
    n = len(centroids)
    angle_noise = np.random.uniform(-ANGLE_JITTER, ANGLE_JITTER, size=n)
    for i, (x, y) in enumerate(centroids):
        ctype = int(class_instance[i]) % len(CELL_MORPHOLOGY)
        major, minor, base_angle = CELL_MORPHOLOGY[ctype]
        angle = base_angle + angle_noise[i]
        patches.append(Ellipse((x, y), width=major * scale, height=minor * scale,
                               angle=angle))
    return patches


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL DRAWING FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def draw_gray(ax, elem, cs, polygon=None, hole=None, lumen=None,
              dashed=False, frac=0.08, layer_boundaries=None):
    centroids = elem.cell_centroids
    circles = _circles(centroids, cs * 0.48)
    col = PatchCollection(circles, alpha=0.82)
    col.set_facecolors([DARK_GRAY_RGB] * len(circles))
    col.set_edgecolors([DARK_GRAY_EDGE] * len(circles))
    col.set_linewidths(0.3)
    ax.add_collection(col)
    _draw_boundary(ax, polygon, hole, lumen, dashed)
    if layer_boundaries is not None:
        _draw_layer_lines(ax, layer_boundaries, centroids, frac)
    _apply_lims(ax, centroids, frac)
    _clean(ax)


def draw_inferno(ax, elem, cs, polygon=None, hole=None, lumen=None,
                 dashed=False, frac=0.08):
    centroids = elem.cell_centroids
    probs = elem.cell_probabilities
    max_probs = probs.max(axis=1)
    norm = Normalize(vmin=max_probs.min() - 0.02, vmax=max_probs.max())
    colors = INFERNO(norm(max_probs))
    circles = _circles(centroids, cs * 0.48)
    col = PatchCollection(circles, alpha=0.88)
    col.set_facecolors(colors)
    col.set_edgecolors([(0, 0, 0, 0.1)] * len(circles))
    col.set_linewidths(0.2)
    ax.add_collection(col)
    _draw_boundary(ax, polygon, hole, lumen, dashed)
    _apply_lims(ax, centroids, frac)
    _clean(ax)


def draw_types(ax, elem, cs, polygon=None, hole=None, lumen=None,
               dashed=False, frac=0.08, layer_boundaries=None):
    centroids = elem.cell_centroids
    class_instance = elem.class_instance
    colors = [PALETTE[int(c) % len(PALETTE)] for c in class_instance]
    ellipses = _ellipses(centroids, class_instance, scale=1.0)
    col = PatchCollection(ellipses, alpha=0.88, match_original=False)
    col.set_facecolors(colors)
    col.set_edgecolors([(0, 0, 0, 0.12)] * len(ellipses))
    col.set_linewidths(0.25)
    ax.add_collection(col)
    _draw_boundary(ax, polygon, hole, lumen, dashed)
    if layer_boundaries is not None:
        _draw_layer_lines(ax, layer_boundaries, centroids, frac)
    _apply_lims(ax, centroids, frac)
    _clean(ax)


# ═════════════════════════════════════════════════════════════════════════════
#  GENERATE THE 6 STRUCTURES  (for Figures 1 & 3)
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("Generating 6 structures")
print("=" * 65)

structs = {}

print("\n1. BranchingStructure + MixOfNCellTypesRule")
np.random.seed(40)
e = BranchingStructure(
    frame_size=650,
    root_position=np.array([60, 325]), root_angle=0.0,
    root_radius=30, wall_thickness=0.45,
    branch_length=120, branch_length_decay=0.72,
    radius_decay=0.62, branching_angle=(0.35, 0.72),
    branching_probability=0.92, max_depth=4, min_radius=4,
    tipical_cell_spacing=4, curvature=0.1,
    rules=MixOfNCellTypesRule(n_cell_types=N, list_N=[3, 6],
                               proportions=[0.55, 0.45]),
).generate()
structs["branching"] = dict(elem=e, cs=4, polygon=e.polygon,
    hole=None, lumen=getattr(e, "lumen", None), dashed=False, frac=0.06,
    layer_boundaries=None)

print("2. LinearLumenStructure + SingleTypeRule")
np.random.seed(30)
e = LinearLumenStructure(
    frame_size=FS, length=400, outer_radius=65, wall_thickness=24,
    n_control_points=5, curvature=0.14,
    fixed_start=np.array([60, 260]), fixed_angle=0.15,
    tipical_cell_spacing=5, smoothing_iterations=2,
    rules=SingleTypeRule(n_cell_types=N, cell_type_ix=1),
).generate()
structs["lumen"] = dict(elem=e, cs=5, polygon=e.polygon,
    hole=None, lumen=e.lumen, dashed=False, frac=0.08, layer_boundaries=None)

print("3. LayeredElement + GradientRule")
np.random.seed(50)
LAYER_BOUNDS = [100, 200, 300, 400]
e = LayeredElement(
    frame_size=FS,
    layer_boundaries=LAYER_BOUNDS,
    tipical_cell_spacing=SP,
    rules=GradientRule(n_cell_types=N, start_type=0, end_type=4,
                       direction="vertical", transition_width=0.35),
).generate()
structs["layered"] = dict(elem=e, cs=SP, polygon=None,
    hole=None, lumen=None, dashed=False, frac=0.02,
    layer_boundaries=LAYER_BOUNDS)

print("4. ClusterElement + GradientRule horizontal")
np.random.seed(60)
e = ClusterElement(
    frame_size=FS, radius=170, n_vertices=(10, 14),
    fixed_center=np.array([250, 250]),
    density_profile="dense_center",
    tipical_cell_spacing=SP, smoothing_iterations=3,
    rules=GradientRule(n_cell_types=N, start_type=7, end_type=5,
                       direction="horizontal", transition_width=0.4),
).generate()
structs["cluster"] = dict(elem=e, cs=SP, polygon=e.polygon,
    hole=None, lumen=None, dashed=True, frac=0.08, layer_boundaries=None)

print("5. VacuolatedStructure + ProbabilityNodeFieldRule")
np.random.seed(20)
e = VacuolatedStructure(
    frame_size=FS, n_vertices=(10, 15), scale=190,
    fixed_center=np.array([[250, 250]]),
    tipical_cell_spacing=SP, hole_scale_factor=0.45, smoothing_iterations=3,
    rules=ProbabilityNodeFieldRule(n_cell_types=N, n_ref_points=5, alpha=0.08),
).generate()
structs["vacuolated"] = dict(elem=e, cs=SP, polygon=e.polygon,
    hole=e.hole, lumen=None, dashed=True, frac=0.08, layer_boundaries=None)

print("6. FibrillarStructure + GradientRule diagonal")
np.random.seed(80)
e = FibrillarStructure(
    frame_size=FS, n_fibers=6, fiber_width=30, fiber_spacing=42,
    orientation=np.pi / 5, waviness=0.09,
    fixed_center=np.array([250, 250]),
    tipical_cell_spacing=SP,
    rules=GradientRule(n_cell_types=N, start_type=3, end_type=6,
                       direction="diagonal", transition_width=0.4),
).generate()
structs["fibrillar"] = dict(elem=e, cs=SP, polygon=e.polygon,
    hole=None, lumen=None, dashed=False, frac=0.06, layer_boundaries=None)

ORDER = ["branching", "lumen", "layered", "cluster", "vacuolated", "fibrillar"]


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1: 1×6 vertical – dark gray structures
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Figure 1: structures in dark gray")
print("=" * 65)

fig, axes = plt.subplots(6, 1, figsize=(5, 30))
fig.patch.set_alpha(0)
for i, key in enumerate(ORDER):
    s = structs[key]
    draw_gray(axes[i], s["elem"], s["cs"], s["polygon"], s["hole"],
              s["lumen"], s["dashed"], s["frac"], s["layer_boundaries"])
plt.subplots_adjust(hspace=0.03)
_save(fig, "structures_gray_1x6")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2: 1×5 vertical – 5 rules on same circular structure (inferno)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Figure 2: 5 rules on identical circular structure")
print("=" * 65)

RULE_FS = 500
RULE_SCALE = 180
RULE_CENTER = np.array([[250, 250]])
RULE_SP = 6

rules_list = [
    ("SingleType",
     SingleTypeRule(n_cell_types=N, cell_type_ix=2)),
    ("Random",
     RandomCellTypeRule(n_cell_types=N)),
    ("GradientLeftRight",
     GradientRule(n_cell_types=N, start_type=1, end_type=5,
                  direction="horizontal", transition_center=0.25,
                  transition_width=0.5, sharpness=2.0)),
    ("DistanceBoundary",
     DistanceBasedRule(n_cell_types=N, reference="boundary",
                       inner_type=3, outer_type=7,
                       transition_width=60, sharpness=3.0)),
    ("ProbabilityField",
     ProbabilityNodeFieldRule(n_cell_types=N, n_ref_points=5, alpha=0.06)),
]

rule_elems = []
for name, rule in rules_list:
    np.random.seed(99)
    e = HistologicalElement(
        frame_size=RULE_FS, n_vertices=(10, 14), scale=RULE_SCALE,
        fixed_center=RULE_CENTER,
        tipical_cell_spacing=RULE_SP, smoothing_iterations=3,
        rules=rule,
    ).generate()
    rule_elems.append((name, e))
    print(f"  {name}: {e.cell_centroids.shape[0]} cells")

fig, axes = plt.subplots(5, 1, figsize=(5, 25))
fig.patch.set_alpha(0)
for i, (name, e) in enumerate(rule_elems):
    draw_inferno(axes[i], e, RULE_SP, polygon=e.polygon, dashed=True, frac=0.08)
plt.subplots_adjust(hspace=0.03)
_save(fig, "rules_inferno_1x5")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: 1×6 vertical – realized cell types as ellipses
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Figure 3: realized cell types (ellipses, high-contrast)")
print("=" * 65)

fig, axes = plt.subplots(6, 1, figsize=(5, 30))
fig.patch.set_alpha(0)
for i, key in enumerate(ORDER):
    s = structs[key]
    np.random.seed(i * 100 + 7)
    draw_types(axes[i], s["elem"], s["cs"], s["polygon"], s["hole"],
               s["lumen"], s["dashed"], s["frac"], s["layer_boundaries"])
plt.subplots_adjust(hspace=0.03)
_save(fig, "structures_types_1x6")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 4: COMPACT COMPOSITE TISSUE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Figure 4: Compact composite tissue")
print("=" * 65)

COMP_FS = 700
np.random.seed(42)

bg = lambda: FrameWideElement(
    frame_size=COMP_FS, tipical_cell_spacing=25,
    rules=SingleTypeRule(n_cell_types=N, cell_type_ix=0),
)

def _branching():
    return BranchingStructure(
        frame_size=COMP_FS,
        root_radius=30 + np.random.uniform(-5, 5),
        wall_thickness=0.45,
        branch_length=120, branch_length_decay=0.72,
        radius_decay=0.62, branching_angle=(0.35, 0.72),
        branching_probability=0.88, max_depth=3, min_radius=5,
        tipical_cell_spacing=5, curvature=0.1,
        rules=MixOfNCellTypesRule(n_cell_types=N, list_N=[2, 3],
                                   proportions=[0.55, 0.45]),
    )

def _lumen():
    return LinearLumenStructure(
        frame_size=COMP_FS,
        length=300 + np.random.uniform(-60, 60),
        outer_radius=35, wall_thickness=14,
        n_control_points=4, curvature=0.12,
        tipical_cell_spacing=5, smoothing_iterations=2,
        rules=SingleTypeRule(n_cell_types=N, cell_type_ix=1),
    )

def _cluster():
    return ClusterElement(
        frame_size=COMP_FS,
        radius=70 + np.random.uniform(-10, 10),
        density_profile="dense_center",
        tipical_cell_spacing=6, smoothing_iterations=2,
        rules=DistanceBasedRule(n_cell_types=N, reference="center",
                                inner_type=6, outer_type=7,
                                transition_width=30, sharpness=1.5),
    )

def _vacuolated():
    return VacuolatedStructure(
        frame_size=COMP_FS,
        scale=90 + np.random.uniform(-10, 10),
        hole_scale_factor=0.45, smoothing_iterations=3,
        tipical_cell_spacing=6,
        rules=DistanceBasedRule(n_cell_types=N, reference="center",
                                inner_type=8, outer_type=9,
                                transition_width=25, sharpness=1.5),
    )

def _fibrillar():
    return FibrillarStructure(
        frame_size=COMP_FS,
        n_fibers=4, fiber_width=22, fiber_spacing=30,
        orientation=np.random.uniform(0, np.pi),
        waviness=0.08,
        tipical_cell_spacing=6,
        rules=GradientRule(n_cell_types=N, start_type=4, end_type=5,
                           direction="diagonal", transition_width=0.4),
    )

fov_dist = FOVDistribution(
    frame_size=COMP_FS, background_element=bg,
    other_elements=[_branching, _lumen, _cluster, _vacuolated, _fibrillar],
    elements_frequency=[1.0, 1.0, 1.0, 1.0, 0.4],
    attempts_at_elements=[5, 6, 9, 10, 2],
)

fov = fov_dist.generate_fov()
print(f"  Composite FOV: {fov.n_cells} cells")

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
fig.patch.set_alpha(0)
cents = fov.cell_centroids
cls = fov.class_instance
np.random.seed(777)
colors = [PALETTE[int(c) % len(PALETTE)] for c in cls]
ellipses = _ellipses(cents, cls, scale=1.0)
col = PatchCollection(ellipses, alpha=0.85, match_original=False)
col.set_facecolors(colors)
col.set_edgecolors([(0, 0, 0, 0.08)] * len(ellipses))
col.set_linewidths(0.12)
ax.add_collection(col)
_apply_lims(ax, cents, 0.01)
_clean(ax)
_save(fig, "composite_tissue_compact")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 5: ZOOM-IN FROM COMPOSITE TISSUE WITH TRANSCRIPT DOTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Figure 5: Zoom-in from composite tissue with transcript dots")
print("=" * 65)

import pandas as pd

np.random.seed(55)
N_GENES_ZOOM = 10

# Create tissue expression for N=10 cell types, 10 genes
tissue_comp = TissueCellTypes()
tissue_comp.generate_types_and_markers(
    n_genes=N_GENES_ZOOM, n_cell_types=N,
    expected_level=15.0, concentration=0.82,
)

# Apply morphological properties to composite FOV
# Sizes derived from CELL_MORPHOLOGY: r = (major+minor)/4, a = 2*minor/(major+minor)
# so that 2*cell_major_axis ≈ visual major, 2*cell_minor_axis ≈ visual minor
comp_props = CellTypesProperties(
    n_cell_types=N,
    sizes=[s * 3 for s in [2.25, 2.0, 2.33, 2.5, 2.38, 1.75, 2.25, 1.95, 2.18, 2.25]],
    size_variation=0.15,
    anisotropy=[0.78, 1.0, 0.60, 1.0, 0.53, 1.0, 0.67, 0.77, 0.80, 1.0],
    anisotropy_variation=0.02,
    relative_rna_concentration=[1.0, 1.2, 1.1, 1.3, 0.9, 1.0, 1.2, 1.1, 1.0, 1.1],
    rna_concentration_variation=0.1,
)
comp_props.apply(fov)

# Generate transcript dots from composite tissue
comp_hybiss = HybISS_Setup(
    tissue=tissue_comp,
    genes_sensitivities=2.5,
    genes_sensitivities_variation=0.3,
    transfer_function=IdentityTransfer(),
)
comp_hybiss.observe_dots(fov)
comp_dots = comp_hybiss.make_pandas_df()
print(f"  Composite tissue: {fov.n_cells} cells, {len(comp_dots)} dots")

# Add dispersion to dots – some scatter outside cell boundaries (unassigned RNA)
comp_dots["x"] += np.random.normal(0, 0.1, len(comp_dots))
comp_dots["y"] += np.random.normal(0, 0.1, len(comp_dots))

# Add background unassigned RNA dots (1% of total)
n_bg_dots = int(len(comp_dots) * 0.01)
bg_df = pd.DataFrame({
    "x": np.random.uniform(0, COMP_FS, n_bg_dots),
    "y": np.random.uniform(0, COMP_FS, n_bg_dots),
    "gene": np.random.choice(tissue_comp.gene_names, n_bg_dots),
    "cell": -1,
})
comp_dots = pd.concat([comp_dots, bg_df], ignore_index=True)

# Find densest region with multiple structure types
crop_r = 50
best_score, best_cx, best_cy = 0, COMP_FS // 2, COMP_FS // 2
comp_cents = fov.cell_centroids
comp_cls = fov.class_instance
for cx_try in range(crop_r + 10, COMP_FS - crop_r - 10, 15):
    for cy_try in range(crop_r + 10, COMP_FS - crop_r - 10, 15):
        m = (
            (comp_cents[:, 0] >= cx_try - crop_r) &
            (comp_cents[:, 0] <= cx_try + crop_r) &
            (comp_cents[:, 1] >= cy_try - crop_r) &
            (comp_cents[:, 1] <= cy_try + crop_r)
        )
        cnt = m.sum()
        if cnt > 20:
            types_in = comp_cls[m]
            n_types = len(np.unique(types_in))
            # Penalize single-type dominance: entropy-like score
            _, counts = np.unique(types_in, return_counts=True)
            max_frac = counts.max() / cnt
            diversity = n_types * (1.0 - max_frac)
            score = cnt * diversity
            if score > best_score:
                best_score = score
                best_cx, best_cy = cx_try, cy_try

cx, cy = best_cx+15, best_cy+20
xlo, xhi = cx - crop_r, cx + crop_r
ylo, yhi = cy - crop_r, cy + crop_r
crop_mask = (
    (comp_cents[:, 0] >= xlo) & (comp_cents[:, 0] <= xhi) &
    (comp_cents[:, 1] >= ylo) & (comp_cents[:, 1] <= yhi)
)
n_crop_cells = crop_mask.sum()
n_crop_types = len(np.unique(comp_cls[crop_mask]))
print(f"  Zoom region: center=({cx},{cy}), {n_crop_cells} cells, {n_crop_types} types")

# Gene color map – distinct from cell type PALETTE
gene_names = sorted(comp_dots["gene"].unique())
GENE_PALETTE = [
    "#000000",  # black
    "#800000",  # maroon
    "#006400",  # dark green
    "#00008B",  # dark blue
    "#FF00FF",  # magenta
    "#008080",  # teal
    "#FF4500",  # orange-red
    "#6A5ACD",  # slate blue
    "#556B2F",  # dark olive
    "#DC143C",  # crimson
]
gene_colors = {g: GENE_PALETTE[i % len(GENE_PALETTE)] for i, g in enumerate(gene_names)}

# ── Figure 5a: composite tissue overview with dashed bounding box ────────
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
fig.patch.set_alpha(0)

np.random.seed(778)
colors_full = [PALETTE[int(c) % len(PALETTE)] for c in comp_cls]
ellipses_full = _ellipses(comp_cents, comp_cls, scale=1.0)
col_full = PatchCollection(ellipses_full, alpha=0.85, match_original=False)
col_full.set_facecolors(colors_full)
col_full.set_edgecolors([(0, 0, 0, 0.5)] * len(ellipses_full))
col_full.set_linewidths(0.3)
ax.add_collection(col_full)

bbox_rect = plt.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                           fill=False, edgecolor="black", linewidth=4.0,
                           linestyle="--", zorder=10)
ax.add_patch(bbox_rect)

_apply_lims(ax, comp_cents, 0.01)
_clean(ax)
_save(fig, "zoom_tissue_overview")

# ── Figure 5b: zoomed crop with faded cell type colors + transcript dots ─
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.patch.set_alpha(0)

# Draw cell ellipses in the crop region (faded colors)
margin = 10
cell_mask = (
    (comp_cents[:, 0] >= xlo - margin) & (comp_cents[:, 0] <= xhi + margin) &
    (comp_cents[:, 1] >= ylo - margin) & (comp_cents[:, 1] <= yhi + margin)
)
np.random.seed(779)
crop_patches = []
crop_colors = []
crop_angle_noise = np.random.uniform(-ANGLE_JITTER, ANGLE_JITTER, cell_mask.sum())
for j, i in enumerate(np.where(cell_mask)[0]):
    ctype = int(comp_cls[i]) % len(CELL_MORPHOLOGY)
    major, minor, base_angle = CELL_MORPHOLOGY[ctype]
    major *= 1.2
    minor *= 1.2
    angle = base_angle + crop_angle_noise[j]
    crop_patches.append(Ellipse(
        (comp_cents[i, 0], comp_cents[i, 1]),
        width=major, height=minor, angle=angle,
    ))
    crop_colors.append(PALETTE[int(comp_cls[i]) % len(PALETTE)])

cell_col = PatchCollection(crop_patches, alpha=0.5, match_original=False)
cell_col.set_facecolors(crop_colors)
cell_col.set_edgecolors([(0, 0, 0, 0.7)] * len(crop_patches))
cell_col.set_linewidths(0.6)
ax.add_collection(cell_col)

# Draw transcript dots colored by gene (full alpha, on top)
dot_mask = (
    (comp_dots["x"] >= xlo) & (comp_dots["x"] <= xhi) &
    (comp_dots["y"] >= ylo) & (comp_dots["y"] <= yhi)
)
n_zoom_dots = dot_mask.sum()
for gene in gene_names:
    gm = dot_mask & (comp_dots["gene"] == gene)
    if gm.sum() > 0:
        ax.scatter(
            comp_dots.loc[gm, "x"], comp_dots.loc[gm, "y"],
            c=[gene_colors[gene]], s=8, alpha=1.0, zorder=5,
            edgecolors="none",
        )
print(f"  Zoom dots in region: {n_zoom_dots}")

ax.set_xlim(xlo, xhi)
ax.set_ylim(ylo, yhi)
_clean(ax)
_save(fig, "zoom_transcripts")

# Gene legend
fig_leg, ax_leg = plt.subplots(figsize=(2.5, 3))
fig_leg.patch.set_alpha(0)
from matplotlib.patches import Patch
handles = [Patch(facecolor=gene_colors[g], edgecolor="none", label=g)
           for g in gene_names]
ax_leg.legend(handles=handles, loc="center", frameon=False, fontsize=8,
              handletextpad=0.4, labelspacing=0.4, ncol=1)
ax_leg.set_axis_off()
_save(fig_leg, "legend_genes")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 6: PGM – PROBABILISTIC GRAPHICAL MODEL
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Figure 6: PGM of generative model")
print("=" * 65)


def _pgm_node(ax, x, y, text, observed=False, fixed=False, size=0.42):
    """Draw a PGM node (circle)."""
    fc = "#d9d9d9" if observed else "white"
    if fixed:
        fc = "#e8e8e8"
    circ = plt.Circle((x, y), size, fc=fc, ec="black", lw=1.2, zorder=3)
    ax.add_patch(circ)
    ax.text(x, y, text, ha="center", va="center", fontsize=11, zorder=4,
            fontstyle="italic" if not fixed else "normal")


def _pgm_arrow(ax, x1, y1, x2, y2, node_r=0.42):
    """Draw a directed arrow between nodes."""
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / dist, dy / dist
    ax.annotate("",
        xy=(x2 - ux * node_r, y2 - uy * node_r),
        xytext=(x1 + ux * node_r, y1 + uy * node_r),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
        zorder=2)


def _pgm_plate(ax, x, y, w, h, label):
    """Draw a plate (rectangle)."""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          fc="none", ec="#555555", lw=1.0, linestyle="-", zorder=1)
    ax.add_patch(rect)
    ax.text(x + w - 0.15, y + 0.18, label, ha="right", va="bottom",
            fontsize=10, color="#555555")


fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_alpha(0)

# Node positions (x, y)
# Top row: hyperparameters / fixed
pos = {
    "E":      (2.0, 7.0),   # Expression matrix
    "sg":     (5.0, 7.0),   # Gene sensitivity
    "pi_c":   (1.0, 5.0),   # Cell type prob vector
    "z_c":    (3.0, 5.0),   # Realized type
    "r_c":    (5.0, 5.0),   # RNA concentration
    "morph":  (7.0, 5.0),   # Cell morphology (major, minor, theta)
    "lam":    (3.5, 3.0),   # Expected count
    "n_cg":   (3.5, 1.2),   # Observed count
    "xy_d":   (6.5, 1.2),   # Dot position
}

# Draw plates
_pgm_plate(ax, 0.2, 0.3, 7.5, 5.6, "$c = 1, \\ldots, C$")   # cell plate
_pgm_plate(ax, 2.5, 0.5, 5.0, 3.3, "$g = 1, \\ldots, G$")    # gene plate

# Draw nodes
_pgm_node(ax, *pos["E"], "$\\mathbf{E}$", fixed=True)
_pgm_node(ax, *pos["sg"], "$s_g$", fixed=False)
_pgm_node(ax, *pos["pi_c"], "$\\pi_c$", fixed=False)
_pgm_node(ax, *pos["z_c"], "$z_c$", fixed=False)
_pgm_node(ax, *pos["r_c"], "$r_c$", fixed=False)
_pgm_node(ax, *pos["morph"], "$\\phi_c$", fixed=False)
_pgm_node(ax, *pos["lam"], "$\\lambda_{cg}$", fixed=False)
_pgm_node(ax, *pos["n_cg"], "$n_{cg}$", observed=True)
_pgm_node(ax, *pos["xy_d"], "$\\mathbf{x}_d$", observed=True)

# Draw arrows
_pgm_arrow(ax, *pos["E"], *pos["lam"])
_pgm_arrow(ax, *pos["sg"], *pos["lam"])
_pgm_arrow(ax, *pos["pi_c"], *pos["z_c"])
_pgm_arrow(ax, *pos["z_c"], *pos["lam"])
_pgm_arrow(ax, *pos["r_c"], *pos["lam"])
_pgm_arrow(ax, *pos["lam"], *pos["n_cg"])
_pgm_arrow(ax, *pos["n_cg"], *pos["xy_d"])
_pgm_arrow(ax, *pos["morph"], *pos["xy_d"])

# Distribution annotations (next to nodes)
ann_kw = dict(fontsize=8, color="#444444", ha="left", va="center")
ax.text(pos["pi_c"][0] + 0.55, pos["pi_c"][1] + 0.25,
        "Rules", **ann_kw)
ax.text(pos["z_c"][0] + 0.55, pos["z_c"][1] + 0.25,
        "$\\sim \\mathrm{Cat}(\\pi_c)$", **ann_kw)
ax.text(pos["r_c"][0] + 0.55, pos["r_c"][1] + 0.25,
        "$\\sim \\mathrm{LogN}(\\mu_r, \\sigma_r)$", **ann_kw)
ax.text(pos["sg"][0] + 0.55, pos["sg"][1] + 0.25,
        "$\\sim \\mathrm{LogN}(\\mu_s, \\sigma_s)$", **ann_kw)
ax.text(pos["lam"][0] + 0.55, pos["lam"][1] + 0.25,
        "$= r_c \\cdot s_g \\cdot E_{g,z_c}$", **ann_kw)
ax.text(pos["n_cg"][0] + 0.55, pos["n_cg"][1] + 0.25,
        "$\\sim \\mathrm{Pois}(\\lambda_{cg})$", **ann_kw)
ax.text(pos["xy_d"][0] + 0.55, pos["xy_d"][1] + 0.25,
        "$\\sim \\mathrm{Cell}(\\phi_c)$", **ann_kw)
ax.text(pos["morph"][0] + 0.55, pos["morph"][1] + 0.25,
        "$(a_c, b_c, \\theta_c)$", **ann_kw)

ax.set_xlim(-0.3, 9.0)
ax.set_ylim(-0.2, 8.0)
ax.set_aspect("equal")
ax.set_axis_off()

_save(fig, "pgm_generative_model")


# ═════════════════════════════════════════════════════════════════════════════
#  LEGENDS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Generating legends")
print("=" * 65)

fig, ax = plt.subplots(figsize=(3.0, 2.8))
fig.patch.set_alpha(0)
handles = [Patch(facecolor=PALETTE[i], edgecolor="none", label=f"Type {i}")
           for i in range(N)]
ax.legend(handles=handles, loc="center", frameon=False, fontsize=9,
          handletextpad=0.5, labelspacing=0.5)
ax.set_axis_off()
_save(fig, "legend_cell_types")

fig, ax = plt.subplots(figsize=(4, 0.5))
fig.patch.set_alpha(0)
grad = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(INFERNO(grad.flatten()).reshape(1, 256, 4), aspect="auto",
          extent=[0, 1, 0, 1])
ax.set_yticks([])
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels(["Low", "", "High"], fontsize=8)
ax.tick_params(length=2, pad=2)
for sp in ax.spines.values():
    sp.set_visible(False)
_save(fig, "legend_inferno_gradient")


print("\n" + "=" * 65)
print(f"All figures saved to  {OUT.resolve()}/")
print("=" * 65)
