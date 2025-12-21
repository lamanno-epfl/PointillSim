#!/usr/bin/env python
"""Generate sample datasets for PointillSim.

This script creates pre-generated sample datasets that are bundled with the package
for easy loading via `from pointillsim.data import load_sample`.
"""

import sys
from pathlib import Path
import numpy as np

# Add the parent directory to path so we can import pointillsim
sys.path.insert(0, str(Path(__file__).parent.parent))

from pointillsim import (
    TissueCellTypes,
    CellTypesProperties,
    HybISS_Setup,
    FOVDistribution,
    FrameWideElement,
    VacuolatedStructure,
    RandomCellTypeRule,
    MixOfNCellTypesRule,
    SingleTypeRule,
    DistanceBasedRule,
    ProbabilityNodeFieldRule,
)


def save_sample(
    name: str,
    fov,
    tissue: TissueCellTypes,
    hybiss: HybISS_Setup,
    description: str = "",
):
    """Save a sample dataset as compressed NPZ."""
    output_dir = Path(__file__).parent.parent / "pointillsim" / "data"
    output_dir.mkdir(exist_ok=True)

    # Get dots DataFrame
    dots_df = hybiss.make_pandas_df()

    # Prepare data dict
    data = {
        # FOV data
        "cell_centroids": fov.cell_centroids,
        "cell_probabilities": fov.cell_probabilities,
        "class_instance_one_hot": fov.class_instance_one_hot,
        # Morphology (if available)
        "cell_minor_axis": getattr(fov, "cell_minor_axis", np.array([])),
        "cell_major_axis": getattr(fov, "cell_major_axis", np.array([])),
        "cell_rotation": getattr(fov, "cell_rotation", np.array([])),
        "cell_rna_concentration": getattr(fov, "cell_rna_concentration", np.array([])),
        # Tissue data
        "expression_matrix": tissue.gene_expression_by_type,
        "gene_names": np.array(tissue.gene_names, dtype=object),
        "cell_type_names": np.array(tissue.cell_type_names, dtype=object),
        # Dots data
        "dots_x": dots_df["x"].values,
        "dots_y": dots_df["y"].values,
        "dots_gene": dots_df["gene"].values.astype(object),
        "dots_cell": dots_df["cell"].values,
        # Metadata
        "description": np.array(description, dtype=object),
    }

    filepath = output_dir / f"{name}.npz"
    np.savez_compressed(filepath, **data)
    print(f"Saved {name} to {filepath}")
    print(f"  - Cells: {len(fov.cell_centroids)}")
    print(f"  - Dots: {len(dots_df)}")
    print(f"  - Genes: {len(tissue.gene_names)}")
    print(f"  - Cell types: {len(tissue.cell_type_names)}")


def generate_simple_fov():
    """Generate a simple FOV with random cell type distribution."""
    print("\nGenerating simple_fov...")
    np.random.seed(42)

    # Create tissue with expression profiles
    tissue = TissueCellTypes()
    tissue.generate_types_and_markers(n_genes=50, n_cell_types=8)

    # Create cell properties
    cell_props = CellTypesProperties(n_cell_types=8)

    # Create FOV distribution with random cell types
    # tipical_cell_spacing controls density (smaller = more cells)
    def background_factory():
        return FrameWideElement(
            frame_size=800,
            rules=RandomCellTypeRule(n_cell_types=8),
            tipical_cell_spacing=20,  # ~20 pixels between cells
        )

    fovd = FOVDistribution(frame_size=800, background_element=background_factory)

    # Generate FOV
    fov = fovd.generate_fov()
    fov.realization()
    cell_props.apply(fov)

    # Setup HybISS and observe
    hybiss = HybISS_Setup(tissue)
    hybiss.measure_gene_expression(fov)
    hybiss.observe_dots(fov)

    save_sample(
        "simple_fov",
        fov,
        tissue,
        hybiss,
        "Simple FOV with 8 cell types randomly distributed. 800x800 pixels, ~50 genes.",
    )


def generate_cortex_like():
    """Generate a cortex-like tissue with layered structure using probability fields."""
    print("\nGenerating cortex_like...")
    np.random.seed(123)

    n_cell_types = 6
    tissue = TissueCellTypes()
    tissue.generate_types_and_markers(n_genes=60, n_cell_types=n_cell_types)
    tissue._cell_type_names = [
        "Layer1_Neuron",
        "Layer2_Neuron",
        "Layer3_Neuron",
        "Layer4_Neuron",
        "Astrocyte",
        "Oligodendrocyte",
    ]

    cell_props = CellTypesProperties(n_cell_types=n_cell_types)

    # Create layered probability field (horizontal layers)
    frame_size = 1000

    # Define reference points for probability interpolation (vertical gradient)
    # We need to cover the frame with reference points at different y-positions
    # Adding points at multiple x positions for better interpolation
    reference_points = np.array([
        [100, 100], [500, 100], [900, 100],   # y=100 (top)
        [100, 300], [500, 300], [900, 300],
        [100, 500], [500, 500], [900, 500],   # y=500 (middle)
        [100, 700], [500, 700], [900, 700],
        [100, 900], [500, 900], [900, 900],   # y=900 (bottom)
    ])

    # Probabilities at each reference point (normalized, not logits)
    # Row order matches reference_points
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    logits = np.array([
        # y=100 (top) - Layer 1 dominant
        [3.0, 0.0, 0.0, 0.0, 1.0, 0.5],
        [3.0, 0.0, 0.0, 0.0, 1.0, 0.5],
        [3.0, 0.0, 0.0, 0.0, 1.0, 0.5],
        # y=300 - Layer 2 dominant
        [1.0, 3.0, 0.5, 0.0, 1.0, 0.5],
        [1.0, 3.0, 0.5, 0.0, 1.0, 0.5],
        [1.0, 3.0, 0.5, 0.0, 1.0, 0.5],
        # y=500 (middle) - Layer 3 dominant
        [0.0, 1.0, 3.0, 1.0, 1.0, 0.5],
        [0.0, 1.0, 3.0, 1.0, 1.0, 0.5],
        [0.0, 1.0, 3.0, 1.0, 1.0, 0.5],
        # y=700 - Layer 4 dominant
        [0.0, 0.0, 1.0, 3.0, 1.0, 0.5],
        [0.0, 0.0, 1.0, 3.0, 1.0, 0.5],
        [0.0, 0.0, 1.0, 3.0, 1.0, 0.5],
        # y=900 (bottom) - Deep layers with more glia
        [0.0, 0.0, 0.0, 2.0, 2.0, 1.5],
        [0.0, 0.0, 0.0, 2.0, 2.0, 1.5],
        [0.0, 0.0, 0.0, 2.0, 2.0, 1.5],
    ])
    ref_probs = softmax(logits)

    prob_rule = ProbabilityNodeFieldRule(
        n_cell_types=n_cell_types,
        n_ref_points=len(reference_points),
        ref_probs=ref_probs,
        reference_points=reference_points,
    )

    def background_factory():
        return FrameWideElement(
            frame_size=frame_size,
            rules=prob_rule,
            tipical_cell_spacing=18,
        )

    fovd = FOVDistribution(frame_size=frame_size, background_element=background_factory)

    fov = fovd.generate_fov()
    fov.realization()
    cell_props.apply(fov)

    hybiss = HybISS_Setup(tissue)
    hybiss.measure_gene_expression(fov)
    hybiss.observe_dots(fov)

    save_sample(
        "cortex_like",
        fov,
        tissue,
        hybiss,
        "Cortex-like tissue with 6 cell types in layered arrangement. Horizontal layers "
        "with neuron types varying by depth and scattered glia.",
    )


def generate_gland_fov():
    """Generate a glandular structure with vacuolated elements."""
    print("\nGenerating gland_fov...")
    np.random.seed(456)

    n_cell_types = 5
    tissue = TissueCellTypes()
    tissue.generate_types_and_markers(n_genes=45, n_cell_types=n_cell_types)
    tissue._cell_type_names = [
        "Epithelial",
        "Secretory",
        "Myoepithelial",
        "Stromal",
        "Immune",
    ]

    cell_props = CellTypesProperties(n_cell_types=n_cell_types)

    frame_size = 900

    # Background: stromal cells with some immune
    def background_factory():
        return FrameWideElement(
            frame_size=frame_size,
            rules=MixOfNCellTypesRule(
                n_cell_types=n_cell_types,
                list_N=[3, 4],  # Stromal + Immune indices
                proportions=[0.8, 0.2],
            ),
            tipical_cell_spacing=30,  # Sparser background
        )

    # Glandular structures: epithelial cells at edge, secretory in middle
    def gland_factory():
        # Distance-based rule: secretory at center (inner_type=1), epithelial at edge (outer_type=0)
        rule = DistanceBasedRule(
            n_cell_types=n_cell_types,
            reference="center",
            inner_type=1,  # Secretory at center
            outer_type=0,  # Epithelial at edge
            transition_width=25,
        )
        return VacuolatedStructure(
            frame_size=frame_size,
            scale=80 + np.random.uniform(-20, 20),  # Random radius ~60-100
            hole_scale_factor=0.3,  # 30% vacuole
            rules=rule,
            tipical_cell_spacing=15,  # Denser in glands
        )

    fovd = FOVDistribution(
        frame_size=frame_size,
        background_element=background_factory,
        other_elements=[gland_factory],
        elements_frequency=[1.0],  # Always place
        attempts_at_elements=4,  # 4 glands
    )

    fov = fovd.generate_fov()
    fov.realization()
    cell_props.apply(fov)

    hybiss = HybISS_Setup(tissue)
    hybiss.measure_gene_expression(fov)
    hybiss.observe_dots(fov)

    save_sample(
        "gland_fov",
        fov,
        tissue,
        hybiss,
        "Glandular tissue with 5 cell types. Multiple vacuolated structures (glands) "
        "with epithelial cells at boundaries and secretory cells in center, "
        "embedded in stromal background.",
    )


def generate_mixed_tissue():
    """Generate a complex tissue with multiple structure types."""
    print("\nGenerating mixed_tissue...")
    np.random.seed(789)

    n_cell_types = 7
    tissue = TissueCellTypes()
    tissue.generate_types_and_markers(n_genes=70, n_cell_types=n_cell_types)
    tissue._cell_type_names = [
        "TypeA",
        "TypeB",
        "TypeC",
        "TypeD",
        "TypeE",
        "TypeF",
        "TypeG",
    ]

    cell_props = CellTypesProperties(n_cell_types=n_cell_types)

    frame_size = 1200

    # Background with gradient - reference points at corners
    reference_points = np.array([
        [100, 100],
        [frame_size - 100, 100],
        [100, frame_size - 100],
        [frame_size - 100, frame_size - 100],
    ])

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    logits = np.array([
        [2.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0],
        [0.0, 2.0, 1.0, 0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 2.0, 1.0, 0.5, 0.5, 0.0],
        [1.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0],
    ])
    ref_probs = softmax(logits)

    bg_rule = ProbabilityNodeFieldRule(
        n_cell_types=n_cell_types,
        n_ref_points=4,
        ref_probs=ref_probs,
        reference_points=reference_points,
    )

    def background_factory():
        return FrameWideElement(
            frame_size=frame_size,
            rules=bg_rule,
            tipical_cell_spacing=25,
        )

    # Small structures with single cell types
    def structure_type1():
        return VacuolatedStructure(
            frame_size=frame_size,
            scale=60 + np.random.uniform(-15, 15),
            hole_scale_factor=0.2,
            rules=SingleTypeRule(n_cell_types=n_cell_types, cell_type_ix=4),  # TypeE
            tipical_cell_spacing=14,
        )

    def structure_type2():
        return VacuolatedStructure(
            frame_size=frame_size,
            scale=50 + np.random.uniform(-10, 10),
            hole_scale_factor=0.0,  # No vacuole - solid cluster
            rules=SingleTypeRule(n_cell_types=n_cell_types, cell_type_ix=5),  # TypeF
            tipical_cell_spacing=14,
        )

    def structure_type3():
        # TypeE (4) at center, TypeG (6) at edge
        rule = DistanceBasedRule(
            n_cell_types=n_cell_types,
            reference="center",
            inner_type=4,  # TypeE at center
            outer_type=6,  # TypeG at edge
            transition_width=20,
        )
        return VacuolatedStructure(
            frame_size=frame_size,
            scale=70 + np.random.uniform(-15, 15),
            hole_scale_factor=0.25,
            rules=rule,
            tipical_cell_spacing=16,
        )

    fovd = FOVDistribution(
        frame_size=frame_size,
        background_element=background_factory,
        other_elements=[structure_type1, structure_type2, structure_type3],
        elements_frequency=[1.0, 1.0, 1.0],  # Always place
        attempts_at_elements=[3, 4, 2],  # Number of each type
    )

    fov = fovd.generate_fov()
    fov.realization()
    cell_props.apply(fov)

    hybiss = HybISS_Setup(tissue)
    hybiss.measure_gene_expression(fov)
    hybiss.observe_dots(fov)

    save_sample(
        "mixed_tissue",
        fov,
        tissue,
        hybiss,
        "Complex tissue with 7 cell types. Features spatial gradient in background, "
        "multiple vacuolated structures with different cell type compositions, "
        "including solid clusters and structures with radial cell type patterns.",
    )


def main():
    """Generate all sample datasets."""
    print("Generating sample datasets for PointillSim...")

    generate_simple_fov()
    generate_cortex_like()
    generate_gland_fov()
    generate_mixed_tissue()

    print("\nDone! Sample datasets generated successfully.")


if __name__ == "__main__":
    main()
