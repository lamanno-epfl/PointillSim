"""
PointillSim Test Suite

This test suite includes:
1. Basic unit tests for core classes
2. Complex integration tests based on notebook patterns
3. Visual output tests that generate plots for user verification

Run with: pytest tests/test_pointillsim.py -v
Run with visuals: pytest tests/test_pointillsim.py -v --visual
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for tests
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_simulations import (
    # Utility functions
    generate_uniform_points_in_circle,
    generate_points_asin_cell,
    one_hot_encode_array,
    lognorm_params_to_mean_std,
    intuitive_rand_lognormal,
    LinearNDInterpolatorExt,
    # Rules
    CellTypeRuleBase,
    RandomCellTypeRule,
    SingleTypeRule,
    MixOfNCellTypesRule,
    ProbabilityNodeFieldRule,
    DeterministicNeighborAssignment,
    DummyRule,
    # Elements
    HistologicalElement,
    FrameWideElement,
    VacuolatedStructure,
    # Core classes
    FOVDistribution,
    FOV,
    CellTypesProperties,
    TissueCellTypes,
    HybISS_Setup,
    # Advanced
    RegionSpec,
    TissueSlice,
    # Transfer functions
    IdentityTransfer,
    AffineNonNegTransfer,
)

from shapely.geometry import Polygon, Point
from shapely import affinity


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def output_dir(tmp_path):
    """Create output directory for visual tests."""
    vis_dir = tmp_path / "visual_outputs"
    vis_dir.mkdir(exist_ok=True)
    return vis_dir


@pytest.fixture
def persistent_output_dir():
    """Create persistent output directory for visual tests (not cleaned up)."""
    vis_dir = Path(__file__).parent / "visual_outputs"
    vis_dir.mkdir(exist_ok=True)
    return vis_dir


@pytest.fixture
def basic_tissue():
    """Create a basic tissue with 10 cell types and 20 genes."""
    tissue = TissueCellTypes()
    tissue.generate_types_and_markers(
        n_genes=20,
        n_cell_types=10,
        expected_level=10.0,
        expected_std_level=3.0,
        concentration=0.9
    )
    return tissue


@pytest.fixture
def basic_cell_props():
    """Create basic cell properties for 10 cell types."""
    return CellTypesProperties(
        n_cell_types=10,
        sizes=13,
        size_variation=1.5,
        anisotropy=0.85,
        anisotropy_variation=0.05
    )


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_uniform_points_in_circle(self):
        """Test point generation within a circle."""
        center = np.array([100.0, 100.0])
        scale = 50.0
        num_points = 1000

        points = generate_uniform_points_in_circle(center, scale, num_points)

        assert points.shape == (num_points, 2)
        # Check points are within expected distance from center
        distances = np.linalg.norm(points - center, axis=1)
        assert np.all(distances <= scale * 1.1)  # Allow small tolerance

    def test_generate_points_asin_cell(self):
        """Test point generation within an ellipse."""
        center = np.array([50.0, 50.0])
        scale = 1.0
        num_points = 100
        major_axis = 10.0
        minor_axis = 5.0
        rotation = np.pi / 4

        points = generate_points_asin_cell(
            center, scale, num_points, major_axis, minor_axis, rotation
        )

        assert points.shape == (num_points, 2)

    def test_one_hot_encode_array(self):
        """Test one-hot encoding."""
        class_ix = np.array([0, 2, 1, 3, 0])
        num_classes = 4

        one_hot = one_hot_encode_array(class_ix, num_classes)

        assert one_hot.shape == (5, 4)
        assert np.all(one_hot.sum(axis=1) == 1)
        assert np.array_equal(np.argmax(one_hot, axis=1), class_ix)

    def test_lognorm_params_conversion(self):
        """Test lognormal parameter conversion."""
        mu, sigma = 10.0, 3.0
        log_mu, log_sigma = lognorm_params_to_mean_std(mu, sigma)

        # Should return valid parameters
        assert np.isfinite(log_mu)
        assert np.isfinite(log_sigma)
        assert log_sigma > 0

    def test_intuitive_rand_lognormal(self):
        """Test lognormal random number generation."""
        mu, sigma = 10.0, 3.0
        samples = intuitive_rand_lognormal(mu, sigma, size=10000)

        # Check mean is approximately correct
        assert np.abs(np.mean(samples) - mu) < 1.0  # Allow some tolerance

    def test_linear_nd_interpolator_ext(self):
        """Test extended linear interpolator with RBF fallback."""
        # Create simple 2D interpolation problem
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        values = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

        interp = LinearNDInterpolatorExt(points, values)

        # Test interior point
        result = interp(np.array([[0.5, 0.5]]))
        assert result.shape == (1, 2)

        # Test point outside convex hull (should use RBF fallback)
        result_outside = interp(np.array([[2.0, 2.0]]))
        assert result_outside.shape == (1, 2)
        assert not np.any(np.isnan(result_outside))


# =============================================================================
# RULE TESTS
# =============================================================================

class TestRules:
    """Tests for cell type assignment rules."""

    def test_random_cell_type_rule(self):
        """Test random cell type assignment."""
        rule = RandomCellTypeRule(n_cell_types=5)
        centroids = np.random.rand(100, 2) * 100

        probs = rule.apply(centroids)

        assert probs.shape == (100, 5)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(probs >= 0)

    def test_single_type_rule(self):
        """Test single cell type assignment."""
        rule = SingleTypeRule(n_cell_types=5, cell_type_ix=2)
        centroids = np.random.rand(50, 2) * 100

        probs = rule.apply(centroids)

        assert probs.shape == (50, 5)
        assert np.all(probs[:, 2] > 0.99)  # Type 2 should dominate

    def test_mix_of_n_cell_types_rule(self):
        """Test mixture of cell types."""
        rule = MixOfNCellTypesRule(
            n_cell_types=10,
            list_N=[1, 3, 5],
            proportions=[0.5, 0.3, 0.2]
        )
        centroids = np.random.rand(100, 2) * 100

        probs = rule.apply(centroids)

        assert probs.shape == (100, 10)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        # Only specified types should have non-zero probability
        for i in range(10):
            if i not in [1, 3, 5]:
                assert np.all(probs[:, i] == 0)

    def test_probability_node_field_rule_with_reference_points(self):
        """Test probability field rule with explicit reference points."""
        n_types = 5
        ref_points = np.array([
            [0, 0], [100, 0], [50, 100]
        ])
        ref_probs = np.array([
            [1, 0, 0, 0, 0],  # Type 0 at origin
            [0, 1, 0, 0, 0],  # Type 1 at (100, 0)
            [0, 0, 1, 0, 0],  # Type 2 at (50, 100)
        ])

        rule = ProbabilityNodeFieldRule(
            n_cell_types=n_types,
            n_ref_points=3,
            reference_points=ref_points,
            ref_probs=ref_probs
        )

        # Test cells near each reference point
        centroids = np.array([[5, 5], [95, 5], [50, 95]])
        probs = rule.apply(centroids)

        assert probs.shape == (3, 5)
        # Cell near origin should favor type 0
        assert probs[0, 0] > probs[0, 1]
        # Cell near (100, 0) should favor type 1
        assert probs[1, 1] > probs[1, 0]

    def test_dummy_rule(self):
        """Test dummy rule that copies probabilities."""
        probs_to_copy = np.random.dirichlet(np.ones(5), 10)
        rule = DummyRule(n_cell_types=5, probs_to_copy=probs_to_copy)

        centroids = np.random.rand(10, 2) * 100
        result = rule.apply(centroids)

        assert np.array_equal(result, probs_to_copy)


# =============================================================================
# HISTOLOGICAL ELEMENT TESTS
# =============================================================================

class TestHistologicalElements:
    """Tests for histological element classes."""

    def test_histological_element_generation(self):
        """Test basic histological element generation."""
        rule = RandomCellTypeRule(n_cell_types=5)
        element = HistologicalElement(
            frame_size=500,
            n_vertices=(8, 12),
            scale=100,
            fixed_center=np.array([250, 250]),
            tipical_cell_spacing=10,
            rules=[rule]
        )

        generated = element.generate()

        assert generated.polygon is not None
        assert generated.cell_centroids is not None
        assert len(generated.cell_centroids) > 0
        assert generated.cell_probabilities.shape[1] == 5

    def test_frame_wide_element(self):
        """Test frame-wide element covers entire frame."""
        rule = RandomCellTypeRule(n_cell_types=3)
        element = FrameWideElement(
            frame_size=200,
            tipical_cell_spacing=15,
            rules=[rule]
        )

        generated = element.generate()

        # Check polygon covers frame
        assert generated.polygon.contains(Point(100, 100))
        assert generated.polygon.contains(Point(10, 10))
        assert generated.polygon.contains(Point(190, 190))

    def test_vacuolated_structure(self):
        """Test vacuolated structure with hole."""
        rule = SingleTypeRule(n_cell_types=5, cell_type_ix=1)
        element = VacuolatedStructure(
            frame_size=500,
            n_vertices=(10, 15),
            scale=100,
            fixed_center=np.array([250, 250]),
            tipical_cell_spacing=8,
            hole_scale_factor=0.5,
            rules=[rule]
        )

        generated = element.generate()

        assert generated.polygon is not None
        assert generated.hole is not None
        # Hole should be smaller than main polygon
        assert generated.hole.area < generated.polygon.area

    def test_element_cell_removal(self):
        """Test cell removal from elements."""
        rule = RandomCellTypeRule(n_cell_types=3)
        element = FrameWideElement(
            frame_size=100,
            tipical_cell_spacing=10,
            rules=[rule]
        ).generate()

        initial_count = len(element.cell_centroids)

        # Remove half the cells
        mask = np.zeros(initial_count, dtype=bool)
        mask[:initial_count // 2] = True
        element.remove_cells(mask)

        assert len(element.cell_centroids) == initial_count - (initial_count // 2)


# =============================================================================
# FOV TESTS
# =============================================================================

class TestFOV:
    """Tests for FOV and FOVDistribution classes."""

    def test_fov_creation(self):
        """Test basic FOV creation."""
        centroids = np.random.rand(50, 2) * 100
        probs = np.random.dirichlet(np.ones(5), 50)

        fov = FOV(centroids, probs)
        fov.realization()

        assert fov.class_instance.shape == (50,)
        assert np.all(fov.class_instance >= 0)
        assert np.all(fov.class_instance < 5)

    def test_fov_pandas_export(self, basic_cell_props):
        """Test FOV export to pandas DataFrame."""
        centroids = np.random.rand(20, 2) * 100
        probs = np.random.dirichlet(np.ones(10), 20)

        fov = FOV(centroids, probs)
        fov.realization()
        basic_cell_props.apply(fov)

        df = fov.make_pandas_df()

        assert isinstance(df, pd.DataFrame)
        assert 'X' in df.columns
        assert 'Y' in df.columns
        assert 'Class ID' in df.columns
        assert len(df) == 20

    def test_fov_distribution(self):
        """Test FOV generation from distribution."""
        n_types = 5
        frame = 200

        bg_rule = RandomCellTypeRule(n_cell_types=n_types)
        bg_proto = lambda: FrameWideElement(
            frame_size=frame, tipical_cell_spacing=15, rules=[bg_rule]
        )

        small_rule = SingleTypeRule(n_cell_types=n_types, cell_type_ix=2)
        small_proto = lambda: HistologicalElement(
            frame_size=frame, scale=40, tipical_cell_spacing=10, rules=[small_rule]
        )

        fovd = FOVDistribution(
            frame_size=frame,
            background_element=bg_proto,
            other_elements=[small_proto],
            elements_frequency=[0.8],
            attempts_at_elements=2
        )

        fov = fovd.generate_fov()

        assert fov.cell_centroids.shape[0] > 0
        assert fov.cell_probabilities.shape[1] == n_types


# =============================================================================
# TISSUE AND HYBISS TESTS
# =============================================================================

class TestTissueAndHybISS:
    """Tests for TissueCellTypes and HybISS_Setup."""

    def test_tissue_generation(self, basic_tissue):
        """Test tissue cell types generation."""
        assert basic_tissue.n_cell_types == 10
        assert basic_tissue.n_genes == 20
        assert basic_tissue.gene_expression_by_type.shape == (20, 10)

    def test_tissue_pandas_export(self, basic_tissue):
        """Test tissue export to pandas."""
        df = basic_tissue.make_pandas_df()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (20, 10)

    def test_hybiss_setup(self, basic_tissue, basic_cell_props):
        """Test HybISS experiment setup."""
        hybiss = HybISS_Setup(
            tissue=basic_tissue,
            genes_sensitivities=1.0,
            genes_sensitivities_variation=0.3
        )

        # Create a simple FOV
        centroids = np.random.rand(30, 2) * 100
        probs = np.random.dirichlet(np.ones(10), 30)
        fov = FOV(centroids, probs)
        fov.realization()
        basic_cell_props.apply(fov)

        # Measure expression
        counts = hybiss.measure_gene_expression(fov)

        assert counts.shape == (30, 20)
        assert np.all(counts >= 0)

    def test_hybiss_dot_observation(self, basic_tissue, basic_cell_props):
        """Test HybISS dot observation."""
        hybiss = HybISS_Setup(
            tissue=basic_tissue,
            genes_sensitivities=1.0,
            genes_sensitivities_variation=0.2
        )

        centroids = np.random.rand(20, 2) * 100
        probs = np.random.dirichlet(np.ones(10), 20)
        fov = FOV(centroids, probs)
        fov.realization()
        basic_cell_props.apply(fov)

        hybiss.observe_dots(fov)
        df = hybiss.make_pandas_df()

        assert isinstance(df, pd.DataFrame)
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'gene' in df.columns
        assert 'cell' in df.columns

    def test_transfer_functions(self, basic_tissue):
        """Test transfer function application."""
        # Identity transfer
        identity = IdentityTransfer()
        result = identity.transform(basic_tissue.gene_expression_by_type)
        assert np.array_equal(result, basic_tissue.gene_expression_by_type)

        # Affine transfer
        affine = AffineNonNegTransfer(
            scales=1.0, offsets=0.0,
            scales_std=0.1, offsets_std=0.05
        )
        result = affine.transform(basic_tissue.gene_expression_by_type)
        assert result.shape == basic_tissue.gene_expression_by_type.shape
        assert np.all(result >= 0)


# =============================================================================
# TISSUE SLICE TESTS (ADVANCED MULTI-REGION)
# =============================================================================

class TestTissueSlice:
    """Tests for multi-region TissueSlice composition."""

    def test_simple_tissue_slice(self):
        """Test basic tissue slice with two regions."""
        frame_size = 200
        n_types = 5

        # Region 1: left half
        rule1 = SingleTypeRule(n_cell_types=n_types, cell_type_ix=0)
        fovd1 = FOVDistribution(
            frame_size=frame_size,
            background_element=lambda: FrameWideElement(
                frame_size=frame_size, tipical_cell_spacing=15, rules=[rule1]
            ),
            other_elements=[],
            elements_frequency=[]
        )
        poly1 = Polygon([(0, 0), (100, 0), (100, 200), (0, 200)])

        # Region 2: right half
        rule2 = SingleTypeRule(n_cell_types=n_types, cell_type_ix=1)
        fovd2 = FOVDistribution(
            frame_size=frame_size,
            background_element=lambda: FrameWideElement(
                frame_size=frame_size, tipical_cell_spacing=15, rules=[rule2]
            ),
            other_elements=[],
            elements_frequency=[]
        )
        poly2 = Polygon([(100, 0), (200, 0), (200, 200), (100, 200)])

        regions = [
            RegionSpec("left", poly1, fovd1, priority=0),
            RegionSpec("right", poly2, fovd2, priority=1),
        ]

        tissue_slice = TissueSlice(frame_size, regions)
        tissue_slice.generate_global()
        tissue_slice.sample_labels()

        assert tissue_slice._cells_xy.shape[0] > 0
        assert tissue_slice._probs.shape[1] == n_types

    def test_tissue_slice_tiling(self):
        """Test FOV tiling from tissue slice."""
        frame_size = 200
        n_types = 3

        rule = RandomCellTypeRule(n_cell_types=n_types)
        fovd = FOVDistribution(
            frame_size=frame_size,
            background_element=lambda: FrameWideElement(
                frame_size=frame_size, tipical_cell_spacing=20, rules=[rule]
            ),
            other_elements=[],
            elements_frequency=[]
        )

        poly = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])
        regions = [RegionSpec("full", poly, fovd, priority=0)]

        tissue_slice = TissueSlice(frame_size, regions)
        tissue_slice.generate_global()

        # Tile into 50x50 FOVs with no overlap
        tiles = tissue_slice.tile_into_fovs(fov_size=50, overlap_frac=0.0)

        # Should get 4x4 = 16 tiles
        assert len(tiles) == 16

        # Each tile should have bbox and indices
        for tile in tiles:
            assert 'bbox' in tile
            assert 'idx' in tile
            assert 'ij' in tile


# =============================================================================
# COMPLEX INTEGRATION TESTS (Based on notebook patterns)
# =============================================================================

class TestComplexIntegration:
    """Complex integration tests based on notebook use cases."""

    def test_cortex_like_layered_structure(self):
        """Test multi-layer tissue generation similar to cortex notebook."""
        np.random.seed(42)
        frame_size = 400
        n_types = 8

        # Create layered regions (simplified cortex)
        layers = []
        y_positions = [0, 80, 160, 240, 320, 400]

        for i in range(len(y_positions) - 1):
            y_start, y_end = y_positions[i], y_positions[i + 1]

            # Each layer has a dominant cell type
            rule = MixOfNCellTypesRule(
                n_cell_types=n_types,
                list_N=[i, (i + 1) % n_types],
                proportions=[0.8, 0.2]
            )

            fovd = FOVDistribution(
                frame_size=frame_size,
                background_element=lambda r=rule: FrameWideElement(
                    frame_size=frame_size, tipical_cell_spacing=12, rules=[r]
                ),
                other_elements=[],
                elements_frequency=[]
            )

            poly = Polygon([
                (0, y_start), (frame_size, y_start),
                (frame_size, y_end), (0, y_end)
            ])

            layers.append(RegionSpec(
                f"Layer_{i}", poly, fovd,
                priority=i, blend_band=10.0
            ))

        tissue_slice = TissueSlice(frame_size, layers)
        tissue_slice.generate_global()
        tissue_slice.sample_labels()

        assert tissue_slice._cells_xy.shape[0] > 500
        assert tissue_slice._probs.shape[1] == n_types

    def test_skin_like_embedded_structures(self):
        """Test tissue with embedded structures like skin notebook."""
        np.random.seed(42)
        frame_size = 300
        n_types = 6

        # Background (dermis-like)
        bg_rule = MixOfNCellTypesRule(
            n_cell_types=n_types,
            list_N=[0, 1],
            proportions=[0.7, 0.3]
        )

        # Follicle-like structures
        follicle_rule = SingleTypeRule(n_cell_types=n_types, cell_type_ix=3)
        follicle_proto = lambda: VacuolatedStructure(
            frame_size=frame_size,
            scale=40,
            fixed_center=np.array([
                np.random.uniform(50, 250),
                np.random.uniform(50, 250)
            ]),
            tipical_cell_spacing=6,
            hole_scale_factor=0.5,
            rules=[follicle_rule]
        )

        # Small vessel-like structures
        vessel_rule = SingleTypeRule(n_cell_types=n_types, cell_type_ix=4)
        vessel_proto = lambda: HistologicalElement(
            frame_size=frame_size,
            scale=20,
            n_vertices=(6, 10),
            tipical_cell_spacing=5,
            rules=[vessel_rule]
        )

        fovd = FOVDistribution(
            frame_size=frame_size,
            background_element=lambda: FrameWideElement(
                frame_size=frame_size, tipical_cell_spacing=15, rules=[bg_rule]
            ),
            other_elements=[follicle_proto, vessel_proto],
            elements_frequency=[0.8, 0.9],
            attempts_at_elements=[3, 5]
        )

        # Generate FOV
        fov = fovd.generate_fov()

        assert fov.cell_centroids.shape[0] > 100
        assert fov.cell_probabilities.shape[1] == n_types

    def test_full_pipeline_end_to_end(self, basic_tissue, basic_cell_props):
        """Test complete pipeline from tissue to dot observation."""
        np.random.seed(42)
        frame_size = 200
        n_types = basic_tissue.n_cell_types

        # Create FOV distribution
        bg_rule = ProbabilityNodeFieldRule(
            n_cell_types=n_types,
            n_ref_points=4,
            alpha=0.1
        )

        small_rule = SingleTypeRule(n_cell_types=n_types, cell_type_ix=5)

        fovd = FOVDistribution(
            frame_size=frame_size,
            background_element=lambda: FrameWideElement(
                frame_size=frame_size, tipical_cell_spacing=12, rules=[bg_rule]
            ),
            other_elements=[
                lambda: HistologicalElement(
                    frame_size=frame_size, scale=50, tipical_cell_spacing=8,
                    rules=[small_rule]
                )
            ],
            elements_frequency=[0.7],
            attempts_at_elements=2
        )

        # Generate FOV
        fov = fovd.generate_fov()

        # Apply cell properties
        basic_cell_props.apply(fov)

        # Create HybISS setup and measure
        hybiss = HybISS_Setup(
            tissue=basic_tissue,
            genes_sensitivities=1.0,
            genes_sensitivities_variation=0.2
        )

        hybiss.observe_dots(fov)

        # Export data
        cells_df = fov.make_pandas_df()
        dots_df = hybiss.make_pandas_df()

        # Validate outputs
        assert len(cells_df) == len(fov.cell_centroids)
        assert len(dots_df) > 0
        assert 'X' in cells_df.columns
        assert 'x' in dots_df.columns
        assert 'gene' in dots_df.columns

    def test_probability_gradient_field(self):
        """Test complex probability gradient across tissue."""
        np.random.seed(42)
        frame_size = 200
        n_types = 4

        # Create a gradient from type 0 (top-left) to type 1 (bottom-right)
        ref_points = np.array([
            [20, 20], [180, 20],
            [20, 180], [180, 180]
        ])

        ref_probs = np.array([
            [1, 0, 0, 0],  # Top-left: type 0
            [0, 0, 1, 0],  # Top-right: type 2
            [0, 1, 0, 0],  # Bottom-left: type 1
            [0, 0, 0, 1],  # Bottom-right: type 3
        ])

        rule = ProbabilityNodeFieldRule(
            n_cell_types=n_types,
            n_ref_points=4,
            reference_points=ref_points,
            ref_probs=ref_probs
        )

        fovd = FOVDistribution(
            frame_size=frame_size,
            background_element=lambda: FrameWideElement(
                frame_size=frame_size, tipical_cell_spacing=10, rules=[rule]
            ),
            other_elements=[],
            elements_frequency=[]
        )

        fov = fovd.generate_fov()

        # Verify gradient behavior
        # Cells in top-left quadrant should favor type 0
        top_left_mask = (fov.cell_centroids[:, 0] < 100) & (fov.cell_centroids[:, 1] < 100)
        if np.any(top_left_mask):
            top_left_probs = fov.cell_probabilities[top_left_mask]
            avg_type0_prob = np.mean(top_left_probs[:, 0])
            assert avg_type0_prob > 0.3  # Should be elevated


# =============================================================================
# VISUAL OUTPUT TESTS
# =============================================================================

class TestVisualOutputs:
    """Tests that generate visual outputs for user verification."""

    @pytest.mark.visual
    def test_visual_basic_fov(self, persistent_output_dir, basic_tissue, basic_cell_props):
        """Generate visual output of a basic FOV."""
        np.random.seed(42)
        frame_size = 500
        n_types = basic_tissue.n_cell_types

        # Create FOV
        rule = ProbabilityNodeFieldRule(n_cell_types=n_types, n_ref_points=5, alpha=0.1)

        fovd = FOVDistribution(
            frame_size=frame_size,
            background_element=lambda: FrameWideElement(
                frame_size=frame_size, tipical_cell_spacing=12, rules=[rule]
            ),
            other_elements=[],
            elements_frequency=[]
        )

        fov = fovd.generate_fov()
        basic_cell_props.apply(fov)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Cell type scatter
        colors = [plt.cm.tab10(i % 10) for i in fov.class_instance]
        axes[0].scatter(
            fov.cell_centroids[:, 0],
            fov.cell_centroids[:, 1],
            c=colors, s=20, alpha=0.7
        )
        axes[0].set_title(f'Cell Types ({len(fov.cell_centroids)} cells)')
        axes[0].set_xlim(0, frame_size)
        axes[0].set_ylim(0, frame_size)
        axes[0].set_aspect('equal')

        # Right: Cell type probability heatmap (type 0)
        scatter = axes[1].scatter(
            fov.cell_centroids[:, 0],
            fov.cell_centroids[:, 1],
            c=fov.cell_probabilities[:, 0],
            cmap='viridis', s=20, alpha=0.8
        )
        plt.colorbar(scatter, ax=axes[1], label='Prob(Type 0)')
        axes[1].set_title('Type 0 Probability Field')
        axes[1].set_xlim(0, frame_size)
        axes[1].set_ylim(0, frame_size)
        axes[1].set_aspect('equal')

        plt.tight_layout()
        plt.savefig(persistent_output_dir / 'test_basic_fov.png', dpi=150)
        plt.close()

        assert (persistent_output_dir / 'test_basic_fov.png').exists()

    @pytest.mark.visual
    def test_visual_embedded_structures(self, persistent_output_dir, basic_tissue, basic_cell_props):
        """Generate visual output of FOV with embedded structures."""
        np.random.seed(42)
        frame_size = 600
        n_types = 8

        # Background
        bg_rule = MixOfNCellTypesRule(n_cell_types=n_types, list_N=[0, 1], proportions=[0.6, 0.4])

        # Vacuolated structures (follicle-like)
        follicle_rule = SingleTypeRule(n_cell_types=n_types, cell_type_ix=3)
        follicle_proto = lambda: VacuolatedStructure(
            frame_size=frame_size,
            scale=np.random.uniform(60, 100),
            tipical_cell_spacing=8,
            hole_scale_factor=np.random.uniform(0.4, 0.7),
            rules=[follicle_rule]
        )

        # Small circular structures
        vessel_rule = SingleTypeRule(n_cell_types=n_types, cell_type_ix=5)
        vessel_proto = lambda: HistologicalElement(
            frame_size=frame_size,
            scale=np.random.uniform(25, 40),
            n_vertices=(8, 12),
            tipical_cell_spacing=6,
            rules=[vessel_rule]
        )

        fovd = FOVDistribution(
            frame_size=frame_size,
            background_element=lambda: FrameWideElement(
                frame_size=frame_size, tipical_cell_spacing=14, rules=[bg_rule]
            ),
            other_elements=[follicle_proto, vessel_proto],
            elements_frequency=[0.9, 0.8],
            attempts_at_elements=[4, 8]
        )

        fov = fovd.generate_fov()

        # Apply properties
        cell_props = CellTypesProperties(
            n_cell_types=n_types,
            sizes=12,
            size_variation=2,
            anisotropy=0.8
        )
        cell_props.apply(fov)

        # HybISS setup
        tissue = TissueCellTypes()
        tissue.generate_types_and_markers(n_genes=30, n_cell_types=n_types)
        hybiss = HybISS_Setup(tissue=tissue)
        hybiss.observe_dots(fov)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Left: Cell types
        colors = [plt.cm.Set1(i % 9) for i in fov.class_instance]
        axes[0].scatter(
            fov.cell_centroids[:, 0],
            fov.cell_centroids[:, 1],
            c=colors, s=15, alpha=0.7
        )
        axes[0].set_title(f'Cell Types ({len(fov.cell_centroids)} cells)')
        axes[0].set_xlim(0, frame_size)
        axes[0].set_ylim(0, frame_size)
        axes[0].set_aspect('equal')

        # Middle: Dots colored by gene
        dots_df = hybiss.make_pandas_df()
        gene_ids = pd.Categorical(dots_df['gene']).codes
        scatter = axes[1].scatter(
            dots_df['x'], dots_df['y'],
            c=gene_ids, cmap='turbo', s=1, alpha=0.5
        )
        axes[1].set_title(f'Transcript Dots ({len(dots_df)} dots)')
        axes[1].set_xlim(0, frame_size)
        axes[1].set_ylim(0, frame_size)
        axes[1].set_aspect('equal')

        # Right: Overlay of cells with ellipses
        # IMPORTANT: Set limits BEFORE adding patches for correct scaling
        axes[2].set_xlim(0, frame_size)
        axes[2].set_ylim(0, frame_size)
        for i in range(min(200, len(fov.cell_centroids))):  # Limit for performance
            ellipse = matplotlib.patches.Ellipse(
                fov.cell_centroids[i],
                width=fov.cell_major_axis[i] * 2,
                height=fov.cell_minor_axis[i] * 2,
                angle=np.degrees(fov.cell_rotation[i]),
                facecolor=fov.cell_colors[i],
                edgecolor='black',
                linewidth=0.3,
                alpha=0.6
            )
            axes[2].add_patch(ellipse)
        axes[2].set_aspect('equal')
        axes[2].set_title('Cell Morphology (subset)')

        plt.tight_layout()
        plt.savefig(persistent_output_dir / 'test_embedded_structures.png', dpi=150)
        plt.close()

        assert (persistent_output_dir / 'test_embedded_structures.png').exists()

    @pytest.mark.visual
    def test_visual_multi_region_tissue(self, persistent_output_dir):
        """Generate visual output of multi-region tissue slice."""
        np.random.seed(42)
        frame_size = 500
        n_types = 6

        # Create three horizontal bands with different cell type compositions
        regions = []
        colors_by_region = [
            [0, 1],  # Top: types 0, 1
            [2, 3],  # Middle: types 2, 3
            [4, 5],  # Bottom: types 4, 5
        ]

        band_height = frame_size // 3

        for i, type_list in enumerate(colors_by_region):
            y_start = i * band_height
            y_end = (i + 1) * band_height

            rule = MixOfNCellTypesRule(
                n_cell_types=n_types,
                list_N=type_list,
                proportions=[0.6, 0.4]
            )

            fovd = FOVDistribution(
                frame_size=frame_size,
                background_element=lambda r=rule: FrameWideElement(
                    frame_size=frame_size, tipical_cell_spacing=10, rules=[r]
                ),
                other_elements=[],
                elements_frequency=[]
            )

            poly = Polygon([
                (0, y_start), (frame_size, y_start),
                (frame_size, y_end), (0, y_end)
            ])

            regions.append(RegionSpec(
                f"Band_{i}", poly, fovd,
                priority=i, blend_band=20.0
            ))

        tissue_slice = TissueSlice(frame_size, regions)
        tissue_slice.generate_global()
        tissue_slice.sample_labels()

        # Apply properties
        cell_props = CellTypesProperties(n_cell_types=n_types)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Left: Cell types by region
        class_ids = np.argmax(tissue_slice._class_onehot, axis=1)
        colors = [plt.cm.Set2(i % 8) for i in class_ids]
        axes[0].scatter(
            tissue_slice._cells_xy[:, 0],
            tissue_slice._cells_xy[:, 1],
            c=colors, s=10, alpha=0.7
        )

        # Draw region boundaries
        for i in range(1, 3):
            axes[0].axhline(y=i * band_height, color='black', linestyle='--', linewidth=1)

        axes[0].set_title(f'Multi-Region Tissue ({len(tissue_slice._cells_xy)} cells)')
        axes[0].set_xlim(0, frame_size)
        axes[0].set_ylim(0, frame_size)
        axes[0].set_aspect('equal')

        # Right: Probability of type 0 (should be high in top band)
        scatter = axes[1].scatter(
            tissue_slice._cells_xy[:, 0],
            tissue_slice._cells_xy[:, 1],
            c=tissue_slice._probs[:, 0],
            cmap='Reds', s=10, alpha=0.8
        )
        plt.colorbar(scatter, ax=axes[1], label='Prob(Type 0)')
        axes[1].set_title('Type 0 Probability (High in Top Band)')
        axes[1].set_xlim(0, frame_size)
        axes[1].set_ylim(0, frame_size)
        axes[1].set_aspect('equal')

        plt.tight_layout()
        plt.savefig(persistent_output_dir / 'test_multi_region_tissue.png', dpi=150)
        plt.close()

        assert (persistent_output_dir / 'test_multi_region_tissue.png').exists()

    @pytest.mark.visual
    def test_visual_complete_simulation(self, persistent_output_dir):
        """Generate comprehensive visual output of complete simulation."""
        np.random.seed(42)
        frame_size = 800
        n_types = 10
        n_genes = 30

        # Create tissue
        tissue = TissueCellTypes()
        tissue.generate_types_and_markers(
            n_genes=n_genes, n_cell_types=n_types,
            expected_level=12.0, expected_std_level=4.0
        )

        # Cell properties
        cell_props = CellTypesProperties(
            n_cell_types=n_types,
            sizes=15,
            size_variation=2.5,
            anisotropy=0.8,
            anisotropy_variation=0.1
        )

        # Complex FOV distribution
        bg_rule = ProbabilityNodeFieldRule(n_cell_types=n_types, n_ref_points=6, alpha=0.05)

        structure_rule = MixOfNCellTypesRule(
            n_cell_types=n_types,
            list_N=[3, 4, 5],
            proportions=[0.5, 0.3, 0.2]
        )

        vacuole_rule = SingleTypeRule(n_cell_types=n_types, cell_type_ix=7)

        fovd = FOVDistribution(
            frame_size=frame_size,
            background_element=lambda: FrameWideElement(
                frame_size=frame_size, tipical_cell_spacing=16, rules=[bg_rule]
            ),
            other_elements=[
                lambda: HistologicalElement(
                    frame_size=frame_size, scale=150,
                    tipical_cell_spacing=10, rules=[structure_rule]
                ),
                lambda: VacuolatedStructure(
                    frame_size=frame_size, scale=80,
                    hole_scale_factor=0.6,
                    tipical_cell_spacing=8, rules=[vacuole_rule]
                ),
            ],
            elements_frequency=[0.5, 0.7],
            attempts_at_elements=[2, 4]
        )

        # Generate FOV
        fov = fovd.generate_fov()
        cell_props.apply(fov)

        # HybISS
        hybiss = HybISS_Setup(
            tissue=tissue,
            genes_sensitivities=1.0,
            genes_sensitivities_variation=0.25
        )
        hybiss.observe_dots(fov)

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))

        # 1. Cell types
        ax1 = fig.add_subplot(2, 3, 1)
        colors = [plt.cm.tab10(i % 10) for i in fov.class_instance]
        ax1.scatter(
            fov.cell_centroids[:, 0],
            fov.cell_centroids[:, 1],
            c=colors, s=8, alpha=0.7
        )
        ax1.set_title(f'Cell Types\n({len(fov.cell_centroids)} cells)')
        ax1.set_aspect('equal')

        # 2. Transcript dots
        ax2 = fig.add_subplot(2, 3, 2)
        dots_df = hybiss.make_pandas_df()
        gene_codes = pd.Categorical(dots_df['gene']).codes
        ax2.scatter(
            dots_df['x'], dots_df['y'],
            c=gene_codes, cmap='turbo', s=0.5, alpha=0.3
        )
        ax2.set_title(f'Transcript Dots\n({len(dots_df)} dots)')
        ax2.set_aspect('equal')

        # 3. Cell morphology (subset)
        ax3 = fig.add_subplot(2, 3, 3)
        # IMPORTANT: Set limits BEFORE adding patches for correct scaling
        ax3.set_xlim(0, frame_size)
        ax3.set_ylim(0, frame_size)
        for i in range(min(300, len(fov.cell_centroids))):
            ellipse = matplotlib.patches.Ellipse(
                fov.cell_centroids[i],
                width=fov.cell_major_axis[i] * 2,
                height=fov.cell_minor_axis[i] * 2,
                angle=np.degrees(fov.cell_rotation[i]),
                facecolor=fov.cell_colors[i],
                edgecolor='black',
                linewidth=0.2,
                alpha=0.5
            )
            ax3.add_patch(ellipse)
        ax3.set_title('Cell Morphology')
        ax3.set_aspect('equal')

        # 4. Gene expression heatmap
        ax4 = fig.add_subplot(2, 3, 4)
        im = ax4.imshow(tissue.gene_expression_by_type, aspect='auto', cmap='viridis')
        ax4.set_xlabel('Cell Types')
        ax4.set_ylabel('Genes')
        ax4.set_title('Gene Expression Matrix')
        plt.colorbar(im, ax=ax4)

        # 5. Cell counts per type
        ax5 = fig.add_subplot(2, 3, 5)
        unique, counts = np.unique(fov.class_instance, return_counts=True)
        ax5.bar(unique, counts, color=[plt.cm.tab10(i % 10) for i in unique])
        ax5.set_xlabel('Cell Type')
        ax5.set_ylabel('Count')
        ax5.set_title('Cell Type Distribution')

        # 6. Dots per cell histogram
        ax6 = fig.add_subplot(2, 3, 6)
        dots_per_cell = dots_df.groupby('cell').size()
        ax6.hist(dots_per_cell, bins=50, edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Dots per Cell')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Transcripts per Cell Distribution')

        plt.tight_layout()
        plt.savefig(persistent_output_dir / 'test_complete_simulation.png', dpi=150)
        plt.close()

        assert (persistent_output_dir / 'test_complete_simulation.png').exists()


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--visual",
        action="store_true",
        default=False,
        help="Run visual output tests"
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "visual: mark test as generating visual output"
    )


def pytest_collection_modifyitems(config, items):
    """Skip visual tests unless --visual flag is provided."""
    if config.getoption("--visual"):
        return

    skip_visual = pytest.mark.skip(reason="need --visual option to run")
    for item in items:
        if "visual" in item.keywords:
            item.add_marker(skip_visual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
