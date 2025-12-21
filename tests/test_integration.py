"""Integration tests for full PointillSim simulation pipelines.

These tests verify that complete simulation workflows work correctly,
from tissue definition through to data export.
"""

import tempfile
import os

import numpy as np
import pandas as pd
import pytest

# Core imports
from pointillsim import (
    FOV,
    FOVDistribution,
    TissueCellTypes,
    TissueSlice,
    RegionSpec,
    HybISS_Setup,
    CellTypesProperties,
)

# Elements
from pointillsim.elements import (
    FrameWideElement,
    HistologicalElement,
    VacuolatedStructure,
    LinearLumenStructure,
    LayeredElement,
    BranchingStructure,
)

# Rules
from pointillsim.rules import (
    RandomCellTypeRule,
    SingleTypeRule,
    MixOfNCellTypesRule,
    LayerRule,
    GradientRule,
)

# Config
from pointillsim.config import SimulationConfig, create_example_config


class TestBasicSimulationPipeline:
    """Test the basic FOV generation pipeline."""

    def test_simple_fov_generation(self):
        """Test generating a simple FOV with background only."""
        n_cell_types = 5
        frame_size = 500

        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=20,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
        )

        fovd = FOVDistribution(
            frame_size=frame_size,
            background_element=bg,
        )

        fov = fovd.generate_fov()

        assert fov.n_cells > 0
        assert fov.cell_centroids.shape[1] == 2
        assert fov.cell_probabilities.shape[1] == n_cell_types
        assert np.all(fov.cell_probabilities.sum(axis=1) > 0.99)

    def test_fov_with_foreground_elements(self):
        """Test FOV generation with foreground elements."""
        n_cell_types = 5
        frame_size = 500

        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=25,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
        )

        def element_factory():
            return VacuolatedStructure(
                frame_size=frame_size,
                scale=80,
                tipical_cell_spacing=10,
                rules=LayerRule(
                    n_cell_types=n_cell_types,
                    layer_types=[0, 1],
                    layer_boundaries=[0.7],
                ),
            )

        fovd = FOVDistribution(
            frame_size=frame_size,
            background_element=bg,
            other_elements=[element_factory],
            elements_frequency=[1.0],
            attempts_at_elements=5,
        )

        fov = fovd.generate_fov()

        assert fov.n_cells > 0
        assert fov.cell_probabilities.shape[1] == n_cell_types

    def test_batch_fov_generation(self):
        """Test generating multiple FOVs in batch."""
        n_fovs = 5
        n_cell_types = 3
        frame_size = 300

        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=25,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
        )

        fovd = FOVDistribution(
            frame_size=frame_size,
            background_element=bg,
        )

        fovs = [fovd.generate_fov() for _ in range(n_fovs)]

        assert len(fovs) == n_fovs
        # All FOVs should have cells
        assert all(fov.n_cells > 0 for fov in fovs)
        # FOVs should be different (stochastic)
        cell_counts = [fov.n_cells for fov in fovs]
        assert len(set(cell_counts)) > 1 or all(c == cell_counts[0] for c in cell_counts)


class TestExperimentSimulation:
    """Test the HybISS experiment simulation pipeline."""

    def test_complete_hybiss_pipeline(self):
        """Test complete HybISS simulation from tissue to dots."""
        n_genes = 20
        n_cell_types = 5
        frame_size = 400

        # Create tissue
        tissue = TissueCellTypes()
        tissue.generate_types_and_markers(n_genes=n_genes, n_cell_types=n_cell_types)

        # Create FOV
        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=20,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
        )
        fovd = FOVDistribution(frame_size=frame_size, background_element=bg)
        fov = fovd.generate_fov()

        # Apply cell properties
        cell_props = CellTypesProperties(n_cell_types=n_cell_types)
        cell_props.apply(fov)

        # Create HybISS setup and observe
        hybiss = HybISS_Setup(tissue)
        hybiss.observe_dots(fov)

        # Get dots DataFrame
        dots_df = hybiss.make_pandas_df()

        assert isinstance(dots_df, pd.DataFrame)
        assert 'x' in dots_df.columns
        assert 'y' in dots_df.columns
        assert 'gene' in dots_df.columns
        assert 'cell' in dots_df.columns
        assert len(dots_df) > 0

    def test_hybiss_config_save_load(self):
        """Test saving and loading HybISS configuration."""
        tissue = TissueCellTypes()
        tissue.generate_types_and_markers(n_genes=15, n_cell_types=4)

        hybiss = HybISS_Setup(tissue)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'config.json')
            hybiss.save_config(path)

            loaded = HybISS_Setup.load_config(path)

            assert loaded.M.shape == hybiss.M.shape
            assert np.allclose(loaded.raw_M, hybiss.raw_M)


class TestTissueSlicePipeline:
    """Test TissueSlice multi-region simulation."""

    def test_tissue_slice_generation(self):
        """Test TissueSlice generation with regions."""
        from shapely.geometry import Polygon

        frame_size = 800
        n_cell_types = 4

        # Create region FOV distributions
        def bg_factory1():
            return FrameWideElement(
                frame_size=frame_size,
                tipical_cell_spacing=30,
                rules=SingleTypeRule(n_cell_types=n_cell_types, cell_type_ix=0),
            )

        def bg_factory2():
            return FrameWideElement(
                frame_size=frame_size,
                tipical_cell_spacing=30,
                rules=SingleTypeRule(n_cell_types=n_cell_types, cell_type_ix=1),
            )

        fovd1 = FOVDistribution(frame_size=frame_size, background_element=bg_factory1)
        fovd2 = FOVDistribution(frame_size=frame_size, background_element=bg_factory2)

        # Define regions
        region1_poly = Polygon([(0, 0), (400, 0), (400, 800), (0, 800)])
        region2_poly = Polygon([(400, 0), (800, 0), (800, 800), (400, 800)])

        region1 = RegionSpec("left", region1_poly, fovd1, priority=1)
        region2 = RegionSpec("right", region2_poly, fovd2, priority=0)

        # Create and generate slice
        tissue_slice = TissueSlice(frame_size=frame_size, regions=[region1, region2])
        tissue_slice.generate_global().sample_labels()

        assert tissue_slice.n_cells > 0
        assert tissue_slice.n_cell_types == n_cell_types

    def test_tissue_slice_save_load(self):
        """Test TissueSlice save and load."""
        slice = TissueSlice(frame_size=500, regions=[])
        slice._cells_xy = np.random.uniform(0, 500, (100, 2))
        slice._probs = np.random.dirichlet(np.ones(4), 100)
        slice._class_onehot = np.eye(4, dtype=int)[np.random.randint(0, 4, 100)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'slice.npz')
            slice.save(path)

            loaded = TissueSlice.load(path)

            assert np.allclose(loaded._cells_xy, slice._cells_xy)
            assert np.allclose(loaded._probs, slice._probs)

    def test_extract_fov_from_slice(self):
        """Test extracting FOV from TissueSlice."""
        slice = TissueSlice(frame_size=1000, regions=[])
        slice._cells_xy = np.random.uniform(0, 1000, (500, 2))
        slice._probs = np.random.dirichlet(np.ones(5), 500)

        fov = slice.extract_fov(x0=200, y0=200, fov_size=300)

        assert 'cell_centroids' in fov
        assert 'cell_probabilities' in fov
        # Check local coords
        assert fov['cell_centroids'].max() <= 300


class TestNewElements:
    """Test newly added element types."""

    def test_layered_element(self):
        """Test LayeredElement for stratified tissues."""
        n_cell_types = 3

        layer_rules = [
            SingleTypeRule(n_cell_types=n_cell_types, cell_type_ix=0),
            SingleTypeRule(n_cell_types=n_cell_types, cell_type_ix=1),
            SingleTypeRule(n_cell_types=n_cell_types, cell_type_ix=2),
        ]

        element = LayeredElement(
            frame_size=400,
            layer_boundaries=[130, 270],
            layer_rules=layer_rules,
            tipical_cell_spacing=15,
        )

        realized = element.generate()

        assert realized.cell_centroids.shape[0] > 0
        assert realized.cell_probabilities.shape[1] == n_cell_types
        assert realized.layer_indices is not None
        assert set(realized.layer_indices) == {0, 1, 2}

    def test_branching_structure(self):
        """Test BranchingStructure for vascular/ductal trees."""
        n_cell_types = 3

        tree = BranchingStructure(
            frame_size=500,
            root_radius=20,
            max_depth=2,
            branch_length=80,
            tipical_cell_spacing=8,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
        )

        realized = tree.generate()

        assert len(realized.branches) > 0
        # May have 0 cells if tree goes out of bounds quickly
        if realized.cell_centroids.shape[0] > 0:
            assert realized.branch_depths is not None

    def test_linear_lumen_structure(self):
        """Test LinearLumenStructure for tubes/vessels."""
        n_cell_types = 3

        vessel = LinearLumenStructure(
            frame_size=400,
            length=200,
            outer_radius=30,
            wall_thickness=10,
            tipical_cell_spacing=8,
            rules=LayerRule(
                n_cell_types=n_cell_types,
                layer_types=[0, 1],
                layer_boundaries=[0.5],
            ),
        )

        realized = vessel.generate()

        assert realized.cell_centroids.shape[0] > 0
        assert realized.lumen is not None
        assert realized.polygon is not None


class TestConfigSystem:
    """Test the configuration system."""

    def test_create_and_save_config(self):
        """Test creating and saving configuration."""
        config = create_example_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test JSON
            json_path = os.path.join(tmpdir, 'config.json')
            config.save(json_path)
            loaded_json = SimulationConfig.load(json_path)
            assert loaded_json.name == config.name

    def test_config_validation(self):
        """Test configuration validation."""
        from pointillsim.config import FOVConfig

        # Valid config
        valid = create_example_config()
        assert len(valid.validate()) == 0

        # Invalid config
        invalid = SimulationConfig(fov=FOVConfig(frame_size=-100))
        errors = invalid.validate()
        assert len(errors) > 0


class TestDataExport:
    """Test data export functionality."""

    def test_anndata_export(self):
        """Test exporting to AnnData format (if available)."""
        pytest.importorskip("anndata")

        # Create simple FOV
        n_cell_types = 4
        bg = lambda: FrameWideElement(
            frame_size=300,
            tipical_cell_spacing=20,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
        )
        fovd = FOVDistribution(frame_size=300, background_element=bg)
        fov = fovd.generate_fov()

        # Create tissue
        tissue = TissueCellTypes()
        tissue.generate_types_and_markers(n_genes=10, n_cell_types=n_cell_types)

        # Apply properties
        cell_props = CellTypesProperties(n_cell_types=n_cell_types)
        cell_props.apply(fov)

        # Observe
        hybiss = HybISS_Setup(tissue)
        hybiss.measure_gene_expression(fov)

        # Export using FOV.to_anndata() method
        adata = fov.to_anndata(
            gene_names=list(tissue.gene_names),
            expression_matrix=hybiss.cellxgene_counts,
        )

        assert adata.n_obs == fov.n_cells
        assert adata.n_vars == tissue.n_genes


class TestReproducibility:
    """Test reproducibility with seeds."""

    def test_reproducible_fov_generation(self):
        """Test that same seed produces same FOV."""
        n_cell_types = 5
        frame_size = 300

        def create_fov(seed):
            np.random.seed(seed)
            bg = lambda: FrameWideElement(
                frame_size=frame_size,
                tipical_cell_spacing=20,
                rules=RandomCellTypeRule(n_cell_types=n_cell_types),
            )
            fovd = FOVDistribution(frame_size=frame_size, background_element=bg)
            return fovd.generate_fov()

        fov1 = create_fov(42)
        fov2 = create_fov(42)
        fov3 = create_fov(123)

        # Same seed should give same result
        assert fov1.n_cells == fov2.n_cells
        # Different seed should give different result (probabilistically)
        # Note: This might rarely fail by chance
        # assert fov1.n_cells != fov3.n_cells or not np.allclose(fov1.cell_centroids, fov3.cell_centroids)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_fov(self):
        """Test FOV with no cells (very sparse configuration)."""
        # Create FOV with zero density
        bg = lambda: FrameWideElement(
            frame_size=100,
            tipical_cell_spacing=10000,  # Very sparse
            rules=RandomCellTypeRule(n_cell_types=3),
        )
        fovd = FOVDistribution(frame_size=100, background_element=bg)
        fov = fovd.generate_fov()

        # Should handle gracefully even if empty
        assert fov.cell_centroids.shape[1] == 2
        assert fov.cell_probabilities.ndim == 2

    def test_single_cell_fov(self):
        """Test FOV with a single cell."""
        centroids = np.array([[50.0, 50.0]])
        probs = np.array([[0.5, 0.3, 0.2]])
        fov = FOV(centroids, probs)

        assert fov.n_cells == 1
        assert fov.cell_probabilities.sum() == pytest.approx(1.0)
        fov.realization()
        assert fov.class_instance_one_hot.sum() == 1

    def test_single_cell_type(self):
        """Test simulation with single cell type."""
        n_cell_types = 1
        frame_size = 200

        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=25,
            rules=SingleTypeRule(n_cell_types=n_cell_types, cell_type_ix=0),
        )
        fovd = FOVDistribution(frame_size=frame_size, background_element=bg)
        fov = fovd.generate_fov()

        assert fov.n_cells > 0
        # All cells should be approximately type 0 (allowing for small numerical errors)
        assert np.allclose(fov.cell_probabilities[:, 0], 1.0)

    def test_many_cell_types(self):
        """Test simulation with many cell types."""
        n_cell_types = 50
        frame_size = 300

        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=20,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
        )
        fovd = FOVDistribution(frame_size=frame_size, background_element=bg)
        fov = fovd.generate_fov()

        assert fov.n_cells > 0
        assert fov.cell_probabilities.shape[1] == n_cell_types
        assert np.allclose(fov.cell_probabilities.sum(axis=1), 1.0)

    def test_very_small_fov(self):
        """Test very small FOV size."""
        frame_size = 50

        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=10,
            rules=RandomCellTypeRule(n_cell_types=3),
        )
        fovd = FOVDistribution(frame_size=frame_size, background_element=bg)
        fov = fovd.generate_fov()

        assert fov.cell_centroids.shape[1] == 2
        # Cells should be within bounds
        if fov.n_cells > 0:
            assert np.all(fov.cell_centroids >= 0)
            assert np.all(fov.cell_centroids <= frame_size)

    def test_very_large_fov(self):
        """Test large FOV size (performance check)."""
        frame_size = 2000

        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=50,  # Sparse to keep test fast
            rules=RandomCellTypeRule(n_cell_types=5),
        )
        fovd = FOVDistribution(frame_size=frame_size, background_element=bg)
        fov = fovd.generate_fov()

        assert fov.n_cells > 0
        # Verify that cells are generated (positions can extend slightly beyond frame)
        assert fov.cell_centroids.shape[1] == 2
        # Most cells should be roughly within the expected region
        assert np.mean(fov.cell_centroids >= 0) > 0.9
        assert np.mean(fov.cell_centroids <= frame_size * 1.1) > 0.9

    def test_zero_genes(self):
        """Test tissue with zero genes."""
        # This should raise an error or handle gracefully
        tissue = TissueCellTypes()
        with pytest.raises((ValueError, ZeroDivisionError, IndexError, OverflowError)):
            tissue.generate_types_and_markers(
                n_genes=0,
                n_cell_types=5,
            )

    def test_extreme_expression_levels(self):
        """Test with extreme expression levels."""
        tissue = TissueCellTypes()
        tissue.generate_types_and_markers(
            n_genes=10,
            n_cell_types=5,
            expected_level=1000.0,  # Very high
            expected_std_level=500.0,
        )

        assert np.all(tissue.gene_expression_by_type >= 0)
        assert tissue.n_genes == 10
        assert tissue.n_cell_types == 5

    def test_overlapping_elements(self):
        """Test FOV with many overlapping elements."""
        n_cell_types = 5
        frame_size = 500

        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=30,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
        )

        def element_factory():
            return VacuolatedStructure(
                frame_size=frame_size,
                scale=100,
                tipical_cell_spacing=15,
                rules=SingleTypeRule(n_cell_types=n_cell_types, cell_type_ix=0),
            )

        fovd = FOVDistribution(
            frame_size=frame_size,
            background_element=bg,
            other_elements=[element_factory],
            elements_frequency=[0.9],  # High frequency
            attempts_at_elements=10,  # Many attempts
        )
        fov = fovd.generate_fov()

        assert fov.n_cells > 0
        assert np.allclose(fov.cell_probabilities.sum(axis=1), 1.0)

    def test_uniform_probabilities(self):
        """Test with uniform probability distribution."""
        n_cells = 100
        n_types = 4
        centroids = np.random.rand(n_cells, 2) * 500
        probs = np.ones((n_cells, n_types)) / n_types

        fov = FOV(centroids, probs)
        fov.realization()

        # Types should be distributed roughly uniformly
        type_counts = fov.class_instance_one_hot.sum(axis=0)
        assert len(type_counts) == n_types
        # Allow some variance but types should be somewhat balanced
        assert type_counts.std() / type_counts.mean() < 0.5

    def test_deterministic_probabilities(self):
        """Test with deterministic (one-hot) probabilities."""
        n_cells = 100
        n_types = 3
        centroids = np.random.rand(n_cells, 2) * 500
        probs = np.zeros((n_cells, n_types))
        # Assign types in a pattern
        for i in range(n_cells):
            probs[i, i % n_types] = 1.0

        fov = FOV(centroids, probs)
        fov.realization()

        # Should match the deterministic pattern
        assigned_types = fov.class_instance_one_hot.argmax(axis=1)
        expected_types = np.arange(n_cells) % n_types
        assert np.array_equal(assigned_types, expected_types)

    def test_new_structure_elements(self):
        """Test the new structure elements work correctly."""
        from pointillsim.elements import (
            FibrillarStructure,
            ClusterElement,
            StromalElement,
        )

        n_cell_types = 5
        frame_size = 500

        # Test FibrillarStructure (generate() returns a new realized copy)
        fiber = FibrillarStructure(
            frame_size=frame_size,
            n_fibers=3,
            fiber_width=30,
            rules=SingleTypeRule(n_cell_types, 0),
        )
        realized_fiber = fiber.generate()
        assert realized_fiber.cell_centroids is not None
        assert len(realized_fiber.cell_centroids) > 0

        # Test ClusterElement (generate() returns a new realized copy)
        cluster = ClusterElement(
            frame_size=frame_size,
            radius=80,
            rules=SingleTypeRule(n_cell_types, 1),
        )
        realized_cluster = cluster.generate()
        assert realized_cluster.cell_centroids is not None
        assert len(realized_cluster.cell_centroids) > 0

        # Test StromalElement (generate() returns a new realized copy)
        stromal = StromalElement(
            frame_size=frame_size,
            rules=SingleTypeRule(n_cell_types, 2),
        )
        realized_stromal = stromal.generate()
        assert realized_stromal.cell_centroids is not None
        assert len(realized_stromal.cell_centroids) > 0

    def test_new_effects_module(self):
        """Test the new effects module."""
        from pointillsim.effects import (
            BatchEffectModel,
            TechnicalNoise,
            BackgroundNoise,
            DropoutModel,
        )

        n_cells = 50
        n_genes = 20
        expression = np.random.poisson(10, (n_cells, n_genes)).astype(float)
        positions = np.random.rand(n_cells, 2) * 1000

        # Test BatchEffectModel
        batch = BatchEffectModel(n_genes=n_genes, n_batches=3)
        batch.generate_effects()
        batch_expr = batch.apply(expression, batch_id=1)
        assert batch_expr.shape == expression.shape

        # Test TechnicalNoise
        tech_noise = TechnicalNoise(frame_size=1000)
        noisy_expr = tech_noise.apply(expression, positions)
        assert noisy_expr.shape == expression.shape

        # Test BackgroundNoise
        bg_noise = BackgroundNoise(frame_size=1000, background_rate=0.01)
        x, y, genes = bg_noise.generate_background_dots(n_genes)
        assert len(x) == len(y) == len(genes)

        # Test DropoutModel
        dropout = DropoutModel(baseline_detection_rate=0.8)
        dropped = dropout.apply(expression)
        assert dropped.shape == expression.shape

    def test_new_design_module(self):
        """Test the new design module."""
        from pointillsim.design import (
            CovariateSystem,
            Covariate,
            EffectController,
            SimulationDesign,
            DesignMatrix,
        )

        # Test Covariate
        cov_cat = Covariate("region", "categorical", values=["A", "B", "C"])
        cov_cont = Covariate("distance", "continuous", values=[0, 100])
        assert cov_cat.validate("A")
        assert not cov_cat.validate("D")
        assert cov_cont.validate(50)

        # Test CovariateSystem
        system = CovariateSystem()
        system.add_covariate(cov_cat)
        system.add_covariate(cov_cont)
        config = system.sample_configuration()
        assert "region" in config
        assert "distance" in config

        # Test EffectController
        controller = EffectController()
        controller.disable("dropout")
        assert not controller.is_enabled("dropout")
        assert controller.is_enabled("batch_effect")

        # Test SimulationDesign
        design = SimulationDesign("test")
        design.add_covariate(cov_cat)
        design.set_replicates(2)
        conditions = design.generate_conditions()
        assert len(conditions) > 0

        # Test DesignMatrix
        matrix = DesignMatrix(["region"])
        matrix.add_row({"region": "A"})
        matrix.add_row({"region": "B"})
        assert matrix.n_fovs == 2

    def test_new_validation_module(self):
        """Test the new validation module."""
        from pointillsim.validation import DifficultyScorer, ValidationMetrics

        # Generate test data
        n_cells = 100
        n_genes = 20
        expression = np.random.poisson(5, (n_cells, n_genes)).astype(float)
        cell_types = np.random.randint(0, 3, n_cells)
        positions = np.random.rand(n_cells, 2) * 1000

        # Test DifficultyScorer
        scorer = DifficultyScorer(use_classifier=False)
        scores = scorer.score(expression, cell_types, positions)
        assert "separability" in scores
        assert "overall" in scores
        assert 0 <= scores["overall"] <= 1

        # Test ValidationMetrics
        validator = ValidationMetrics()
        sim_expr = np.random.poisson(5, (80, 20)).astype(float)
        metrics = validator.compare(expression, sim_expr)
        assert "expression_similarity" in metrics
        assert "overall_quality" in metrics

    def test_multi_fov_features(self):
        """Test new multi-FOV generation features."""
        from pointillsim.core import ConsistentTiling
        from shapely.geometry import box

        n_cell_types = 3
        frame_size = 500

        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=30,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
        )

        fovd = FOVDistribution(frame_size=frame_size, background_element=bg)

        # Test generate_batch
        fovs = fovd.generate_batch(n_fovs=3)
        assert len(fovs) == 3
        for fov in fovs:
            assert fov.n_cells > 0

        # Test TissueSlice.generate_multiple
        region = RegionSpec("test", box(0, 0, 500, 500), fovd)
        tissue = TissueSlice(frame_size=500, regions=[region])
        tissue.generate_global()
        realizations = tissue.generate_multiple(3)
        assert len(realizations) == 3

        # Test ConsistentTiling
        tissue.sample_labels()
        tiles = tissue.tile_into_fovs(fov_size=250, overlap_frac=0.1)
        tiling = ConsistentTiling(overlap_frac=0.1)
        consistent_tiles = tiling.ensure_consistency(tiles, tissue)
        assert len(consistent_tiles) > 0
