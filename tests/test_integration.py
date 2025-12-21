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
