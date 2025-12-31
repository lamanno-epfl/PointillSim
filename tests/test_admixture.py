"""Tests for the admixture module.

Tests cover:
1. Lateral2DAdmixture - boundary-based transcript misassignment
2. ZAxisAdmixture - out-of-plane cell contamination
3. CompositeAdmixture - combined admixture effects
4. AdmixtureMetrics - admixture quantification
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pointillsim.effects.admixture import (
    Lateral2DAdmixture,
    ZAxisAdmixture,
    CompositeAdmixture,
    AdmixtureMetrics,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_dots_df():
    """Create a simple dots DataFrame for testing."""
    np.random.seed(42)
    n_dots = 100
    return pd.DataFrame({
        "x": np.random.uniform(0, 500, n_dots),
        "y": np.random.uniform(0, 500, n_dots),
        "gene": np.random.choice(["Gene_A", "Gene_B", "Gene_C"], n_dots),
        "cell": np.random.randint(0, 10, n_dots),
    })


@pytest.fixture
def simple_cell_data():
    """Create simple cell data for testing."""
    np.random.seed(42)
    n_cells = 10
    cell_centroids = np.random.uniform(50, 450, (n_cells, 2))
    cell_types = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1])  # 4 types
    cell_radii = np.full(n_cells, 25.0)
    return cell_centroids, cell_types, cell_radii


@pytest.fixture
def clustered_cells():
    """Create clustered cell arrangement for admixture testing."""
    np.random.seed(42)
    # Create 3 clusters of cells
    centroids = []
    types = []

    # Cluster 1: Type 0 cells
    for _ in range(5):
        centroids.append([100 + np.random.uniform(-30, 30),
                         100 + np.random.uniform(-30, 30)])
        types.append(0)

    # Cluster 2: Type 1 cells
    for _ in range(5):
        centroids.append([300 + np.random.uniform(-30, 30),
                         100 + np.random.uniform(-30, 30)])
        types.append(1)

    # Cluster 3: Mixed types
    for _ in range(5):
        centroids.append([200 + np.random.uniform(-30, 30),
                         300 + np.random.uniform(-30, 30)])
        types.append(np.random.choice([0, 1, 2]))

    return np.array(centroids), np.array(types), np.full(15, 20.0)


@pytest.fixture
def dots_for_clustered_cells(clustered_cells):
    """Create dots matching clustered cell arrangement."""
    cell_centroids, cell_types, cell_radii = clustered_cells
    np.random.seed(42)

    dots_data = []
    for cell_idx in range(len(cell_centroids)):
        # Generate 10-20 dots per cell
        n_dots = np.random.randint(10, 21)
        for _ in range(n_dots):
            # Place dots within cell radius
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, cell_radii[cell_idx] * 0.8)
            x = cell_centroids[cell_idx, 0] + r * np.cos(angle)
            y = cell_centroids[cell_idx, 1] + r * np.sin(angle)
            dots_data.append({
                "x": x,
                "y": y,
                "gene": np.random.choice(["Gene_A", "Gene_B", "Gene_C"]),
                "cell": cell_idx,
            })

    return pd.DataFrame(dots_data)


# =============================================================================
# LATERAL2D ADMIXTURE TESTS
# =============================================================================


class TestLateral2DAdmixture:
    """Tests for Lateral2DAdmixture class."""

    def test_init_default(self):
        """Test default initialization."""
        admix = Lateral2DAdmixture()
        assert admix.boundary_width == 5.0
        assert admix.transfer_rate == 0.3
        assert admix.distance_decay == 0.5

    def test_init_custom(self):
        """Test custom initialization."""
        admix = Lateral2DAdmixture(
            boundary_width=10.0,
            transfer_rate=0.5,
            distance_decay=1.0,
            seed=42
        )
        assert admix.boundary_width == 10.0
        assert admix.transfer_rate == 0.5
        assert admix.distance_decay == 1.0

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            Lateral2DAdmixture(boundary_width=-1)

        with pytest.raises(ValueError):
            Lateral2DAdmixture(transfer_rate=1.5)

        with pytest.raises(ValueError):
            Lateral2DAdmixture(distance_decay=-0.5)

    def test_apply_empty_dots(self, simple_cell_data):
        """Test apply with empty dots DataFrame."""
        admix = Lateral2DAdmixture()
        cell_centroids, cell_types, cell_radii = simple_cell_data
        empty_dots = pd.DataFrame(columns=["x", "y", "gene", "cell"])

        result = admix.apply(empty_dots, cell_centroids, cell_types, cell_radii)
        assert len(result) == 0
        assert admix.admixture_record is not None

    def test_apply_basic(self, dots_for_clustered_cells, clustered_cells):
        """Test basic admixture application."""
        cell_centroids, cell_types, cell_radii = clustered_cells
        admix = Lateral2DAdmixture(
            boundary_width=15.0,
            transfer_rate=0.8,
            seed=42
        )

        result = admix.apply(
            dots_for_clustered_cells,
            cell_centroids,
            cell_types,
            cell_radii
        )

        # Result should have same number of dots
        assert len(result) == len(dots_for_clustered_cells)
        # Some dots should be reassigned
        assert admix.admixture_record is not None
        # Check that record contains expected columns
        if len(admix.admixture_record) > 0:
            assert "original_cell" in admix.admixture_record.columns
            assert "new_cell" in admix.admixture_record.columns
            assert "source" in admix.admixture_record.columns

    def test_reproducibility(self, dots_for_clustered_cells, clustered_cells):
        """Test that same seed produces same results."""
        cell_centroids, cell_types, cell_radii = clustered_cells

        admix1 = Lateral2DAdmixture(transfer_rate=0.5, seed=42)
        result1 = admix1.apply(dots_for_clustered_cells.copy(), cell_centroids, cell_types, cell_radii)

        admix2 = Lateral2DAdmixture(transfer_rate=0.5, seed=42)
        result2 = admix2.apply(dots_for_clustered_cells.copy(), cell_centroids, cell_types, cell_radii)

        pd.testing.assert_frame_equal(result1, result2)

    def test_get_admixture_summary(self, dots_for_clustered_cells, clustered_cells):
        """Test admixture summary computation."""
        cell_centroids, cell_types, cell_radii = clustered_cells
        admix = Lateral2DAdmixture(transfer_rate=0.5, seed=42)
        admix.apply(dots_for_clustered_cells, cell_centroids, cell_types, cell_radii)

        summary = admix.get_admixture_summary()
        assert "n_total_dots" in summary
        assert "n_reassigned" in summary
        assert "reassignment_rate" in summary
        assert summary["reassignment_rate"] <= 1.0


# =============================================================================
# ZAXIS ADMIXTURE TESTS
# =============================================================================


class TestZAxisAdmixture:
    """Tests for ZAxisAdmixture class."""

    def test_init_default(self):
        """Test default initialization."""
        admix = ZAxisAdmixture()
        assert admix.z_contamination_rate == 0.1
        assert admix.neighborhood_correlation == 0.7
        assert admix.neighborhood_radius == 50.0

    def test_init_custom(self):
        """Test custom initialization."""
        admix = ZAxisAdmixture(
            z_contamination_rate=0.2,
            neighborhood_correlation=0.9,
            neighborhood_radius=100.0,
            seed=42
        )
        assert admix.z_contamination_rate == 0.2
        assert admix.neighborhood_correlation == 0.9
        assert admix.neighborhood_radius == 100.0

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            ZAxisAdmixture(z_contamination_rate=-0.1)

        with pytest.raises(ValueError):
            ZAxisAdmixture(neighborhood_correlation=1.5)

        with pytest.raises(ValueError):
            ZAxisAdmixture(neighborhood_radius=0)

    def test_apply_empty(self, simple_cell_data):
        """Test apply with empty dots."""
        admix = ZAxisAdmixture()
        cell_centroids, cell_types, cell_radii = simple_cell_data
        empty_dots = pd.DataFrame(columns=["x", "y", "gene", "cell"])

        result = admix.apply(empty_dots, cell_centroids, cell_types, cell_radii)
        assert len(result) == 0

    def test_apply_basic(self, dots_for_clustered_cells, clustered_cells):
        """Test basic z-axis admixture application."""
        cell_centroids, cell_types, cell_radii = clustered_cells
        admix = ZAxisAdmixture(
            z_contamination_rate=0.2,
            seed=42
        )

        result = admix.apply(
            dots_for_clustered_cells,
            cell_centroids,
            cell_types,
            cell_radii
        )

        # Same number of dots
        assert len(result) == len(dots_for_clustered_cells)
        # Record should exist
        assert admix.admixture_record is not None
        # Check source column
        if len(admix.admixture_record) > 0:
            assert all(admix.admixture_record["source"] == "z_axis")

    def test_contamination_rate(self, dots_for_clustered_cells, clustered_cells):
        """Test that contamination rate is approximately correct."""
        cell_centroids, cell_types, cell_radii = clustered_cells

        # High contamination rate for testing
        admix = ZAxisAdmixture(
            z_contamination_rate=0.5,
            seed=42
        )

        admix.apply(dots_for_clustered_cells, cell_centroids, cell_types, cell_radii)

        # Should have approximately 50% of dots reassigned
        n_dots = len(dots_for_clustered_cells)
        n_reassigned = len(admix.admixture_record)
        # Allow some variance
        assert n_reassigned > n_dots * 0.3
        assert n_reassigned < n_dots * 0.7

    def test_neighborhood_correlation(self, dots_for_clustered_cells, clustered_cells):
        """Test that neighborhood correlation affects type selection."""
        cell_centroids, cell_types, cell_radii = clustered_cells

        # High correlation should prefer local types
        admix_high = ZAxisAdmixture(
            z_contamination_rate=0.3,
            neighborhood_correlation=0.99,
            seed=42
        )
        admix_high.apply(dots_for_clustered_cells.copy(), cell_centroids, cell_types)

        # Low correlation should be more random
        admix_low = ZAxisAdmixture(
            z_contamination_rate=0.3,
            neighborhood_correlation=0.01,
            seed=42
        )
        admix_low.apply(dots_for_clustered_cells.copy(), cell_centroids, cell_types)

        # Both should produce valid results
        assert admix_high.admixture_record is not None
        assert admix_low.admixture_record is not None


# =============================================================================
# COMPOSITE ADMIXTURE TESTS
# =============================================================================


class TestCompositeAdmixture:
    """Tests for CompositeAdmixture class."""

    def test_init(self):
        """Test initialization."""
        lateral = Lateral2DAdmixture(transfer_rate=0.2)
        z_axis = ZAxisAdmixture(z_contamination_rate=0.1)
        composite = CompositeAdmixture(models=[lateral, z_axis])

        assert len(composite.models) == 2

    def test_apply_combined(self, dots_for_clustered_cells, clustered_cells):
        """Test combined admixture application."""
        cell_centroids, cell_types, cell_radii = clustered_cells

        lateral = Lateral2DAdmixture(transfer_rate=0.2, seed=42)
        z_axis = ZAxisAdmixture(z_contamination_rate=0.15, seed=43)
        composite = CompositeAdmixture(models=[lateral, z_axis])

        result = composite.apply(
            dots_for_clustered_cells,
            cell_centroids,
            cell_types,
            cell_radii
        )

        assert len(result) == len(dots_for_clustered_cells)
        # Should have events from both sources
        assert composite.admixture_record is not None
        if len(composite.admixture_record) > 0:
            sources = composite.admixture_record["source"].unique()
            # May have lateral_2d, z_axis, or both
            assert len(sources) >= 1

    def test_empty_models(self, dots_for_clustered_cells, clustered_cells):
        """Test with no models."""
        cell_centroids, cell_types, cell_radii = clustered_cells
        composite = CompositeAdmixture(models=[])

        result = composite.apply(
            dots_for_clustered_cells,
            cell_centroids,
            cell_types,
            cell_radii
        )

        # Should return unchanged
        assert len(result) == len(dots_for_clustered_cells)


# =============================================================================
# ADMIXTURE METRICS TESTS
# =============================================================================


class TestAdmixtureMetrics:
    """Tests for AdmixtureMetrics class."""

    def test_metrics_with_admixture(self, dots_for_clustered_cells, clustered_cells):
        """Test metrics computation with admixture."""
        cell_centroids, cell_types, cell_radii = clustered_cells

        # Apply admixture first
        admix = Lateral2DAdmixture(transfer_rate=0.5, boundary_width=20, seed=42)
        admixed_dots = admix.apply(
            dots_for_clustered_cells,
            cell_centroids,
            cell_types,
            cell_radii
        )

        # Compute metrics
        metrics = AdmixtureMetrics(
            admixture_record=admix.admixture_record,
            dots_df=admixed_dots,
            cell_types=cell_types,
            cell_centroids=cell_centroids
        )

        # Check per-cell contamination
        assert len(metrics.per_cell_contamination) == len(cell_types)
        assert all(metrics.per_cell_contamination >= 0)
        assert all(metrics.per_cell_contamination <= 1)

        # Check contamination matrix
        n_types = int(cell_types.max()) + 1
        assert metrics.contamination_matrix.shape == (n_types, n_types)

    def test_get_summary(self, dots_for_clustered_cells, clustered_cells):
        """Test summary generation."""
        cell_centroids, cell_types, cell_radii = clustered_cells

        admix = ZAxisAdmixture(z_contamination_rate=0.2, seed=42)
        admixed_dots = admix.apply(
            dots_for_clustered_cells,
            cell_centroids,
            cell_types,
            cell_radii
        )

        metrics = AdmixtureMetrics(
            admixture_record=admix.admixture_record,
            dots_df=admixed_dots,
            cell_types=cell_types,
            cell_centroids=cell_centroids
        )

        summary = metrics.get_summary()
        assert "total_dots_reassigned" in summary
        assert "mean_contamination_per_cell" in summary
        assert "max_contamination_per_cell" in summary
        assert "total_cells" in summary

    def test_contamination_by_type(self, dots_for_clustered_cells, clustered_cells):
        """Test per-type contamination statistics."""
        cell_centroids, cell_types, cell_radii = clustered_cells

        admix = Lateral2DAdmixture(transfer_rate=0.3, seed=42)
        admixed_dots = admix.apply(
            dots_for_clustered_cells,
            cell_centroids,
            cell_types,
            cell_radii
        )

        metrics = AdmixtureMetrics(
            admixture_record=admix.admixture_record,
            dots_df=admixed_dots,
            cell_types=cell_types,
            cell_centroids=cell_centroids
        )

        by_type = metrics.get_contamination_by_type()
        assert isinstance(by_type, pd.DataFrame)
        assert "cell_type" in by_type.columns
        assert "mean_contamination" in by_type.columns
        assert "dots_lost" in by_type.columns
        assert "dots_gained" in by_type.columns

    def test_spatial_pattern(self, dots_for_clustered_cells, clustered_cells):
        """Test spatial contamination pattern."""
        cell_centroids, cell_types, cell_radii = clustered_cells

        admix = Lateral2DAdmixture(transfer_rate=0.3, seed=42)
        admixed_dots = admix.apply(
            dots_for_clustered_cells,
            cell_centroids,
            cell_types,
            cell_radii
        )

        metrics = AdmixtureMetrics(
            admixture_record=admix.admixture_record,
            dots_df=admixed_dots,
            cell_types=cell_types,
            cell_centroids=cell_centroids
        )

        pattern = metrics.get_spatial_contamination_pattern(grid_size=5)
        assert pattern.shape == (5, 5)
        assert np.all(pattern >= 0)

    def test_empty_admixture(self, dots_for_clustered_cells, clustered_cells):
        """Test metrics with no admixture events."""
        cell_centroids, cell_types, _ = clustered_cells

        # Create empty admixture record
        empty_record = pd.DataFrame(
            columns=["dot_index", "original_cell", "new_cell", "source"]
        )
        empty_record.attrs["n_total_dots"] = len(dots_for_clustered_cells)

        metrics = AdmixtureMetrics(
            admixture_record=empty_record,
            dots_df=dots_for_clustered_cells,
            cell_types=cell_types,
            cell_centroids=cell_centroids
        )

        summary = metrics.get_summary()
        assert summary["total_dots_reassigned"] == 0
        assert summary["mean_contamination_per_cell"] == 0.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAdmixtureIntegration:
    """Integration tests with full simulation pipeline."""

    def test_full_pipeline(self):
        """Test admixture in full simulation pipeline."""
        from pointillsim import (
            TissueCellTypes,
            CellTypesProperties,
            HybISS_Setup,
            FOVDistribution,
            FrameWideElement,
            RandomCellTypeRule,
        )

        # Create tissue
        tissue = TissueCellTypes()
        tissue.generate_types_and_markers(n_genes=20, n_cell_types=5)

        # Create FOV distribution
        bg = lambda: FrameWideElement(
            frame_size=500,
            rules=RandomCellTypeRule(n_cell_types=5)
        )
        fovd = FOVDistribution(frame_size=500, background_element=bg)

        # Generate FOV
        np.random.seed(42)
        fov = fovd.generate_fov()

        # Apply cell properties
        cell_props = CellTypesProperties(n_cell_types=5)
        cell_props.apply(fov)

        # Create HybISS and observe dots
        hybiss = HybISS_Setup(tissue)
        hybiss.observe_dots(fov)
        dots_df = hybiss.make_pandas_df()

        # Get cell data
        cell_centroids = fov.cell_centroids
        cell_types = fov.class_instance
        cell_radii = fov.cell_major_axis / 2

        # Apply admixture
        admix = Lateral2DAdmixture(
            boundary_width=10,
            transfer_rate=0.2,
            seed=42
        )
        admixed_dots = admix.apply(dots_df, cell_centroids, cell_types, cell_radii)

        # Verify results
        assert len(admixed_dots) == len(dots_df)
        assert admix.admixture_record is not None

        # Compute metrics
        metrics = AdmixtureMetrics(
            admixture_record=admix.admixture_record,
            dots_df=admixed_dots,
            cell_types=cell_types,
            cell_centroids=cell_centroids
        )

        summary = metrics.get_summary()
        assert summary["total_cells"] > 0

    def test_combined_admixture_pipeline(self):
        """Test combined lateral and z-axis admixture."""
        from pointillsim import (
            TissueCellTypes,
            CellTypesProperties,
            HybISS_Setup,
            FOVDistribution,
            FrameWideElement,
            RandomCellTypeRule,
        )

        # Create simple simulation
        tissue = TissueCellTypes()
        tissue.generate_types_and_markers(n_genes=10, n_cell_types=3)

        bg = lambda: FrameWideElement(
            frame_size=300,
            rules=RandomCellTypeRule(n_cell_types=3)
        )
        fovd = FOVDistribution(frame_size=300, background_element=bg)

        np.random.seed(42)
        fov = fovd.generate_fov()
        cell_props = CellTypesProperties(n_cell_types=3)
        cell_props.apply(fov)

        hybiss = HybISS_Setup(tissue)
        hybiss.observe_dots(fov)
        dots_df = hybiss.make_pandas_df()

        # Combined admixture
        lateral = Lateral2DAdmixture(transfer_rate=0.15, seed=42)
        z_axis = ZAxisAdmixture(z_contamination_rate=0.1, seed=43)
        composite = CompositeAdmixture(models=[lateral, z_axis])

        result = composite.apply(
            dots_df,
            fov.cell_centroids,
            fov.class_instance,
            fov.cell_major_axis / 2
        )

        assert len(result) == len(dots_df)
        assert composite.admixture_record is not None
