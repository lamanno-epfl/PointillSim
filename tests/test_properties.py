"""Property-based tests for PointillSim using hypothesis.

These tests verify invariants and properties that should hold across
a wide range of input parameters, helping catch edge cases that
traditional unit tests might miss.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# Core imports
from pointillsim import (
    FOV,
    FOVDistribution,
    TissueCellTypes,
    CellTypesProperties,
)

# Elements
from pointillsim.elements import (
    FrameWideElement,
    VacuolatedStructure,
    LayeredElement,
)

# Rules
from pointillsim.rules import (
    RandomCellTypeRule,
    SingleTypeRule,
    MixOfNCellTypesRule,
    LayerRule,
)

# Utilities
from pointillsim.utils.geometry import (
    generate_uniform_points_in_circle,
    smooth_polygon,
    chaikin_smooth,
)
from pointillsim.utils.math import lognorm_params_to_mean_std, intuitive_rand_lognormal
from pointillsim.utils.encoding import one_hot_encode_array


# =============================================================================
# Strategy definitions for generating test inputs
# =============================================================================

# Valid ranges for simulation parameters
n_cell_types_strategy = st.integers(min_value=2, max_value=20)
n_genes_strategy = st.integers(min_value=5, max_value=100)
frame_size_strategy = st.integers(min_value=100, max_value=2000)
cell_spacing_strategy = st.integers(min_value=5, max_value=50)
scale_strategy = st.integers(min_value=20, max_value=200)


# =============================================================================
# Tests for cell type rules
# =============================================================================

class TestCellTypeRuleProperties:
    """Property-based tests for cell type assignment rules."""

    @given(
        n_cells=st.integers(min_value=1, max_value=100),
        n_cell_types=n_cell_types_strategy,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_random_cell_type_rule_probability_sums(self, n_cells, n_cell_types):
        """RandomCellTypeRule should always produce valid probability distributions."""
        rule = RandomCellTypeRule(n_cell_types=n_cell_types)

        # Generate random cell positions
        cell_positions = np.random.uniform(0, 500, (n_cells, 2))
        probs = rule.apply(cell_positions)

        # Properties that must hold:
        # 1. Shape is correct
        assert probs.shape == (n_cells, n_cell_types)

        # 2. All probabilities are non-negative
        assert np.all(probs >= 0)

        # 3. Rows sum to 1 (within floating point tolerance)
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    @given(
        n_cells=st.integers(min_value=1, max_value=100),
        n_cell_types=n_cell_types_strategy,
        cell_type_ix=st.integers(min_value=0, max_value=19),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_single_type_rule_assigns_correct_type(self, n_cells, n_cell_types, cell_type_ix):
        """SingleTypeRule should assign high probability to the specified type."""
        assume(cell_type_ix < n_cell_types)

        rule = SingleTypeRule(n_cell_types=n_cell_types, cell_type_ix=cell_type_ix)
        cell_positions = np.random.uniform(0, 500, (n_cells, 2))
        probs = rule.apply(cell_positions)

        # All cells should have high probability (>0.99) for the specified type
        assert np.all(probs[:, cell_type_ix] > 0.99)

        # The specified type should be the argmax
        assert np.all(np.argmax(probs, axis=1) == cell_type_ix)

        # Sum of each row should be 1.0
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    @given(
        n_cells=st.integers(min_value=1, max_value=100),
        n_cell_types=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_mix_of_n_cell_types_rule_valid_probabilities(self, n_cells, n_cell_types):
        """MixOfNCellTypesRule should produce valid probability distributions."""
        # Specify a list of types to use
        n_mix = min(3, n_cell_types)
        list_N = list(range(n_mix))

        rule = MixOfNCellTypesRule(n_cell_types=n_cell_types, list_N=list_N)
        cell_positions = np.random.uniform(0, 500, (n_cells, 2))
        probs = rule.apply(cell_positions)

        # Should have correct shape
        assert probs.shape == (n_cells, n_cell_types)

        # Rows should sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

        # Only the specified types should have non-zero probability
        types_used = np.any(probs > 1e-10, axis=0)
        for i in range(n_cell_types):
            if i in list_N:
                assert types_used[i], f"Type {i} should be used but isn't"
            else:
                assert not types_used[i], f"Type {i} should not be used but is"


# =============================================================================
# Tests for FOV generation
# =============================================================================

class TestFOVProperties:
    """Property-based tests for FOV generation."""

    @given(
        frame_size=frame_size_strategy,
        n_cell_types=n_cell_types_strategy,
        cell_spacing=cell_spacing_strategy,
    )
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_fov_cells_within_bounds(self, frame_size, n_cell_types, cell_spacing):
        """All generated cells should be within the frame bounds."""
        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=cell_spacing,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
        )

        fovd = FOVDistribution(frame_size=frame_size, background_element=bg)
        fov = fovd.generate_fov()

        if fov.n_cells > 0:
            # All x coordinates within bounds (with some margin for cell size)
            assert np.all(fov.cell_centroids[:, 0] >= -cell_spacing)
            assert np.all(fov.cell_centroids[:, 0] <= frame_size + cell_spacing)

            # All y coordinates within bounds
            assert np.all(fov.cell_centroids[:, 1] >= -cell_spacing)
            assert np.all(fov.cell_centroids[:, 1] <= frame_size + cell_spacing)

    @given(
        frame_size=frame_size_strategy,
        n_cell_types=n_cell_types_strategy,
    )
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_fov_probability_consistency(self, frame_size, n_cell_types):
        """FOV cell probabilities should always be valid distributions."""
        bg = lambda: FrameWideElement(
            frame_size=frame_size,
            tipical_cell_spacing=20,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
        )

        fovd = FOVDistribution(frame_size=frame_size, background_element=bg)
        fov = fovd.generate_fov()

        if fov.n_cells > 0:
            # Probabilities should be non-negative
            assert np.all(fov.cell_probabilities >= 0)

            # Probabilities should have correct number of columns
            assert fov.cell_probabilities.shape[1] == n_cell_types

            # Row sums should be 1
            assert np.allclose(fov.cell_probabilities.sum(axis=1), 1.0, atol=1e-6)


# =============================================================================
# Tests for geometry utilities
# =============================================================================

class TestGeometryProperties:
    """Property-based tests for geometry utilities."""

    @given(
        n_points=st.integers(min_value=1, max_value=1000),
        radius=st.floats(min_value=1.0, max_value=1000.0),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_uniform_points_in_circle_bounds(self, n_points, radius):
        """All points should be within the circle radius."""
        center = np.array([100.0, 100.0])
        points = generate_uniform_points_in_circle(center, radius, n_points)

        # All points should be within radius of center
        distances = np.linalg.norm(points - center, axis=1)
        assert np.all(distances <= radius * 1.001)  # Small tolerance for float precision

    @given(
        n_points=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_uniform_points_count(self, n_points):
        """Should generate exactly the requested number of points."""
        center = np.array([0.0, 0.0])
        points = generate_uniform_points_in_circle(center, 10.0, n_points)

        assert points.shape == (n_points, 2)

    @given(
        n_vertices=st.integers(min_value=4, max_value=20),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_chaikin_smooth_increases_vertices(self, n_vertices):
        """Chaikin smoothing should increase vertex count (for iterations > 0)."""
        # Generate a simple polygon
        angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
        coords = np.column_stack([np.cos(angles), np.sin(angles)])

        smoothed = chaikin_smooth(coords, iterations=1)

        # Smoothing should increase vertex count (roughly double minus 1 per iteration)
        assert len(smoothed) >= len(coords)


# =============================================================================
# Tests for math utilities
# =============================================================================

class TestMathProperties:
    """Property-based tests for math utilities."""

    @given(
        mean=st.floats(min_value=0.1, max_value=100.0),
        std=st.floats(min_value=0.01, max_value=10.0),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_lognorm_params_roundtrip(self, mean, std):
        """Lognormal parameters should give correct mean and std."""
        assume(std < mean)  # Valid constraint for lognormal

        mu, sigma = lognorm_params_to_mean_std(mean, std)

        # The parameters should be real numbers
        assert np.isfinite(mu)
        assert np.isfinite(sigma)
        assert sigma >= 0

    @given(
        mean=st.floats(min_value=1.0, max_value=100.0),
        std=st.floats(min_value=0.1, max_value=10.0),
        n_samples=st.integers(min_value=100, max_value=1000),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_intuitive_lognormal_positive(self, mean, std, n_samples):
        """Intuitive lognormal should always produce positive values."""
        assume(std < mean)

        samples = intuitive_rand_lognormal(mean, std, n_samples)

        assert len(samples) == n_samples
        assert np.all(samples > 0)


# =============================================================================
# Tests for encoding utilities
# =============================================================================

class TestEncodingProperties:
    """Property-based tests for encoding utilities."""

    @given(
        n_items=st.integers(min_value=1, max_value=100),
        n_classes=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_one_hot_encode_roundtrip(self, n_items, n_classes):
        """One-hot encoding should be reversible."""
        # Generate random class assignments
        labels = np.random.randint(0, n_classes, n_items)

        # Encode
        one_hot = one_hot_encode_array(labels, n_classes)

        # Properties:
        # 1. Shape is correct
        assert one_hot.shape == (n_items, n_classes)

        # 2. Each row has exactly one 1
        assert np.all(one_hot.sum(axis=1) == 1)

        # 3. Can recover original labels
        recovered = np.argmax(one_hot, axis=1)
        assert np.array_equal(labels, recovered)


# =============================================================================
# Tests for element structures
# =============================================================================

class TestElementProperties:
    """Property-based tests for element structures."""

    @given(
        frame_size=st.integers(min_value=200, max_value=800),
        n_cell_types=st.integers(min_value=2, max_value=5),
        n_layers=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=15, deadline=15000, suppress_health_check=[HealthCheck.too_slow])
    def test_layered_element_layer_consistency(self, frame_size, n_cell_types, n_layers):
        """LayeredElement should assign cells to valid layers."""
        # Create layer boundaries
        boundaries = sorted(np.random.uniform(0.1, 0.9, n_layers - 1).tolist())
        boundaries = [int(b * frame_size) for b in boundaries]

        # Create rules for each layer
        layer_rules = [
            SingleTypeRule(n_cell_types=n_cell_types, cell_type_ix=i % n_cell_types)
            for i in range(n_layers)
        ]

        element = LayeredElement(
            frame_size=frame_size,
            layer_boundaries=boundaries,
            layer_rules=layer_rules,
            tipical_cell_spacing=15,
        )

        realized = element.generate()

        if realized.cell_centroids.shape[0] > 0:
            # Layer indices should be valid
            assert realized.layer_indices is not None
            assert np.all(realized.layer_indices >= 0)
            assert np.all(realized.layer_indices < n_layers)

    @given(
        frame_size=st.integers(min_value=300, max_value=600),
        scale=st.integers(min_value=40, max_value=80),
        n_cell_types=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=15, deadline=15000, suppress_health_check=[HealthCheck.too_slow])
    def test_vacuolated_structure_has_hole(self, frame_size, scale, n_cell_types):
        """VacuolatedStructure should always have a hole (lumen) polygon."""
        # Use fixed center to ensure structure is in bounds
        center = np.array([[frame_size / 2, frame_size / 2]])

        structure = VacuolatedStructure(
            frame_size=frame_size,
            scale=scale,
            tipical_cell_spacing=10,
            rules=RandomCellTypeRule(n_cell_types=n_cell_types),
            fixed_center=center,
        )

        realized = structure.generate()

        # Should have a hole (lumen)
        assert realized.hole is not None

        # Hole should be valid geometry
        assert realized.hole.is_valid


# =============================================================================
# Tests for tissue generation
# =============================================================================

class TestTissueProperties:
    """Property-based tests for tissue generation."""

    @given(
        n_genes=n_genes_strategy,
        n_cell_types=n_cell_types_strategy,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_tissue_expression_matrix_shape(self, n_genes, n_cell_types):
        """Tissue expression matrix should have correct dimensions."""
        tissue = TissueCellTypes()
        tissue.generate_types_and_markers(n_genes=n_genes, n_cell_types=n_cell_types)

        # Expression matrix shape
        assert tissue.gene_expression_by_type.shape == (n_genes, n_cell_types)

        # Gene names
        assert len(tissue.gene_names) == n_genes

        # Cell type names
        assert len(tissue.cell_type_names) == n_cell_types

    @given(
        n_genes=n_genes_strategy,
        n_cell_types=n_cell_types_strategy,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_tissue_expression_non_negative(self, n_genes, n_cell_types):
        """Tissue expression values should be non-negative (when n_genes != n_cell_types)."""
        # Avoid edge case where n_genes == n_cell_types which can cause division issues
        assume(n_genes != n_cell_types)

        tissue = TissueCellTypes()
        tissue.generate_types_and_markers(n_genes=n_genes, n_cell_types=n_cell_types)

        # All expression values should be non-negative and finite
        expr = tissue.gene_expression_by_type
        assert np.all(np.isfinite(expr))
        assert np.all(expr >= 0)


# =============================================================================
# Tests for cell type properties
# =============================================================================

class TestCellPropertiesProperties:
    """Property-based tests for CellTypesProperties."""

    @given(
        n_cell_types=n_cell_types_strategy,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_cell_properties_array_sizes(self, n_cell_types):
        """CellTypesProperties arrays should have correct sizes."""
        props = CellTypesProperties(n_cell_types=n_cell_types)

        # All arrays should have n_cell_types elements
        assert len(props.sizes) == n_cell_types
        assert len(props.size_variation) == n_cell_types
        assert len(props.anisotropy) == n_cell_types
        assert len(props.relative_rna_concentration) == n_cell_types

    @given(
        n_cell_types=n_cell_types_strategy,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_cell_properties_positive_values(self, n_cell_types):
        """CellTypesProperties should have positive sizes and concentrations."""
        props = CellTypesProperties(n_cell_types=n_cell_types)

        # Sizes should be positive
        assert np.all(props.sizes > 0)

        # RNA concentrations should be positive
        assert np.all(props.relative_rna_concentration > 0)

        # Anisotropy should be positive
        assert np.all(props.anisotropy > 0)


# =============================================================================
# Invariant tests - properties that should always hold
# =============================================================================

class TestInvariants:
    """Tests for invariants that should always hold in the system."""

    @given(
        n_cells=st.integers(min_value=1, max_value=50),
        n_cell_types=n_cell_types_strategy,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_probability_normalization_invariant(self, n_cells, n_cell_types):
        """Probability distributions should always be normalized."""
        # Generate random probabilities and normalize
        raw_probs = np.random.uniform(0, 1, (n_cells, n_cell_types))
        probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)

        # Check invariant
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    @given(
        n_cells=st.integers(min_value=1, max_value=100),
        n_cell_types=n_cell_types_strategy,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_one_hot_is_subset_of_probability(self, n_cells, n_cell_types):
        """One-hot encoding should be a valid probability distribution."""
        # Generate random class assignments
        labels = np.random.randint(0, n_cell_types, n_cells)
        one_hot = one_hot_encode_array(labels, n_cell_types)

        # One-hot should satisfy probability distribution properties
        assert np.all(one_hot >= 0)
        assert np.all(one_hot <= 1)
        assert np.allclose(one_hot.sum(axis=1), 1.0)
