"""Covariate-based cell type assignment rules."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Any, Union
import numpy as np
from numpy.typing import NDArray

from .base import CellTypeRuleBase


class CovariateBasedRule(CellTypeRuleBase):
    """Rule that assigns cell types based on covariate values.

    Allows cell type probabilities to vary based on spatial or contextual
    covariates, enabling complex patterns tied to measurable factors.

    Parameters
    ----------
    n_cell_types : int
        Number of cell types.
    covariate_name : str
        Name of the covariate to use for assignment.
    mapping : dict or callable
        How covariate values map to cell type probabilities.
        If dict: maps covariate values to probability arrays or type indices.
        If callable: function(covariate_value) -> probability array or type index.
    default_probs : np.ndarray, optional
        Default probability distribution when covariate is not available.
        If None, uses uniform distribution.
    covariate_getter : callable, optional
        Function to compute covariate values from cell positions.
        Signature: (cell_centroids: np.ndarray) -> np.ndarray of covariate values.
        If None, covariate values must be passed via kwargs.

    Examples
    --------
    >>> # Map tissue regions to cell type probabilities
    >>> region_mapping = {
    ...     "cortex": [0.8, 0.1, 0.1],
    ...     "medulla": [0.2, 0.6, 0.2],
    ...     "boundary": [0.3, 0.3, 0.4],
    ... }
    >>> rule = CovariateBasedRule(
    ...     n_cell_types=3,
    ...     covariate_name="region",
    ...     mapping=region_mapping,
    ... )

    >>> # Use a distance-based covariate computed from positions
    >>> def distance_from_center(centroids):
    ...     center = centroids.mean(axis=0)
    ...     return np.linalg.norm(centroids - center, axis=1)
    >>>
    >>> def distance_to_probs(distance):
    ...     if distance < 100:
    ...         return [0.9, 0.05, 0.05]
    ...     elif distance < 200:
    ...         return [0.3, 0.5, 0.2]
    ...     else:
    ...         return [0.1, 0.2, 0.7]
    >>>
    >>> rule = CovariateBasedRule(
    ...     n_cell_types=3,
    ...     covariate_name="distance",
    ...     mapping=distance_to_probs,
    ...     covariate_getter=distance_from_center,
    ... )
    """

    def __init__(
        self,
        n_cell_types: int,
        covariate_name: str,
        mapping: Union[Dict[Any, Any], Callable],
        default_probs: Optional[NDArray[np.floating]] = None,
        covariate_getter: Optional[Callable[[NDArray], NDArray]] = None,
    ) -> None:
        super().__init__(n_cell_types)
        self.covariate_name = covariate_name
        self.mapping = mapping
        self.covariate_getter = covariate_getter

        if default_probs is not None:
            if len(default_probs) != n_cell_types:
                raise ValueError(
                    f"default_probs length ({len(default_probs)}) must match n_cell_types ({n_cell_types})"
                )
            self.default_probs = np.array(default_probs) / np.sum(default_probs)
        else:
            self.default_probs = np.ones(n_cell_types) / n_cell_types

        self._element_center: Optional[NDArray] = None
        self._element_scale: Optional[float] = None

    def adapt_rule_to_element(self, element) -> "CovariateBasedRule":
        """Adapt the rule to an element by storing its geometry.

        Parameters
        ----------
        element : HistologicalElement
            The element to adapt to.

        Returns
        -------
        self
            For method chaining.
        """
        self._element_center = np.array(element.center).flatten()
        self._element_scale = element.scale
        self.is_adapted = True
        return self

    def apply(
        self,
        cell_centroids: NDArray[np.floating],
        current_probs: Optional[NDArray[np.floating]] = None,
        mix: float = 0.9,
        **kwargs,
    ) -> NDArray[np.floating]:
        """Apply covariate-based cell type assignment.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions, shape (n_cells, 2).
        current_probs : np.ndarray, optional
            Existing probabilities to blend with.
        mix : float, optional
            Blending weight when combining with current_probs. Default 0.9.
        **kwargs : dict
            Additional parameters. Can include:
            - covariate_values: np.ndarray of covariate values per cell
            - {covariate_name}: single covariate value applied to all cells

        Returns
        -------
        np.ndarray
            Probability matrix, shape (n_cells, n_cell_types).
        """
        n_cells = cell_centroids.shape[0]
        probs = np.zeros((n_cells, self.n_cell_types), dtype=float)

        if n_cells == 0:
            return probs

        # Get covariate values
        covariate_values = self._get_covariate_values(cell_centroids, **kwargs)

        # Apply mapping to each cell
        for i in range(n_cells):
            cov_value = covariate_values[i]
            probs[i] = self._map_to_probs(cov_value)

        # Normalize
        probs = np.maximum(probs, 1e-12)
        probs = probs / probs.sum(axis=1, keepdims=True)

        if current_probs is not None:
            probs = mix * probs + (1 - mix) * current_probs
            probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def _get_covariate_values(
        self,
        cell_centroids: NDArray[np.floating],
        **kwargs,
    ) -> NDArray:
        """Get covariate values for all cells.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions.
        **kwargs : dict
            May contain covariate values.

        Returns
        -------
        np.ndarray
            Covariate values for each cell.
        """
        n_cells = len(cell_centroids)

        # Check for direct covariate_values in kwargs
        if "covariate_values" in kwargs:
            values = kwargs["covariate_values"]
            if len(values) != n_cells:
                raise ValueError(
                    f"covariate_values length ({len(values)}) must match n_cells ({n_cells})"
                )
            return np.array(values)

        # Check for named covariate in kwargs
        if self.covariate_name in kwargs:
            value = kwargs[self.covariate_name]
            # Single value - apply to all cells
            return np.full(n_cells, value)

        # Use covariate_getter if available
        if self.covariate_getter is not None:
            return self.covariate_getter(cell_centroids)

        # No covariate values available - return None values
        return np.array([None] * n_cells)

    def _map_to_probs(self, covariate_value: Any) -> NDArray[np.floating]:
        """Map a covariate value to probability array.

        Parameters
        ----------
        covariate_value : Any
            The covariate value for a single cell.

        Returns
        -------
        np.ndarray
            Probability array of shape (n_cell_types,).
        """
        if covariate_value is None:
            return self.default_probs.copy()

        # Apply mapping
        if callable(self.mapping):
            result = self.mapping(covariate_value)
        elif isinstance(self.mapping, dict):
            if covariate_value in self.mapping:
                result = self.mapping[covariate_value]
            else:
                return self.default_probs.copy()
        else:
            raise ValueError("mapping must be a dict or callable")

        # Convert result to probability array
        if isinstance(result, int):
            # Single type index
            probs = np.zeros(self.n_cell_types)
            probs[result] = 1.0
            return probs
        else:
            # Probability array
            probs = np.array(result, dtype=float)
            if len(probs) != self.n_cell_types:
                raise ValueError(
                    f"Mapping returned {len(probs)} values, expected {self.n_cell_types}"
                )
            return probs / probs.sum()


class ThresholdCovariateRule(CellTypeRuleBase):
    """Rule that assigns cell types based on covariate thresholds.

    Creates step-like patterns where cell type changes at specific
    threshold values of a continuous covariate.

    Parameters
    ----------
    n_cell_types : int
        Number of cell types.
    covariate_name : str
        Name of the covariate.
    thresholds : list of float
        Threshold values where type changes occur, sorted ascending.
    type_indices : list of int
        Cell type indices for each region. Length should be len(thresholds) + 1.
    covariate_getter : callable, optional
        Function to compute covariate from positions.
    transition_width : float, optional
        Width of transition zone for smooth boundaries. Default 0.0 (sharp).
    sharpness : float, optional
        Steepness of transitions when transition_width > 0. Default 2.0.

    Examples
    --------
    >>> # Distance-based zones: <50 -> type 0, 50-150 -> type 1, >150 -> type 2
    >>> rule = ThresholdCovariateRule(
    ...     n_cell_types=3,
    ...     covariate_name="distance",
    ...     thresholds=[50, 150],
    ...     type_indices=[0, 1, 2],
    ...     covariate_getter=lambda pts: np.linalg.norm(pts - center, axis=1),
    ... )
    """

    def __init__(
        self,
        n_cell_types: int,
        covariate_name: str,
        thresholds: List[float],
        type_indices: List[int],
        covariate_getter: Optional[Callable[[NDArray], NDArray]] = None,
        transition_width: float = 0.0,
        sharpness: float = 2.0,
    ) -> None:
        super().__init__(n_cell_types)
        self.covariate_name = covariate_name
        self.thresholds = sorted(thresholds)
        self.type_indices = type_indices
        self.covariate_getter = covariate_getter
        self.transition_width = transition_width
        self.sharpness = sharpness

        if len(type_indices) != len(thresholds) + 1:
            raise ValueError(
                f"type_indices length ({len(type_indices)}) must be len(thresholds) + 1 "
                f"({len(thresholds) + 1})"
            )

        for ti in type_indices:
            if ti < 0 or ti >= n_cell_types:
                raise ValueError(f"type_index {ti} out of range [0, {n_cell_types})")

    def adapt_rule_to_element(self, element) -> "ThresholdCovariateRule":
        """Adapt the rule to an element.

        Parameters
        ----------
        element : HistologicalElement
            The element to adapt to.

        Returns
        -------
        self
            For method chaining.
        """
        self.is_adapted = True
        return self

    def apply(
        self,
        cell_centroids: NDArray[np.floating],
        current_probs: Optional[NDArray[np.floating]] = None,
        mix: float = 0.9,
        **kwargs,
    ) -> NDArray[np.floating]:
        """Apply threshold-based cell type assignment.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions, shape (n_cells, 2).
        current_probs : np.ndarray, optional
            Existing probabilities to blend with.
        mix : float, optional
            Blending weight. Default 0.9.
        **kwargs : dict
            May contain covariate_values or named covariate.

        Returns
        -------
        np.ndarray
            Probability matrix, shape (n_cells, n_cell_types).
        """
        n_cells = cell_centroids.shape[0]
        probs = np.zeros((n_cells, self.n_cell_types), dtype=float)

        if n_cells == 0:
            return probs

        # Get covariate values
        covariate_values = self._get_covariate_values(cell_centroids, **kwargs)

        if self.transition_width <= 0:
            # Sharp transitions
            for i, value in enumerate(covariate_values):
                zone = self._find_zone(value)
                probs[i, self.type_indices[zone]] = 1.0
        else:
            # Soft transitions using sigmoid
            for i, value in enumerate(covariate_values):
                probs[i] = self._compute_soft_probs(value)

        # Normalize
        probs = np.maximum(probs, 1e-12)
        probs = probs / probs.sum(axis=1, keepdims=True)

        if current_probs is not None:
            probs = mix * probs + (1 - mix) * current_probs
            probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def _get_covariate_values(
        self,
        cell_centroids: NDArray[np.floating],
        **kwargs,
    ) -> NDArray[np.floating]:
        """Get covariate values for all cells."""
        n_cells = len(cell_centroids)

        if "covariate_values" in kwargs:
            return np.array(kwargs["covariate_values"])

        if self.covariate_name in kwargs:
            return np.full(n_cells, kwargs[self.covariate_name])

        if self.covariate_getter is not None:
            return self.covariate_getter(cell_centroids)

        raise ValueError(
            f"No covariate values provided. Pass '{self.covariate_name}' or "
            "provide a covariate_getter."
        )

    def _find_zone(self, value: float) -> int:
        """Find which zone a value falls into.

        Parameters
        ----------
        value : float
            Covariate value.

        Returns
        -------
        int
            Zone index (0 to len(thresholds)).
        """
        for i, threshold in enumerate(self.thresholds):
            if value < threshold:
                return i
        return len(self.thresholds)

    def _compute_soft_probs(self, value: float) -> NDArray[np.floating]:
        """Compute soft probability distribution with transitions.

        Parameters
        ----------
        value : float
            Covariate value.

        Returns
        -------
        np.ndarray
            Probability array.
        """
        n_zones = len(self.thresholds) + 1
        zone_probs = np.zeros(n_zones)

        for i in range(n_zones):
            prob = 1.0

            # Lower boundary (if not first zone)
            if i > 0:
                threshold = self.thresholds[i - 1]
                normalized = (value - threshold) / max(self.transition_width, 1e-8)
                prob *= 1.0 / (1.0 + np.exp(-self.sharpness * normalized))

            # Upper boundary (if not last zone)
            if i < n_zones - 1:
                threshold = self.thresholds[i]
                normalized = (threshold - value) / max(self.transition_width, 1e-8)
                prob *= 1.0 / (1.0 + np.exp(-self.sharpness * normalized))

            zone_probs[i] = prob

        # Normalize zone probabilities
        zone_probs = zone_probs / (zone_probs.sum() + 1e-12)

        # Map to cell type probabilities
        probs = np.zeros(self.n_cell_types)
        for i, type_idx in enumerate(self.type_indices):
            probs[type_idx] += zone_probs[i]

        return probs


class SpatialCovariateRule(CellTypeRuleBase):
    """Rule that computes covariates from spatial position.

    Convenience class that wraps common spatial covariate computations
    like distance from center, normalized position, or density.

    Parameters
    ----------
    n_cell_types : int
        Number of cell types.
    covariate_type : str
        Type of spatial covariate:
        - "distance_from_center": Distance from element center
        - "normalized_x": X position normalized to element bounds
        - "normalized_y": Y position normalized to element bounds
        - "radial_position": Normalized radial distance (0=center, 1=edge)
        - "angle": Angle from center in radians
    mapping : callable
        Function mapping covariate value to probability array or type index.

    Examples
    --------
    >>> # Radial gradient with 3 zones
    >>> def radial_mapping(r):
    ...     if r < 0.3:
    ...         return [0.9, 0.05, 0.05]
    ...     elif r < 0.7:
    ...         return [0.2, 0.7, 0.1]
    ...     else:
    ...         return [0.1, 0.1, 0.8]
    >>>
    >>> rule = SpatialCovariateRule(
    ...     n_cell_types=3,
    ...     covariate_type="radial_position",
    ...     mapping=radial_mapping,
    ... )
    """

    def __init__(
        self,
        n_cell_types: int,
        covariate_type: str,
        mapping: Callable[[float], Any],
    ) -> None:
        super().__init__(n_cell_types)
        valid_types = {
            "distance_from_center",
            "normalized_x",
            "normalized_y",
            "radial_position",
            "angle",
        }
        if covariate_type not in valid_types:
            raise ValueError(f"covariate_type must be one of {valid_types}")

        self.covariate_type = covariate_type
        self.mapping = mapping
        self._center: Optional[NDArray] = None
        self._scale: Optional[float] = None
        self._bounds: Optional[NDArray] = None

    def adapt_rule_to_element(self, element) -> "SpatialCovariateRule":
        """Adapt the rule to an element.

        Parameters
        ----------
        element : HistologicalElement
            The element to adapt to.

        Returns
        -------
        self
            For method chaining.
        """
        self._center = np.array(element.center).flatten()
        self._scale = element.scale
        if hasattr(element, 'polygon') and element.polygon is not None:
            bounds = element.polygon.bounds
            self._bounds = np.array([bounds[0], bounds[1], bounds[2], bounds[3]])
        else:
            self._bounds = np.array([
                self._center[0] - self._scale,
                self._center[1] - self._scale,
                self._center[0] + self._scale,
                self._center[1] + self._scale,
            ])
        self.is_adapted = True
        return self

    def apply(
        self,
        cell_centroids: NDArray[np.floating],
        current_probs: Optional[NDArray[np.floating]] = None,
        mix: float = 0.9,
        **kwargs,
    ) -> NDArray[np.floating]:
        """Apply spatial covariate-based cell type assignment.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions, shape (n_cells, 2).
        current_probs : np.ndarray, optional
            Existing probabilities to blend with.
        mix : float, optional
            Blending weight. Default 0.9.
        **kwargs : dict
            Additional parameters (unused).

        Returns
        -------
        np.ndarray
            Probability matrix, shape (n_cells, n_cell_types).
        """
        if not self.is_adapted:
            raise ValueError("Rule not adapted to element. Call adapt_rule_to_element first.")

        n_cells = cell_centroids.shape[0]
        probs = np.zeros((n_cells, self.n_cell_types), dtype=float)

        if n_cells == 0:
            return probs

        # Compute covariate values
        covariate_values = self._compute_covariate(cell_centroids)

        # Apply mapping
        for i, value in enumerate(covariate_values):
            result = self.mapping(value)
            if isinstance(result, int):
                probs[i, result] = 1.0
            else:
                probs[i] = np.array(result)

        # Normalize
        probs = np.maximum(probs, 1e-12)
        probs = probs / probs.sum(axis=1, keepdims=True)

        if current_probs is not None:
            probs = mix * probs + (1 - mix) * current_probs
            probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def _compute_covariate(self, cell_centroids: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute spatial covariate values.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions, shape (n_cells, 2).

        Returns
        -------
        np.ndarray
            Covariate values for each cell.
        """
        if self.covariate_type == "distance_from_center":
            return np.linalg.norm(cell_centroids - self._center, axis=1)

        elif self.covariate_type == "normalized_x":
            width = self._bounds[2] - self._bounds[0]
            return (cell_centroids[:, 0] - self._bounds[0]) / max(width, 1e-8)

        elif self.covariate_type == "normalized_y":
            height = self._bounds[3] - self._bounds[1]
            return (cell_centroids[:, 1] - self._bounds[1]) / max(height, 1e-8)

        elif self.covariate_type == "radial_position":
            distances = np.linalg.norm(cell_centroids - self._center, axis=1)
            return distances / max(self._scale, 1e-8)

        elif self.covariate_type == "angle":
            relative = cell_centroids - self._center
            return np.arctan2(relative[:, 1], relative[:, 0])

        else:
            raise ValueError(f"Unknown covariate_type: {self.covariate_type}")
