"""Composite and distance-based cell type assignment rules."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point

from .base import CellTypeRuleBase


class DistanceBasedRule(CellTypeRuleBase):
    """Rule that assigns cell types based on distance from element center or boundary.

    Creates radial or concentric patterns by assigning cell type probabilities
    based on each cell's distance from a reference point (usually element center)
    or from the element boundary.

    Parameters
    ----------
    n_cell_types : int
        Number of cell types.
    reference : str, optional
        What to measure distance from: "center" or "boundary". Default "center".
    inner_type : int, optional
        Cell type index for cells close to reference. Default 0.
    outer_type : int, optional
        Cell type index for cells far from reference. Default 1.
    transition_distance : float, optional
        Distance at which transition occurs (in pixels). If None, uses
        half the element scale. Default None.
    transition_width : float, optional
        Width of the transition zone for smooth blending. Default 20.0.
    sharpness : float, optional
        Controls steepness of transition. Higher = sharper. Default 1.0.

    Examples
    --------
    >>> rule = DistanceBasedRule(
    ...     n_cell_types=5,
    ...     inner_type=0,
    ...     outer_type=2,
    ...     transition_distance=100
    ... )
    >>> element = HistologicalElement(scale=200, rules=rule)
    """

    def __init__(
        self,
        n_cell_types: int,
        reference: str = "center",
        inner_type: int = 0,
        outer_type: int = 1,
        transition_distance: Optional[float] = None,
        transition_width: float = 20.0,
        sharpness: float = 1.0,
    ) -> None:
        super().__init__(n_cell_types)
        if reference not in ("center", "boundary"):
            raise ValueError("reference must be 'center' or 'boundary'")
        self.reference = reference
        self.inner_type = inner_type
        self.outer_type = outer_type
        self.transition_distance = transition_distance
        self.transition_width = transition_width
        self.sharpness = sharpness
        self._center: Optional[NDArray] = None
        self._polygon = None
        self._scale: Optional[float] = None

    def adapt_rule_to_element(self, element) -> DistanceBasedRule:
        """Adapt the rule to an element by storing its center and polygon.

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
        self._polygon = element.polygon
        self._scale = element.scale
        self.is_adapted = True
        return self

    def apply(
        self,
        cell_centroids: NDArray[np.floating],
        current_probs: Optional[NDArray[np.floating]] = None,
        mix: float = 0.9,
    ) -> NDArray[np.floating]:
        """Apply distance-based cell type assignment.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions, shape (n_cells, 2).
        current_probs : np.ndarray, optional
            Existing probabilities to blend with.
        mix : float, optional
            Blending weight. Default 0.9.

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

        # Calculate distances
        if self.reference == "center":
            distances = np.linalg.norm(cell_centroids - self._center, axis=1)
        else:  # boundary
            distances = np.array([
                self._polygon.boundary.distance(Point(x, y))
                for x, y in cell_centroids
            ])

        # Determine transition distance
        trans_dist = self.transition_distance
        if trans_dist is None:
            trans_dist = self._scale / 2 if self._scale else 100.0

        # Calculate sigmoid transition
        # sigmoid: 1/(1 + exp(-sharpness * (d - trans_dist) / trans_width))
        normalized = (distances - trans_dist) / max(self.transition_width, 1e-8)
        weights = 1.0 / (1.0 + np.exp(-self.sharpness * normalized))

        # Assign probabilities
        probs[:, self.inner_type] = 1.0 - weights
        probs[:, self.outer_type] = weights

        # Normalize
        probs = np.maximum(probs, 1e-12)
        probs = probs / probs.sum(axis=1, keepdims=True)

        if current_probs is not None:
            probs = mix * probs + (1 - mix) * current_probs
            probs = probs / probs.sum(axis=1, keepdims=True)

        return probs


class CompositeRule(CellTypeRuleBase):
    """Rule that combines multiple rules with configurable weights.

    Allows combining different cell type assignment strategies,
    either by averaging probabilities or by spatial masking.

    Parameters
    ----------
    rules : list of CellTypeRuleBase
        Rules to combine.
    weights : list of float, optional
        Weights for each rule (will be normalized). If None, equal weights.
    mode : str, optional
        How to combine: "average" for weighted average of probabilities,
        "sequential" to apply rules in order. Default "average".

    Examples
    --------
    >>> rule1 = RandomCellTypeRule(n_cell_types=5)
    >>> rule2 = SingleTypeRule(n_cell_types=5, cell_type_ix=0)
    >>> composite = CompositeRule(
    ...     rules=[rule1, rule2],
    ...     weights=[0.7, 0.3],
    ...     mode="average"
    ... )
    """

    def __init__(
        self,
        rules: List[CellTypeRuleBase],
        weights: Optional[List[float]] = None,
        mode: str = "average",
    ) -> None:
        if not rules:
            raise ValueError("At least one rule must be provided")

        n_cell_types = rules[0].n_cell_types
        for rule in rules:
            if rule.n_cell_types != n_cell_types:
                raise ValueError("All rules must have the same n_cell_types")

        super().__init__(n_cell_types)
        self.rules = rules
        self.mode = mode

        if weights is None:
            self.weights = np.ones(len(rules)) / len(rules)
        else:
            if len(weights) != len(rules):
                raise ValueError("weights must have same length as rules")
            self.weights = np.array(weights) / np.sum(weights)

    def adapt_rule_to_element(self, element) -> CompositeRule:
        """Adapt all contained rules to the element.

        Parameters
        ----------
        element : HistologicalElement
            The element to adapt to.

        Returns
        -------
        self
            For method chaining.
        """
        for rule in self.rules:
            rule.adapt_rule_to_element(element)
        self.is_adapted = True
        return self

    def apply(
        self,
        cell_centroids: NDArray[np.floating],
        current_probs: Optional[NDArray[np.floating]] = None,
        **kwargs,
    ) -> NDArray[np.floating]:
        """Apply combined rules to assign cell type probabilities.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions, shape (n_cells, 2).
        current_probs : np.ndarray, optional
            Existing probabilities (used if mode="sequential").
        **kwargs : dict
            Additional arguments passed to individual rules.

        Returns
        -------
        np.ndarray
            Combined probability matrix, shape (n_cells, n_cell_types).
        """
        n_cells = cell_centroids.shape[0]

        if n_cells == 0:
            return np.empty((0, self.n_cell_types))

        if self.mode == "average":
            combined = np.zeros((n_cells, self.n_cell_types))
            for rule, weight in zip(self.rules, self.weights):
                probs = rule.apply(cell_centroids, current_probs=None, **kwargs)
                combined += weight * probs

            # Normalize
            combined = np.maximum(combined, 1e-12)
            combined = combined / combined.sum(axis=1, keepdims=True)
            return combined

        elif self.mode == "sequential":
            probs = current_probs
            for rule in self.rules:
                probs = rule.apply(cell_centroids, current_probs=probs, **kwargs)
            return probs

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class LayerRule(CellTypeRuleBase):
    """Rule that assigns cell types in concentric layers based on distance from center.

    Creates layered patterns similar to cortical layers or onion-like structures,
    where different cell types occupy concentric rings at different distances
    from the element center.

    Parameters
    ----------
    n_cell_types : int
        Number of cell types.
    layer_types : list of int
        Cell type indices for each layer, from center outward.
        E.g., [0, 1, 2] means type 0 in center, type 1 in middle, type 2 at edge.
    layer_boundaries : list of float, optional
        Normalized boundaries between layers as fractions of element scale.
        E.g., [0.3, 0.7] creates 3 layers at 0-30%, 30-70%, 70-100% of radius.
        If None, layers are evenly spaced.
    transition_width : float, optional
        Width of transition zone between layers (in pixels). Default 10.0.
    sharpness : float, optional
        Controls steepness of transitions. Higher = sharper. Default 2.0.

    Examples
    --------
    >>> # Create 3 concentric layers
    >>> rule = LayerRule(
    ...     n_cell_types=5,
    ...     layer_types=[0, 2, 4],  # Types for inner, middle, outer
    ...     layer_boundaries=[0.4, 0.7],  # Transition at 40% and 70% of radius
    ... )
    """

    def __init__(
        self,
        n_cell_types: int,
        layer_types: List[int],
        layer_boundaries: Optional[List[float]] = None,
        transition_width: float = 10.0,
        sharpness: float = 2.0,
    ) -> None:
        super().__init__(n_cell_types)

        if len(layer_types) < 2:
            raise ValueError("At least 2 layer types are required")

        for lt in layer_types:
            if lt < 0 or lt >= n_cell_types:
                raise ValueError(f"layer_type {lt} out of range [0, {n_cell_types})")

        self.layer_types = layer_types
        self.n_layers = len(layer_types)

        # Set up layer boundaries
        if layer_boundaries is None:
            # Evenly spaced layers
            self.layer_boundaries = [
                (i + 1) / self.n_layers for i in range(self.n_layers - 1)
            ]
        else:
            if len(layer_boundaries) != self.n_layers - 1:
                raise ValueError(
                    f"layer_boundaries must have {self.n_layers - 1} values for {self.n_layers} layers"
                )
            self.layer_boundaries = sorted(layer_boundaries)

        self.transition_width = transition_width
        self.sharpness = sharpness
        self._center: Optional[NDArray] = None
        self._scale: Optional[float] = None

    def adapt_rule_to_element(self, element) -> "LayerRule":
        """Adapt the rule to an element by storing its center and scale.

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
        self.is_adapted = True
        return self

    def apply(
        self,
        cell_centroids: NDArray[np.floating],
        current_probs: Optional[NDArray[np.floating]] = None,
        mix: float = 0.9,
    ) -> NDArray[np.floating]:
        """Apply layered cell type assignment based on distance from center.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions, shape (n_cells, 2).
        current_probs : np.ndarray, optional
            Existing probabilities to blend with.
        mix : float, optional
            Blending weight. Default 0.9.

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

        # Calculate normalized distances (0 = center, 1 = at scale radius)
        distances = np.linalg.norm(cell_centroids - self._center, axis=1)
        normalized_distances = distances / max(self._scale, 1e-8)

        # Convert boundaries to absolute distances
        abs_boundaries = [b * self._scale for b in self.layer_boundaries]

        # Compute layer membership probabilities using soft transitions
        # For each boundary, compute sigmoid transition
        layer_probs = np.zeros((n_cells, self.n_layers))

        for i in range(self.n_layers):
            # Start with full probability
            layer_prob = np.ones(n_cells)

            # Apply lower boundary (if not innermost layer)
            if i > 0:
                boundary = abs_boundaries[i - 1]
                normalized = (distances - boundary) / max(self.transition_width, 1e-8)
                lower_weight = 1.0 / (1.0 + np.exp(-self.sharpness * normalized))
                layer_prob *= lower_weight

            # Apply upper boundary (if not outermost layer)
            if i < self.n_layers - 1:
                boundary = abs_boundaries[i]
                normalized = (boundary - distances) / max(self.transition_width, 1e-8)
                upper_weight = 1.0 / (1.0 + np.exp(-self.sharpness * normalized))
                layer_prob *= upper_weight

            layer_probs[:, i] = layer_prob

        # Normalize layer probabilities
        layer_probs = np.maximum(layer_probs, 1e-12)
        layer_probs = layer_probs / layer_probs.sum(axis=1, keepdims=True)

        # Map layer probabilities to cell type probabilities
        for i, cell_type in enumerate(self.layer_types):
            probs[:, cell_type] += layer_probs[:, i]

        # Normalize
        probs = np.maximum(probs, 1e-12)
        probs = probs / probs.sum(axis=1, keepdims=True)

        if current_probs is not None:
            probs = mix * probs + (1 - mix) * current_probs
            probs = probs / probs.sum(axis=1, keepdims=True)

        return probs


class GradientRule(CellTypeRuleBase):
    """Rule that assigns cell types along a linear gradient direction.

    Creates directional patterns where cell type probabilities vary along
    a specified axis (e.g., left-to-right, top-to-bottom, or diagonal).

    Parameters
    ----------
    n_cell_types : int
        Number of cell types.
    start_type : int
        Cell type index at the start of the gradient.
    end_type : int
        Cell type index at the end of the gradient.
    direction : str or tuple, optional
        Gradient direction. Can be:
        - "horizontal" or "x": left to right
        - "vertical" or "y": bottom to top
        - "diagonal": bottom-left to top-right
        - tuple (dx, dy): custom direction vector
        Default "horizontal".
    transition_center : float, optional
        Normalized position (0-1) where transition occurs. Default 0.5.
    transition_width : float, optional
        Width of transition zone as fraction of element extent. Default 0.3.
    sharpness : float, optional
        Controls steepness of transition. Higher = sharper. Default 2.0.

    Examples
    --------
    >>> # Horizontal gradient from type 0 (left) to type 2 (right)
    >>> rule = GradientRule(
    ...     n_cell_types=5,
    ...     start_type=0,
    ...     end_type=2,
    ...     direction="horizontal",
    ... )
    >>>
    >>> # Diagonal gradient
    >>> rule = GradientRule(
    ...     n_cell_types=5,
    ...     start_type=1,
    ...     end_type=3,
    ...     direction=(1, 1),  # Custom direction
    ... )
    """

    def __init__(
        self,
        n_cell_types: int,
        start_type: int,
        end_type: int,
        direction: Union[str, Tuple[float, float]] = "horizontal",
        transition_center: float = 0.5,
        transition_width: float = 0.3,
        sharpness: float = 2.0,
    ) -> None:
        super().__init__(n_cell_types)

        if start_type < 0 or start_type >= n_cell_types:
            raise ValueError(f"start_type {start_type} out of range [0, {n_cell_types})")
        if end_type < 0 or end_type >= n_cell_types:
            raise ValueError(f"end_type {end_type} out of range [0, {n_cell_types})")

        self.start_type = start_type
        self.end_type = end_type
        self.transition_center = transition_center
        self.transition_width = transition_width
        self.sharpness = sharpness

        # Parse direction
        if isinstance(direction, str):
            direction_map = {
                "horizontal": (1, 0),
                "x": (1, 0),
                "vertical": (0, 1),
                "y": (0, 1),
                "diagonal": (1, 1),
            }
            if direction.lower() not in direction_map:
                raise ValueError(
                    f"Unknown direction '{direction}'. Use 'horizontal', 'vertical', "
                    "'diagonal', or a (dx, dy) tuple."
                )
            self.direction = np.array(direction_map[direction.lower()], dtype=float)
        else:
            self.direction = np.array(direction, dtype=float)

        # Normalize direction
        norm = np.linalg.norm(self.direction)
        if norm < 1e-8:
            raise ValueError("Direction vector cannot be zero")
        self.direction = self.direction / norm

        self._center: Optional[NDArray] = None
        self._scale: Optional[float] = None
        self._polygon = None

    def adapt_rule_to_element(self, element) -> "GradientRule":
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
        self._center = np.array(element.center).flatten()
        self._scale = element.scale
        self._polygon = element.polygon
        self.is_adapted = True
        return self

    def apply(
        self,
        cell_centroids: NDArray[np.floating],
        current_probs: Optional[NDArray[np.floating]] = None,
        mix: float = 0.9,
    ) -> NDArray[np.floating]:
        """Apply linear gradient cell type assignment.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions, shape (n_cells, 2).
        current_probs : np.ndarray, optional
            Existing probabilities to blend with.
        mix : float, optional
            Blending weight. Default 0.9.

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

        # Project cell positions onto gradient direction
        # relative to element center
        relative_positions = cell_centroids - self._center
        projections = np.dot(relative_positions, self.direction)

        # Normalize projections to [0, 1] range based on element scale
        # Cells at -scale in direction map to 0, at +scale map to 1
        normalized_projections = (projections / self._scale + 1) / 2
        normalized_projections = np.clip(normalized_projections, 0, 1)

        # Apply sigmoid transition
        trans_width = max(self.transition_width * self._scale, 1e-8)
        trans_center = self.transition_center * self._scale * 2 - self._scale

        normalized = (projections - trans_center) / trans_width
        weights = 1.0 / (1.0 + np.exp(-self.sharpness * normalized))

        # Assign probabilities
        probs[:, self.start_type] = 1.0 - weights
        probs[:, self.end_type] = weights

        # Normalize
        probs = np.maximum(probs, 1e-12)
        probs = probs / probs.sum(axis=1, keepdims=True)

        if current_probs is not None:
            probs = mix * probs + (1 - mix) * current_probs
            probs = probs / probs.sum(axis=1, keepdims=True)

        return probs
