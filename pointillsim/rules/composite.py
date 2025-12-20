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
