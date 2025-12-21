"""Frame-wide histological elements."""

import copy
import numpy as np
from shapely.geometry import Polygon

from .base import HistologicalElement
from ..rules.base import DummyRule


class FrameWideElement(HistologicalElement):
    """Histological element that spans the entire field of view.

    Used as a background element to fill the entire FOV with cells.
    The bounding polygon is a rectangle covering the full frame.

    Parameters
    ----------
    frame_size : int, optional
        Size of the FOV in pixels. Default is 5000.
    n_vertices : int, optional
        Ignored for this class (always 4). Default is 4.
    scale : int, optional
        Ignored for this class. Default is 5000.
    center : np.ndarray, optional
        Ignored for this class (always frame center).
    tipical_cell_spacing : float, optional
        Average distance between cells. Default is 8.
    rules : CellTypeRuleBase or list
        Rules for cell type assignment.
    """

    def __init__(
        self,
        frame_size=5000,
        n_vertices=4,
        scale=5000,
        center=None,
        tipical_cell_spacing=8,
        rules=None,
    ):
        super().__init__(
            frame_size=frame_size,
            n_vertices=n_vertices,
            scale=scale,
            fixed_center=center,
            tipical_cell_spacing=tipical_cell_spacing,
            smoothing_iterations=0,
            rules=rules,
        )

    def generate_bounding_polygon(self):
        """Generate a rectangular polygon covering the entire frame.

        Returns
        -------
        shapely.Polygon
            Rectangle from (0,0) to (frame_size, frame_size).
        """
        self.center = np.array([self.frame_size / 2, self.frame_size / 2])
        return Polygon(
            [
                (0, 0),
                (self.frame_size, 0),
                (self.frame_size, self.frame_size),
                (0, self.frame_size),
                (0, 0),
            ]
        )


class FrameWideUpdater(FrameWideElement):
    """Element that applies rules to all cells from previously generated elements.

    Used to apply global modifications (like neighbor-based rules) to the
    combined cell population from multiple elements. Collects cells from
    context_knowledge and applies its rules to the aggregate.

    Parameters
    ----------
    rules : CellTypeRuleBase or list
        Rules to apply to the combined cell population.
    include_self : bool, optional
        If True, include cells from previous FrameWideUpdater elements in the
        aggregate. If False, skip them to avoid double-processing.
        Default is False.

    Notes
    -----
    This element must be used after other elements in FOVDistribution
    as it requires context_knowledge from previously generated elements.

    Examples
    --------
    >>> from pointillsim.rules import DeterministicNeighborAssignment
    >>> updater = FrameWideUpdater(
    ...     rules=DeterministicNeighborAssignment(n_cell_types=5),
    ... )
    >>> # Use in FOVDistribution with other_elements
    """

    def __init__(self, rules, include_self: bool = False):
        super().__init__(rules=rules)
        self.include_self = include_self
        self._used = False

    def apply_rules(self, rules):
        """Apply rules to cell centroids."""
        probs = None
        for rule in rules:
            probs = rule.apply(self.cell_centroids, current_probs=probs)

        if probs is None:
            n_cells = self.cell_centroids.shape[0]
            n_types = 0
            if rules:
                try:
                    n_types = rules[0].n_cell_types
                except (IndexError, AttributeError):
                    n_types = 0
            probs = np.empty((n_cells, n_types))

        return probs

    def generate(self, **kwargs):
        """Generate by aggregating cells from context knowledge.

        Parameters
        ----------
        **kwargs : dict
            Must include 'context_knowledge' with list of realized elements.

        Returns
        -------
        FrameWideUpdater
            New instance with aggregated cells and updated probabilities.
        """
        other = copy.deepcopy(self)
        other.polygon = None
        other.cell_centroids = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules
        other._used = True

        realized_elements = kwargs.get("context_knowledge", [])

        if not realized_elements:
            # No elements to aggregate
            other.polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
            other.cell_centroids = np.empty((0, 2))
            other.cell_probabilities = np.empty((0, 0))
            return other

        # Filter elements based on include_self setting
        elements_to_process = []
        for elem in realized_elements:
            if not other.include_self and isinstance(elem, FrameWideUpdater):
                continue
            elements_to_process.append(elem)

        if not elements_to_process:
            # All elements were filtered out
            bg = realized_elements[0]
            other.polygon = bg.polygon if hasattr(bg, 'polygon') else Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
            other.cell_centroids = np.empty((0, 2))
            other.cell_probabilities = np.empty((0, 0))
            return other

        bg = elements_to_process[0]
        other.polygon = bg.polygon if hasattr(bg, 'polygon') else Polygon(
            [(0, 0), (other.frame_size, 0), (other.frame_size, other.frame_size), (0, other.frame_size)]
        )

        # Collect cells and probabilities from all elements
        centroids_list = []
        probs_list = []
        for elem in elements_to_process:
            if hasattr(elem, 'cell_centroids') and elem.cell_centroids is not None and len(elem.cell_centroids) > 0:
                centroids_list.append(elem.cell_centroids)
                if hasattr(elem, 'cell_probabilities') and elem.cell_probabilities is not None:
                    probs_list.append(elem.cell_probabilities)

        if not centroids_list:
            other.cell_centroids = np.empty((0, 2))
            other.cell_probabilities = np.empty((0, 0))
            return other

        other.cell_centroids = np.row_stack(centroids_list)

        # Determine number of cell types and pad probabilities if needed
        if probs_list:
            n_types = max(p.shape[1] for p in probs_list if p.shape[0] > 0)
            padded_probs = []
            for p in probs_list:
                if p.shape[1] < n_types:
                    padded = np.zeros((p.shape[0], n_types))
                    padded[:, :p.shape[1]] = p
                    padded_probs.append(padded)
                else:
                    padded_probs.append(p)
            other.probs_to_update = np.row_stack(padded_probs)
        else:
            n_types = other.rules[0].n_cell_types if other.rules else 0
            other.probs_to_update = np.zeros((len(other.cell_centroids), n_types))

        # Adapt rules and create DummyRule with correct n_cell_types
        tmp = [i.adapt_rule_to_element(other) for i in other.rules]
        other.rules = [
            DummyRule(
                n_cell_types=other.probs_to_update.shape[1],  # Fixed: use probs shape, not centroids
                probs_to_copy=other.probs_to_update,
            )
        ] + tmp
        other.cell_probabilities = other.apply_rules(other.rules)
        return other

    def is_inside(self, points):
        """Always returns True for all points.

        This ensures FrameWideUpdater doesn't remove cells from other elements.
        """
        return np.ones(len(points), dtype=bool)

    def is_outside(self, points):
        """Always returns False for all points."""
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix):
        """Remove cells at specified indices."""
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]
        if hasattr(self, 'probs_to_update') and self.probs_to_update is not None:
            self.probs_to_update = self.probs_to_update[~bool_ix]
