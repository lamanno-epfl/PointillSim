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
            frame_size, n_vertices, scale, center, tipical_cell_spacing, rules
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

    Notes
    -----
    This element must be used after other elements in FOVDistribution
    as it requires context_knowledge from previously generated elements.

    Warning
    -------
    Currently has a limitation where it only works once per FOV generation.
    """

    def __init__(self, rules):
        super().__init__(rules=rules)

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
        """Generate by aggregating cells from context knowledge."""
        other = copy.deepcopy(self)
        other.polygon = None
        other.cell_centroids = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        realized_elements = kwargs["context_knowledge"]
        bg = realized_elements[0]
        other.polygon = bg.polygon
        other.cell_centroids = np.row_stack(
            [i.cell_centroids for i in realized_elements]
        )
        other.probs_to_update = np.row_stack(
            [i.cell_probabilities for i in realized_elements]
        )
        tmp = [i.adapt_rule_to_element(other) for i in other.rules]
        other.rules = [
            DummyRule(
                n_cell_types=other.cell_centroids.shape[1],
                probs_to_copy=other.probs_to_update,
            )
        ] + tmp
        other.cell_probabilities = other.apply_rules(other.rules)
        return other

    def is_inside(self, points):
        """Always returns True (hack to remove the rest)."""
        return np.ones(len(points), dtype=bool)

    def is_outside(self, points):
        """Always returns False."""
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix):
        """Remove cells at specified indices."""
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]
