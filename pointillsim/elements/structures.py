"""Specialized histological structures."""

import copy
import time
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.affinity import scale as shapely_scale

from ..utils.geometry import generate_uniform_points_in_circle


class VacuolatedStructure:
    """Histological element with a central hole (vacuole or lumen).

    Represents ring-shaped tissue structures such as glands, vessels,
    or any structure with a hollow center. Cells are generated in the
    annular region between the outer polygon and inner hole.

    Parameters
    ----------
    frame_size : int, optional
        Size of the FOV in pixels. Default is 5000.
    n_vertices : int or tuple, optional
        Number of vertices for the outer polygon. Default is (10, 15).
    scale : float, optional
        Approximate outer radius. Default is 200.
    fixed_center : np.ndarray, optional
        Fixed center position, or None for random placement.
    tipical_cell_spacing : float, optional
        Average cell spacing. Default is 8.
    hole_scale_factor : float, optional
        Size of the inner hole relative to outer polygon.
        0.75 means the hole is 75% the size of the outer boundary.
        Default is 0.75.
    rules : CellTypeRuleBase or list
        Rules for cell type assignment.

    Attributes
    ----------
    hole : shapely.Polygon
        The inner hole polygon.
    """
    def __init__(
        self,
        frame_size=5000,
        n_vertices=(10, 15),
        scale=200,
        fixed_center=None,
        tipical_cell_spacing=8,
        hole_scale_factor=0.75,
        rules=None,
    ):
        self.n_vertices = n_vertices
        self.frame_size = frame_size
        self.scale = scale
        self.fixed_center = fixed_center
        self.tipical_cell_spacing = tipical_cell_spacing
        self.hole_scale_factor = hole_scale_factor
        self.polygon = None
        self.hole = None
        self.cell_centroids = None

        if rules is None:
            raise ValueError("No rules provided")
        self.rules = rules
        if not isinstance(self.rules, list):
            self.rules = [self.rules]
        self.original_rules = self.rules
        self.rng = np.random.default_rng(seed=int(time.time() * 1e6))
        self._class_instance_one_hot = None

    def generate(self, **kwargs):
        """Generate a new realization of this vacuolated structure."""
        other = copy.deepcopy(self)
        other.polygon = None
        other.hole = None
        other.cell_centroids = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        other.polygon, other.hole = other.generate_polygon_andhole()
        other.cell_centroids = other.generate_cell_centroids()
        other.rules = [i.adapt_rule_to_element(other) for i in other.rules]
        other.cell_probabilities = other.apply_rules(other.rules)
        return other

    def generate_polygon_andhole(self):
        """Generate outer polygon and scaled inner hole.

        Returns
        -------
        tuple
            (polygon, hole) where both are shapely.Polygon objects.
            The hole is the outer polygon scaled by hole_scale_factor.
        """
        num_points = (
            np.random.randint(self.n_vertices[0], self.n_vertices[1])
            if isinstance(self.n_vertices, tuple)
            else self.n_vertices
        )
        if self.fixed_center is None or self.fixed_center is False:
            self.center = np.random.uniform(
                self.frame_size * 0.1, self.frame_size * 0.9, (1, 2)
            )
        else:
            self.center = self.fixed_center
        points = generate_uniform_points_in_circle(self.center, self.scale, num_points)
        polygon = Polygon(points).convex_hull
        hole = shapely_scale(
            polygon,
            xfact=self.hole_scale_factor,
            yfact=self.hole_scale_factor,
            origin=polygon.centroid,
        )
        return polygon, hole

    @property
    def bounding_box(self):
        """tuple: Bounding box of the polygon."""
        return self.polygon.bounds

    @property
    def class_instance(self):
        """np.ndarray: Sampled cell type indices."""
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def class_instance_one_hot(self):
        """np.ndarray: One-hot encoded sampled cell types."""
        if self._class_instance_one_hot is None:
            self._class_instance_one_hot = self.rng.multinomial(
                n=1, pvals=self.cell_probabilities
            )
        return self._class_instance_one_hot

    @property
    def ML_class(self):
        """np.ndarray: Maximum likelihood cell type."""
        return np.argmax(self.cell_probabilities, axis=1)

    def is_inside(self, points, consider_hole=False):
        """Check which points are inside the element (optionally excluding hole)."""
        if consider_hole:
            list_bool = []
            for point in points:
                px = Point(point[0], point[1])
                list_bool.append(
                    self.polygon.contains(px) and not self.hole.contains(px)
                )
            return np.array(list_bool, dtype=bool)
        else:
            return np.array(
                [self.polygon.contains(Point(point[0], point[1])) for point in points],
                dtype=bool,
            )

    def is_outside(self, points):
        """Check which points are outside the element."""
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix):
        """Remove cells at specified indices."""
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]

    def generate_cell_centroids(self):
        """Generate cell centroids in the annular region (excluding hole)."""
        x = np.arange(
            self.bounding_box[0], self.bounding_box[2], self.tipical_cell_spacing
        )
        y = np.arange(
            self.bounding_box[1],
            self.bounding_box[3],
            self.tipical_cell_spacing * np.sin(np.pi / 3),
        )
        X, Y = np.meshgrid(x, y)
        X[::2] += self.tipical_cell_spacing / 2.0
        points = np.stack((X.flatten(), Y.flatten()), axis=1)

        points = points[self.is_inside(points, consider_hole=True)]

        points += np.random.normal(0, self.tipical_cell_spacing / 5.0, points.shape)
        return points

    def apply_rules(self, rules):
        """Apply rules to compute cell type probabilities."""
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
