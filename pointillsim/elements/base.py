"""Base histological element class."""

import copy
import time
import numpy as np
from shapely.geometry import Point, Polygon

from ..utils.geometry import generate_uniform_points_in_circle


class HistologicalElement:
    """Base class for tissue regions that generate cell populations.

    A histological element represents a bounded region of tissue with its own
    cell population and cell type assignment rules. Elements generate polygonal
    boundaries and populate them with cell centroids on a quasi-hexagonal grid.

    Parameters
    ----------
    frame_size : int, optional
        Size of the containing field of view in pixels. Default is 5000.
    n_vertices : int or tuple, optional
        Number of vertices for the bounding polygon. If tuple (min, max),
        randomly sampled. Default is (10, 15).
    scale : float, optional
        Approximate radius of the element. Default is 200.
    fixed_center : np.ndarray, optional
        If provided, fixes the element center. Otherwise randomly placed.
    tipical_cell_spacing : float, optional
        Average distance between cell centroids. Default is 8.
    rules : CellTypeRuleBase or list, optional
        Rule(s) for assigning cell type probabilities. Required.

    Attributes
    ----------
    polygon : shapely.Polygon
        The element's bounding polygon (after generate()).
    cell_centroids : np.ndarray
        Cell centroid positions, shape (n_cells, 2).
    cell_probabilities : np.ndarray
        Cell type probabilities, shape (n_cells, n_cell_types).

    Raises
    ------
    ValueError
        If no rules are provided.

    See Also
    --------
    FrameWideElement : Element spanning the entire frame.
    VacuolatedStructure : Element with a central hole.

    Examples
    --------
    >>> from pointillsim.rules import RandomCellTypeRule
    >>> rule = RandomCellTypeRule(n_cell_types=5)
    >>> element = HistologicalElement(frame_size=1000, scale=200, rules=rule)
    >>> realized = element.generate()
    >>> realized.cell_centroids.shape[1]
    2
    """
    def __init__(
        self,
        frame_size=5000,
        n_vertices=(10, 15),
        scale=200,
        fixed_center=None,
        tipical_cell_spacing=8,
        rules=None,
    ):
        self.n_vertices = n_vertices
        self.frame_size = frame_size
        self.scale = scale
        self.fixed_center = fixed_center
        self.tipical_cell_spacing = tipical_cell_spacing
        self.polygon = None
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
        """Generate a new realization of this histological element.

        Creates a deep copy of the element template and generates:
        1. A random bounding polygon
        2. Cell centroids within the polygon
        3. Cell type probabilities via the configured rules

        Parameters
        ----------
        **kwargs : dict
            Additional context passed to subclasses (e.g., context_knowledge
            for FrameWideUpdater).

        Returns
        -------
        HistologicalElement
            A new element instance with generated geometry and cells.
        """
        other = copy.deepcopy(self)
        other.polygon = None
        other.cell_centroids = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        other.polygon = other.generate_bounding_polygon()
        other.cell_centroids = other.generate_cell_centroids()
        other.rules = [i.adapt_rule_to_element(other) for i in other.rules]
        other.cell_probabilities = other.apply_rules(other.rules)
        return other

    def generate_bounding_polygon(self):
        """Generate a convex polygon boundary for this element.

        Creates a random convex hull from points distributed within a circle
        centered at the element's center with radius equal to scale.

        Returns
        -------
        shapely.Polygon
            Convex polygon defining the element boundary.
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
        return polygon

    @property
    def bounding_box(self):
        """tuple: Bounding box (minx, miny, maxx, maxy) of the polygon."""
        return self.polygon.bounds

    @property
    def class_instance(self):
        """np.ndarray: Sampled cell type indices from probabilities."""
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def class_instance_one_hot(self):
        """np.ndarray: One-hot encoded sampled cell types.

        Lazily sampled from cell_probabilities using multinomial distribution.
        """
        if self._class_instance_one_hot is None:
            self._class_instance_one_hot = self.rng.multinomial(
                n=1, pvals=self.cell_probabilities
            )
        return self._class_instance_one_hot

    @property
    def ML_class(self):
        """np.ndarray: Maximum likelihood cell type (argmax of probabilities)."""
        return np.argmax(self.cell_probabilities, axis=1)

    def is_inside(self, points):
        """Check which points are inside the element polygon.

        Parameters
        ----------
        points : np.ndarray
            Point coordinates, shape (n_points, 2).

        Returns
        -------
        np.ndarray
            Boolean array of length n_points.
        """
        return np.array(
            [self.polygon.contains(Point(point[0], point[1])) for point in points],
            dtype=bool,
        )

    def is_outside(self, points):
        """Check which points are outside the element polygon."""
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix):
        """Remove cells at specified indices.

        Parameters
        ----------
        bool_ix : np.ndarray
            Boolean mask where True indicates cells to remove.
        """
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]

    def generate_cell_centroids(self):
        """Generate cell centroids on a quasi-hexagonal grid within the polygon.

        Creates a grid of points with alternating row offsets (hexagonal packing),
        removes points outside the polygon, and adds jitter for natural appearance.

        Returns
        -------
        np.ndarray
            Cell centroid coordinates, shape (n_cells, 2).

        Notes
        -----
        The hexagonal arrangement provides efficient packing while the jitter
        prevents artificial regularity in the final simulation.
        """
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

        points = points[self.is_inside(points)]

        points += np.random.normal(0, self.tipical_cell_spacing / 5.0, points.shape)
        return points

    def apply_rules(self, rules):
        """Apply a sequence of rules to compute cell type probabilities.

        Parameters
        ----------
        rules : list of CellTypeRuleBase
            Rules to apply sequentially.

        Returns
        -------
        np.ndarray
            Cell type probability matrix, shape (n_cells, n_cell_types).
        """
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
