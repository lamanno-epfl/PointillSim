"""Specialized histological structures."""

import copy
import time
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point, Polygon, LineString
from shapely.affinity import scale as shapely_scale
from shapely.ops import unary_union

from ..utils.geometry import generate_uniform_points_in_circle, smooth_polygon


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
    smoothing_iterations : int, optional
        Number of Chaikin smoothing iterations to apply to both the outer
        polygon and the inner hole. 0 means no smoothing. Default is 0.
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
        smoothing_iterations=0,
        rules=None,
    ):
        self.n_vertices = n_vertices
        self.frame_size = frame_size
        self.scale = scale
        self.fixed_center = fixed_center
        self.tipical_cell_spacing = tipical_cell_spacing
        self.hole_scale_factor = hole_scale_factor
        self.smoothing_iterations = smoothing_iterations
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

        If smoothing_iterations > 0, applies Chaikin smoothing to both
        the outer polygon and the inner hole.

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

        # Apply Chaikin smoothing if requested
        if self.smoothing_iterations > 0:
            polygon = smooth_polygon(
                polygon, iterations=self.smoothing_iterations, preserve_area=True
            )

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


class LinearLumenStructure:
    """Tube-like histological element with a central lumen (hollow channel).

    Represents elongated hollow structures such as blood vessels, ducts,
    and tubular glands. Cells are generated in the wall region between
    the outer boundary and the inner lumen.

    The structure is defined by a centerline path (straight or curved),
    with configurable wall thickness and lumen diameter. The centerline
    can be automatically generated as a random path or explicitly specified.

    Parameters
    ----------
    frame_size : int, optional
        Size of the FOV in pixels. Default is 5000.
    length : float, optional
        Length of the tube along its centerline. Default is 400.
    outer_radius : float, optional
        Radius from centerline to outer wall. Default is 50.
    wall_thickness : float, optional
        Thickness of the tube wall. The inner lumen radius is
        outer_radius - wall_thickness. Default is 15.
    n_control_points : int, optional
        Number of control points for curved centerlines. Higher values
        create more sinuous paths. Default is 3 (straight line).
    curvature : float, optional
        Maximum perpendicular deviation from straight path, as fraction
        of length. Default is 0.1 (10% of length).
    fixed_start : np.ndarray, optional
        Fixed starting point (x, y) or None for random placement.
    fixed_angle : float, optional
        Fixed orientation angle in radians, or None for random.
    tipical_cell_spacing : float, optional
        Average cell spacing. Default is 8.
    smoothing_iterations : int, optional
        Chaikin smoothing iterations for the tube walls. Default is 2.
    rules : CellTypeRuleBase or list
        Rules for cell type assignment.

    Attributes
    ----------
    polygon : shapely.Polygon
        The outer wall polygon.
    lumen : shapely.Polygon
        The inner lumen polygon.
    centerline : np.ndarray
        Control points defining the tube centerline.

    Examples
    --------
    >>> from pointillsim.rules import LayerRule
    >>> rule = LayerRule(n_cell_types=3, layer_types=[0, 1, 2])
    >>> vessel = LinearLumenStructure(
    ...     length=300, outer_radius=40, wall_thickness=12,
    ...     curvature=0.15, rules=rule
    ... )
    >>> realized = vessel.generate()
    >>> print(f"Generated {realized.cell_centroids.shape[0]} cells")
    """

    def __init__(
        self,
        frame_size: int = 5000,
        length: float = 400,
        outer_radius: float = 50,
        wall_thickness: float = 15,
        n_control_points: int = 3,
        curvature: float = 0.1,
        fixed_start: Optional[NDArray[np.floating]] = None,
        fixed_angle: Optional[float] = None,
        tipical_cell_spacing: float = 8,
        smoothing_iterations: int = 2,
        rules=None,
    ) -> None:
        if rules is None:
            raise ValueError("No rules provided")

        if wall_thickness >= outer_radius:
            raise ValueError(
                f"wall_thickness ({wall_thickness}) must be less than "
                f"outer_radius ({outer_radius})"
            )

        self.frame_size = frame_size
        self.length = length
        self.outer_radius = outer_radius
        self.wall_thickness = wall_thickness
        self.inner_radius = outer_radius - wall_thickness
        self.n_control_points = max(2, n_control_points)
        self.curvature = curvature
        self.fixed_start = fixed_start
        self.fixed_angle = fixed_angle
        self.tipical_cell_spacing = tipical_cell_spacing
        self.smoothing_iterations = smoothing_iterations

        self.rules = rules if isinstance(rules, list) else [rules]
        self.original_rules = self.rules

        self.polygon = None
        self.lumen = None
        self.centerline = None
        self.cell_centroids = None
        self.cell_probabilities = None

        self.rng = np.random.default_rng(seed=int(time.time() * 1e6))
        self._class_instance_one_hot = None

    def generate(self, **kwargs) -> "LinearLumenStructure":
        """Generate a new realization of this tube structure.

        Returns
        -------
        LinearLumenStructure
            A new instance with generated geometry and cells.
        """
        other = copy.deepcopy(self)
        other.polygon = None
        other.lumen = None
        other.centerline = None
        other.cell_centroids = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        other.centerline = other._generate_centerline()
        other.polygon, other.lumen = other._generate_tube_polygons()
        other.cell_centroids = other.generate_cell_centroids()
        other.rules = [i.adapt_rule_to_element(other) for i in other.rules]
        other.cell_probabilities = other.apply_rules(other.rules)
        return other

    def _generate_centerline(self) -> NDArray[np.floating]:
        """Generate the tube centerline as a series of control points.

        Returns
        -------
        np.ndarray
            Control points of shape (n_control_points, 2).
        """
        # Determine starting point
        margin = self.outer_radius + self.length * 0.1
        if self.fixed_start is not None:
            start = np.asarray(self.fixed_start).flatten()[:2]
        else:
            start = np.random.uniform(
                margin, self.frame_size - margin, size=2
            )

        # Determine orientation angle
        if self.fixed_angle is not None:
            angle = self.fixed_angle
        else:
            angle = np.random.uniform(0, 2 * np.pi)

        # Generate control points along the path
        t_values = np.linspace(0, 1, self.n_control_points)
        base_direction = np.array([np.cos(angle), np.sin(angle)])
        perp_direction = np.array([-np.sin(angle), np.cos(angle)])

        points = []
        max_deviation = self.length * self.curvature

        for i, t in enumerate(t_values):
            # Position along main axis
            pos = start + t * self.length * base_direction

            # Add perpendicular deviation (except at endpoints for smooth caps)
            if 0 < i < len(t_values) - 1:
                deviation = np.random.uniform(-max_deviation, max_deviation)
                pos = pos + deviation * perp_direction

            points.append(pos)

        return np.array(points)

    def _generate_tube_polygons(self) -> Tuple[Polygon, Polygon]:
        """Generate outer wall and inner lumen polygons from centerline.

        Uses the centerline to create a buffered tube shape with proper
        wall thickness.

        Returns
        -------
        tuple
            (outer_polygon, lumen_polygon)
        """
        # Create smooth centerline using interpolation
        if len(self.centerline) >= 3:
            # Interpolate more points for smoother curves
            from scipy.interpolate import splprep, splev

            # Fit spline to control points
            t = np.linspace(0, 1, len(self.centerline))
            try:
                tck, _ = splprep(
                    [self.centerline[:, 0], self.centerline[:, 1]],
                    s=0, k=min(3, len(self.centerline) - 1)
                )
                # Evaluate at more points
                t_fine = np.linspace(0, 1, max(50, len(self.centerline) * 10))
                smooth_x, smooth_y = splev(t_fine, tck)
                smooth_centerline = np.column_stack([smooth_x, smooth_y])
            except Exception:
                # Fallback to linear interpolation
                smooth_centerline = self.centerline
        else:
            smooth_centerline = self.centerline

        # Create LineString from centerline
        line = LineString(smooth_centerline)

        # Buffer to create tube shape
        outer_polygon = line.buffer(
            self.outer_radius,
            cap_style=1,  # Round caps
            join_style=1,  # Round joins
            resolution=16
        )
        lumen_polygon = line.buffer(
            self.inner_radius,
            cap_style=1,
            join_style=1,
            resolution=16
        )

        # Apply smoothing if requested
        if self.smoothing_iterations > 0:
            outer_polygon = smooth_polygon(
                outer_polygon,
                iterations=self.smoothing_iterations,
                preserve_area=True
            )
            lumen_polygon = smooth_polygon(
                lumen_polygon,
                iterations=self.smoothing_iterations,
                preserve_area=True
            )

        return outer_polygon, lumen_polygon

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """tuple: Bounding box of the outer polygon."""
        return self.polygon.bounds

    @property
    def center(self) -> NDArray[np.floating]:
        """np.ndarray: Centroid of the tube structure."""
        return np.array([self.polygon.centroid.x, self.polygon.centroid.y])

    @property
    def scale(self) -> float:
        """float: Effective scale (outer_radius) for rule compatibility."""
        return self.outer_radius

    @property
    def class_instance(self) -> NDArray[np.integer]:
        """np.ndarray: Sampled cell type indices."""
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def class_instance_one_hot(self) -> NDArray[np.integer]:
        """np.ndarray: One-hot encoded sampled cell types."""
        if self._class_instance_one_hot is None:
            self._class_instance_one_hot = self.rng.multinomial(
                n=1, pvals=self.cell_probabilities
            )
        return self._class_instance_one_hot

    @property
    def ML_class(self) -> NDArray[np.integer]:
        """np.ndarray: Maximum likelihood cell type."""
        return np.argmax(self.cell_probabilities, axis=1)

    def is_inside(
        self, points: NDArray[np.floating], consider_lumen: bool = False
    ) -> NDArray[np.bool_]:
        """Check which points are inside the element.

        Parameters
        ----------
        points : np.ndarray
            Point coordinates, shape (n_points, 2).
        consider_lumen : bool, optional
            If True, points inside the lumen are considered outside.
            Default is False.

        Returns
        -------
        np.ndarray
            Boolean array of length n_points.
        """
        if consider_lumen:
            result = []
            for point in points:
                px = Point(point[0], point[1])
                result.append(
                    self.polygon.contains(px) and not self.lumen.contains(px)
                )
            return np.array(result, dtype=bool)
        else:
            return np.array(
                [self.polygon.contains(Point(p[0], p[1])) for p in points],
                dtype=bool
            )

    def is_outside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are outside the element."""
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix: NDArray[np.bool_]) -> None:
        """Remove cells at specified indices."""
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]

    def generate_cell_centroids(self) -> NDArray[np.floating]:
        """Generate cell centroids in the tube wall (excluding lumen).

        Returns
        -------
        np.ndarray
            Cell centroid coordinates, shape (n_cells, 2).
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

        # Keep only points in the wall (inside outer, outside lumen)
        points = points[self.is_inside(points, consider_lumen=True)]

        # Add jitter
        points += np.random.normal(0, self.tipical_cell_spacing / 5.0, points.shape)
        return points

    def apply_rules(self, rules) -> NDArray[np.floating]:
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

    def distance_from_lumen(
        self, points: Optional[NDArray[np.floating]] = None
    ) -> NDArray[np.floating]:
        """Calculate distance from points to the lumen boundary.

        Useful for LayerRule to create concentric cell type patterns
        based on distance from the lumen.

        Parameters
        ----------
        points : np.ndarray, optional
            Point coordinates. If None, uses cell_centroids.

        Returns
        -------
        np.ndarray
            Distance from each point to the nearest lumen boundary point.
        """
        if points is None:
            points = self.cell_centroids

        lumen_boundary = self.lumen.exterior
        distances = np.array([
            lumen_boundary.distance(Point(p[0], p[1])) for p in points
        ])
        return distances

    def normalized_wall_position(
        self, points: Optional[NDArray[np.floating]] = None
    ) -> NDArray[np.floating]:
        """Calculate normalized position within the wall (0=lumen, 1=outer).

        Useful for creating smooth gradients across the wall thickness.

        Parameters
        ----------
        points : np.ndarray, optional
            Point coordinates. If None, uses cell_centroids.

        Returns
        -------
        np.ndarray
            Normalized position (0-1) for each point within the wall.
        """
        if points is None:
            points = self.cell_centroids

        dist_from_lumen = self.distance_from_lumen(points)
        # Normalize by wall thickness
        normalized = np.clip(dist_from_lumen / self.wall_thickness, 0, 1)
        return normalized
