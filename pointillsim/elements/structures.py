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


class LayeredElement:
    """Stratified tissue element with horizontal layers.

    Represents tissues with distinct horizontal layers such as cerebral cortex,
    skin epidermis, or retinal layers. Each layer can have different cell type
    compositions defined by separate rules.

    The element creates a rectangular region spanning the frame width with
    specified layer boundaries. Different rules can be assigned to each layer
    for complete control over cell type distributions.

    Parameters
    ----------
    frame_size : int, optional
        Size of the FOV in pixels. Default is 5000.
    layer_boundaries : list of float, optional
        Y-coordinates of layer boundaries (in pixels from top). Should be
        sorted in ascending order. Creates n+1 layers for n boundaries.
        If None, uses [frame_size/2] for two equal layers. Default is None.
    layer_rules : list of CellTypeRuleBase, optional
        Rules for each layer. Must have len(layer_boundaries) + 1 rules.
        If None, uses the default `rules` parameter for all layers.
    margin : float, optional
        Horizontal margin from frame edges (in pixels). Default is 0.
    tipical_cell_spacing : float, optional
        Average cell spacing. Default is 8.
    orientation : str, optional
        Layer orientation: 'horizontal' (y-axis layers) or 'vertical'
        (x-axis layers). Default is 'horizontal'.
    rules : CellTypeRuleBase or list, optional
        Default rule(s) for all layers if layer_rules not specified.
        Required if layer_rules is None.

    Attributes
    ----------
    polygon : shapely.Polygon
        The element's bounding polygon.
    cell_centroids : np.ndarray
        Cell centroid positions, shape (n_cells, 2).
    cell_probabilities : np.ndarray
        Cell type probabilities, shape (n_cells, n_cell_types).
    layer_indices : np.ndarray
        Layer index (0, 1, 2, ...) for each cell.

    Examples
    --------
    >>> from pointillsim.rules import SingleTypeRule, MixOfNCellTypesRule
    >>> # Create 3-layer epidermis: stratum corneum, spinosum, basale
    >>> layer_rules = [
    ...     SingleTypeRule(n_cell_types=3, cell_type_ix=0),  # Corneum
    ...     MixOfNCellTypesRule(n_cell_types=3, list_N=[0, 1], proportions=[0.3, 0.7]),
    ...     SingleTypeRule(n_cell_types=3, cell_type_ix=2),  # Basale
    ... ]
    >>> epidermis = LayeredElement(
    ...     frame_size=1000,
    ...     layer_boundaries=[200, 600],  # 3 layers
    ...     layer_rules=layer_rules,
    ...     tipical_cell_spacing=12,
    ... )
    >>> realized = epidermis.generate()
    """

    def __init__(
        self,
        frame_size: int = 5000,
        layer_boundaries: Optional[list] = None,
        layer_rules: Optional[list] = None,
        margin: float = 0,
        tipical_cell_spacing: float = 8,
        orientation: str = "horizontal",
        rules=None,
    ) -> None:
        self.frame_size = frame_size
        self.margin = margin
        self.tipical_cell_spacing = tipical_cell_spacing
        self.orientation = orientation

        # Set up layer boundaries
        if layer_boundaries is None:
            self.layer_boundaries = [frame_size / 2]
        else:
            self.layer_boundaries = sorted(layer_boundaries)

        # Validate and set up rules
        n_layers = len(self.layer_boundaries) + 1

        if layer_rules is not None:
            if len(layer_rules) != n_layers:
                raise ValueError(
                    f"layer_rules must have {n_layers} rules for "
                    f"{len(self.layer_boundaries)} boundaries, got {len(layer_rules)}"
                )
            self.layer_rules = layer_rules
            # Use the first layer rule as default for compatibility
            self.rules = layer_rules[0] if not isinstance(layer_rules[0], list) else layer_rules[0]
        elif rules is not None:
            self.layer_rules = None
            self.rules = rules if isinstance(rules, list) else [rules]
        else:
            raise ValueError("Either 'rules' or 'layer_rules' must be provided")

        if not isinstance(self.rules, list):
            self.rules = [self.rules]

        self.original_rules = self.rules
        self.original_layer_rules = self.layer_rules

        self.polygon = None
        self.cell_centroids = None
        self.cell_probabilities = None
        self.layer_indices = None

        self.rng = np.random.default_rng(seed=int(time.time() * 1e6))
        self._class_instance_one_hot = None

    @property
    def n_layers(self) -> int:
        """int: Number of layers in this element."""
        return len(self.layer_boundaries) + 1

    @property
    def scale(self) -> float:
        """float: Effective scale for rule compatibility."""
        return self.frame_size / 2

    @property
    def center(self) -> NDArray[np.floating]:
        """np.ndarray: Center of the element."""
        return np.array([[self.frame_size / 2, self.frame_size / 2]])

    def generate(self, **kwargs) -> "LayeredElement":
        """Generate a new realization of this layered element.

        Returns
        -------
        LayeredElement
            A new instance with generated geometry and cells.
        """
        other = copy.deepcopy(self)
        other.polygon = None
        other.cell_centroids = None
        other.cell_probabilities = None
        other.layer_indices = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules
        other.layer_rules = other.original_layer_rules

        other.polygon = other._generate_polygon()
        other.cell_centroids = other.generate_cell_centroids()
        other.layer_indices = other._assign_layer_indices()
        other.cell_probabilities = other._apply_layer_rules()
        return other

    def _generate_polygon(self) -> Polygon:
        """Generate the bounding polygon for this layered element.

        Returns
        -------
        shapely.Polygon
            Rectangle spanning the frame (with optional margins).
        """
        x_min = self.margin
        x_max = self.frame_size - self.margin
        y_min = 0
        y_max = self.frame_size

        return Polygon([
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ])

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """tuple: Bounding box of the polygon."""
        return self.polygon.bounds

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

    def is_inside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are inside the element."""
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
        if self.layer_indices is not None:
            self.layer_indices = self.layer_indices[~bool_ix]

    def generate_cell_centroids(self) -> NDArray[np.floating]:
        """Generate cell centroids on a quasi-hexagonal grid.

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

        # Keep points inside polygon
        points = points[self.is_inside(points)]

        # Add jitter
        points += np.random.normal(0, self.tipical_cell_spacing / 5.0, points.shape)
        return points

    def _assign_layer_indices(self) -> NDArray[np.integer]:
        """Assign each cell to a layer based on its position.

        Returns
        -------
        np.ndarray
            Layer index (0, 1, 2, ...) for each cell.
        """
        if self.orientation == "horizontal":
            positions = self.cell_centroids[:, 1]  # Y-coordinate
        else:
            positions = self.cell_centroids[:, 0]  # X-coordinate

        # Digitize returns indices of the bins
        # For n boundaries, we get n+1 bins (layers)
        indices = np.digitize(positions, self.layer_boundaries)
        return indices

    def _apply_layer_rules(self) -> NDArray[np.floating]:
        """Apply rules to compute cell type probabilities per layer.

        Returns
        -------
        np.ndarray
            Cell type probability matrix, shape (n_cells, n_cell_types).
        """
        n_cells = self.cell_centroids.shape[0]

        if self.layer_rules is not None:
            # Determine n_cell_types from first rule
            first_rule = self.layer_rules[0]
            if isinstance(first_rule, list):
                first_rule = first_rule[0]
            n_cell_types = first_rule.n_cell_types

            probs = np.zeros((n_cells, n_cell_types))

            for layer_idx in range(self.n_layers):
                mask = self.layer_indices == layer_idx

                if not np.any(mask):
                    continue

                layer_rule = self.layer_rules[layer_idx]
                if not isinstance(layer_rule, list):
                    layer_rule = [layer_rule]

                # Adapt rules to this element
                adapted_rules = [r.adapt_rule_to_element(self) for r in layer_rule]

                # Get cell positions for this layer
                layer_points = self.cell_centroids[mask]

                # Apply rules
                layer_probs = None
                for rule in adapted_rules:
                    layer_probs = rule.apply(layer_points, current_probs=layer_probs)

                probs[mask] = layer_probs

            return probs
        else:
            # Use default rules for all layers
            adapted_rules = [r.adapt_rule_to_element(self) for r in self.rules]
            probs = None
            for rule in adapted_rules:
                probs = rule.apply(self.cell_centroids, current_probs=probs)

            if probs is None:
                n_types = self.rules[0].n_cell_types if self.rules else 0
                probs = np.empty((n_cells, n_types))

            return probs

    def get_layer_boundary_y(self, layer_idx: int) -> Tuple[float, float]:
        """Get the Y-coordinate boundaries for a specific layer.

        Parameters
        ----------
        layer_idx : int
            Layer index (0 to n_layers - 1).

        Returns
        -------
        tuple
            (y_min, y_max) boundaries for the layer.
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise ValueError(f"layer_idx must be 0 to {self.n_layers - 1}")

        all_boundaries = [0] + list(self.layer_boundaries) + [self.frame_size]
        return (all_boundaries[layer_idx], all_boundaries[layer_idx + 1])

    def normalized_layer_position(
        self, points: Optional[NDArray[np.floating]] = None
    ) -> NDArray[np.floating]:
        """Calculate normalized position within each layer (0=top, 1=bottom).

        Useful for creating gradients within layers.

        Parameters
        ----------
        points : np.ndarray, optional
            Point coordinates. If None, uses cell_centroids.

        Returns
        -------
        np.ndarray
            Normalized position (0-1) within each point's layer.
        """
        if points is None:
            points = self.cell_centroids

        if self.orientation == "horizontal":
            positions = points[:, 1]
        else:
            positions = points[:, 0]

        layer_indices = np.digitize(positions, self.layer_boundaries)
        all_boundaries = [0] + list(self.layer_boundaries) + [self.frame_size]

        normalized = np.zeros(len(positions))
        for i in range(len(positions)):
            layer_idx = layer_indices[i]
            y_min = all_boundaries[layer_idx]
            y_max = all_boundaries[layer_idx + 1]
            layer_thickness = y_max - y_min
            if layer_thickness > 0:
                normalized[i] = (positions[i] - y_min) / layer_thickness
            else:
                normalized[i] = 0.5

        return np.clip(normalized, 0, 1)

    def apply_rules(self, rules) -> NDArray[np.floating]:
        """Apply rules to compute cell type probabilities.

        This is a compatibility method. For layer-specific rules,
        use layer_rules parameter in constructor.
        """
        probs = None
        for rule in rules:
            probs = rule.apply(self.cell_centroids, current_probs=probs)

        if probs is None:
            n_cells = self.cell_centroids.shape[0]
            n_types = rules[0].n_cell_types if rules else 0
            probs = np.empty((n_cells, n_types))

        return probs


class BranchingStructure:
    """Tree-like branching histological structure.

    Represents branching tissue architectures such as vascular trees,
    neural dendrites, bronchial airways, or ductal systems. The structure
    starts from a root point and recursively branches into smaller segments.

    Each branch is a tube with configurable radius that can decrease with
    depth. Cells are generated in the tube walls, and cell type rules can
    be applied based on branch depth or position within the wall.

    Parameters
    ----------
    frame_size : int, optional
        Size of the FOV in pixels. Default is 5000.
    root_position : np.ndarray, optional
        Starting position (x, y) of the tree root. If None, placed at frame edge.
    root_angle : float, optional
        Initial growth angle in radians. If None, grows toward center.
    root_radius : float, optional
        Radius of the main trunk. Default is 40.
    wall_thickness : float, optional
        Wall thickness as fraction of radius. Default is 0.4.
    branch_length : float, optional
        Length of each branch segment. Default is 150.
    branch_length_decay : float, optional
        Factor by which branch length decreases each generation. Default is 0.8.
    radius_decay : float, optional
        Factor by which radius decreases each generation. Default is 0.7.
    branching_angle : tuple, optional
        Range of branching angles in radians (min, max). Default is (0.3, 0.7).
    branching_probability : float, optional
        Probability of branching at each generation. Default is 0.8.
    max_depth : int, optional
        Maximum branching depth. Default is 4.
    min_radius : float, optional
        Minimum branch radius (stops branching below this). Default is 5.
    tipical_cell_spacing : float, optional
        Average cell spacing. Default is 8.
    curvature : float, optional
        Random curvature of branches. Default is 0.1.
    rules : CellTypeRuleBase or list
        Rules for cell type assignment. Required.

    Attributes
    ----------
    polygon : shapely.Polygon
        Union of all branch polygons.
    lumen : shapely.Polygon
        Union of all branch lumens.
    branches : list
        List of branch dictionaries with 'start', 'end', 'radius', 'depth'.
    cell_centroids : np.ndarray
        Cell centroid positions.
    branch_depths : np.ndarray
        Branching depth for each cell (useful for depth-based rules).

    Examples
    --------
    >>> from pointillsim.rules import LayerRule
    >>> rule = LayerRule(n_cell_types=3, layer_types=[0, 1, 2])
    >>> tree = BranchingStructure(
    ...     frame_size=1000,
    ...     root_radius=30,
    ...     max_depth=3,
    ...     rules=rule,
    ... )
    >>> realized = tree.generate()
    >>> print(f"Generated tree with {len(realized.branches)} branches")
    """

    def __init__(
        self,
        frame_size: int = 5000,
        root_position: Optional[NDArray[np.floating]] = None,
        root_angle: Optional[float] = None,
        root_radius: float = 40,
        wall_thickness: float = 0.4,
        branch_length: float = 150,
        branch_length_decay: float = 0.8,
        radius_decay: float = 0.7,
        branching_angle: Tuple[float, float] = (0.3, 0.7),
        branching_probability: float = 0.8,
        max_depth: int = 4,
        min_radius: float = 5,
        tipical_cell_spacing: float = 8,
        curvature: float = 0.1,
        rules=None,
    ) -> None:
        if rules is None:
            raise ValueError("No rules provided")

        self.frame_size = frame_size
        self.root_position = root_position
        self.root_angle = root_angle
        self.root_radius = root_radius
        self.wall_thickness_ratio = wall_thickness
        self.branch_length = branch_length
        self.branch_length_decay = branch_length_decay
        self.radius_decay = radius_decay
        self.branching_angle = branching_angle
        self.branching_probability = branching_probability
        self.max_depth = max_depth
        self.min_radius = min_radius
        self.tipical_cell_spacing = tipical_cell_spacing
        self.curvature = curvature

        self.rules = rules if isinstance(rules, list) else [rules]
        self.original_rules = self.rules

        self.polygon = None
        self.lumen = None
        self.branches = []
        self.cell_centroids = None
        self.cell_probabilities = None
        self.branch_depths = None

        self.rng = np.random.default_rng(seed=int(time.time() * 1e6))
        self._class_instance_one_hot = None

    @property
    def scale(self) -> float:
        """float: Effective scale for rule compatibility."""
        return self.root_radius

    @property
    def center(self) -> NDArray[np.floating]:
        """np.ndarray: Center of the structure (centroid of polygon)."""
        if self.polygon is not None:
            return np.array([[self.polygon.centroid.x, self.polygon.centroid.y]])
        return np.array([[self.frame_size / 2, self.frame_size / 2]])

    @property
    def wall_thickness(self) -> float:
        """float: Wall thickness of root branch."""
        return self.root_radius * self.wall_thickness_ratio

    def generate(self, **kwargs) -> "BranchingStructure":
        """Generate a new realization of this branching structure.

        Returns
        -------
        BranchingStructure
            A new instance with generated geometry and cells.
        """
        other = copy.deepcopy(self)
        other.polygon = None
        other.lumen = None
        other.branches = []
        other.cell_centroids = None
        other.cell_probabilities = None
        other.branch_depths = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        # Generate tree structure
        other._generate_tree()

        # Generate polygons from branches
        other.polygon, other.lumen = other._generate_polygons()

        # Generate cells
        other.cell_centroids, other.branch_depths = other._generate_cells()

        # Apply rules
        other.rules = [r.adapt_rule_to_element(other) for r in other.rules]
        other.cell_probabilities = other.apply_rules(other.rules)

        return other

    def _generate_tree(self) -> None:
        """Generate the branching tree structure recursively."""
        # Determine root position
        if self.root_position is not None:
            start = np.asarray(self.root_position).flatten()[:2]
        else:
            # Place at edge of frame, pointing inward
            edge = np.random.choice(['left', 'right', 'top', 'bottom'])
            margin = self.root_radius * 2
            if edge == 'left':
                start = np.array([margin, np.random.uniform(margin, self.frame_size - margin)])
            elif edge == 'right':
                start = np.array([self.frame_size - margin, np.random.uniform(margin, self.frame_size - margin)])
            elif edge == 'top':
                start = np.array([np.random.uniform(margin, self.frame_size - margin), margin])
            else:  # bottom
                start = np.array([np.random.uniform(margin, self.frame_size - margin), self.frame_size - margin])

        # Determine root angle
        if self.root_angle is not None:
            angle = self.root_angle
        else:
            # Point toward center
            center = np.array([self.frame_size / 2, self.frame_size / 2])
            direction = center - start
            angle = np.arctan2(direction[1], direction[0])

        # Recursively generate branches
        self._grow_branch(start, angle, self.root_radius, self.branch_length, depth=0)

    def _grow_branch(
        self,
        start: NDArray[np.floating],
        angle: float,
        radius: float,
        length: float,
        depth: int,
    ) -> None:
        """Recursively grow a branch and its children.

        Parameters
        ----------
        start : np.ndarray
            Starting position of the branch.
        angle : float
            Growth direction in radians.
        radius : float
            Branch radius.
        length : float
            Branch length.
        depth : int
            Current branching depth.
        """
        if depth > self.max_depth or radius < self.min_radius:
            return

        # Add curvature
        angle_variation = np.random.uniform(-self.curvature, self.curvature) * np.pi

        # Calculate end point
        direction = np.array([np.cos(angle + angle_variation), np.sin(angle + angle_variation)])
        end = start + direction * length

        # Check bounds
        if not (0 < end[0] < self.frame_size and 0 < end[1] < self.frame_size):
            return

        # Store this branch
        self.branches.append({
            'start': start.copy(),
            'end': end.copy(),
            'radius': radius,
            'depth': depth,
            'angle': angle,
        })

        # Determine if we branch
        if np.random.random() < self.branching_probability:
            # Two child branches
            new_length = length * self.branch_length_decay
            new_radius = radius * self.radius_decay

            # Left branch
            left_angle = angle + np.random.uniform(*self.branching_angle)
            self._grow_branch(end, left_angle, new_radius, new_length, depth + 1)

            # Right branch
            right_angle = angle - np.random.uniform(*self.branching_angle)
            self._grow_branch(end, right_angle, new_radius, new_length, depth + 1)
        else:
            # Continue straight (with slight variation)
            new_length = length * self.branch_length_decay
            new_radius = radius * self.radius_decay
            self._grow_branch(end, angle, new_radius, new_length, depth + 1)

    def _generate_polygons(self) -> Tuple[Polygon, Polygon]:
        """Generate outer and lumen polygons from branches.

        Returns
        -------
        tuple
            (outer_polygon, lumen_polygon)
        """
        if not self.branches:
            # Return empty polygons
            return Polygon(), Polygon()

        outer_parts = []
        lumen_parts = []

        for branch in self.branches:
            line = LineString([branch['start'], branch['end']])
            outer_radius = branch['radius']
            inner_radius = outer_radius * (1 - self.wall_thickness_ratio)

            outer = line.buffer(outer_radius, cap_style=1, resolution=8)
            inner = line.buffer(max(inner_radius, 1), cap_style=1, resolution=8)

            outer_parts.append(outer)
            lumen_parts.append(inner)

        # Union all parts
        outer_polygon = unary_union(outer_parts)
        lumen_polygon = unary_union(lumen_parts)

        # Apply smoothing
        outer_polygon = smooth_polygon(outer_polygon, iterations=1, preserve_area=True)
        lumen_polygon = smooth_polygon(lumen_polygon, iterations=1, preserve_area=True)

        return outer_polygon, lumen_polygon

    def _generate_cells(self) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Generate cell centroids in the branch walls.

        Returns
        -------
        tuple
            (cell_centroids, branch_depths)
        """
        if not self.branches or self.polygon.is_empty:
            return np.empty((0, 2)), np.empty(0, dtype=int)

        # Generate grid points
        bounds = self.polygon.bounds
        x = np.arange(bounds[0], bounds[2], self.tipical_cell_spacing)
        y = np.arange(bounds[1], bounds[3], self.tipical_cell_spacing * np.sin(np.pi / 3))
        X, Y = np.meshgrid(x, y)
        X[::2] += self.tipical_cell_spacing / 2.0
        points = np.stack((X.flatten(), Y.flatten()), axis=1)

        # Keep points in wall (inside outer, outside lumen)
        inside_outer = np.array([
            self.polygon.contains(Point(p[0], p[1])) for p in points
        ])
        inside_lumen = np.array([
            self.lumen.contains(Point(p[0], p[1])) for p in points
        ])
        mask = inside_outer & ~inside_lumen
        points = points[mask]

        # Add jitter
        points += np.random.normal(0, self.tipical_cell_spacing / 5.0, points.shape)

        # Assign branch depth to each cell
        depths = self._assign_branch_depths(points)

        return points, depths

    def _assign_branch_depths(self, points: NDArray[np.floating]) -> NDArray[np.integer]:
        """Assign branching depth to each cell based on nearest branch.

        Parameters
        ----------
        points : np.ndarray
            Cell centroid positions.

        Returns
        -------
        np.ndarray
            Branch depth for each cell.
        """
        depths = np.zeros(len(points), dtype=int)

        for i, point in enumerate(points):
            min_dist = float('inf')
            best_depth = 0

            for branch in self.branches:
                line = LineString([branch['start'], branch['end']])
                dist = line.distance(Point(point[0], point[1]))

                if dist < min_dist:
                    min_dist = dist
                    best_depth = branch['depth']

            depths[i] = best_depth

        return depths

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """tuple: Bounding box of the polygon."""
        if self.polygon is not None and not self.polygon.is_empty:
            return self.polygon.bounds
        return (0, 0, self.frame_size, self.frame_size)

    @property
    def class_instance(self) -> NDArray[np.integer]:
        """np.ndarray: Sampled cell type indices."""
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def class_instance_one_hot(self) -> NDArray[np.integer]:
        """np.ndarray: One-hot encoded sampled cell types."""
        if self._class_instance_one_hot is None:
            if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
                self._class_instance_one_hot = self.rng.multinomial(
                    n=1, pvals=self.cell_probabilities
                )
            else:
                self._class_instance_one_hot = np.empty((0, 0), dtype=int)
        return self._class_instance_one_hot

    @property
    def ML_class(self) -> NDArray[np.integer]:
        """np.ndarray: Maximum likelihood cell type."""
        if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
            return np.argmax(self.cell_probabilities, axis=1)
        return np.empty(0, dtype=int)

    def is_inside(
        self, points: NDArray[np.floating], consider_lumen: bool = False
    ) -> NDArray[np.bool_]:
        """Check which points are inside the structure."""
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
        """Check which points are outside the structure."""
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix: NDArray[np.bool_]) -> None:
        """Remove cells at specified indices."""
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]
        if self.branch_depths is not None:
            self.branch_depths = self.branch_depths[~bool_ix]

    def apply_rules(self, rules) -> NDArray[np.floating]:
        """Apply rules to compute cell type probabilities."""
        if self.cell_centroids is None or len(self.cell_centroids) == 0:
            return np.empty((0, rules[0].n_cell_types if rules else 0))

        probs = None
        for rule in rules:
            probs = rule.apply(self.cell_centroids, current_probs=probs)

        if probs is None:
            n_cells = self.cell_centroids.shape[0]
            n_types = rules[0].n_cell_types if rules else 0
            probs = np.empty((n_cells, n_types))

        return probs

    def normalized_depth(self) -> NDArray[np.floating]:
        """Calculate normalized branching depth (0=root, 1=deepest).

        Useful for depth-based cell type gradients.

        Returns
        -------
        np.ndarray
            Normalized depth (0-1) for each cell.
        """
        if self.branch_depths is None or len(self.branch_depths) == 0:
            return np.empty(0)

        max_depth = self.branch_depths.max()
        if max_depth == 0:
            return np.zeros_like(self.branch_depths, dtype=float)

        return self.branch_depths / max_depth


class FibrillarStructure:
    """Elongated fibrous histological structure.

    Represents fibrillar tissue components such as collagen bundles, muscle fibers,
    nerve tracts, or other elongated structures with aligned cells. The structure
    consists of parallel or slightly wavy fibers that can span the field of view.

    Parameters
    ----------
    frame_size : int, optional
        Size of the FOV in pixels. Default is 5000.
    n_fibers : int, optional
        Number of parallel fibers. Default is 5.
    fiber_width : float, optional
        Width of each fiber. Default is 30.
    fiber_spacing : float, optional
        Spacing between fiber centers. Default is 50.
    orientation : float, optional
        Orientation angle in radians. If None, random. Default is None.
    waviness : float, optional
        Amount of wave/undulation in fibers (0-1). Default is 0.1.
    fixed_center : np.ndarray, optional
        Center position of the fiber bundle. If None, centered in FOV.
    tipical_cell_spacing : float, optional
        Average cell spacing. Default is 8.
    rules : CellTypeRuleBase or list
        Rules for cell type assignment. Required.

    Attributes
    ----------
    polygon : shapely.Polygon
        Union of all fiber polygons.
    fibers : list of shapely.Polygon
        Individual fiber polygons.
    cell_centroids : np.ndarray
        Cell centroid positions.
    fiber_indices : np.ndarray
        Fiber index (0 to n_fibers-1) for each cell.

    Examples
    --------
    >>> from pointillsim.rules import SingleTypeRule
    >>> rule = SingleTypeRule(n_cell_types=3, cell_type_ix=1)
    >>> collagen = FibrillarStructure(
    ...     frame_size=500, n_fibers=4, fiber_width=20,
    ...     orientation=np.pi/4, rules=rule
    ... )
    >>> realized = collagen.generate()
    """

    def __init__(
        self,
        frame_size: int = 5000,
        n_fibers: int = 5,
        fiber_width: float = 30,
        fiber_spacing: float = 50,
        orientation: Optional[float] = None,
        waviness: float = 0.1,
        fixed_center: Optional[NDArray[np.floating]] = None,
        tipical_cell_spacing: float = 8,
        rules=None,
    ) -> None:
        if rules is None:
            raise ValueError("No rules provided")

        self.frame_size = frame_size
        self.n_fibers = n_fibers
        self.fiber_width = fiber_width
        self.fiber_spacing = fiber_spacing
        self.orientation = orientation
        self.waviness = waviness
        self.fixed_center = fixed_center
        self.tipical_cell_spacing = tipical_cell_spacing

        self.rules = rules if isinstance(rules, list) else [rules]
        self.original_rules = self.rules

        self.polygon = None
        self.fibers = []
        self.cell_centroids = None
        self.cell_probabilities = None
        self.fiber_indices = None

        self.rng = np.random.default_rng(seed=int(time.time() * 1e6))
        self._class_instance_one_hot = None

    @property
    def scale(self) -> float:
        """float: Effective scale for rule compatibility."""
        return self.n_fibers * self.fiber_spacing / 2

    @property
    def center(self) -> NDArray[np.floating]:
        """np.ndarray: Center of the structure."""
        if self.fixed_center is not None:
            return np.array([self.fixed_center]).reshape(1, 2)
        return np.array([[self.frame_size / 2, self.frame_size / 2]])

    def generate(self, **kwargs) -> "FibrillarStructure":
        """Generate a new realization of this fibrillar structure."""
        other = copy.deepcopy(self)
        other.polygon = None
        other.fibers = []
        other.cell_centroids = None
        other.cell_probabilities = None
        other.fiber_indices = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        # Set orientation
        if other.orientation is None:
            other.orientation = np.random.uniform(0, np.pi)

        # Generate fibers
        other.fibers = other._generate_fibers()
        other.polygon = unary_union(other.fibers) if other.fibers else Polygon()

        # Generate cells
        other.cell_centroids, other.fiber_indices = other._generate_cells()

        # Apply rules
        other.rules = [r.adapt_rule_to_element(other) for r in other.rules]
        other.cell_probabilities = other.apply_rules(other.rules)

        return other

    def _generate_fibers(self) -> list:
        """Generate individual fiber polygons."""
        fibers = []
        center = self.center.flatten()

        # Direction vectors
        direction = np.array([np.cos(self.orientation), np.sin(self.orientation)])
        perp = np.array([-np.sin(self.orientation), np.cos(self.orientation)])

        # Fiber length (diagonal of frame to ensure coverage)
        fiber_length = self.frame_size * 1.5

        # Starting offset for centering the bundle
        bundle_width = (self.n_fibers - 1) * self.fiber_spacing
        start_offset = -bundle_width / 2

        for i in range(self.n_fibers):
            # Offset perpendicular to direction
            offset = start_offset + i * self.fiber_spacing
            fiber_center = center + offset * perp

            # Create wavy centerline
            n_points = max(10, int(fiber_length / 50))
            t = np.linspace(-0.5, 0.5, n_points)
            points = []

            for ti in t:
                pos = fiber_center + ti * fiber_length * direction
                # Add waviness
                wave = np.sin(ti * 4 * np.pi) * self.waviness * self.fiber_spacing
                pos = pos + wave * perp
                points.append(pos)

            # Create polygon by buffering the line
            line = LineString(points)
            fiber_poly = line.buffer(self.fiber_width / 2, cap_style=2)
            fibers.append(fiber_poly)

        return fibers

    def _generate_cells(self) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Generate cell centroids within fibers."""
        if not self.fibers or self.polygon.is_empty:
            return np.empty((0, 2)), np.empty(0, dtype=int)

        bounds = self.polygon.bounds
        x = np.arange(bounds[0], bounds[2], self.tipical_cell_spacing)
        y = np.arange(bounds[1], bounds[3], self.tipical_cell_spacing * np.sin(np.pi / 3))
        X, Y = np.meshgrid(x, y)
        X[::2] += self.tipical_cell_spacing / 2.0
        points = np.stack((X.flatten(), Y.flatten()), axis=1)

        # Keep points inside any fiber
        mask = np.array([self.polygon.contains(Point(p[0], p[1])) for p in points])
        points = points[mask]

        # Add jitter
        points += np.random.normal(0, self.tipical_cell_spacing / 5.0, points.shape)

        # Assign fiber indices
        fiber_indices = self._assign_fiber_indices(points)

        return points, fiber_indices

    def _assign_fiber_indices(self, points: NDArray[np.floating]) -> NDArray[np.integer]:
        """Assign each cell to the nearest fiber."""
        indices = np.zeros(len(points), dtype=int)
        for i, point in enumerate(points):
            px = Point(point[0], point[1])
            min_dist = float('inf')
            for j, fiber in enumerate(self.fibers):
                if fiber.contains(px):
                    indices[i] = j
                    break
                dist = fiber.exterior.distance(px)
                if dist < min_dist:
                    min_dist = dist
                    indices[i] = j
        return indices

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """tuple: Bounding box of the polygon."""
        if self.polygon is not None and not self.polygon.is_empty:
            return self.polygon.bounds
        return (0, 0, self.frame_size, self.frame_size)

    @property
    def class_instance(self) -> NDArray[np.integer]:
        """np.ndarray: Sampled cell type indices."""
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def class_instance_one_hot(self) -> NDArray[np.integer]:
        """np.ndarray: One-hot encoded sampled cell types."""
        if self._class_instance_one_hot is None:
            if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
                self._class_instance_one_hot = self.rng.multinomial(
                    n=1, pvals=self.cell_probabilities
                )
            else:
                self._class_instance_one_hot = np.empty((0, 0), dtype=int)
        return self._class_instance_one_hot

    @property
    def ML_class(self) -> NDArray[np.integer]:
        """np.ndarray: Maximum likelihood cell type."""
        if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
            return np.argmax(self.cell_probabilities, axis=1)
        return np.empty(0, dtype=int)

    def is_inside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are inside the structure."""
        return np.array([self.polygon.contains(Point(p[0], p[1])) for p in points], dtype=bool)

    def is_outside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are outside the structure."""
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix: NDArray[np.bool_]) -> None:
        """Remove cells at specified indices."""
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]
        if self.fiber_indices is not None:
            self.fiber_indices = self.fiber_indices[~bool_ix]

    def apply_rules(self, rules) -> NDArray[np.floating]:
        """Apply rules to compute cell type probabilities."""
        if self.cell_centroids is None or len(self.cell_centroids) == 0:
            return np.empty((0, rules[0].n_cell_types if rules else 0))

        probs = None
        for rule in rules:
            probs = rule.apply(self.cell_centroids, current_probs=probs)

        if probs is None:
            n_cells = self.cell_centroids.shape[0]
            n_types = rules[0].n_cell_types if rules else 0
            probs = np.empty((n_cells, n_types))

        return probs


class ClusterElement:
    """Clustered group of cells with shared properties.

    Represents localized cellular aggregates such as lymphoid follicles,
    tumor nodules, inflammatory infiltrates, or any discrete cluster of
    cells. The cluster has an approximately circular shape with optional
    density gradients from center to periphery.

    Parameters
    ----------
    frame_size : int, optional
        Size of the FOV in pixels. Default is 5000.
    radius : float, optional
        Approximate radius of the cluster. Default is 100.
    n_vertices : int or tuple, optional
        Number of vertices for the polygon boundary. Default is (8, 12).
    fixed_center : np.ndarray, optional
        Center position. If None, random placement.
    density_profile : str, optional
        Cell density profile: 'uniform', 'dense_center', 'dense_edge'.
        Default is 'uniform'.
    tipical_cell_spacing : float, optional
        Average cell spacing. Default is 8.
    smoothing_iterations : int, optional
        Chaikin smoothing iterations. Default is 2.
    rules : CellTypeRuleBase or list
        Rules for cell type assignment. Required.

    Attributes
    ----------
    polygon : shapely.Polygon
        The cluster boundary polygon.
    cell_centroids : np.ndarray
        Cell centroid positions.

    Examples
    --------
    >>> from pointillsim.rules import MixOfNCellTypesRule
    >>> rule = MixOfNCellTypesRule(n_cell_types=4, list_N=[0, 1], proportions=[0.7, 0.3])
    >>> follicle = ClusterElement(
    ...     radius=80, density_profile='dense_center', rules=rule
    ... )
    >>> realized = follicle.generate()
    """

    def __init__(
        self,
        frame_size: int = 5000,
        radius: float = 100,
        n_vertices: Union[int, Tuple[int, int]] = (8, 12),
        fixed_center: Optional[NDArray[np.floating]] = None,
        density_profile: str = "uniform",
        tipical_cell_spacing: float = 8,
        smoothing_iterations: int = 2,
        rules=None,
    ) -> None:
        if rules is None:
            raise ValueError("No rules provided")

        self.frame_size = frame_size
        self.radius = radius
        self.n_vertices = n_vertices
        self.fixed_center = fixed_center
        self.density_profile = density_profile
        self.tipical_cell_spacing = tipical_cell_spacing
        self.smoothing_iterations = smoothing_iterations

        self.rules = rules if isinstance(rules, list) else [rules]
        self.original_rules = self.rules

        self.polygon = None
        self.cell_centroids = None
        self.cell_probabilities = None

        self.rng = np.random.default_rng(seed=int(time.time() * 1e6))
        self._class_instance_one_hot = None

    @property
    def scale(self) -> float:
        """float: Effective scale for rule compatibility."""
        return self.radius

    @property
    def center(self) -> NDArray[np.floating]:
        """np.ndarray: Center of the cluster."""
        if self._center is not None:
            return self._center.reshape(1, 2)
        return np.array([[self.frame_size / 2, self.frame_size / 2]])

    def generate(self, **kwargs) -> "ClusterElement":
        """Generate a new realization of this cluster element."""
        other = copy.deepcopy(self)
        other.polygon = None
        other.cell_centroids = None
        other.cell_probabilities = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        # Set center
        if other.fixed_center is not None:
            other._center = np.asarray(other.fixed_center).flatten()[:2]
        else:
            margin = other.radius * 1.5
            other._center = np.random.uniform(margin, other.frame_size - margin, size=2)

        # Generate polygon
        other.polygon = other._generate_polygon()

        # Generate cells
        other.cell_centroids = other._generate_cells()

        # Apply rules
        other.rules = [r.adapt_rule_to_element(other) for r in other.rules]
        other.cell_probabilities = other.apply_rules(other.rules)

        return other

    def _generate_polygon(self) -> Polygon:
        """Generate the cluster boundary polygon."""
        n_verts = (
            np.random.randint(self.n_vertices[0], self.n_vertices[1])
            if isinstance(self.n_vertices, tuple)
            else self.n_vertices
        )
        points = generate_uniform_points_in_circle(
            self._center.reshape(1, 2), self.radius, n_verts
        )
        polygon = Polygon(points).convex_hull

        if self.smoothing_iterations > 0:
            polygon = smooth_polygon(polygon, iterations=self.smoothing_iterations, preserve_area=True)

        return polygon

    def _generate_cells(self) -> NDArray[np.floating]:
        """Generate cell centroids within the cluster."""
        bounds = self.polygon.bounds
        spacing = self.tipical_cell_spacing

        if self.density_profile == 'dense_center':
            spacing = spacing * 0.8  # Slightly denser overall
        elif self.density_profile == 'dense_edge':
            spacing = spacing * 0.9

        x = np.arange(bounds[0], bounds[2], spacing)
        y = np.arange(bounds[1], bounds[3], spacing * np.sin(np.pi / 3))
        X, Y = np.meshgrid(x, y)
        X[::2] += spacing / 2.0
        points = np.stack((X.flatten(), Y.flatten()), axis=1)

        # Keep points inside polygon
        mask = np.array([self.polygon.contains(Point(p[0], p[1])) for p in points])
        points = points[mask]

        # Apply density profile
        if self.density_profile == 'dense_center':
            # Higher probability of keeping cells near center
            distances = np.linalg.norm(points - self._center, axis=1)
            max_dist = distances.max() if len(distances) > 0 else 1
            keep_prob = 1 - 0.5 * (distances / max_dist) ** 2
            keep_mask = np.random.random(len(points)) < keep_prob
            points = points[keep_mask]
        elif self.density_profile == 'dense_edge':
            # Higher probability near edge
            distances = np.linalg.norm(points - self._center, axis=1)
            max_dist = distances.max() if len(distances) > 0 else 1
            keep_prob = 0.5 + 0.5 * (distances / max_dist) ** 2
            keep_mask = np.random.random(len(points)) < keep_prob
            points = points[keep_mask]

        # Add jitter
        points += np.random.normal(0, self.tipical_cell_spacing / 5.0, points.shape)
        return points

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """tuple: Bounding box of the polygon."""
        if self.polygon is not None:
            return self.polygon.bounds
        return (0, 0, self.frame_size, self.frame_size)

    @property
    def class_instance(self) -> NDArray[np.integer]:
        """np.ndarray: Sampled cell type indices."""
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def class_instance_one_hot(self) -> NDArray[np.integer]:
        """np.ndarray: One-hot encoded sampled cell types."""
        if self._class_instance_one_hot is None:
            if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
                self._class_instance_one_hot = self.rng.multinomial(
                    n=1, pvals=self.cell_probabilities
                )
            else:
                self._class_instance_one_hot = np.empty((0, 0), dtype=int)
        return self._class_instance_one_hot

    @property
    def ML_class(self) -> NDArray[np.integer]:
        """np.ndarray: Maximum likelihood cell type."""
        if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
            return np.argmax(self.cell_probabilities, axis=1)
        return np.empty(0, dtype=int)

    def is_inside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are inside the structure."""
        return np.array([self.polygon.contains(Point(p[0], p[1])) for p in points], dtype=bool)

    def is_outside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are outside the structure."""
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix: NDArray[np.bool_]) -> None:
        """Remove cells at specified indices."""
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]

    def apply_rules(self, rules) -> NDArray[np.floating]:
        """Apply rules to compute cell type probabilities."""
        if self.cell_centroids is None or len(self.cell_centroids) == 0:
            return np.empty((0, rules[0].n_cell_types if rules else 0))

        probs = None
        for rule in rules:
            probs = rule.apply(self.cell_centroids, current_probs=probs)

        if probs is None:
            n_cells = self.cell_centroids.shape[0]
            n_types = rules[0].n_cell_types if rules else 0
            probs = np.empty((n_cells, n_types))

        return probs

    def normalized_distance_from_center(
        self, points: Optional[NDArray[np.floating]] = None
    ) -> NDArray[np.floating]:
        """Calculate normalized distance from cluster center (0=center, 1=edge)."""
        if points is None:
            points = self.cell_centroids
        if points is None or len(points) == 0:
            return np.empty(0)

        distances = np.linalg.norm(points - self._center, axis=1)
        return np.clip(distances / self.radius, 0, 1)


class GlandularUnit:
    """Glandular/acinar structure with secretory units.

    Represents glandular tissue components such as acini, alveoli, or
    secretory tubules. The structure consists of multiple connected
    secretory units (acini) arranged around a central duct or lumen.

    Parameters
    ----------
    frame_size : int, optional
        Size of the FOV in pixels. Default is 5000.
    n_acini : int, optional
        Number of acinar units. Default is 5.
    acinus_radius : float, optional
        Radius of each acinus. Default is 50.
    arrangement : str, optional
        Acini arrangement: 'circular', 'linear', 'random'. Default is 'circular'.
    central_duct : bool, optional
        Whether to include a central duct. Default is True.
    duct_radius : float, optional
        Radius of central duct. Default is 20.
    fixed_center : np.ndarray, optional
        Center position. If None, random placement.
    tipical_cell_spacing : float, optional
        Average cell spacing. Default is 8.
    rules : CellTypeRuleBase or list
        Rules for cell type assignment. Required.

    Attributes
    ----------
    polygon : shapely.Polygon
        Union of all acinar polygons.
    acini : list of shapely.Polygon
        Individual acinar polygons.
    duct : shapely.Polygon
        Central duct polygon (if central_duct=True).
    cell_centroids : np.ndarray
        Cell centroid positions.
    acinus_indices : np.ndarray
        Acinus index for each cell (-1 for duct cells).
    """

    def __init__(
        self,
        frame_size: int = 5000,
        n_acini: int = 5,
        acinus_radius: float = 50,
        arrangement: str = "circular",
        central_duct: bool = True,
        duct_radius: float = 20,
        fixed_center: Optional[NDArray[np.floating]] = None,
        tipical_cell_spacing: float = 8,
        rules=None,
    ) -> None:
        if rules is None:
            raise ValueError("No rules provided")

        self.frame_size = frame_size
        self.n_acini = n_acini
        self.acinus_radius = acinus_radius
        self.arrangement = arrangement
        self.central_duct = central_duct
        self.duct_radius = duct_radius
        self.fixed_center = fixed_center
        self.tipical_cell_spacing = tipical_cell_spacing

        self.rules = rules if isinstance(rules, list) else [rules]
        self.original_rules = self.rules

        self.polygon = None
        self.acini = []
        self.duct = None
        self.cell_centroids = None
        self.cell_probabilities = None
        self.acinus_indices = None

        self.rng = np.random.default_rng(seed=int(time.time() * 1e6))
        self._class_instance_one_hot = None
        self._center = None

    @property
    def scale(self) -> float:
        """float: Effective scale for rule compatibility."""
        return self.acinus_radius * 2

    @property
    def center(self) -> NDArray[np.floating]:
        """np.ndarray: Center of the glandular unit."""
        if self._center is not None:
            return self._center.reshape(1, 2)
        return np.array([[self.frame_size / 2, self.frame_size / 2]])

    def generate(self, **kwargs) -> "GlandularUnit":
        """Generate a new realization of this glandular unit."""
        other = copy.deepcopy(self)
        other.polygon = None
        other.acini = []
        other.duct = None
        other.cell_centroids = None
        other.cell_probabilities = None
        other.acinus_indices = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        # Set center
        if other.fixed_center is not None:
            other._center = np.asarray(other.fixed_center).flatten()[:2]
        else:
            margin = other.acinus_radius * 3
            other._center = np.random.uniform(margin, other.frame_size - margin, size=2)

        # Generate structure
        other.acini, other.duct = other._generate_acini()
        other.polygon = unary_union(other.acini + ([other.duct] if other.duct else []))

        # Generate cells
        other.cell_centroids, other.acinus_indices = other._generate_cells()

        # Apply rules
        other.rules = [r.adapt_rule_to_element(other) for r in other.rules]
        other.cell_probabilities = other.apply_rules(other.rules)

        return other

    def _generate_acini(self) -> Tuple[list, Optional[Polygon]]:
        """Generate acinar polygons and central duct."""
        acini = []

        if self.arrangement == 'circular':
            # Arrange acini in a circle around center
            angles = np.linspace(0, 2 * np.pi, self.n_acini, endpoint=False)
            arrangement_radius = self.acinus_radius * 1.5

            for angle in angles:
                acinus_center = self._center + arrangement_radius * np.array([np.cos(angle), np.sin(angle)])
                acinus = Point(acinus_center).buffer(self.acinus_radius, resolution=16)
                acini.append(acinus)

        elif self.arrangement == 'linear':
            # Arrange acini in a line
            direction = np.random.uniform(0, np.pi)
            dir_vec = np.array([np.cos(direction), np.sin(direction)])
            spacing = self.acinus_radius * 2.2
            start = self._center - (self.n_acini - 1) / 2 * spacing * dir_vec

            for i in range(self.n_acini):
                acinus_center = start + i * spacing * dir_vec
                acinus = Point(acinus_center).buffer(self.acinus_radius, resolution=16)
                acini.append(acinus)

        else:  # random
            for _ in range(self.n_acini):
                offset = np.random.uniform(-self.acinus_radius * 2, self.acinus_radius * 2, size=2)
                acinus_center = self._center + offset
                acinus = Point(acinus_center).buffer(self.acinus_radius, resolution=16)
                acini.append(acinus)

        # Generate central duct
        duct = None
        if self.central_duct:
            duct = Point(self._center).buffer(self.duct_radius, resolution=16)

        return acini, duct

    def _generate_cells(self) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Generate cell centroids within the glandular unit."""
        if self.polygon.is_empty:
            return np.empty((0, 2)), np.empty(0, dtype=int)

        bounds = self.polygon.bounds
        x = np.arange(bounds[0], bounds[2], self.tipical_cell_spacing)
        y = np.arange(bounds[1], bounds[3], self.tipical_cell_spacing * np.sin(np.pi / 3))
        X, Y = np.meshgrid(x, y)
        X[::2] += self.tipical_cell_spacing / 2.0
        points = np.stack((X.flatten(), Y.flatten()), axis=1)

        # Keep points inside polygon
        mask = np.array([self.polygon.contains(Point(p[0], p[1])) for p in points])
        points = points[mask]

        # Add jitter
        points += np.random.normal(0, self.tipical_cell_spacing / 5.0, points.shape)

        # Assign acinus indices
        acinus_indices = self._assign_acinus_indices(points)

        return points, acinus_indices

    def _assign_acinus_indices(self, points: NDArray[np.floating]) -> NDArray[np.integer]:
        """Assign each cell to an acinus or duct."""
        indices = np.full(len(points), -1, dtype=int)

        for i, point in enumerate(points):
            px = Point(point[0], point[1])

            # Check if in duct first
            if self.duct is not None and self.duct.contains(px):
                indices[i] = -1
                continue

            # Check each acinus
            for j, acinus in enumerate(self.acini):
                if acinus.contains(px):
                    indices[i] = j
                    break

        return indices

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """tuple: Bounding box of the polygon."""
        if self.polygon is not None and not self.polygon.is_empty:
            return self.polygon.bounds
        return (0, 0, self.frame_size, self.frame_size)

    @property
    def class_instance(self) -> NDArray[np.integer]:
        """np.ndarray: Sampled cell type indices."""
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def class_instance_one_hot(self) -> NDArray[np.integer]:
        """np.ndarray: One-hot encoded sampled cell types."""
        if self._class_instance_one_hot is None:
            if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
                self._class_instance_one_hot = self.rng.multinomial(
                    n=1, pvals=self.cell_probabilities
                )
            else:
                self._class_instance_one_hot = np.empty((0, 0), dtype=int)
        return self._class_instance_one_hot

    @property
    def ML_class(self) -> NDArray[np.integer]:
        """np.ndarray: Maximum likelihood cell type."""
        if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
            return np.argmax(self.cell_probabilities, axis=1)
        return np.empty(0, dtype=int)

    def is_inside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are inside the structure."""
        return np.array([self.polygon.contains(Point(p[0], p[1])) for p in points], dtype=bool)

    def is_outside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are outside the structure."""
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix: NDArray[np.bool_]) -> None:
        """Remove cells at specified indices."""
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]
        if self.acinus_indices is not None:
            self.acinus_indices = self.acinus_indices[~bool_ix]

    def apply_rules(self, rules) -> NDArray[np.floating]:
        """Apply rules to compute cell type probabilities."""
        if self.cell_centroids is None or len(self.cell_centroids) == 0:
            return np.empty((0, rules[0].n_cell_types if rules else 0))

        probs = None
        for rule in rules:
            probs = rule.apply(self.cell_centroids, current_probs=probs)

        if probs is None:
            n_cells = self.cell_centroids.shape[0]
            n_types = rules[0].n_cell_types if rules else 0
            probs = np.empty((n_cells, n_types))

        return probs


class InterfaceElement:
    """Tissue boundary/interface element.

    Represents transition zones between different tissue types such as
    epithelial-stromal interfaces, tumor-normal boundaries, or any
    biological interface where two distinct regions meet.

    Parameters
    ----------
    frame_size : int, optional
        Size of the FOV in pixels. Default is 5000.
    interface_position : float, optional
        Position of the interface (0-1 fraction of frame). Default is 0.5.
    interface_width : float, optional
        Width of the transition zone. Default is 50.
    orientation : str, optional
        Interface orientation: 'horizontal', 'vertical', 'diagonal'. Default is 'horizontal'.
    waviness : float, optional
        Amount of wave/undulation in the interface (0-1). Default is 0.1.
    n_waves : int, optional
        Number of wave cycles across the interface. Default is 3.
    tipical_cell_spacing : float, optional
        Average cell spacing. Default is 8.
    rules_side_a : CellTypeRuleBase or list, optional
        Rules for side A (before interface). If None, uses main rules.
    rules_side_b : CellTypeRuleBase or list, optional
        Rules for side B (after interface). If None, uses main rules.
    rules : CellTypeRuleBase or list
        Default rules for both sides. Required if side-specific rules not provided.

    Attributes
    ----------
    polygon : shapely.Polygon
        The full element polygon (entire frame).
    interface_line : np.ndarray
        Points defining the interface boundary.
    cell_centroids : np.ndarray
        Cell centroid positions.
    side_indices : np.ndarray
        Side index (0=A, 1=B) for each cell.
    interface_distances : np.ndarray
        Signed distance from interface for each cell (negative=A, positive=B).
    """

    def __init__(
        self,
        frame_size: int = 5000,
        interface_position: float = 0.5,
        interface_width: float = 50,
        orientation: str = "horizontal",
        waviness: float = 0.1,
        n_waves: int = 3,
        tipical_cell_spacing: float = 8,
        rules_side_a=None,
        rules_side_b=None,
        rules=None,
    ) -> None:
        if rules is None and (rules_side_a is None or rules_side_b is None):
            raise ValueError("Either 'rules' or both 'rules_side_a' and 'rules_side_b' must be provided")

        self.frame_size = frame_size
        self.interface_position = interface_position
        self.interface_width = interface_width
        self.orientation = orientation
        self.waviness = waviness
        self.n_waves = n_waves
        self.tipical_cell_spacing = tipical_cell_spacing

        # Set up rules
        if rules_side_a is not None:
            self.rules_side_a = rules_side_a if isinstance(rules_side_a, list) else [rules_side_a]
        else:
            self.rules_side_a = rules if isinstance(rules, list) else [rules]

        if rules_side_b is not None:
            self.rules_side_b = rules_side_b if isinstance(rules_side_b, list) else [rules_side_b]
        else:
            self.rules_side_b = rules if isinstance(rules, list) else [rules]

        self.rules = self.rules_side_a  # Default for compatibility
        self.original_rules_a = self.rules_side_a
        self.original_rules_b = self.rules_side_b

        self.polygon = None
        self.interface_line = None
        self.cell_centroids = None
        self.cell_probabilities = None
        self.side_indices = None
        self.interface_distances = None

        self.rng = np.random.default_rng(seed=int(time.time() * 1e6))
        self._class_instance_one_hot = None

    @property
    def scale(self) -> float:
        """float: Effective scale for rule compatibility."""
        return self.frame_size / 2

    @property
    def center(self) -> NDArray[np.floating]:
        """np.ndarray: Center of the element."""
        return np.array([[self.frame_size / 2, self.frame_size / 2]])

    def generate(self, **kwargs) -> "InterfaceElement":
        """Generate a new realization of this interface element."""
        other = copy.deepcopy(self)
        other.polygon = None
        other.interface_line = None
        other.cell_centroids = None
        other.cell_probabilities = None
        other.side_indices = None
        other.interface_distances = None
        other._class_instance_one_hot = None
        other.rules_side_a = other.original_rules_a
        other.rules_side_b = other.original_rules_b

        # Generate polygon and interface
        other.polygon = Polygon([
            (0, 0), (other.frame_size, 0),
            (other.frame_size, other.frame_size), (0, other.frame_size)
        ])
        other.interface_line = other._generate_interface_line()

        # Generate cells
        other.cell_centroids = other._generate_cells()
        other.side_indices, other.interface_distances = other._compute_side_assignments()

        # Apply rules
        other.cell_probabilities = other._apply_side_rules()

        return other

    def _generate_interface_line(self) -> NDArray[np.floating]:
        """Generate the wavy interface line."""
        n_points = 100

        if self.orientation == 'horizontal':
            base_y = self.frame_size * self.interface_position
            x = np.linspace(0, self.frame_size, n_points)
            wave = np.sin(x / self.frame_size * 2 * np.pi * self.n_waves)
            y = base_y + wave * self.frame_size * self.waviness
            return np.column_stack([x, y])

        elif self.orientation == 'vertical':
            base_x = self.frame_size * self.interface_position
            y = np.linspace(0, self.frame_size, n_points)
            wave = np.sin(y / self.frame_size * 2 * np.pi * self.n_waves)
            x = base_x + wave * self.frame_size * self.waviness
            return np.column_stack([x, y])

        else:  # diagonal
            t = np.linspace(0, 1, n_points)
            base_x = t * self.frame_size
            base_y = t * self.frame_size
            wave = np.sin(t * 2 * np.pi * self.n_waves)
            offset = wave * self.frame_size * self.waviness
            x = base_x + offset * 0.707
            y = base_y - offset * 0.707
            return np.column_stack([x, y])

    def _generate_cells(self) -> NDArray[np.floating]:
        """Generate cell centroids."""
        x = np.arange(0, self.frame_size, self.tipical_cell_spacing, dtype=float)
        y = np.arange(0, self.frame_size, self.tipical_cell_spacing * np.sin(np.pi / 3), dtype=float)
        X, Y = np.meshgrid(x, y)
        X[::2] += self.tipical_cell_spacing / 2.0
        points = np.stack((X.flatten(), Y.flatten()), axis=1)

        # Add jitter
        points += np.random.normal(0, self.tipical_cell_spacing / 5.0, points.shape)
        return points

    def _compute_side_assignments(self) -> Tuple[NDArray[np.integer], NDArray[np.floating]]:
        """Compute which side each cell is on and distance from interface."""
        interface_line = LineString(self.interface_line)

        distances = np.zeros(len(self.cell_centroids))
        sides = np.zeros(len(self.cell_centroids), dtype=int)

        for i, point in enumerate(self.cell_centroids):
            px = Point(point[0], point[1])
            dist = interface_line.distance(px)

            # Determine side based on orientation
            if self.orientation == 'horizontal':
                # Interpolate interface Y at this X
                idx = np.searchsorted(self.interface_line[:, 0], point[0])
                idx = np.clip(idx, 1, len(self.interface_line) - 1)
                interface_y = np.interp(point[0], self.interface_line[:, 0], self.interface_line[:, 1])
                if point[1] < interface_y:
                    sides[i] = 0
                    distances[i] = -dist
                else:
                    sides[i] = 1
                    distances[i] = dist
            elif self.orientation == 'vertical':
                interface_x = np.interp(point[1], self.interface_line[:, 1], self.interface_line[:, 0])
                if point[0] < interface_x:
                    sides[i] = 0
                    distances[i] = -dist
                else:
                    sides[i] = 1
                    distances[i] = dist
            else:  # diagonal
                # Use distance from diagonal line y=x
                if point[1] < point[0]:
                    sides[i] = 0
                    distances[i] = -dist
                else:
                    sides[i] = 1
                    distances[i] = dist

        return sides, distances

    def _apply_side_rules(self) -> NDArray[np.floating]:
        """Apply rules to each side separately."""
        n_cells = len(self.cell_centroids)
        n_cell_types = self.rules_side_a[0].n_cell_types

        probs = np.zeros((n_cells, n_cell_types))

        # Side A
        mask_a = self.side_indices == 0
        if np.any(mask_a):
            adapted_rules = [r.adapt_rule_to_element(self) for r in self.rules_side_a]
            probs_a = None
            for rule in adapted_rules:
                probs_a = rule.apply(self.cell_centroids[mask_a], current_probs=probs_a)
            probs[mask_a] = probs_a

        # Side B
        mask_b = self.side_indices == 1
        if np.any(mask_b):
            adapted_rules = [r.adapt_rule_to_element(self) for r in self.rules_side_b]
            probs_b = None
            for rule in adapted_rules:
                probs_b = rule.apply(self.cell_centroids[mask_b], current_probs=probs_b)
            probs[mask_b] = probs_b

        return probs

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """tuple: Bounding box of the polygon."""
        return (0, 0, self.frame_size, self.frame_size)

    @property
    def class_instance(self) -> NDArray[np.integer]:
        """np.ndarray: Sampled cell type indices."""
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def class_instance_one_hot(self) -> NDArray[np.integer]:
        """np.ndarray: One-hot encoded sampled cell types."""
        if self._class_instance_one_hot is None:
            if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
                self._class_instance_one_hot = self.rng.multinomial(
                    n=1, pvals=self.cell_probabilities
                )
            else:
                self._class_instance_one_hot = np.empty((0, 0), dtype=int)
        return self._class_instance_one_hot

    @property
    def ML_class(self) -> NDArray[np.integer]:
        """np.ndarray: Maximum likelihood cell type."""
        if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
            return np.argmax(self.cell_probabilities, axis=1)
        return np.empty(0, dtype=int)

    def is_inside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are inside the element (always True for frame-wide)."""
        return np.ones(len(points), dtype=bool)

    def is_outside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are outside the element."""
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix: NDArray[np.bool_]) -> None:
        """Remove cells at specified indices."""
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]
        if self.side_indices is not None:
            self.side_indices = self.side_indices[~bool_ix]
        if self.interface_distances is not None:
            self.interface_distances = self.interface_distances[~bool_ix]

    def apply_rules(self, rules) -> NDArray[np.floating]:
        """Apply rules to compute cell type probabilities."""
        if self.cell_centroids is None or len(self.cell_centroids) == 0:
            return np.empty((0, rules[0].n_cell_types if rules else 0))

        probs = None
        for rule in rules:
            probs = rule.apply(self.cell_centroids, current_probs=probs)

        if probs is None:
            n_cells = self.cell_centroids.shape[0]
            n_types = rules[0].n_cell_types if rules else 0
            probs = np.empty((n_cells, n_types))

        return probs


class StromalElement:
    """Background stromal/connective tissue element.

    Represents stromal tissue components such as connective tissue stroma,
    mesenchyme, or extracellular matrix-rich regions. Features lower cell
    density than parenchymal tissue and can include embedded structures.

    Parameters
    ----------
    frame_size : int, optional
        Size of the FOV in pixels. Default is 5000.
    cell_density : float, optional
        Relative cell density (0-1). 1.0 = normal spacing, 0.5 = half density.
        Default is 0.6.
    heterogeneity : float, optional
        Spatial heterogeneity in cell density (0-1). Default is 0.3.
    exclude_regions : list of shapely.Polygon, optional
        Regions where no cells should be placed.
    tipical_cell_spacing : float, optional
        Base cell spacing before density adjustment. Default is 12.
    rules : CellTypeRuleBase or list
        Rules for cell type assignment. Required.

    Attributes
    ----------
    polygon : shapely.Polygon
        The stromal region polygon.
    cell_centroids : np.ndarray
        Cell centroid positions.

    Examples
    --------
    >>> from pointillsim.rules import MixOfNCellTypesRule
    >>> rule = MixOfNCellTypesRule(
    ...     n_cell_types=4, list_N=[2, 3],  # fibroblasts and immune cells
    ...     proportions=[0.8, 0.2]
    ... )
    >>> stroma = StromalElement(
    ...     frame_size=500, cell_density=0.5, rules=rule
    ... )
    >>> realized = stroma.generate()
    """

    def __init__(
        self,
        frame_size: int = 5000,
        cell_density: float = 0.6,
        heterogeneity: float = 0.3,
        exclude_regions: Optional[list] = None,
        tipical_cell_spacing: float = 12,
        rules=None,
    ) -> None:
        if rules is None:
            raise ValueError("No rules provided")

        self.frame_size = frame_size
        self.cell_density = np.clip(cell_density, 0.1, 1.0)
        self.heterogeneity = np.clip(heterogeneity, 0, 1)
        self.exclude_regions = exclude_regions or []
        self.tipical_cell_spacing = tipical_cell_spacing

        self.rules = rules if isinstance(rules, list) else [rules]
        self.original_rules = self.rules

        self.polygon = None
        self.cell_centroids = None
        self.cell_probabilities = None

        self.rng = np.random.default_rng(seed=int(time.time() * 1e6))
        self._class_instance_one_hot = None

    @property
    def scale(self) -> float:
        """float: Effective scale for rule compatibility."""
        return self.frame_size / 2

    @property
    def center(self) -> NDArray[np.floating]:
        """np.ndarray: Center of the element."""
        return np.array([[self.frame_size / 2, self.frame_size / 2]])

    def generate(self, **kwargs) -> "StromalElement":
        """Generate a new realization of this stromal element."""
        other = copy.deepcopy(self)
        other.polygon = None
        other.cell_centroids = None
        other.cell_probabilities = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        # Generate polygon (frame minus exclusions)
        other.polygon = other._generate_polygon()

        # Generate cells
        other.cell_centroids = other._generate_cells()

        # Apply rules
        other.rules = [r.adapt_rule_to_element(other) for r in other.rules]
        other.cell_probabilities = other.apply_rules(other.rules)

        return other

    def _generate_polygon(self) -> Polygon:
        """Generate the stromal region polygon."""
        frame = Polygon([
            (0, 0), (self.frame_size, 0),
            (self.frame_size, self.frame_size), (0, self.frame_size)
        ])

        # Subtract excluded regions
        result = frame
        for region in self.exclude_regions:
            if hasattr(region, 'polygon') and region.polygon is not None:
                result = result.difference(region.polygon)
            elif isinstance(region, Polygon):
                result = result.difference(region)

        return result

    def _generate_cells(self) -> NDArray[np.floating]:
        """Generate cell centroids with variable density."""
        # Adjust spacing based on density
        base_spacing = self.tipical_cell_spacing / np.sqrt(self.cell_density)

        x = np.arange(0, self.frame_size, base_spacing, dtype=float)
        y = np.arange(0, self.frame_size, base_spacing * np.sin(np.pi / 3), dtype=float)
        X, Y = np.meshgrid(x, y)
        X[::2] += base_spacing / 2.0
        points = np.stack((X.flatten(), Y.flatten()), axis=1)

        # Keep points inside polygon
        mask = np.array([self.polygon.contains(Point(p[0], p[1])) for p in points])
        points = points[mask]

        # Apply heterogeneity (spatially-correlated random thinning)
        if self.heterogeneity > 0 and len(points) > 0:
            noise_scale = self.frame_size / 4
            # Compute spatially-correlated noise
            local_noise = np.sin(points[:, 0] / noise_scale) * np.cos(points[:, 1] / noise_scale)
            local_noise = (local_noise + 1) / 2  # 0-1 range

            # Probability of keeping each cell
            keep_prob = 1 - self.heterogeneity * (1 - local_noise)
            keep_mask = np.random.random(len(points)) < keep_prob
            points = points[keep_mask]

        # Add jitter
        if len(points) > 0:
            points += np.random.normal(0, base_spacing / 4.0, points.shape)

        return points

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """tuple: Bounding box of the polygon."""
        if self.polygon is not None and not self.polygon.is_empty:
            return self.polygon.bounds
        return (0, 0, self.frame_size, self.frame_size)

    @property
    def class_instance(self) -> NDArray[np.integer]:
        """np.ndarray: Sampled cell type indices."""
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def class_instance_one_hot(self) -> NDArray[np.integer]:
        """np.ndarray: One-hot encoded sampled cell types."""
        if self._class_instance_one_hot is None:
            if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
                self._class_instance_one_hot = self.rng.multinomial(
                    n=1, pvals=self.cell_probabilities
                )
            else:
                self._class_instance_one_hot = np.empty((0, 0), dtype=int)
        return self._class_instance_one_hot

    @property
    def ML_class(self) -> NDArray[np.integer]:
        """np.ndarray: Maximum likelihood cell type."""
        if self.cell_probabilities is not None and len(self.cell_probabilities) > 0:
            return np.argmax(self.cell_probabilities, axis=1)
        return np.empty(0, dtype=int)

    def is_inside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are inside the structure."""
        return np.array([self.polygon.contains(Point(p[0], p[1])) for p in points], dtype=bool)

    def is_outside(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check which points are outside the structure."""
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix: NDArray[np.bool_]) -> None:
        """Remove cells at specified indices."""
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]

    def apply_rules(self, rules) -> NDArray[np.floating]:
        """Apply rules to compute cell type probabilities."""
        if self.cell_centroids is None or len(self.cell_centroids) == 0:
            return np.empty((0, rules[0].n_cell_types if rules else 0))

        probs = None
        for rule in rules:
            probs = rule.apply(self.cell_centroids, current_probs=probs)

        if probs is None:
            n_cells = self.cell_centroids.shape[0]
            n_types = rules[0].n_cell_types if rules else 0
            probs = np.empty((n_cells, n_types))

        return probs
