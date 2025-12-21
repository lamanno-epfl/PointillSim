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
