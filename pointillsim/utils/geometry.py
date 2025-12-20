"""Geometric utility functions for point generation and polygon manipulation."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


def chaikin_smooth(
    coords: NDArray[np.floating],
    iterations: int = 2,
    closed: bool = True,
) -> NDArray[np.floating]:
    """Apply Chaikin's corner-cutting algorithm to smooth a polygon.

    Chaikin's algorithm iteratively replaces each edge with two new points,
    creating a smoother curve that approaches a quadratic B-spline.

    Parameters
    ----------
    coords : np.ndarray
        Polygon coordinates, shape (n_vertices, 2).
    iterations : int, optional
        Number of smoothing iterations. More iterations = smoother.
        Default 2.
    closed : bool, optional
        If True, treats the polygon as closed (connects last to first).
        Default True.

    Returns
    -------
    np.ndarray
        Smoothed coordinates with more vertices.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> smoothed = chaikin_smooth(coords, iterations=2)
    >>> len(smoothed) > len(coords)
    True

    Notes
    -----
    Each iteration roughly doubles the number of vertices. The smoothing
    factor is 0.25 (quarter-point), which produces a quadratic B-spline
    approximation.
    """
    if len(coords) < 3:
        return coords.copy()

    result = coords.copy()

    for _ in range(iterations):
        n = len(result)
        if n < 2:
            break

        new_coords = []
        for i in range(n):
            p0 = result[i]
            p1 = result[(i + 1) % n] if closed else result[min(i + 1, n - 1)]

            # Chaikin's quarter-point rule
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1

            new_coords.append(q)
            if closed or i < n - 1:
                new_coords.append(r)

        result = np.array(new_coords)

    return result


def smooth_polygon(
    polygon,
    iterations: int = 2,
    preserve_area: bool = True,
):
    """Smooth a Shapely polygon using Chaikin's algorithm.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon to smooth.
    iterations : int, optional
        Number of smoothing iterations. Default 2.
    preserve_area : bool, optional
        If True, scale the smoothed polygon to preserve original area.
        Default True.

    Returns
    -------
    shapely.geometry.Polygon
        Smoothed polygon.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> from pointillsim.utils.geometry import smooth_polygon
    >>> square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> smoothed = smooth_polygon(square, iterations=3)
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.affinity import scale as shapely_scale

    if polygon.is_empty:
        return polygon

    # Get exterior coordinates (excluding closing point)
    coords = np.array(polygon.exterior.coords[:-1])
    original_area = polygon.area

    # Smooth exterior
    smoothed_exterior = chaikin_smooth(coords, iterations=iterations, closed=True)

    # Handle holes if any
    smoothed_holes = []
    for interior in polygon.interiors:
        hole_coords = np.array(interior.coords[:-1])
        smoothed_hole = chaikin_smooth(hole_coords, iterations=iterations, closed=True)
        smoothed_holes.append(smoothed_hole)

    # Create new polygon
    if smoothed_holes:
        smoothed = ShapelyPolygon(smoothed_exterior, smoothed_holes)
    else:
        smoothed = ShapelyPolygon(smoothed_exterior)

    # Preserve area if requested
    if preserve_area and smoothed.area > 0:
        scale_factor = np.sqrt(original_area / smoothed.area)
        centroid = smoothed.centroid
        smoothed = shapely_scale(
            smoothed, xfact=scale_factor, yfact=scale_factor, origin=centroid
        )

    return smoothed


def add_edge_noise(
    coords: NDArray[np.floating],
    noise_std: float = 1.0,
    seed: Optional[int] = None,
) -> NDArray[np.floating]:
    """Add Gaussian noise to polygon vertex positions.

    Parameters
    ----------
    coords : np.ndarray
        Polygon coordinates, shape (n_vertices, 2).
    noise_std : float, optional
        Standard deviation of noise. Default 1.0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Perturbed coordinates.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, coords.shape)
    return coords + noise


def generate_uniform_points_in_circle(
    center: NDArray[np.floating],
    scale: float,
    num_points: int,
) -> NDArray[np.floating]:
    """Generate points uniformly distributed within a circle.

    Uses rejection sampling with normal distribution to achieve uniform
    spatial distribution within the circular region.

    Parameters
    ----------
    center : np.ndarray
        Center coordinates of the circle, shape (1, 2) or (2,).
    scale : float
        Radius of the circle.
    num_points : int
        Number of points to generate.

    Returns
    -------
    np.ndarray
        Array of shape (num_points, 2) with x, y coordinates.

    Examples
    --------
    >>> points = generate_uniform_points_in_circle(np.array([[500, 500]]), 100, 50)
    >>> points.shape
    (50, 2)
    """
    points = np.random.normal(0, 1, (num_points, center.shape[-1]))
    samples = (
        0.5
        * np.sqrt(np.random.uniform(0, scale**2, (num_points, 1)))
        * points
        / np.linalg.norm(points, axis=1, keepdims=True)
    )
    points = center + samples
    return points


def generate_points_asin_cell(
    center,
    scale,
    num_points,
    major_axis,
    minor_axis,
    rotation_angle,
    center_balance=0.4,
    smeer=0.4,
):
    """Generate points distributed within an ellipse simulating RNA dot positions in a cell.

    Creates a spatial distribution of points that mimics the appearance of
    transcript dots within a cell, with configurable density toward the center.

    Parameters
    ----------
    center : np.ndarray
        Cell centroid coordinates, shape (2,).
    scale : float
        Overall scaling factor for the distribution.
    num_points : int
        Number of points (transcript dots) to generate.
    major_axis : float
        Length of the ellipse major axis.
    minor_axis : float
        Length of the ellipse minor axis.
    rotation_angle : float
        Rotation angle of the ellipse in radians.
    center_balance : float, optional
        Controls density distribution from center to edge. Lower values
        concentrate points toward the center. Default is 0.4.
    smeer : float, optional
        Noise parameter controlling point spread. Default is 0.4.

    Returns
    -------
    np.ndarray
        Array of shape (num_points, 2) with x, y coordinates of dots.
    """
    points = np.random.normal(0, 1, (num_points, center.shape[-1]))
    renorm = np.linalg.norm(points, axis=1, keepdims=True)
    samples = (
        (0.5)
        * np.random.uniform(
            0, (scale * (1 + smeer / 2.0)) ** (2.0 / center_balance), (num_points, 1)
        )
        ** (center_balance * 0.5)
        * points
        / (renorm + smeer)
    )
    samples[:, 0] = samples[:, 0] * major_axis
    samples[:, 1] = samples[:, 1] * minor_axis
    # Rotate the points
    points = np.zeros_like(samples)
    points[:, 0] = samples[:, 0] * np.cos(rotation_angle) - samples[:, 1] * np.sin(
        rotation_angle
    )
    points[:, 1] = samples[:, 0] * np.sin(rotation_angle) + samples[:, 1] * np.cos(
        rotation_angle
    )
    return center + samples
