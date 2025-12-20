"""Geometric utility functions for point generation."""

import numpy as np


def generate_uniform_points_in_circle(center, scale, num_points):
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
