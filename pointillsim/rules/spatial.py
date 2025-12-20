"""Spatial-based cell type assignment rules."""

import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

from .base import CellTypeRuleBase
from ..utils.geometry import generate_uniform_points_in_circle
from ..utils.interpolation import LinearNDInterpolatorExt


class ProbabilityNodeFieldRule(CellTypeRuleBase):
    """Rule that creates spatial probability gradients using reference points.

    Defines a probability field over space by interpolating between
    reference points with known cell type probabilities. Creates smooth
    spatial transitions between different cell type compositions.

    This is useful for simulating tissue regions with gradual cell type
    transitions, such as cortical layers or tissue boundaries.

    Parameters
    ----------
    n_cell_types : int, optional
        Number of cell types. Default is 3.
    n_ref_points : int, optional
        Number of reference points for interpolation. Default is 4.
    alpha : float, optional
        Dirichlet concentration parameter for reference point probabilities.
        Lower values create more peaked distributions (one dominant type),
        higher values create more uniform distributions. Default is 0.05.
    ref_probs : np.ndarray, optional
        Explicit probabilities at reference points, shape (n_ref_points, n_cell_types).
        If None, randomly generated from Dirichlet(alpha).
    element : HistologicalElement, optional
        Element to adapt to immediately.
    reference_points : np.ndarray, optional
        Explicit reference point positions, shape (n_ref_points, 2).
    """

    def __init__(
        self,
        n_cell_types=3,
        n_ref_points=4,
        alpha=0.05,
        ref_probs=None,
        element=None,
        reference_points=None,
    ):
        super().__init__(n_cell_types)
        self.n_ref_points = n_ref_points
        self.alpha = alpha
        if ref_probs is None:
            self.ref_probs = np.random.dirichlet(
                self.alpha * np.ones(self.n_cell_types), self.n_ref_points
            )
        else:
            self.ref_probs = ref_probs
        if element is not None:
            self.adapt_rule_to_element(element)
        if reference_points is not None:
            self.reference_points = reference_points
            self.is_adapted = True
        else:
            self.reference_points = None

    def adapt_rule_to_element(self, element):
        """Adapt the rule by generating reference points within the element.

        Creates reference points distributed within the element's geometry
        if not explicitly provided. Points are arranged in a convex hull
        pattern to ensure good interpolation coverage.

        Parameters
        ----------
        element : HistologicalElement
            Element providing center and scale for reference point placement.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        if self.reference_points is not None:
            self.is_adapted = True
            return self

        self.is_adapted = True
        vertices = generate_uniform_points_in_circle(
            element.center, element.scale, self.n_ref_points
        )
        if self.n_ref_points <= 2:
            self.reference_points = vertices
        else:
            vertices = np.row_stack([vertices, vertices[0:1, :]])
            cvxh = np.array(Polygon(vertices).convex_hull.exterior.coords)[:-1]
            if cvxh.shape[0] >= self.n_ref_points:
                polygon_vertices = cvxh[: self.n_ref_points]
            elif self.n_ref_points - cvxh.shape[0] == 1:
                polygon_vertices = np.row_stack(
                    [cvxh, (cvxh[-1:, :] + cvxh[:1, :]) / 2.0]
                )
            elif self.n_ref_points - cvxh.shape[0] == 2:
                polygon_vertices = np.row_stack(
                    [cvxh, (cvxh[-1:, :] + cvxh[:1, :]) / 2.0]
                )
                polygon_vertices = np.row_stack(
                    [
                        polygon_vertices,
                        (polygon_vertices[-1:, :] + polygon_vertices[:1, :]) / 2.0,
                    ]
                )
            else:
                raise ValueError("Weird corner case")
            self.reference_points = polygon_vertices
        return self

    def apply(self, cell_centroids, current_probs=None, mix=0.9):
        """Interpolate cell type probabilities from reference point field.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions, shape (n_cells, 2).
        current_probs : np.ndarray, optional
            Existing probabilities to blend with interpolated values.
        mix : float, optional
            Blending weight for interpolated vs existing probabilities.
            1.0 uses only interpolated, 0.0 uses only existing. Default 0.9.

        Returns
        -------
        np.ndarray
            Interpolated and normalized probability matrix.

        Raises
        ------
        ValueError
            If rule has not been adapted to an element.
        """
        if self.is_adapted is False:
            raise ValueError("Rule not adapted to element")
        else:
            interpolator = LinearNDInterpolatorExt(self.reference_points, self.ref_probs)
            interpolated_probs = interpolator(cell_centroids)
            if np.isnan(interpolated_probs).any():
                plt.plot(self.reference_points[:, 0], self.reference_points[:, 1], "-o")
                raise ValueError("Interpolated probabilities contain NaNs")
            if current_probs is not None:
                interpolated_probs = (
                    mix * interpolated_probs + (1 - mix) * current_probs
                )
            interpolated_probs = np.maximum(interpolated_probs, 1e-12)

            cell_probs = interpolated_probs / (
                interpolated_probs.sum(axis=1)[:, np.newaxis] + 1e-8
            )

            return cell_probs


class SingleTypeRule(CellTypeRuleBase):
    """Rule that assigns all cells to a single cell type.

    Useful for uniform regions where all cells belong to the same type,
    such as specific tissue compartments or structures.

    Parameters
    ----------
    n_cell_types : int
        Total number of cell types in the system.
    cell_type_ix : int, optional
        Index of the cell type to assign. If None, randomly selected.

    Examples
    --------
    >>> rule = SingleTypeRule(n_cell_types=10, cell_type_ix=3)
    >>> probs = rule.apply(centroids)
    >>> np.all(probs.argmax(axis=1) == 3)
    True
    """

    def __init__(self, n_cell_types, cell_type_ix=None):
        super().__init__(n_cell_types=n_cell_types)
        if cell_type_ix is None:
            self.cell_type_ix = np.random.randint(self.n_cell_types)
        else:
            self.cell_type_ix = cell_type_ix

    def apply(self, cell_centroids, current_probs=None, mix=0.9):
        """Assign single cell type probability (near 1.0) to all cells.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions, shape (n_cells, 2).
        current_probs : np.ndarray, optional
            If provided, blends with existing probabilities.
        mix : float, optional
            Blending weight. Default 0.9.

        Returns
        -------
        np.ndarray
            Probability matrix with specified type having probability ~1.
        """
        new_probs = np.zeros((len(cell_centroids), self.n_cell_types), dtype=float)
        new_probs[:, self.cell_type_ix] = 0.9999999
        if current_probs is None:
            cell_probs = new_probs
        else:
            cell_probs = mix * new_probs + (1 - mix) * current_probs
        cell_probs = np.maximum(cell_probs, 1e-12)

        cell_probs = cell_probs / (cell_probs.sum(axis=1)[:, np.newaxis] + 1e-8)

        return cell_probs
