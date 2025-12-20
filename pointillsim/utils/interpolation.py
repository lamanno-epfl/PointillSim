"""Interpolation utilities."""

import numpy as np
import scipy as sp


class LinearNDInterpolatorExt(sp.interpolate.LinearNDInterpolator):
    """Extended linear interpolator with RBF fallback for extrapolation.

    Wraps scipy's LinearNDInterpolator but handles points outside the
    convex hull by falling back to RBFInterpolator instead of returning NaN.

    Parameters
    ----------
    points : np.ndarray
        Data point coordinates, shape (n_points, n_dims).
    values : np.ndarray
        Data values at each point, shape (n_points,) or (n_points, n_values).

    Notes
    -----
    This is particularly useful for spatial interpolation of cell type
    probabilities where query points may fall slightly outside the
    reference point convex hull.
    """

    def __init__(self, points, values):
        super().__init__(points, values)
        self.funcinterp = sp.interpolate.LinearNDInterpolator(points, values)
        self.funcnearest = sp.interpolate.RBFInterpolator(points, values)

    def __call__(self, *args):
        t = self.funcinterp(*args)
        return np.where(
            np.any(np.isnan(t), axis=1)[:, None], self.funcnearest(*args), t
        )
