"""Interpolation utilities."""

# import numpy as np
# import scipy as sp


# class LinearNDInterpolatorExt(sp.interpolate.LinearNDInterpolator):
#     """Extended linear interpolator with RBF fallback for extrapolation.

#     Wraps scipy's LinearNDInterpolator but handles points outside the
#     convex hull by falling back to RBFInterpolator instead of returning NaN.

#     Parameters
#     ----------
#     points : np.ndarray
#         Data point coordinates, shape (n_points, n_dims).
#     values : np.ndarray
#         Data values at each point, shape (n_points,) or (n_points, n_values).

#     Notes
#     -----
#     This is particularly useful for spatial interpolation of cell type
#     probabilities where query points may fall slightly outside the
#     reference point convex hull.
#     """

#     def __init__(self, points, values):
#         super().__init__(points, values)
#         self.funcinterp = sp.interpolate.LinearNDInterpolator(points, values)
#         self.funcnearest = sp.interpolate.RBFInterpolator(points, values)

#     def __call__(self, *args):
#         t = self.funcinterp(*args)
#         return np.where(
#             np.any(np.isnan(t), axis=1)[:, None], self.funcnearest(*args), t
#         )
# pointillsim/utils/interpolation.py

import numpy as np
import scipy as sp
from scipy.spatial import QhullError

class LinearNDInterpolatorExt:
    def __init__(self, points, values):
        self.points = np.asarray(points)
        self.values = np.asarray(values)

        # linear (may fail)
        self.funcinterp = None
        try:
            self.funcinterp = sp.interpolate.LinearNDInterpolator(self.points, self.values)
        except QhullError:
            self.funcinterp = None

        # RBF fallback (works with 2 points if degree=0)
        self.funcrbf = sp.interpolate.RBFInterpolator(self.points, self.values, degree=0)

    def __call__(self, *args):
        # Support both calling styles:
        # 1) interp(xi) where xi is (N,D)
        # 2) interp(x, y, ...) like scipy sometimes allows
        if len(args) == 1:
            xi = np.asarray(args[0])
        else:
            xi = np.column_stack([np.asarray(a).ravel() for a in args])

        if self.funcinterp is None:
            return self.funcrbf(xi)

        t = self.funcinterp(xi)

        # If output is 1D, handle NaNs without axis=1 issues
        if t.ndim == 1:
            nan_mask = np.isnan(t)
            if nan_mask.any():
                t[nan_mask] = self.funcrbf(xi[nan_mask]).reshape(-1)
            return t

        nan_rows = np.any(np.isnan(t), axis=1)
        if nan_rows.any():
            t[nan_rows] = self.funcrbf(xi[nan_rows])
        return t