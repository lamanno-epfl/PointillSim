"""Encoding utility functions."""

import numpy as np


def one_hot_encode_array(class_ix, num_classes):
    """Convert an array of class indices to one-hot encoded matrix.

    Parameters
    ----------
    class_ix : np.ndarray
        Array of integer class indices.
    num_classes : int
        Total number of classes.

    Returns
    -------
    np.ndarray
        One-hot encoded matrix of shape (len(class_ix), num_classes).

    Examples
    --------
    >>> one_hot_encode_array([0, 2, 1], 3)
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])
    """
    one_hot_matrix = np.zeros((len(class_ix), num_classes))
    one_hot_matrix[np.arange(len(class_ix)), class_ix] = 1
    return one_hot_matrix


def unfold_int_matrix(a):
    """Unfold an integer count matrix into row and column index lists.

    For each element a[i,j] with value k, appends k copies of (i, j)
    to the output lists. Useful for expanding count matrices into
    individual observation indices.

    Parameters
    ----------
    a : np.ndarray
        2D integer array of counts, shape (n_rows, n_cols).

    Returns
    -------
    tuple
        (unfolded_0, unfolded_1) where each is a list of lists.
        unfolded_0[i] contains row indices, unfolded_1[i] contains column indices
        for row i of the input matrix.

    Examples
    --------
    >>> a = np.array([[2, 1], [0, 3]])
    >>> rows, cols = unfold_int_matrix(a)
    >>> rows[0]  # Row 0: 2 copies of col 0, 1 copy of col 1
    [0, 0, 0]
    >>> cols[0]
    [0, 0, 1]
    """
    unfolded_0 = []
    unfolded_1 = []
    for i in range(a.shape[0]):
        unfolded_per_row1 = []
        unfolded_per_row0 = []
        for j in range(a.shape[1]):
            for k in range(a[i, j]):
                unfolded_per_row1.append(j)
                unfolded_per_row0.append(i)
        unfolded_1.append(unfolded_per_row1)
        unfolded_0.append(unfolded_per_row0)
    return unfolded_0, unfolded_1
