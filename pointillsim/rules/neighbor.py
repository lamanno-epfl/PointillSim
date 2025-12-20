"""Neighbor-based cell type assignment rules."""

import numpy as np
from sklearn.neighbors import KDTree

from .base import CellTypeRuleBase


class DeterministicNeighborAssignment(CellTypeRuleBase):
    """Rule that modifies cell types based on neighbor relationships.

    Cells near a specified "focus" type are reassigned to a "target" type.
    Useful for simulating cell-cell interactions or boundary effects.

    Parameters
    ----------
    n_cell_types : int
        Total number of cell types.
    close_to : int, optional
        Cell type index that triggers neighbor reassignment.
    becomes : int, optional
        Cell type index that neighbors are reassigned to.
    threshold_probdist : float, optional
        Probability distance threshold for identifying focus cells. Default 0.3.
    allow_selftype_change : bool, optional
        If True, focus cells can also be reassigned. Default False.
    mix : float, optional
        Blending weight for new probabilities. Default 0.9.
    nneigh : int, optional
        Number of nearest neighbors to consider. Default 2.

    Examples
    --------
    >>> # Cells neighboring type 0 become type 5
    >>> rule = DeterministicNeighborAssignment(10, close_to=0, becomes=5)
    """

    def __init__(
        self,
        n_cell_types,
        close_to=None,
        becomes=None,
        threshold_probdist=0.3,
        allow_selftype_change=False,
        mix=0.9,
        nneigh=2,
    ):
        super().__init__(n_cell_types=n_cell_types)
        if close_to is None:
            self.close_to = np.random.randint(0, n_cell_types)
        else:
            self.close_to = close_to
        if becomes is None:
            self.becomes = np.random.randint(0, n_cell_types)
        else:
            self.becomes = becomes
        self.close_to_onehot = np.zeros(self.n_cell_types)
        self.close_to_onehot[self.close_to] = 1
        self.becomes_onehot = np.zeros(self.n_cell_types)
        self.becomes_onehot[self.becomes] = 1
        self.threshold = threshold_probdist
        self.allow_selftype_change = allow_selftype_change
        self.mix = mix
        self.nneigh = nneigh

    def apply(self, cell_centroids, current_probs):
        """Apply neighbor-based cell type reassignment.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions for KDTree queries.
        current_probs : np.ndarray
            Required - existing probabilities to modify.

        Returns
        -------
        np.ndarray
            Modified probability matrix.
        """
        new_probs = np.copy(current_probs)
        kd = KDTree(cell_centroids)
        dist_prob = np.sqrt(np.sum((current_probs - self.close_to_onehot) ** 2, 1))
        type_of_focus = dist_prob < self.threshold
        dist_, ind_ = kd.query(cell_centroids, k=1 + self.nneigh)
        for i in range(self.nneigh):
            dist, ind = dist_[:, 1 + i], ind_[:, 1 + i]
            if not self.allow_selftype_change:
                nearestneigh_is_focustype = type_of_focus[ind]
                bool_ = type_of_focus & ~nearestneigh_is_focustype
            else:
                bool_ = type_of_focus
            ixes = ind[bool_]
            new_probs[ixes, :] = (1 - self.mix) * new_probs[
                ixes, :
            ] + self.mix * self.becomes_onehot

            cell_probs = np.maximum(new_probs, 1e-12)
            cell_probs = cell_probs / (cell_probs.sum(axis=1)[:, np.newaxis] + 1e-8)

        return cell_probs
