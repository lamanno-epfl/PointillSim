"""Random-based cell type assignment rules."""

import numpy as np
from .base import CellTypeRuleBase


class RandomCellTypeRule(CellTypeRuleBase):
    """Rule that assigns cell types randomly using a Dirichlet distribution.

    Each cell receives independent random probabilities drawn from a
    symmetric Dirichlet distribution, creating heterogeneous cell type
    mixtures.

    Parameters
    ----------
    n_cell_types : int, optional
        Number of cell types. Default is 3.

    Examples
    --------
    >>> rule = RandomCellTypeRule(n_cell_types=5)
    >>> centroids = np.random.rand(100, 2) * 1000
    >>> probs = rule.apply(centroids)
    >>> probs.shape
    (100, 5)
    """

    def __init__(self, n_cell_types=3):
        super().__init__(n_cell_types)

    def apply(self, cell_centroids, current_probs=None):
        """Assign random cell type probabilities using Dirichlet distribution.

        If current_probs is provided, adds Gaussian noise to existing
        probabilities and renormalizes.
        """
        if current_probs is None:
            probs = np.random.dirichlet(np.ones(self.n_cell_types), len(cell_centroids))
        else:
            probs = current_probs + np.random.normal(0, 0.1, current_probs.shape)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs


class MixOfNCellTypesRule(CellTypeRuleBase):
    """Rule that assigns a fixed mixture of N specific cell types.

    All cells in the element receive the same probability distribution
    over a subset of cell types, useful for regions with known cell
    type composition.

    Parameters
    ----------
    n_cell_types : int, optional
        Total number of cell types in the system. Default is 3.
    N : int, optional
        Number of cell types to include in the mixture. Randomly selected
        if list_N is not provided.
    list_N : array-like, optional
        Specific indices of cell types to include. Overrides N if provided.
    proportions : array-like, optional
        Proportions for each selected cell type. If None, drawn from
        Dirichlet distribution.

    Raises
    ------
    ValueError
        If N and list_N are inconsistent.

    Examples
    --------
    >>> rule = MixOfNCellTypesRule(n_cell_types=10, list_N=[0, 3, 7], proportions=[0.5, 0.3, 0.2])
    """
    def __init__(self, n_cell_types=3, N=None, list_N=None, proportions=None):
        super().__init__(n_cell_types)

        if list_N is None and N is not None:
            self.N = N
            self.list_N = np.random.choice(np.arange(N), replace=False)
        elif list_N is not None and (N is None or N == len(list_N)):
            self.list_N = np.array(list_N)
        else:
            raise ValueError("N and list_N must be consistent")

        if proportions is not None:
            self.proportions = np.array(proportions)
        else:
            self.proportions = np.random.dirichlet(np.ones(len(self.list_N)), 1)[0]
        self.probs = np.zeros(self.n_cell_types)
        self.probs[self.list_N] = self.proportions

    def apply(self, cell_centroids, current_probs=None):
        """Apply fixed cell type proportions to all cells.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell positions, shape (n_cells, 2).
        current_probs : np.ndarray, optional
            If provided, adds to existing probabilities and renormalizes.

        Returns
        -------
        np.ndarray
            Probability matrix with fixed proportions for all cells.
        """
        probs = self.probs * np.ones(cell_centroids.shape[0])[:, None]
        if current_probs is None:
            return probs
        else:
            probs = current_probs + probs
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            return probs
