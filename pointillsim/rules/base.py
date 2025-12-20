"""Base classes for cell type assignment rules."""

import numpy as np


class CellTypeRuleBase:
    """Abstract base class for cell type assignment rules.

    Rules define how cell type probabilities are assigned to cells within
    histological elements. Multiple rules can be composed sequentially,
    with each rule potentially modifying the probabilities from previous rules.

    Parameters
    ----------
    n_cell_types : int, optional
        Number of possible cell types. Default is 3.

    Attributes
    ----------
    n_cell_types : int
        Number of cell types this rule operates on.
    is_adapted : bool
        Whether the rule has been adapted to a specific element.

    Notes
    -----
    Subclasses must implement the `apply` method. The `adapt_rule_to_element`
    method can be overridden to perform element-specific initialization.

    See Also
    --------
    RandomCellTypeRule : Random Dirichlet-distributed assignments.
    SingleTypeRule : Assigns all cells to one type.
    ProbabilityNodeFieldRule : Spatial gradient-based assignments.
    """

    def __init__(self, n_cell_types=3):
        self.n_cell_types = n_cell_types
        self.is_adapted = False

    def apply(self, cell_centroids, current_probs=None, **kwargs):
        """Apply the rule to assign cell type probabilities.

        Parameters
        ----------
        cell_centroids : np.ndarray
            Cell centroid coordinates, shape (n_cells, 2).
        current_probs : np.ndarray, optional
            Existing probability matrix from previous rules, shape (n_cells, n_cell_types).
            If None, this is the first rule in the chain.
        **kwargs : dict
            Additional rule-specific parameters.

        Returns
        -------
        np.ndarray
            Updated probability matrix, shape (n_cells, n_cell_types).
            Each row sums to 1.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def adapt_rule_to_element(self, element):
        """Adapt the rule to a specific histological element.

        Called before apply() to perform element-specific initialization,
        such as setting reference points based on element geometry.

        Parameters
        ----------
        element : HistologicalElement
            The element this rule will be applied to.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        self.is_adapted = True
        return self


class DummyRule(CellTypeRuleBase):
    """Rule that returns pre-specified probabilities.

    Used internally for composing rules, particularly in FrameWideUpdater
    to carry forward existing probabilities.

    Parameters
    ----------
    n_cell_types : int
        Number of cell types.
    probs_to_copy : np.ndarray, optional
        Probability matrix to return.
    """
    def __init__(self, n_cell_types, probs_to_copy=None):
        super().__init__(n_cell_types)
        self.probs_to_copy = probs_to_copy

    def apply(self, cell_centroids, current_probs=None, **kwargs):
        """Return stored probabilities, ignoring inputs."""
        return self.probs_to_copy

    def adapt_rule_to_element(self, element):
        """No adaptation needed."""
        return self
