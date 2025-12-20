"""Field of View classes."""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class FOV:
    """Container for a realized Field of View with cells and their properties.

    Holds cell positions, type probabilities, and sampled type assignments.
    Can be extended with morphological properties via CellTypesProperties.

    Parameters
    ----------
    cell_centroids : np.ndarray
        Cell centroid coordinates, shape (n_cells, 2).
    cell_probabilities : np.ndarray
        Cell type probability matrix, shape (n_cells, n_cell_types).

    Attributes
    ----------
    cell_centroids : np.ndarray
        Cell positions.
    cell_probabilities : np.ndarray
        Soft cell type assignments (probabilities).
    class_instance_one_hot : np.ndarray
        Hard cell type assignments after realization(), one-hot encoded.
    cell_minor_axis : np.ndarray
        Cell minor axis lengths (after CellTypesProperties.apply).
    cell_major_axis : np.ndarray
        Cell major axis lengths (after CellTypesProperties.apply).
    cell_rotation : np.ndarray
        Cell rotation angles (after CellTypesProperties.apply).
    cell_rna_concentration : np.ndarray
        Relative RNA concentration per cell (after CellTypesProperties.apply).
    cell_colors : list
        RGBA colors per cell (after CellTypesProperties.apply).
    """

    def __init__(
        self,
        cell_centroids: NDArray[np.floating],
        cell_probabilities: NDArray[np.floating],
    ) -> None:
        # Input validation
        cell_centroids = np.asarray(cell_centroids)
        cell_probabilities = np.asarray(cell_probabilities)

        if cell_centroids.ndim != 2 or cell_centroids.shape[1] != 2:
            raise ValueError(
                f"cell_centroids must have shape (n_cells, 2), got {cell_centroids.shape}"
            )
        if cell_probabilities.ndim != 2:
            raise ValueError(
                f"cell_probabilities must be 2D, got shape {cell_probabilities.shape}"
            )
        if cell_centroids.shape[0] != cell_probabilities.shape[0]:
            raise ValueError(
                f"Number of cells must match: centroids has {cell_centroids.shape[0]}, "
                f"probabilities has {cell_probabilities.shape[0]}"
            )

        self.cell_centroids = cell_centroids
        self.cell_probabilities = cell_probabilities
        self.rng = np.random.default_rng()
        self.class_instance_one_hot: Optional[NDArray[np.integer]] = None

    def realization(self) -> None:
        """Sample hard cell type assignments from probabilities.

        Draws one cell type per cell from the multinomial distribution
        defined by cell_probabilities.
        """
        self.class_instance_one_hot = self.rng.multinomial(
            n=1, pvals=self.cell_probabilities
        )

    @property
    def class_instance(self) -> NDArray[np.integer]:
        """np.ndarray: Integer cell type indices from one-hot encoding."""
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def n_cells(self) -> int:
        """int: Number of cells in the FOV."""
        return self.cell_centroids.shape[0]

    @property
    def n_cell_types(self) -> int:
        """int: Number of cell types."""
        return self.cell_probabilities.shape[1]

    def add_noise(
        self,
        position_std: float = 1.0,
        probability_std: float = 0.0,
        seed: Optional[int] = None,
    ) -> FOV:
        """Add Gaussian noise to cell positions and/or probabilities.

        Parameters
        ----------
        position_std : float, optional
            Standard deviation of position noise in pixels. Default 1.0.
        probability_std : float, optional
            Standard deviation of probability noise (before renormalization).
            Default 0.0 (no probability noise).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        FOV
            Self for method chaining.
        """
        rng = np.random.default_rng(seed)

        if position_std > 0:
            self.cell_centroids = self.cell_centroids + rng.normal(
                0, position_std, self.cell_centroids.shape
            )

        if probability_std > 0:
            noise = rng.normal(0, probability_std, self.cell_probabilities.shape)
            noisy_probs = self.cell_probabilities + noise
            noisy_probs = np.maximum(noisy_probs, 1e-12)
            self.cell_probabilities = noisy_probs / noisy_probs.sum(
                axis=1, keepdims=True
            )

        return self

    def subsample(
        self,
        fraction: Optional[float] = None,
        n_cells: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> FOV:
        """Randomly subsample cells from the FOV.

        Parameters
        ----------
        fraction : float, optional
            Fraction of cells to keep (0-1). Mutually exclusive with n_cells.
        n_cells : int, optional
            Exact number of cells to keep. Mutually exclusive with fraction.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        FOV
            Self with subsampled cells (for method chaining).

        Raises
        ------
        ValueError
            If neither or both fraction and n_cells are specified.
        """
        if (fraction is None) == (n_cells is None):
            raise ValueError("Specify exactly one of 'fraction' or 'n_cells'")

        rng = np.random.default_rng(seed)
        total = self.cell_centroids.shape[0]

        if fraction is not None:
            n_keep = int(total * fraction)
        else:
            n_keep = min(n_cells, total)

        if n_keep >= total:
            return self

        indices = rng.choice(total, size=n_keep, replace=False)
        indices = np.sort(indices)

        self.cell_centroids = self.cell_centroids[indices]
        self.cell_probabilities = self.cell_probabilities[indices]

        if self.class_instance_one_hot is not None:
            self.class_instance_one_hot = self.class_instance_one_hot[indices]

        # Also subsample morphological properties if they exist
        for attr in [
            "cell_minor_axis",
            "cell_major_axis",
            "cell_rotation",
            "cell_rna_concentration",
        ]:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr)[indices])

        if hasattr(self, "cell_colors"):
            self.cell_colors = [self.cell_colors[i] for i in indices]

        return self

    def make_pandas_df(self) -> pd.DataFrame:
        """Export FOV data as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for position, class ID, one-hot encoding,
            probabilities, and morphological properties (if applied).
        """
        data = (
            [*self.cell_centroids.T]
            + [*self.class_instance[:, None].T]
            + [*self.class_instance_one_hot.T]
            + [*self.cell_probabilities.T]
        )
        col_names = (
            ["X", "Y", "Class ID"]
            + [f"Class {i}" for i in range(self.class_instance_one_hot.shape[1])]
            + [f"ProbClass{i}" for i in range(self.cell_probabilities.shape[1])]
        )

        try:
            data.append(self.cell_minor_axis)
            col_names.append("Minor Axis")
            data.append(self.cell_major_axis)
            col_names.append("Major Axis")
            data.append(self.cell_rotation)
            col_names.append("Rotation")
            data.append(self.cell_rna_concentration)
            col_names.append("RNA Concentration")
            data.append(np.array([str(i) for i in self.cell_colors]))
            col_names.append("Color_string")
        except AttributeError:
            pass

        series_dict = {
            col_names[i]: pd.Series(data[i], dtype=data[i].dtype.str)
            for i in range(len(data))
        }
        return pd.DataFrame(series_dict)


class FOVDistribution:
    """Stochastic generator for Fields of View with multiple histological elements.

    Defines a probabilistic model for generating FOVs by combining a background
    element with randomly placed foreground elements. Elements can overlap,
    with foreground elements taking precedence and removing background cells.

    Parameters
    ----------
    frame_size : int, optional
        Size of the FOV in pixels. Default is 5000.
    background_element : callable, optional
        Factory function returning a HistologicalElement for the background.
        Should return a FrameWideElement or similar.
    other_elements : list of callable, optional
        Factory functions for foreground elements.
    elements_frequency : list of float, optional
        Probability of placing each foreground element type (0 to 1).
    attempts_at_elements : int or list, optional
        Number of placement attempts per element type. Higher values create
        more instances of that element type. Default is 1.

    Examples
    --------
    >>> from pointillsim.elements import FrameWideElement, HistologicalElement
    >>> from pointillsim.rules import RandomCellTypeRule, SingleTypeRule
    >>> bg = lambda: FrameWideElement(frame_size=1000, rules=RandomCellTypeRule(5))
    >>> fg = lambda: HistologicalElement(scale=100, rules=SingleTypeRule(5, 0))
    >>> fovd = FOVDistribution(
    ...     frame_size=1000,
    ...     background_element=bg,
    ...     other_elements=[fg],
    ...     elements_frequency=[0.5]
    ... )
    >>> fov = fovd.generate_fov()
    """

    def __init__(
        self,
        frame_size: int = 5000,
        background_element: Optional[Callable] = None,
        other_elements: Optional[List[Callable]] = None,
        elements_frequency: Optional[List[float]] = None,
        attempts_at_elements: Union[int, List[int]] = 1,
    ) -> None:
        self.frame_size = frame_size
        self.background_element = background_element
        self.other_elements = other_elements if other_elements is not None else []
        self.elements_frequency = (
            elements_frequency if elements_frequency is not None else []
        )
        self.attempts_at_elements = attempts_at_elements

    def generate_fov(self) -> FOV:
        """Generate a single FOV realization.

        Creates a background element, then attempts to place foreground elements
        according to their frequencies. Overlapping regions are handled by
        removing background cells that fall within foreground elements.

        Returns
        -------
        FOV
            A realized FOV with cell centroids and type probabilities.
        """
        bg = self.background_element().generate()
        realized_elements = [bg]

        for n, element in enumerate(self.other_elements):
            if isinstance(self.attempts_at_elements, list) or isinstance(
                self.attempts_at_elements, tuple
            ):
                attempts = int(self.attempts_at_elements[n])
            else:
                attempts = int(self.attempts_at_elements)
            for _ in range(attempts):
                if np.random.rand() <= self.elements_frequency[n]:
                    el = element().generate(context_knowledge=realized_elements)
                    el.remove_cells(bg.is_outside(el.cell_centroids))
                    for k in realized_elements:
                        k.remove_cells(el.is_inside(k.cell_centroids))
                    realized_elements.append(el)

        list_of_cents = [i.cell_centroids for i in realized_elements]
        list_of_probs = [i.cell_probabilities for i in realized_elements]

        n_types = 0
        for p in list_of_probs:
            n_types = max(n_types, p.shape[1])

        final_cents = []
        final_probs = []
        for C, P in zip(list_of_cents, list_of_probs):
            if C.shape[0] == 0:
                continue

            if P.shape[1] == 0:
                final_probs.append(np.zeros((C.shape[0], n_types)))
            else:
                final_probs.append(P)

            final_cents.append(C)

        if not final_cents:
            cell_centroids = np.empty((0, 2))
            cell_probabilities = np.empty((0, n_types))
        else:
            cell_centroids = np.row_stack(final_cents)
            cell_probabilities = np.row_stack(final_probs)

        fov = FOV(cell_centroids, cell_probabilities)
        fov.realization()
        return fov
