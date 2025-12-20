"""Plotting utilities for PointillSim visualizations."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray


def plot_fov(
    fov,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    colormap: str = "turbo",
    point_size: float = 10.0,
    alpha: float = 0.8,
    show_colorbar: bool = True,
    title: Optional[str] = None,
    color_by: str = "class",
) -> Tuple[Figure, Axes]:
    """Plot a Field of View with cells colored by type.

    Parameters
    ----------
    fov : FOV
        The field of view to visualize.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple, optional
        Figure size if creating new figure. Default (8, 8).
    colormap : str, optional
        Matplotlib colormap name. Default "turbo".
    point_size : float, optional
        Size of cell markers. Default 10.0.
    alpha : float, optional
        Transparency of markers. Default 0.8.
    show_colorbar : bool, optional
        Whether to show colorbar. Default True.
    title : str, optional
        Plot title. Default None.
    color_by : str, optional
        How to color cells: "class" for sampled types, "probability" for
        max probability class, "entropy" for probability entropy.
        Default "class".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> from pointillsim import FOVDistribution
    >>> from pointillsim.viz import plot_fov
    >>> fov = fovd.generate_fov()
    >>> fig, ax = plot_fov(fov)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    x = fov.cell_centroids[:, 0]
    y = fov.cell_centroids[:, 1]

    cmap = plt.get_cmap(colormap)

    if color_by == "class":
        if fov.class_instance_one_hot is None:
            fov.realization()
        classes = fov.class_instance
        n_types = fov.cell_probabilities.shape[1]
        colors = [cmap(c / max(n_types - 1, 1)) for c in classes]
        scatter = ax.scatter(x, y, c=classes, cmap=colormap, s=point_size, alpha=alpha)
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Cell Type")
    elif color_by == "probability":
        max_prob_class = np.argmax(fov.cell_probabilities, axis=1)
        n_types = fov.cell_probabilities.shape[1]
        scatter = ax.scatter(
            x, y, c=max_prob_class, cmap=colormap, s=point_size, alpha=alpha
        )
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Max Probability Class")
    elif color_by == "entropy":
        probs = fov.cell_probabilities
        probs = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(probs * np.log2(probs), axis=1)
        scatter = ax.scatter(x, y, c=entropy, cmap="viridis", s=point_size, alpha=alpha)
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Entropy (bits)")
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"FOV: {fov.n_cells} cells, {fov.n_cell_types} types")

    return fig, ax


def plot_expression_matrix(
    tissue,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 8),
    colormap: str = "viridis",
    show_colorbar: bool = True,
    title: Optional[str] = None,
    log_scale: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot gene expression matrix as a heatmap.

    Parameters
    ----------
    tissue : TissueCellTypes
        The tissue object with expression profiles.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple, optional
        Figure size if creating new figure. Default (10, 8).
    colormap : str, optional
        Matplotlib colormap name. Default "viridis".
    show_colorbar : bool, optional
        Whether to show colorbar. Default True.
    title : str, optional
        Plot title. Default None.
    log_scale : bool, optional
        If True, plot log(expression + 1). Default False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> from pointillsim import TissueCellTypes
    >>> from pointillsim.viz import plot_expression_matrix
    >>> tissue = TissueCellTypes()
    >>> tissue.generate_types_and_markers(n_genes=50, n_cell_types=10)
    >>> fig, ax = plot_expression_matrix(tissue)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    data = tissue.gene_expression_by_type
    if log_scale:
        data = np.log1p(data)

    im = ax.imshow(data, aspect="auto", cmap=colormap)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        label = "log(Expression + 1)" if log_scale else "Expression Level"
        cbar.set_label(label)

    # Set ticks if not too many
    if tissue.n_cell_types <= 20:
        ax.set_xticks(range(tissue.n_cell_types))
        ax.set_xticklabels(tissue.cell_type_names, rotation=45, ha="right")
    else:
        ax.set_xlabel("Cell Type")

    if tissue.n_genes <= 50:
        ax.set_yticks(range(tissue.n_genes))
        ax.set_yticklabels(tissue.gene_names)
    else:
        ax.set_ylabel("Gene")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Expression Matrix: {tissue.n_genes} genes x {tissue.n_cell_types} types")

    plt.tight_layout()
    return fig, ax
