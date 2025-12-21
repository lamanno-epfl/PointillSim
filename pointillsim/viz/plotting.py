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


def plot_tissue_slice(
    tissue_slice,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 10),
    colormap: str = "turbo",
    point_size: float = 2.0,
    alpha: float = 0.7,
    show_colorbar: bool = True,
    show_regions: bool = True,
    region_alpha: float = 0.1,
    title: Optional[str] = None,
    color_by: str = "class",
    show_grid: bool = False,
    grid_size: Optional[int] = None,
) -> Tuple[Figure, Axes]:
    """Plot a TissueSlice with all cells colored by type.

    Visualizes the global tissue composition across all regions,
    with optional region boundary overlays and FOV grid lines.

    Parameters
    ----------
    tissue_slice : TissueSlice
        The tissue slice to visualize (after generate_global()).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple, optional
        Figure size if creating new figure. Default (10, 10).
    colormap : str, optional
        Matplotlib colormap name. Default "turbo".
    point_size : float, optional
        Size of cell markers. Default 2.0 (smaller for large slices).
    alpha : float, optional
        Transparency of cell markers. Default 0.7.
    show_colorbar : bool, optional
        Whether to show colorbar. Default True.
    show_regions : bool, optional
        Whether to show region boundaries. Default True.
    region_alpha : float, optional
        Transparency of region fill. Default 0.1.
    title : str, optional
        Plot title. Default None.
    color_by : str, optional
        How to color cells: "class" for sampled types, "probability" for
        max probability class. Default "class".
    show_grid : bool, optional
        Whether to show FOV tile grid lines. Default False.
    grid_size : int, optional
        FOV size for grid lines. Required if show_grid is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> from pointillsim import TissueSlice
    >>> from pointillsim.viz import plot_tissue_slice
    >>> slice.generate_global().sample_labels()
    >>> fig, ax = plot_tissue_slice(slice, show_regions=True)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Check if slice has been generated
    if tissue_slice._cells_xy is None or len(tissue_slice._cells_xy) == 0:
        ax.text(
            0.5, 0.5, "No cells generated.\nCall generate_global() first.",
            ha='center', va='center', transform=ax.transAxes, fontsize=12
        )
        ax.set_xlim(0, tissue_slice.frame_size)
        ax.set_ylim(0, tissue_slice.frame_size)
        return fig, ax

    x = tissue_slice._cells_xy[:, 0]
    y = tissue_slice._cells_xy[:, 1]

    # Determine colors
    if color_by == "class":
        if tissue_slice._class_onehot is None:
            # Sample labels if not done
            tissue_slice.sample_labels()
        classes = np.argmax(tissue_slice._class_onehot, axis=1)
        n_types = tissue_slice._probs.shape[1] if tissue_slice._probs.shape[1] > 0 else 1
        scatter = ax.scatter(
            x, y, c=classes, cmap=colormap, s=point_size, alpha=alpha,
            vmin=0, vmax=max(n_types - 1, 1)
        )
    elif color_by == "probability":
        max_prob_class = np.argmax(tissue_slice._probs, axis=1)
        n_types = tissue_slice._probs.shape[1] if tissue_slice._probs.shape[1] > 0 else 1
        scatter = ax.scatter(
            x, y, c=max_prob_class, cmap=colormap, s=point_size, alpha=alpha,
            vmin=0, vmax=max(n_types - 1, 1)
        )
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Cell Type")

    # Draw region boundaries
    if show_regions:
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
        import matplotlib.colors as mcolors

        patches = []
        colors = []
        for i, region in enumerate(tissue_slice.regions):
            if region.polygon is not None and not region.polygon.is_empty:
                # Get polygon exterior coords
                if hasattr(region.polygon, 'exterior'):
                    coords = np.array(region.polygon.exterior.coords)
                    patch = MplPolygon(coords, closed=True)
                    patches.append(patch)
                    colors.append(plt.cm.Set3(i % 12))

        if patches:
            # Add patches with transparent fill
            for patch, color in zip(patches, colors):
                patch.set_facecolor((*color[:3], region_alpha))
                patch.set_edgecolor(color[:3])
                patch.set_linewidth(1.5)
                ax.add_patch(patch)

    # Draw grid lines
    if show_grid and grid_size is not None:
        for x_line in range(0, tissue_slice.frame_size + 1, grid_size):
            ax.axvline(x=x_line, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        for y_line in range(0, tissue_slice.frame_size + 1, grid_size):
            ax.axhline(y=y_line, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    ax.set_xlim(0, tissue_slice.frame_size)
    ax.set_ylim(0, tissue_slice.frame_size)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)
    else:
        n_cells = len(tissue_slice._cells_xy)
        n_regions = len(tissue_slice.regions)
        ax.set_title(f"TissueSlice: {n_cells:,} cells, {n_regions} regions")

    return fig, ax


def plot_probability_field(
    fov_or_element,
    cell_type: int = 0,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    colormap: str = "viridis",
    resolution: int = 100,
    show_colorbar: bool = True,
    show_cells: bool = True,
    cell_color: str = "white",
    cell_size: float = 5.0,
    cell_alpha: float = 0.5,
    title: Optional[str] = None,
    interpolation: str = "bilinear",
) -> Tuple[Figure, Axes]:
    """Plot cell type probability as a continuous spatial field.

    Interpolates cell type probabilities to create a smooth
    probability density visualization across the FOV.

    Parameters
    ----------
    fov_or_element : FOV or HistologicalElement
        Object with cell_centroids and cell_probabilities attributes.
    cell_type : int, optional
        Index of cell type to visualize. Default 0.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple, optional
        Figure size if creating new figure. Default (8, 8).
    colormap : str, optional
        Matplotlib colormap name. Default "viridis".
    resolution : int, optional
        Number of grid points per axis for interpolation. Default 100.
    show_colorbar : bool, optional
        Whether to show colorbar. Default True.
    show_cells : bool, optional
        Whether to overlay cell positions. Default True.
    cell_color : str, optional
        Color for cell markers. Default "white".
    cell_size : float, optional
        Size of cell markers. Default 5.0.
    cell_alpha : float, optional
        Transparency of cell markers. Default 0.5.
    title : str, optional
        Plot title. Default None.
    interpolation : str, optional
        Matplotlib imshow interpolation. Default "bilinear".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> from pointillsim.viz import plot_probability_field
    >>> # Plot probability of cell type 0
    >>> fig, ax = plot_probability_field(fov, cell_type=0)
    >>> plt.show()
    """
    from scipy.interpolate import griddata

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get data from FOV or element
    centroids = fov_or_element.cell_centroids
    probs = fov_or_element.cell_probabilities

    if len(centroids) == 0:
        ax.text(
            0.5, 0.5, "No cells to plot.",
            ha='center', va='center', transform=ax.transAxes, fontsize=12
        )
        return fig, ax

    # Extract probability for specific cell type
    if cell_type >= probs.shape[1]:
        raise ValueError(
            f"cell_type {cell_type} out of range. "
            f"FOV has {probs.shape[1]} cell types."
        )

    prob_values = probs[:, cell_type]

    # Determine frame bounds
    if hasattr(fov_or_element, 'frame_size'):
        frame_size = fov_or_element.frame_size
        x_min, y_min = 0, 0
        x_max, y_max = frame_size, frame_size
    else:
        x_min, y_min = centroids.min(axis=0) - 10
        x_max, y_max = centroids.max(axis=0) + 10

    # Create interpolation grid
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate probabilities
    Zi = griddata(
        centroids, prob_values, (Xi, Yi),
        method='linear', fill_value=0.0
    )

    # Plot the field
    extent = [x_min, x_max, y_min, y_max]
    im = ax.imshow(
        Zi, extent=extent, origin='lower',
        cmap=colormap, vmin=0, vmax=1,
        interpolation=interpolation, aspect='equal'
    )

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"P(Cell Type {cell_type})")

    # Overlay cells
    if show_cells:
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c=cell_color, s=cell_size, alpha=cell_alpha,
            edgecolors='none'
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Probability Field: Cell Type {cell_type}")

    return fig, ax


def plot_element_polygon(
    element,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    fill_color: str = "lightblue",
    edge_color: str = "blue",
    fill_alpha: float = 0.3,
    edge_width: float = 2.0,
    show_cells: bool = True,
    cell_colormap: str = "turbo",
    cell_size: float = 10.0,
    cell_alpha: float = 0.8,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot a histological element with its polygon boundary.

    Visualizes the element's bounding polygon along with cell
    positions colored by type.

    Parameters
    ----------
    element : HistologicalElement or similar
        Element with polygon and cell_centroids attributes.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple, optional
        Figure size if creating new figure. Default (8, 8).
    fill_color : str, optional
        Polygon fill color. Default "lightblue".
    edge_color : str, optional
        Polygon edge color. Default "blue".
    fill_alpha : float, optional
        Polygon fill transparency. Default 0.3.
    edge_width : float, optional
        Polygon edge width. Default 2.0.
    show_cells : bool, optional
        Whether to show cell positions. Default True.
    cell_colormap : str, optional
        Colormap for cells. Default "turbo".
    cell_size : float, optional
        Size of cell markers. Default 10.0.
    cell_alpha : float, optional
        Transparency of cell markers. Default 0.8.
    title : str, optional
        Plot title. Default None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    from matplotlib.patches import Polygon as MplPolygon

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot polygon(s)
    if element.polygon is not None and not element.polygon.is_empty:
        if hasattr(element.polygon, 'exterior'):
            # Single polygon
            coords = np.array(element.polygon.exterior.coords)
            patch = MplPolygon(
                coords, closed=True,
                facecolor=fill_color, alpha=fill_alpha,
                edgecolor=edge_color, linewidth=edge_width
            )
            ax.add_patch(patch)
        elif hasattr(element.polygon, 'geoms'):
            # MultiPolygon
            for geom in element.polygon.geoms:
                if hasattr(geom, 'exterior'):
                    coords = np.array(geom.exterior.coords)
                    patch = MplPolygon(
                        coords, closed=True,
                        facecolor=fill_color, alpha=fill_alpha,
                        edgecolor=edge_color, linewidth=edge_width
                    )
                    ax.add_patch(patch)

    # Plot hole/lumen if present
    if hasattr(element, 'hole') and element.hole is not None:
        if hasattr(element.hole, 'exterior'):
            coords = np.array(element.hole.exterior.coords)
            patch = MplPolygon(
                coords, closed=True,
                facecolor='white', alpha=0.9,
                edgecolor='gray', linewidth=1.0
            )
            ax.add_patch(patch)

    if hasattr(element, 'lumen') and element.lumen is not None:
        if hasattr(element.lumen, 'exterior'):
            coords = np.array(element.lumen.exterior.coords)
            patch = MplPolygon(
                coords, closed=True,
                facecolor='white', alpha=0.9,
                edgecolor='gray', linewidth=1.0
            )
            ax.add_patch(patch)
        elif hasattr(element.lumen, 'geoms'):
            for geom in element.lumen.geoms:
                if hasattr(geom, 'exterior'):
                    coords = np.array(geom.exterior.coords)
                    patch = MplPolygon(
                        coords, closed=True,
                        facecolor='white', alpha=0.9,
                        edgecolor='gray', linewidth=1.0
                    )
                    ax.add_patch(patch)

    # Plot cells
    if show_cells and element.cell_centroids is not None and len(element.cell_centroids) > 0:
        x = element.cell_centroids[:, 0]
        y = element.cell_centroids[:, 1]

        if element.cell_probabilities is not None:
            classes = np.argmax(element.cell_probabilities, axis=1)
            n_types = element.cell_probabilities.shape[1]
            ax.scatter(
                x, y, c=classes, cmap=cell_colormap,
                s=cell_size, alpha=cell_alpha,
                vmin=0, vmax=max(n_types - 1, 1)
            )
        else:
            ax.scatter(x, y, c='blue', s=cell_size, alpha=cell_alpha)

    # Set limits
    bounds = element.bounding_box
    margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
    ax.set_aspect('equal')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if title:
        ax.set_title(title)
    else:
        n_cells = len(element.cell_centroids) if element.cell_centroids is not None else 0
        ax.set_title(f"{element.__class__.__name__}: {n_cells} cells")

    return fig, ax
