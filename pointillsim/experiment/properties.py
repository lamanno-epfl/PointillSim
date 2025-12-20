"""Cell morphological properties."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class CellTypesProperties:
    """Morphological and visual properties for each cell type.

    Defines how cells of each type appear: their size, shape (anisotropy),
    RNA content, and display color. Properties can be specified per-type
    or as a single value applied to all types.

    Parameters
    ----------
    n_cell_types : int
        Number of cell types.
    colordict : dict or matplotlib.colors.Colormap, optional
        Color mapping {type_index: RGBA tuple}. If Colormap, samples from it.
    gene_colordict : dict, optional
        Color mapping for genes (for visualization).
    name_dict : dict, optional
        Human-readable names for cell types.
    sizes : float or array-like, optional
        Mean cell radius per type. Default is 13.
    size_variation : float or array-like, optional
        Standard deviation of cell radius. Default is 1.
    anisotropy : float or array-like, optional
        Cell elongation factor (1.0 = circular, <1 = elongated). Default 0.85.
    anisotropy_variation : float or array-like, optional
        Std of anisotropy. Default 0.05.
    relative_rna_concentration : float or array-like, optional
        Relative RNA content multiplier per type. Default 1.0.
    rna_concentration_variation : float or array-like, optional
        Std of RNA concentration. Default 0.05.

    Examples
    --------
    >>> props = CellTypesProperties(
    ...     n_cell_types=5,
    ...     sizes=[10, 12, 8, 15, 11],
    ...     anisotropy=0.8
    ... )
    >>> props.apply(fov)  # Adds morphological attributes to fov
    """

    def __init__(
        self,
        n_cell_types,
        colordict=None,
        gene_colordict=None,
        name_dict=None,
        sizes=13,
        size_variation=1,
        anisotropy=0.85,
        anisotropy_variation=0.05,
        relative_rna_concentration=1.0,
        rna_concentration_variation=0.05,
    ):
        self.n_cell_types = n_cell_types
        if colordict is None:
            self.colordict = {i: plt.cm.tab20(i) for i in range(n_cell_types)}
        elif isinstance(colordict, matplotlib.colors.Colormap):
            self.colordict = {
                i: colordict(float(i) / n_cell_types) for i in range(n_cell_types)
            }
        elif isinstance(colordict, dict):
            self.colordict = colordict
        else:
            raise ValueError("colordict must be a dictionary or a colormap")

        if gene_colordict is None:
            self.gene_colordict = {i: plt.cm.tab20(i / 200.0) for i in range(200)}
        elif isinstance(gene_colordict, dict):
            self.gene_colordict = gene_colordict
        else:
            raise ValueError("gene_colordict must be a dictionary or a colormap")

        if name_dict is None:
            self.name_dict = {i: f"Type {i+1}" for i in range(self.n_cell_types)}

        if isinstance(sizes, (int, float)):
            self.sizes = np.ones(n_cell_types) * sizes
        else:
            self.sizes = sizes

        if isinstance(size_variation, (int, float)):
            self.size_variation = np.ones(n_cell_types) * size_variation
        else:
            self.size_variation = size_variation

        if isinstance(anisotropy, (int, float)):
            self.anisotropy = np.ones(n_cell_types) * anisotropy
        else:
            self.anisotropy = anisotropy

        if isinstance(anisotropy_variation, (int, float)):
            self.anisotropy_variation = np.ones(n_cell_types) * anisotropy_variation
        else:
            self.anisotropy_variation = anisotropy_variation

        if isinstance(relative_rna_concentration, (int, float)):
            self.relative_rna_concentration = (
                np.ones(n_cell_types) * relative_rna_concentration
            )
        else:
            self.relative_rna_concentration = relative_rna_concentration

        if isinstance(rna_concentration_variation, (int, float)):
            self.rna_concentration_variation = (
                np.ones(n_cell_types) * rna_concentration_variation
            )
        else:
            self.rna_concentration_variation = rna_concentration_variation

    def apply(self, fov):
        """Apply morphological properties to cells in a FOV.

        Samples cell-specific sizes, shapes, and RNA concentrations based on
        each cell's type, and assigns display colors.

        Parameters
        ----------
        fov : FOV
            Field of view to augment with morphological properties.

        Notes
        -----
        After calling this method, the FOV will have these new attributes:
        - cell_minor_axis, cell_major_axis: ellipse dimensions
        - cell_rotation: random orientation angles
        - cell_rna_concentration: per-cell RNA levels
        - cell_colors: display colors based on cell type
        """
        scell_sizes = fov.class_instance_one_hot @ self.sizes
        scell_anisotropy = fov.class_instance_one_hot @ self.anisotropy
        scell_size_variation = fov.class_instance_one_hot @ self.size_variation
        scell_relative_rna_concentration = (
            fov.class_instance_one_hot @ self.relative_rna_concentration
        )
        scell_rna_concentration_variation = (
            fov.class_instance_one_hot @ self.rna_concentration_variation
        )
        scell_anisotropy_variation = (
            fov.class_instance_one_hot @ self.anisotropy_variation
        )
        scell_rotation = np.random.uniform(
            0, 2 * np.pi, fov.class_instance_one_hot.shape[0]
        )

        scell_radius = np.clip(
            np.random.normal(scell_sizes, scell_size_variation),
            scell_sizes * 0.2,
            scell_sizes * 3,
        )
        scell_anisotropy_realized = np.clip(
            np.random.normal(scell_anisotropy, scell_anisotropy_variation),
            0.2,
            2 * scell_anisotropy,
        )
        scell_concentration_realized = np.random.normal(
            scell_relative_rna_concentration, scell_rna_concentration_variation
        )

        fov.cell_minor_axis = scell_radius * scell_anisotropy_realized
        fov.cell_major_axis = scell_radius * (2 - scell_anisotropy_realized)
        fov.cell_rotation = scell_rotation
        fov.cell_rna_concentration = scell_concentration_realized
        fov.cell_colors = [self.colordict[i] for i in fov.class_instance]
