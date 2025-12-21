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

    def to_anndata(
        self,
        gene_names: Optional[List[str]] = None,
        expression_matrix: Optional[NDArray[np.floating]] = None,
        include_spatial: bool = True,
    ):
        """Export FOV data as an AnnData object.

        Creates an AnnData object suitable for analysis with scanpy and
        other single-cell analysis tools. Spatial coordinates are stored
        in obsm['spatial'].

        Parameters
        ----------
        gene_names : list of str, optional
            Gene names for the expression matrix columns. If None and
            expression_matrix is provided, uses 'Gene_0', 'Gene_1', etc.
        expression_matrix : np.ndarray, optional
            Gene expression matrix, shape (n_cells, n_genes). If None,
            uses cell_probabilities as a placeholder.
        include_spatial : bool, optional
            If True, stores spatial coordinates in obsm['spatial'].
            Default True.

        Returns
        -------
        anndata.AnnData
            AnnData object with:
            - X: expression matrix or cell probabilities
            - obs: cell metadata (class_id, morphology if available)
            - var: gene metadata
            - obsm['spatial']: spatial coordinates (if include_spatial)
            - uns['cell_type_probabilities']: soft assignments

        Raises
        ------
        ImportError
            If anndata is not installed.

        Examples
        --------
        >>> fov = fov_distribution.generate_fov()
        >>> adata = fov.to_anndata()
        >>> import scanpy as sc
        >>> sc.pl.embedding(adata, basis='spatial', color='class_id')
        """
        try:
            import anndata
        except ImportError:
            raise ImportError(
                "anndata is required for to_anndata(). "
                "Install with: pip install pointillsim[anndata]"
            )

        from scipy import sparse

        n_cells = self.n_cells

        # Prepare expression matrix
        if expression_matrix is not None:
            X = np.asarray(expression_matrix)
            if X.shape[0] != n_cells:
                raise ValueError(
                    f"expression_matrix has {X.shape[0]} cells, expected {n_cells}"
                )
            n_genes = X.shape[1]
        else:
            # Use cell probabilities as placeholder
            X = self.cell_probabilities.copy()
            n_genes = self.n_cell_types

        # Gene names
        if gene_names is not None:
            if len(gene_names) != n_genes:
                raise ValueError(
                    f"gene_names has {len(gene_names)} entries, expected {n_genes}"
                )
            var_names = gene_names
        elif expression_matrix is not None:
            var_names = [f"Gene_{i}" for i in range(n_genes)]
        else:
            var_names = [f"CellType_{i}" for i in range(n_genes)]

        # Build obs DataFrame
        obs_data = {"cell_id": [f"cell_{i}" for i in range(n_cells)]}

        if self.class_instance_one_hot is not None:
            obs_data["class_id"] = self.class_instance

        # Add morphological properties if available
        for attr, col_name in [
            ("cell_minor_axis", "minor_axis"),
            ("cell_major_axis", "major_axis"),
            ("cell_rotation", "rotation"),
            ("cell_rna_concentration", "rna_concentration"),
        ]:
            if hasattr(self, attr):
                obs_data[col_name] = getattr(self, attr)

        obs = pd.DataFrame(obs_data)
        obs.index = obs["cell_id"]

        # Build var DataFrame
        var = pd.DataFrame({"gene_name": var_names})
        var.index = var_names

        # Create AnnData object
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Add spatial coordinates
        if include_spatial:
            adata.obsm["spatial"] = self.cell_centroids.copy()

        # Store cell type probabilities in uns
        adata.uns["cell_type_probabilities"] = self.cell_probabilities.copy()

        # Store one-hot encoding if available
        if self.class_instance_one_hot is not None:
            adata.obsm["cell_type_one_hot"] = self.class_instance_one_hot.copy()

        return adata

    def to_spatialdata(
        self,
        gene_names: Optional[List[str]] = None,
        expression_matrix: Optional[NDArray[np.floating]] = None,
        cell_radius: Optional[Union[float, NDArray[np.floating]]] = None,
        include_shapes: bool = True,
    ):
        """Export FOV data as a SpatialData object.

        Creates a SpatialData object compatible with the scverse ecosystem
        for spatial transcriptomics analysis.

        Parameters
        ----------
        gene_names : list of str, optional
            Gene names for the expression matrix columns. If None and
            expression_matrix is provided, uses 'Gene_0', 'Gene_1', etc.
        expression_matrix : np.ndarray, optional
            Gene expression matrix, shape (n_cells, n_genes). If None,
            uses cell_probabilities as a placeholder.
        cell_radius : float or np.ndarray, optional
            Radius for cell circles in the shapes layer. If float, uses
            same radius for all cells. If array, must have length n_cells.
            If None, uses 5.0 as default or derives from cell_major_axis
            if available.
        include_shapes : bool, optional
            If True, includes cell shapes as circles. Default True.

        Returns
        -------
        spatialdata.SpatialData
            SpatialData object with:
            - shapes['cells']: GeoDataFrame with cell circle geometries
            - tables['adata']: AnnData with expression and metadata

        Raises
        ------
        ImportError
            If spatialdata or geopandas is not installed.

        Examples
        --------
        >>> fov = fov_distribution.generate_fov()
        >>> sdata = fov.to_spatialdata()
        >>> sdata.pl.render_shapes('cells', color='class_id').pl.show()

        Notes
        -----
        Requires the spatialdata package: pip install spatialdata
        """
        try:
            import spatialdata as sd
            from spatialdata.models import ShapesModel, TableModel
        except ImportError:
            raise ImportError(
                "spatialdata is required for to_spatialdata(). "
                "Install with: pip install spatialdata"
            )

        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError:
            raise ImportError(
                "geopandas is required for to_spatialdata(). "
                "Install with: pip install geopandas"
            )

        n_cells = self.n_cells

        # First create the AnnData object using existing method
        adata = self.to_anndata(
            gene_names=gene_names,
            expression_matrix=expression_matrix,
            include_spatial=True,
        )

        # Determine cell radii
        if cell_radius is not None:
            if isinstance(cell_radius, (int, float)):
                radii = np.full(n_cells, float(cell_radius))
            else:
                radii = np.asarray(cell_radius)
                if len(radii) != n_cells:
                    raise ValueError(
                        f"cell_radius array has {len(radii)} elements, expected {n_cells}"
                    )
        elif hasattr(self, "cell_major_axis"):
            # Use half of major axis as radius
            radii = self.cell_major_axis / 2.0
        else:
            radii = np.full(n_cells, 5.0)

        sdata_components = {}

        if include_shapes and n_cells > 0:
            # Create GeoDataFrame with circle geometries
            geometries = [
                Point(x, y) for x, y in self.cell_centroids
            ]
            shapes_df = gpd.GeoDataFrame(
                {"radius": radii},
                geometry=geometries,
            )
            shapes_df.index = [f"cell_{i}" for i in range(n_cells)]

            # Parse through ShapesModel
            shapes_for_sdata = ShapesModel.parse(shapes_df)
            sdata_components["shapes"] = {"cells": shapes_for_sdata}

            # Link table to shapes
            adata.obs["region"] = pd.Categorical(["cells"] * n_cells)
            adata.obs["instance_id"] = shapes_df.index.tolist()

        # Parse AnnData through TableModel
        adata_for_sdata = TableModel.parse(
            adata,
            region="cells" if include_shapes else None,
            region_key="region" if include_shapes else None,
            instance_key="instance_id" if include_shapes else None,
        )
        sdata_components["tables"] = {"adata": adata_for_sdata}

        # Create SpatialData object
        sdata = sd.SpatialData(**sdata_components)

        return sdata


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

    def generate_batch(
        self,
        n_fovs: int,
        seeds: Optional[List[int]] = None,
        parallel: bool = False,
        n_jobs: int = -1,
        progress: bool = False,
    ) -> List[FOV]:
        """Generate multiple FOV realizations.

        Efficiently generates many FOVs, optionally in parallel.

        Parameters
        ----------
        n_fovs : int
            Number of FOVs to generate.
        seeds : list of int, optional
            Random seeds for each FOV. If None, uses sequential seeds.
        parallel : bool, optional
            If True, use parallel processing. Default is False.
        n_jobs : int, optional
            Number of parallel jobs. Default -1 uses all cores.
        progress : bool, optional
            If True, show progress bar. Default is False.

        Returns
        -------
        list of FOV
            List of generated FOV objects.

        Examples
        --------
        >>> fovd = FOVDistribution(frame_size=1000, ...)
        >>> fovs = fovd.generate_batch(10)
        >>> len(fovs)
        10
        """
        if seeds is None:
            seeds = list(range(n_fovs))

        fovs = []

        if parallel:
            try:
                from joblib import Parallel, delayed

                def _generate_single(seed):
                    np.random.seed(seed)
                    return self.generate_fov()

                fovs = Parallel(n_jobs=n_jobs)(
                    delayed(_generate_single)(seed) for seed in seeds
                )
            except ImportError:
                # Fall back to sequential if joblib not available
                parallel = False

        if not parallel:
            iterator = seeds
            if progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(seeds, desc="Generating FOVs")
                except ImportError:
                    pass

            for seed in iterator:
                np.random.seed(seed)
                fovs.append(self.generate_fov())

        return fovs
