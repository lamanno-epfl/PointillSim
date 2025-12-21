"""Tissue and expression profile classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from shapely.geometry import Polygon, Point

from ..utils.math import intuitive_rand_lognormal


class TissueCellTypes:
    """Gene expression profiles for cell types in a tissue.

    Defines the expected gene expression levels for each cell type,
    which are used to simulate realistic transcript counts.

    Parameters
    ----------
    gene_expression_by_type : np.ndarray, optional
        Pre-defined expression matrix, shape (n_genes, n_cell_types).
        If None, use generate_types_and_markers() to create one.

    Attributes
    ----------
    gene_expression_by_type : np.ndarray
        Expression matrix (genes x cell_types).
    colordict : dict
        Cell type colors for visualization.
    gene_colordict : dict
        Gene colors for visualization.
    """

    def __init__(
        self, gene_expression_by_type: Optional[NDArray[np.floating]] = None
    ) -> None:
        self.gene_expression_by_type = gene_expression_by_type
        self.concentration: Optional[float] = None
        self._cell_type_names: Optional[List[str]] = None
        self._gene_names: Optional[List[str]] = None
        self.colordict: Dict[int, Tuple] = {}
        self.gene_colordict: Dict[int, Tuple] = {}

    @classmethod
    def load_from_csv(
        cls,
        filepath: str,
        gene_col: Optional[str] = None,
        transpose: bool = False,
    ) -> TissueCellTypes:
        """Load expression profiles from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to CSV file with expression matrix.
        gene_col : str, optional
            Column name containing gene names. If None, uses index.
        transpose : bool, optional
            If True, transpose the matrix (rows become columns).
            Use when file has genes as columns and cell types as rows.

        Returns
        -------
        TissueCellTypes
            Instance with loaded expression matrix.

        Examples
        --------
        >>> tissue = TissueCellTypes.load_from_csv("expression.csv")
        >>> tissue = TissueCellTypes.load_from_csv("expression.csv", transpose=True)
        """
        df = pd.read_csv(filepath, index_col=0 if gene_col is None else None)

        if gene_col is not None:
            df = df.set_index(gene_col)

        if transpose:
            df = df.T

        instance = cls(gene_expression_by_type=df.values.astype(float))
        instance._gene_names = list(df.index)
        instance._cell_type_names = list(df.columns)

        n_genes = len(instance._gene_names)
        n_cell_types = len(instance._cell_type_names)

        instance.colordict = {
            i: plt.cm.turbo(float(i) / n_cell_types) for i in range(n_cell_types)
        }
        instance.gene_colordict = {
            i: plt.cm.turbo(float(i) / n_genes) for i in range(n_genes)
        }

        return instance

    @classmethod
    def load_from_anndata(
        cls,
        adata,
        layer: Optional[str] = None,
        cell_type_key: str = "cell_type",
        aggregate: str = "mean",
    ) -> "TissueCellTypes":
        """Load expression profiles from an AnnData object.

        Aggregates single-cell expression data by cell type to create
        cell type-level expression profiles suitable for simulation.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object with expression data. Must have cell type
            annotations in obs.
        layer : str, optional
            Layer to use for expression values. If None, uses adata.X.
        cell_type_key : str, optional
            Column in adata.obs containing cell type labels. Default "cell_type".
        aggregate : str, optional
            Aggregation method: "mean", "median", or "sum". Default "mean".

        Returns
        -------
        TissueCellTypes
            Instance with expression profiles aggregated by cell type.

        Raises
        ------
        ImportError
            If anndata is not installed.
        ValueError
            If cell_type_key is not found in adata.obs.
        ValueError
            If aggregate method is invalid.

        Examples
        --------
        >>> import anndata
        >>> adata = anndata.read_h5ad("scrnaseq_data.h5ad")
        >>> tissue = TissueCellTypes.load_from_anndata(adata, cell_type_key="celltype")
        >>> tissue.n_cell_types
        10
        """
        try:
            import anndata as ad
        except ImportError:
            raise ImportError(
                "anndata is required for load_from_anndata(). "
                "Install with: pip install pointillsim[anndata]"
            )

        if cell_type_key not in adata.obs.columns:
            raise ValueError(
                f"Cell type key '{cell_type_key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        if aggregate not in ("mean", "median", "sum"):
            raise ValueError(
                f"aggregate must be 'mean', 'median', or 'sum', got '{aggregate}'"
            )

        # Get expression matrix
        if layer is not None:
            if layer not in adata.layers:
                raise ValueError(
                    f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}"
                )
            X = adata.layers[layer]
        else:
            X = adata.X

        # Convert sparse matrix to dense if needed
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)

        # Get cell type labels
        cell_types = adata.obs[cell_type_key].values
        unique_types = np.unique(cell_types)
        n_cell_types = len(unique_types)
        n_genes = X.shape[1]

        # Aggregate expression by cell type
        expression_by_type = np.zeros((n_genes, n_cell_types))

        for i, ct in enumerate(unique_types):
            mask = cell_types == ct
            if aggregate == "mean":
                expression_by_type[:, i] = X[mask].mean(axis=0)
            elif aggregate == "median":
                expression_by_type[:, i] = np.median(X[mask], axis=0)
            else:  # sum
                expression_by_type[:, i] = X[mask].sum(axis=0)

        # Create instance
        instance = cls(gene_expression_by_type=expression_by_type)
        instance._gene_names = list(adata.var_names)
        instance._cell_type_names = [str(ct) for ct in unique_types]

        instance.colordict = {
            i: plt.cm.turbo(float(i) / n_cell_types) for i in range(n_cell_types)
        }
        instance.gene_colordict = {
            i: plt.cm.turbo(float(i) / n_genes) for i in range(n_genes)
        }

        return instance

    def generate_types_and_markers(
        self,
        n_genes: int,
        n_cell_types: int,
        expected_level: float = 8.0,
        expected_std_level: float = 3.0,
        concentration: float = 0.90,
    ) -> NDArray[np.floating]:
        """Generate random gene expression profiles with marker gene patterns.

        Creates a sparse expression matrix where each gene is predominantly
        expressed in one or a few cell types (marker genes), with expression
        levels drawn from lognormal distributions.

        Parameters
        ----------
        n_genes : int
            Number of genes to simulate.
        n_cell_types : int
            Number of cell types.
        expected_level : float, optional
            Mean expression level across genes. Default 8.0.
        expected_std_level : float, optional
            Std of expression levels. Default 3.0.
        concentration : float, optional
            Sparsity parameter (0-1). Higher values create more distinct
            marker genes. Default 0.90.

        Returns
        -------
        np.ndarray
            Gene expression matrix, shape (n_genes, n_cell_types).
        """
        self.expected_level = expected_level
        self.concentration = concentration
        tmp = np.round(
            np.random.dirichlet(
                np.ones(n_cell_types) * (1 - self.concentration), n_genes
            ),
            int(2.5 * np.log10(n_genes)),
        )
        self.fuzzy_sparsity_pattern = tmp / tmp.sum(axis=0)

        self.scale_of_the_gene = intuitive_rand_lognormal(
            expected_level, expected_std_level, n_genes
        )
        self.gene_expression_by_type = (
            self.scale_of_the_gene[:, None] * self.fuzzy_sparsity_pattern
        )

        self.colordict = {
            i: plt.cm.turbo(float(i) / n_cell_types) for i in range(n_cell_types)
        }

        self.gene_colordict = {
            i: plt.cm.turbo(float(i) / n_genes) for i in range(n_genes)
        }

        ixs = np.argsort(self.gene_expression_by_type.argmax(1))
        self.gene_expression_by_type = self.gene_expression_by_type[ixs, :]

        return self.gene_expression_by_type

    @property
    def n_cell_types(self):
        """int: Number of cell types."""
        return self.gene_expression_by_type.shape[1]

    @property
    def n_genes(self):
        """int: Number of genes."""
        return self.gene_expression_by_type.shape[0]

    @property
    def cell_type_names(self) -> NDArray[np.str_]:
        """np.ndarray: Cell type names (custom if loaded, else 'Type 0', ...)."""
        if self._cell_type_names is not None:
            return np.array(self._cell_type_names)
        return np.array([f"Type {i}" for i in range(self.n_cell_types)])

    @property
    def gene_names(self) -> NDArray[np.str_]:
        """np.ndarray: Gene names (custom if loaded, else 'Gene 0', ...)."""
        if self._gene_names is not None:
            return np.array(self._gene_names)
        return np.array([f"Gene {i}" for i in range(self.n_genes)])

    def make_pandas_df(self):
        """Export expression matrix as DataFrame with named rows and columns."""
        return pd.DataFrame(
            self.gene_expression_by_type,
            columns=self.cell_type_names,
            index=self.gene_names,
        )


@dataclass
class RegionSpec:
    """Specification for a tissue region within a TissueSlice.

    Attributes
    ----------
    name : str
        Human-readable name for the region.
    polygon : shapely.Polygon
        Spatial boundary of the region.
    fovdist : FOVDistribution
        Generator for cells within this region.
    priority : int
        Overlap resolution priority (higher wins). Default 0.
    blend_band : float
        Width in pixels for boundary blending. Default 0.0 (no blending).
    """
    name: str
    polygon: Polygon
    fovdist: "FOVDistribution"  # Forward reference
    priority: int = 0
    blend_band: float = 0.0


class TissueSlice:
    """Composer for multi-region tissue sections with consistent FOV tiling.

    Combines multiple region-specific FOVDistributions on a global canvas,
    resolving overlaps by priority. Supports exporting overlapping FOV tiles
    where shared cells have identical properties.

    Parameters
    ----------
    frame_size : int
        Size of the global canvas in pixels.
    regions : list of RegionSpec
        Region definitions to compose.

    Attributes
    ----------
    _cells_xy : np.ndarray
        Global cell positions after generate_global().
    _probs : np.ndarray
        Global cell type probabilities.
    _class_onehot : np.ndarray
        Sampled cell types after sample_labels().

    Examples
    --------
    >>> cortex_region = RegionSpec("cortex", cortex_polygon, cortex_fovd, priority=1)
    >>> white_matter = RegionSpec("wm", wm_polygon, wm_fovd, priority=0)
    >>> slice = TissueSlice(10000, [cortex_region, white_matter])
    >>> slice.generate_global().sample_labels()
    >>> tiles = slice.tile_into_fovs(1000, overlap_frac=0.1)
    """
    def __init__(self, frame_size: int, regions: list):
        self.frame_size = frame_size
        self.regions = sorted(regions, key=lambda r: r.priority)
        self._cells_xy = None
        self._probs = None
        self._class_onehot = None

    def generate_global(self):
        """Generate cells for all regions and merge into global canvas.

        Processes regions in priority order, clipping cells to region
        boundaries and optionally applying boundary blending.

        Returns
        -------
        self
            For method chaining.
        """
        realized = []
        occupied_mask = None

        for reg in self.regions:
            fov = reg.fovdist.generate_fov()
            xy = fov.cell_centroids
            P = fov.cell_probabilities

            inside = np.array([reg.polygon.contains(Point(x, y)) for x, y in xy], bool)
            xy, P = xy[inside], P[inside]

            if reg.blend_band > 0:
                bdist = np.array([reg.polygon.boundary.distance(Point(x, y)) for x, y in xy])
                w = np.clip(bdist / reg.blend_band, 0.0, 1.0)[:, None]
                hard = np.eye(P.shape[1])[np.argmax(P, 1)]
                P = w * P + (1.0 - w) * hard

            realized.append({"xy": xy, "P": P})

        if not realized:
            self._cells_xy = np.empty((0, 2))
            self._probs = np.empty((0, 0))
            self._class_onehot = None
            return self

        n_types = 0
        for r in realized:
            if r["P"].shape[0] > 0:
                n_types = max(n_types, r["P"].shape[1])

        all_xy_list = []
        all_P_list = []
        for r in realized:
            if r["xy"].shape[0] == 0:
                continue

            all_xy_list.append(r["xy"])

            P = r["P"]
            if P.shape[1] < n_types:
                P_padded = np.zeros((P.shape[0], n_types))
                P_padded[:, :P.shape[1]] = P
                all_P_list.append(P_padded)
            else:
                all_P_list.append(P)

        if not all_xy_list:
            self._cells_xy = np.empty((0, 2))
            self._probs = np.empty((0, n_types))
            self._class_onehot = None
            return self

        all_xy = np.row_stack(all_xy_list)
        all_P = np.row_stack(all_P_list)

        key = np.round(all_xy, 3).view([('', all_xy.dtype)] * 2).flatten()
        unique_keys_reversed, last_indices_from_reversed = np.unique(key[::-1], return_index=True)
        keep_idx = (len(key) - 1 - last_indices_from_reversed)
        keep_idx.sort()

        self._cells_xy = all_xy[keep_idx]
        self._probs = all_P[keep_idx]
        self._class_onehot = None
        return self

    def sample_labels(self, rng=None):
        """Sample hard cell type labels from probabilities.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        self
            For method chaining.
        """
        rng = np.random.default_rng() if rng is None else rng
        n, k = self._probs.shape

        if k == 0:
            self._class_onehot = np.empty((n, 0), dtype=int)
            return self

        draws = np.zeros(n, dtype=int)
        uniform_probs = np.ones(k) / k

        for i, p in enumerate(self._probs):
            p_sum = p.sum()
            if p_sum > 1e-8:
                p_normalized = p / p_sum
            else:
                p_normalized = uniform_probs

            try:
                draws[i] = rng.choice(k, p=p_normalized)
            except ValueError:
                draws[i] = np.argmax(p_normalized)

        self._class_onehot = np.eye(k, dtype=int)[draws]
        return self

    def tile_into_fovs(self, fov_size: int, overlap_frac: float = 0.10):
        """Partition global canvas into overlapping FOV tiles.

        Creates a grid of FOV windows with specified overlap. Cells in
        overlap regions appear in multiple tiles with identical indices.

        Parameters
        ----------
        fov_size : int
            Size of each FOV tile in pixels.
        overlap_frac : float, optional
            Fraction of overlap between adjacent tiles. Default 0.10.

        Returns
        -------
        list of dict
            Each dict contains:
            - 'ij': (col, row) grid position
            - 'bbox': (x0, y0, x1, y1) bounding box
            - 'idx': np.ndarray of global cell indices in this tile
        """
        step = int(round(fov_size * (1.0 - overlap_frac)))
        xs = list(range(0, self.frame_size - fov_size + 1, step))
        ys = list(range(0, self.frame_size - fov_size + 1, step))
        tiles = []
        X, Y = self._cells_xy[:, 0], self._cells_xy[:, 1]
        for j, y0 in enumerate(ys):
            for i, x0 in enumerate(xs):
                x1, y1 = x0 + fov_size, y0 + fov_size
                keep = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
                tiles.append({
                    "ij": (i, j),
                    "bbox": (x0, y0, x1, y1),
                    "idx": np.where(keep)[0]
                })
        return tiles

    def extract_fov(
        self,
        x0: float,
        y0: float,
        fov_size: int,
        local_coords: bool = True,
    ):
        """Extract a single FOV from a specific location.

        Creates an FOV-like object containing cells within the specified
        bounding box. Useful for extracting specific regions of interest
        or for custom tiling strategies.

        Parameters
        ----------
        x0 : float
            X coordinate of the FOV bottom-left corner.
        y0 : float
            Y coordinate of the FOV bottom-left corner.
        fov_size : int
            Size of the extracted FOV in pixels.
        local_coords : bool, optional
            If True, translate coordinates to local FOV space (0 to fov_size).
            If False, keep global coordinates. Default True.

        Returns
        -------
        dict
            Dictionary containing:
            - 'cell_centroids': np.ndarray of cell positions
            - 'cell_probabilities': np.ndarray of probabilities
            - 'class_instance': np.ndarray of sampled types (if available)
            - 'global_indices': np.ndarray of indices in the global slice
            - 'bbox': (x0, y0, x1, y1) bounding box

        Examples
        --------
        >>> slice.generate_global().sample_labels()
        >>> fov = slice.extract_fov(x0=1000, y0=2000, fov_size=500)
        >>> print(f"Extracted {len(fov['cell_centroids'])} cells")
        """
        if self._cells_xy is None:
            raise ValueError("No cells generated. Call generate_global() first.")

        x1, y1 = x0 + fov_size, y0 + fov_size
        X, Y = self._cells_xy[:, 0], self._cells_xy[:, 1]
        mask = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
        indices = np.where(mask)[0]

        centroids = self._cells_xy[mask].copy()
        if local_coords:
            centroids[:, 0] -= x0
            centroids[:, 1] -= y0

        result = {
            'cell_centroids': centroids,
            'cell_probabilities': self._probs[mask].copy(),
            'global_indices': indices,
            'bbox': (x0, y0, x1, y1),
            'frame_size': fov_size,
        }

        if self._class_onehot is not None:
            result['class_onehot'] = self._class_onehot[mask].copy()
            result['class_instance'] = np.argmax(self._class_onehot[mask], axis=1)

        return result

    def save(self, filepath: str, format: str = 'auto'):
        """Save the TissueSlice to a file.

        Serializes the tissue slice including cell positions, probabilities,
        and sampled labels. Region specifications are not saved (they contain
        non-serializable FOVDistribution objects).

        Parameters
        ----------
        filepath : str
            Path to save the file.
        format : str, optional
            File format: 'npz', 'pickle', 'hdf5', or 'auto' (from extension).
            Default 'auto'.

        Notes
        -----
        - 'npz': Compressed NumPy format, portable, recommended for sharing
        - 'pickle': Python pickle, preserves all attributes but less portable
        - 'hdf5': HDF5 format, requires h5py, good for very large slices

        Examples
        --------
        >>> slice.generate_global().sample_labels()
        >>> slice.save('my_tissue.npz')
        >>> # or
        >>> slice.save('my_tissue.h5', format='hdf5')
        """
        if format == 'auto':
            if filepath.endswith('.npz'):
                format = 'npz'
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'
            elif filepath.endswith('.h5') or filepath.endswith('.hdf5'):
                format = 'hdf5'
            else:
                format = 'npz'

        if format == 'npz':
            self._save_npz(filepath)
        elif format == 'pickle':
            self._save_pickle(filepath)
        elif format == 'hdf5':
            self._save_hdf5(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _save_npz(self, filepath: str):
        """Save to compressed NumPy format."""
        data = {
            'frame_size': np.array([self.frame_size]),
            'cells_xy': self._cells_xy if self._cells_xy is not None else np.empty((0, 2)),
            'probs': self._probs if self._probs is not None else np.empty((0, 0)),
        }
        if self._class_onehot is not None:
            data['class_onehot'] = self._class_onehot
        np.savez_compressed(filepath, **data)

    def _save_pickle(self, filepath: str):
        """Save to pickle format."""
        import pickle
        data = {
            'frame_size': self.frame_size,
            '_cells_xy': self._cells_xy,
            '_probs': self._probs,
            '_class_onehot': self._class_onehot,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def _save_hdf5(self, filepath: str):
        """Save to HDF5 format."""
        import h5py
        with h5py.File(filepath, 'w') as f:
            f.attrs['frame_size'] = self.frame_size
            if self._cells_xy is not None:
                f.create_dataset('cells_xy', data=self._cells_xy, compression='gzip')
            if self._probs is not None:
                f.create_dataset('probs', data=self._probs, compression='gzip')
            if self._class_onehot is not None:
                f.create_dataset('class_onehot', data=self._class_onehot, compression='gzip')

    @classmethod
    def load(cls, filepath: str, format: str = 'auto') -> 'TissueSlice':
        """Load a TissueSlice from a file.

        Loads cell positions and probabilities from a saved file.
        Region specifications are not restored (they must be recreated
        if needed for further generation).

        Parameters
        ----------
        filepath : str
            Path to the saved file.
        format : str, optional
            File format: 'npz', 'pickle', 'hdf5', or 'auto' (from extension).
            Default 'auto'.

        Returns
        -------
        TissueSlice
            Loaded tissue slice with cells and probabilities.

        Examples
        --------
        >>> slice = TissueSlice.load('my_tissue.npz')
        >>> print(f"Loaded {len(slice._cells_xy)} cells")
        """
        if format == 'auto':
            if filepath.endswith('.npz'):
                format = 'npz'
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'
            elif filepath.endswith('.h5') or filepath.endswith('.hdf5'):
                format = 'hdf5'
            else:
                format = 'npz'

        if format == 'npz':
            return cls._load_npz(filepath)
        elif format == 'pickle':
            return cls._load_pickle(filepath)
        elif format == 'hdf5':
            return cls._load_hdf5(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def _load_npz(cls, filepath: str) -> 'TissueSlice':
        """Load from compressed NumPy format."""
        data = np.load(filepath)
        frame_size = int(data['frame_size'][0])
        obj = cls(frame_size=frame_size, regions=[])
        obj._cells_xy = data['cells_xy']
        obj._probs = data['probs']
        if 'class_onehot' in data:
            obj._class_onehot = data['class_onehot']
        return obj

    @classmethod
    def _load_pickle(cls, filepath: str) -> 'TissueSlice':
        """Load from pickle format."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        obj = cls(frame_size=data['frame_size'], regions=[])
        obj._cells_xy = data['_cells_xy']
        obj._probs = data['_probs']
        obj._class_onehot = data['_class_onehot']
        return obj

    @classmethod
    def _load_hdf5(cls, filepath: str) -> 'TissueSlice':
        """Load from HDF5 format."""
        import h5py
        with h5py.File(filepath, 'r') as f:
            frame_size = int(f.attrs['frame_size'])
            obj = cls(frame_size=frame_size, regions=[])
            if 'cells_xy' in f:
                obj._cells_xy = f['cells_xy'][:]
            if 'probs' in f:
                obj._probs = f['probs'][:]
            if 'class_onehot' in f:
                obj._class_onehot = f['class_onehot'][:]
        return obj

    @property
    def n_cells(self) -> int:
        """int: Number of cells in the tissue slice."""
        if self._cells_xy is None:
            return 0
        return len(self._cells_xy)

    @property
    def n_cell_types(self) -> int:
        """int: Number of cell types."""
        if self._probs is None or self._probs.shape[1] == 0:
            return 0
        return self._probs.shape[1]

    @property
    def cell_centroids(self):
        """np.ndarray: Cell centroid positions (for compatibility)."""
        return self._cells_xy

    @property
    def cell_probabilities(self):
        """np.ndarray: Cell type probabilities (for compatibility)."""
        return self._probs

    def generate_multiple(
        self,
        n_realizations: int,
        seeds: Optional[List[int]] = None,
        resample_cells: bool = False,
        resample_labels: bool = True,
    ) -> List[Dict]:
        """Generate multiple realizations of the tissue slice.

        Creates multiple versions of the tissue, either by resampling
        the cell positions or just the cell type labels.

        Parameters
        ----------
        n_realizations : int
            Number of realizations to generate.
        seeds : list of int, optional
            Random seeds for each realization. If None, uses sequential seeds.
        resample_cells : bool, optional
            If True, regenerate cell positions for each realization.
            If False, only resample labels. Default is False.
        resample_labels : bool, optional
            If True, resample cell type labels. Default is True.

        Returns
        -------
        list of dict
            Each dict contains:
            - 'cell_centroids': np.ndarray of positions
            - 'cell_probabilities': np.ndarray of probabilities
            - 'class_onehot': np.ndarray of sampled types
            - 'seed': the seed used

        Examples
        --------
        >>> slice = TissueSlice(5000, regions).generate_global()
        >>> realizations = slice.generate_multiple(10, resample_labels=True)
        >>> for r in realizations:
        ...     print(f"Realization with seed {r['seed']}: {len(r['cell_centroids'])} cells")
        """
        if self._cells_xy is None:
            raise ValueError("No cells generated. Call generate_global() first.")

        if seeds is None:
            seeds = list(range(n_realizations))

        realizations = []

        for seed in seeds:
            rng = np.random.default_rng(seed)

            if resample_cells:
                # Regenerate from scratch
                self.generate_global()
                centroids = self._cells_xy.copy()
                probs = self._probs.copy()
            else:
                centroids = self._cells_xy.copy()
                probs = self._probs.copy()

            if resample_labels:
                # Resample labels
                n, k = probs.shape
                class_onehot = np.zeros((n, k), dtype=int)
                for i, p in enumerate(probs):
                    p_sum = p.sum()
                    if p_sum > 1e-8:
                        p_normalized = p / p_sum
                    else:
                        p_normalized = np.ones(k) / k
                    try:
                        choice = rng.choice(k, p=p_normalized)
                    except ValueError:
                        choice = np.argmax(p_normalized)
                    class_onehot[i, choice] = 1
            else:
                if self._class_onehot is not None:
                    class_onehot = self._class_onehot.copy()
                else:
                    # Sample once if not already done
                    self.sample_labels(rng)
                    class_onehot = self._class_onehot.copy()

            realizations.append({
                'cell_centroids': centroids,
                'cell_probabilities': probs,
                'class_onehot': class_onehot,
                'class_instance': np.argmax(class_onehot, axis=1),
                'seed': seed,
            })

        return realizations


class ConsistentTiling:
    """Utility for creating consistent cell properties across tile boundaries.

    When tiling a tissue into overlapping FOVs, cells that appear in multiple
    tiles should have identical properties. This class ensures consistency
    by using cell indices to synchronize properties.

    Parameters
    ----------
    overlap_frac : float, optional
        Fraction of overlap between adjacent tiles. Default is 0.1.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> tiling = ConsistentTiling(overlap_frac=0.15)
    >>> tiles = slice.tile_into_fovs(1000, overlap_frac=0.15)
    >>> consistent_tiles = tiling.ensure_consistency(tiles, slice)
    """

    def __init__(self, overlap_frac: float = 0.1, seed: Optional[int] = None):
        self.overlap_frac = overlap_frac
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._property_cache = {}

    def ensure_consistency(
        self,
        tiles: List[Dict],
        tissue_slice: "TissueSlice",
        properties: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict]:
        """Ensure cells in overlapping regions have consistent properties.

        Parameters
        ----------
        tiles : list of dict
            Tiles from TissueSlice.tile_into_fovs().
        tissue_slice : TissueSlice
            The source tissue slice.
        properties : dict of np.ndarray, optional
            Additional per-cell properties to include. Keys are property names,
            values are arrays of shape (n_global_cells,).

        Returns
        -------
        list of dict
            Tiles with consistent properties added.
        """
        # Cache global properties
        if properties is not None:
            self._property_cache.update(properties)

        # Process each tile
        for tile in tiles:
            global_idx = tile['idx']

            # Add core properties from tissue slice
            tile['cell_centroids'] = tissue_slice._cells_xy[global_idx].copy()
            tile['cell_probabilities'] = tissue_slice._probs[global_idx].copy()

            if tissue_slice._class_onehot is not None:
                tile['class_onehot'] = tissue_slice._class_onehot[global_idx].copy()
                tile['class_instance'] = np.argmax(tile['class_onehot'], axis=1)

            # Transform to local coordinates
            x0, y0, _, _ = tile['bbox']
            tile['cell_centroids_local'] = tile['cell_centroids'].copy()
            tile['cell_centroids_local'][:, 0] -= x0
            tile['cell_centroids_local'][:, 1] -= y0

            # Add cached properties
            for prop_name, prop_values in self._property_cache.items():
                tile[prop_name] = prop_values[global_idx].copy()

        return tiles

    def generate_consistent_morphology(
        self,
        tissue_slice: "TissueSlice",
        cell_properties,
    ) -> Dict[str, np.ndarray]:
        """Generate morphological properties consistently across the slice.

        Parameters
        ----------
        tissue_slice : TissueSlice
            The source tissue slice.
        cell_properties : CellTypesProperties
            Cell type morphology definitions.

        Returns
        -------
        dict
            Dictionary of property arrays keyed by property name.
        """
        if tissue_slice._class_onehot is None:
            raise ValueError("Labels not sampled. Call sample_labels() first.")

        n_cells = len(tissue_slice._cells_xy)
        cell_types = np.argmax(tissue_slice._class_onehot, axis=1)

        # Generate morphological properties
        properties = {
            'minor_axis': np.zeros(n_cells),
            'major_axis': np.zeros(n_cells),
            'rotation': np.zeros(n_cells),
            'rna_concentration': np.zeros(n_cells),
        }

        for cell_type in range(tissue_slice.n_cell_types):
            mask = cell_types == cell_type
            n_type = mask.sum()

            if n_type == 0:
                continue

            # Get properties for this cell type
            mean_minor = cell_properties.cell_minor_axis_mean[cell_type]
            std_minor = cell_properties.cell_minor_axis_std[cell_type]
            mean_major = cell_properties.cell_major_axis_mean[cell_type]
            std_major = cell_properties.cell_major_axis_std[cell_type]
            concentration = cell_properties.cell_rna_concentration[cell_type]

            # Sample with consistent RNG
            properties['minor_axis'][mask] = self.rng.normal(mean_minor, std_minor, n_type)
            properties['major_axis'][mask] = self.rng.normal(mean_major, std_major, n_type)
            properties['rotation'][mask] = self.rng.uniform(0, 2 * np.pi, n_type)
            properties['rna_concentration'][mask] = concentration

        # Ensure positive values
        properties['minor_axis'] = np.maximum(properties['minor_axis'], 1)
        properties['major_axis'] = np.maximum(properties['major_axis'], 1)

        # Store in cache
        self._property_cache.update(properties)

        return properties

    def reset_cache(self):
        """Clear the property cache."""
        self._property_cache = {}
