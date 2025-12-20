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
