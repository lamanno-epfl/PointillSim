"""HybISS experiment simulation."""

import numpy as np
import pandas as pd

from .transfer import IdentityTransfer
from ..utils.math import lognorm_params_to_mean_std
from ..utils.geometry import generate_points_asin_cell


class HybISS_Setup:
    """Simulator for HybISS (Hybridization-based In Situ Sequencing) experiments.

    Models the process of observing RNA transcripts as spatial dots within cells,
    including gene-specific detection sensitivity and Poisson sampling noise.

    Parameters
    ----------
    tissue : TissueCellTypes
        Tissue defining gene expression profiles per cell type.
    genes_sensitivities : float or np.ndarray, optional
        Mean detection sensitivity per gene. Default 1.0.
    genes_sensitivities_variation : float or np.ndarray, optional
        Std of sensitivity (lognormal). Default 0.3.
    transfer_function : TransferFunctionBase, optional
        Function to transform expression values before observation.
        Default is IdentityTransfer().

    Attributes
    ----------
    M : np.ndarray
        Transformed expression matrix (genes x cell_types).
    cellxgene_counts : np.ndarray
        Observed transcript counts per cell per gene (after measure_gene_expression).

    Examples
    --------
    >>> from pointillsim.core import TissueCellTypes
    >>> tissue = TissueCellTypes()
    >>> tissue.generate_types_and_markers(50, 10)
    >>> hybiss = HybISS_Setup(tissue)
    >>> hybiss.observe_dots(fov)
    >>> dots_df = hybiss.make_pandas_df()
    """

    def __init__(
        self,
        tissue,
        genes_sensitivities=1.0,
        genes_sensitivities_variation=0.3,
        transfer_function=None,
    ):
        if transfer_function is None:
            transfer_function = IdentityTransfer()

        self.tissue = tissue
        self.transfer_function = transfer_function
        self.raw_M = tissue.gene_expression_by_type
        self.M = self.transfer_function.transform(self.raw_M)
        self.rng = np.random.default_rng()
        self.cellxgene_counts = None

        if isinstance(genes_sensitivities_variation, float):
            self.genes_sensitivities_variation = (
                np.ones(self.M.shape[0]) * genes_sensitivities_variation
            )
        else:
            self.genes_sensitivities_variation = genes_sensitivities_variation

        if isinstance(genes_sensitivities, float):
            tmp = np.ones(self.M.shape[0]) * genes_sensitivities
            self.genes_sensitivities = np.random.lognormal(
                *lognorm_params_to_mean_std(tmp, self.genes_sensitivities_variation)
            )
        else:
            self.genes_sensitivities = genes_sensitivities

    def measure_gene_expression(self, fov):
        """Simulate transcript count observations for each cell.

        Computes expected counts from cell type, RNA concentration, and gene
        sensitivity, then samples actual counts from Poisson distribution.

        Parameters
        ----------
        fov : FOV
            Field of view with cell assignments and RNA concentrations.

        Returns
        -------
        np.ndarray
            Observed counts matrix, shape (n_cells, n_genes).
        """
        cellxgene_expectation = (
            fov.cell_rna_concentration[:, None]
            * (fov.class_instance_one_hot @ self.M.T)
            * self.genes_sensitivities
        )
        self.cellxgene_counts = self.rng.poisson(cellxgene_expectation)
        self.cellxtotal_counts = self.cellxgene_counts.sum(axis=1)
        return self.cellxgene_counts

    def observe_dots(self, fov):
        """Generate spatial dot positions for observed transcripts.

        Places transcript dots within each cell's elliptical boundary
        according to the count matrix from measure_gene_expression.

        Parameters
        ----------
        fov : FOV
            Field of view with cell morphology (requires cell_major_axis, etc.).

        Notes
        -----
        After calling, use make_pandas_df() to get dot coordinates with
        gene identities.
        """
        self.measure_gene_expression(fov)

        self.dot_belongsto_by_cells = []
        self.dot_isgene_by_cells = []
        self.dot_xs_by_cells = []
        self.dot_ys_by_cells = []

        for i in range(self.cellxgene_counts.shape[0]):
            xs, ys = generate_points_asin_cell(
                fov.cell_centroids[i],
                1.0,
                self.cellxtotal_counts[i],
                fov.cell_major_axis[i],
                fov.cell_minor_axis[i],
                fov.cell_rotation[i],
            ).T
            self.dot_xs_by_cells.append(xs)
            self.dot_ys_by_cells.append(ys)
            tmp_is = []
            tmp_js = []
            for j in range(self.cellxgene_counts.shape[1]):
                for _ in range(self.cellxgene_counts[i, j]):
                    tmp_js.append(j)
                    tmp_is.append(i)
            self.dot_belongsto_by_cells.append(tmp_is)
            self.dot_isgene_by_cells.append(tmp_js)

    def make_pandas_df(self):
        """Export observed dots as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: x, y (coordinates), gene (name), cell (index).
        """
        cell_ixs = np.concatenate(self.dot_belongsto_by_cells).astype(int)
        gene_ixs = np.concatenate(self.dot_isgene_by_cells).astype(int)
        return pd.DataFrame(
            {
                "x": np.concatenate(self.dot_xs_by_cells),
                "y": np.concatenate(self.dot_ys_by_cells),
                "gene": pd.Series(self.tissue.gene_names[gene_ixs], dtype=str),
                "cell": pd.Series(cell_ixs, dtype=int),
            }
        )
