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

    def save_config(self, filepath: str, format: str = 'auto'):
        """Save experiment configuration to a file.

        Saves all experiment parameters including gene sensitivities,
        transfer function settings, and tissue expression profiles.
        This allows reproducing the exact experimental setup.

        Parameters
        ----------
        filepath : str
            Path to save the configuration.
        format : str, optional
            File format: 'json', 'yaml', 'npz', or 'auto' (from extension).
            Default 'auto'.

        Notes
        -----
        - 'json': Human-readable, editable, recommended for configuration
        - 'yaml': Human-readable, requires pyyaml
        - 'npz': NumPy format, includes all numerical data

        Examples
        --------
        >>> hybiss.save_config('experiment_config.json')
        >>> hybiss.save_config('experiment.yaml')
        """
        if format == 'auto':
            if filepath.endswith('.json'):
                format = 'json'
            elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                format = 'yaml'
            elif filepath.endswith('.npz'):
                format = 'npz'
            else:
                format = 'json'

        config = self._build_config_dict()

        if format == 'json':
            self._save_json(filepath, config)
        elif format == 'yaml':
            self._save_yaml(filepath, config)
        elif format == 'npz':
            self._save_npz(filepath, config)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _build_config_dict(self):
        """Build configuration dictionary."""
        config = {
            'experiment_type': 'HybISS',
            'n_genes': int(self.M.shape[0]),
            'n_cell_types': int(self.M.shape[1]),
            'gene_names': self.tissue.gene_names.tolist() if hasattr(self.tissue.gene_names, 'tolist') else list(self.tissue.gene_names),
            'cell_type_names': self.tissue.cell_type_names.tolist() if hasattr(self.tissue.cell_type_names, 'tolist') else list(self.tissue.cell_type_names),
            'genes_sensitivities': self.genes_sensitivities.tolist(),
            'genes_sensitivities_variation': self.genes_sensitivities_variation.tolist(),
            'transfer_function': {
                'type': self.transfer_function.__class__.__name__,
            },
            'expression_matrix': self.raw_M.tolist(),
        }

        # Add transfer function parameters if available
        if hasattr(self.transfer_function, 'scale'):
            config['transfer_function']['scale'] = float(self.transfer_function.scale)
        if hasattr(self.transfer_function, 'offset'):
            config['transfer_function']['offset'] = float(self.transfer_function.offset)

        return config

    def _save_json(self, filepath: str, config: dict):
        """Save to JSON format."""
        import json
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    def _save_yaml(self, filepath: str, config: dict):
        """Save to YAML format."""
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def _save_npz(self, filepath: str, config: dict):
        """Save to NPZ format with arrays."""
        np.savez_compressed(
            filepath,
            n_genes=np.array([config['n_genes']]),
            n_cell_types=np.array([config['n_cell_types']]),
            gene_names=np.array(config['gene_names']),
            cell_type_names=np.array(config['cell_type_names']),
            genes_sensitivities=np.array(config['genes_sensitivities']),
            genes_sensitivities_variation=np.array(config['genes_sensitivities_variation']),
            expression_matrix=np.array(config['expression_matrix']),
            transfer_function_type=np.array([config['transfer_function']['type']]),
        )

    @classmethod
    def load_config(cls, filepath: str, format: str = 'auto'):
        """Load experiment configuration from a file.

        Reconstructs a HybISS_Setup from saved configuration.
        Requires recreating the tissue object from saved expression data.

        Parameters
        ----------
        filepath : str
            Path to the configuration file.
        format : str, optional
            File format: 'json', 'yaml', 'npz', or 'auto'.
            Default 'auto'.

        Returns
        -------
        HybISS_Setup
            Reconstructed experiment setup.

        Examples
        --------
        >>> hybiss = HybISS_Setup.load_config('experiment_config.json')
        """
        from ..core import TissueCellTypes

        if format == 'auto':
            if filepath.endswith('.json'):
                format = 'json'
            elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                format = 'yaml'
            elif filepath.endswith('.npz'):
                format = 'npz'
            else:
                format = 'json'

        if format == 'json':
            config = cls._load_json(filepath)
        elif format == 'yaml':
            config = cls._load_yaml(filepath)
        elif format == 'npz':
            config = cls._load_npz(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Recreate tissue
        tissue = TissueCellTypes()
        tissue._gene_names = config['gene_names']
        tissue._cell_type_names = config['cell_type_names']
        tissue.gene_expression_by_type = np.array(config['expression_matrix'])

        # Recreate transfer function
        tf_type = config['transfer_function']['type']
        if tf_type == 'IdentityTransfer':
            transfer_function = IdentityTransfer()
        else:
            # Default to identity if unknown
            transfer_function = IdentityTransfer()

        # Create HybISS_Setup
        obj = cls(
            tissue=tissue,
            genes_sensitivities=np.array(config['genes_sensitivities']),
            genes_sensitivities_variation=np.array(config['genes_sensitivities_variation']),
            transfer_function=transfer_function,
        )

        return obj

    @staticmethod
    def _load_json(filepath: str) -> dict:
        """Load from JSON format."""
        import json
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def _load_yaml(filepath: str) -> dict:
        """Load from YAML format."""
        import yaml
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_npz(filepath: str) -> dict:
        """Load from NPZ format."""
        data = np.load(filepath, allow_pickle=True)
        return {
            'n_genes': int(data['n_genes'][0]),
            'n_cell_types': int(data['n_cell_types'][0]),
            'gene_names': data['gene_names'].tolist(),
            'cell_type_names': data['cell_type_names'].tolist(),
            'genes_sensitivities': data['genes_sensitivities'].tolist(),
            'genes_sensitivities_variation': data['genes_sensitivities_variation'].tolist(),
            'expression_matrix': data['expression_matrix'].tolist(),
            'transfer_function': {'type': str(data['transfer_function_type'][0])},
        }
