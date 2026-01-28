"""Validation and difficulty assessment tools for simulated data.

This module provides tools for:
1. Assessing the difficulty of cell type classification tasks
2. Validating simulated data against reference datasets
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Optional, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


class DifficultyScorer:
    """Quantify the difficulty of a cell type classification task.

    This class assesses classification difficulty using:
    - Expression profile separability (silhouette score)
    - Spatial mixing (nearest neighbor purity)
    - Cell type balance

    Parameters
    ----------
    tissue : TissueCellTypes
        Tissue with gene expression profiles.

    Examples
    --------
    >>> from pointillsim import TissueCellTypes
    >>> tissue = TissueCellTypes()
    >>> tissue.generate_types_and_markers(n_genes=50, n_cell_types=5)
    >>> scorer = DifficultyScorer(tissue)
    >>> difficulty = scorer.score(fov, fov.class_instance)
    """

    def __init__(self, tissue):
        """Initialize difficulty scorer.

        Parameters
        ----------
        tissue : TissueCellTypes
            Tissue with gene expression profiles.
        """
        self.tissue = tissue
        self.expression_matrix = tissue.gene_expression_by_type

    def score(
        self,
        fov,
        cell_types: Optional[NDArray[np.integer]] = None,
        cell_counts: Optional[NDArray[np.floating]] = None
    ) -> Dict[str, float]:
        """Compute difficulty metrics for the given FOV.

        Parameters
        ----------
        fov : FOV
            Field of view to score.
        cell_types : array, optional
            Cell type labels. If None, uses fov.class_instance.
        cell_counts : array, optional
            Cell × gene count matrix for expression-based metrics.
            If None, uses ground truth expression profiles.

        Returns
        -------
        dict
            Dictionary with difficulty metrics:
            - separability: Expression profile separability (0-1, higher is easier)
            - spatial_mixing: Spatial mixing score (0-1, higher is more mixed/harder)
            - type_balance: Cell type balance (0-1, 1 is perfectly balanced)
            - overall_difficulty: Combined difficulty score (0-1, higher is harder)
        """
        if cell_types is None:
            cell_types = fov.class_instance

        # 1. Expression separability using silhouette score
        if cell_counts is not None and cell_counts.sum() > 0:
            # Use actual counts if provided
            X = np.log1p(cell_counts)
        else:
            # Use ground truth expression profiles
            X = self.expression_matrix[cell_types]

        # Only compute silhouette if we have multiple cell types
        unique_types = np.unique(cell_types)
        if len(unique_types) > 1 and len(cell_types) > len(unique_types):
            try:
                silhouette = silhouette_score(X, cell_types, metric='euclidean')
                # Transform from [-1, 1] to [0, 1]
                separability = (silhouette + 1) / 2
            except:
                separability = 0.5
        else:
            separability = 1.0

        # 2. Spatial mixing: How mixed are cell types spatially?
        spatial_mixing = self._compute_spatial_mixing(
            fov.cell_centroids, cell_types, k=10
        )

        # 3. Type balance: How balanced are cell type proportions?
        type_balance = self._compute_type_balance(cell_types)

        # 4. Overall difficulty (higher = harder)
        # Invert separability (high separability = easy = low difficulty)
        overall_difficulty = (
            (1 - separability) * 0.5 +  # Less separable = harder
            spatial_mixing * 0.3 +       # More mixing = harder
            (1 - type_balance) * 0.2     # Imbalanced = harder
        )

        return {
            'separability': separability,
            'spatial_mixing': spatial_mixing,
            'type_balance': type_balance,
            'overall_difficulty': overall_difficulty,
        }

    def _compute_spatial_mixing(
        self,
        centroids: NDArray[np.floating],
        cell_types: NDArray[np.integer],
        k: int = 10
    ) -> float:
        """Compute spatial mixing score using k-NN purity.

        Parameters
        ----------
        centroids : array
            Cell centroids (n_cells, 2).
        cell_types : array
            Cell type labels.
        k : int
            Number of neighbors to consider.

        Returns
        -------
        float
            Mixing score (0 = unmixed, 1 = fully mixed).
        """
        if len(centroids) < k + 1:
            return 0.0

        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nbrs.fit(centroids)
        _, indices = nbrs.kneighbors(centroids)

        # For each cell, compute fraction of neighbors with different type
        mixing_scores = []
        for i, neighbors in enumerate(indices):
            neighbors = neighbors[1:]  # Exclude self
            own_type = cell_types[i]
            different_type = np.sum(cell_types[neighbors] != own_type)
            mixing_scores.append(different_type / k)

        return np.mean(mixing_scores)

    def _compute_type_balance(self, cell_types: NDArray[np.integer]) -> float:
        """Compute cell type balance using Shannon entropy.

        Parameters
        ----------
        cell_types : array
            Cell type labels.

        Returns
        -------
        float
            Balance score (0 = completely imbalanced, 1 = perfectly balanced).
        """
        unique, counts = np.unique(cell_types, return_counts=True)
        n_types = len(unique)

        if n_types <= 1:
            return 1.0

        # Compute Shannon entropy
        proportions = counts / counts.sum()
        entropy = -np.sum(proportions * np.log(proportions + 1e-10))

        # Normalize by maximum entropy (uniform distribution)
        max_entropy = np.log(n_types)
        balance = entropy / max_entropy if max_entropy > 0 else 1.0

        return balance


class ValidationMetrics:
    """Compare simulated data to reference data for validation.

    This class computes similarity metrics between simulated and reference
    datasets to assess how realistic the simulation is.

    Examples
    --------
    >>> validator = ValidationMetrics()
    >>> metrics = validator.compare(
    ...     sim_counts, sim_types, ref_counts, ref_types
    ... )
    """

    def __init__(self):
        """Initialize validation metrics."""
        pass

    def compare(
        self,
        sim_counts: NDArray[np.floating],
        sim_types: NDArray[np.integer],
        ref_counts: NDArray[np.floating],
        ref_types: NDArray[np.integer],
    ) -> Dict[str, float]:
        """Compare simulated and reference datasets.

        Parameters
        ----------
        sim_counts : array
            Simulated count matrix (n_cells × n_genes).
        sim_types : array
            Simulated cell type labels.
        ref_counts : array
            Reference count matrix (n_cells × n_genes).
        ref_types : array
            Reference cell type labels.

        Returns
        -------
        dict
            Dictionary with validation metrics:
            - expression_similarity: Wasserstein distance between mean expressions (0-1)
            - variance_similarity: Similarity of gene variances (0-1)
            - correlation_similarity: Correlation of gene-gene correlations (0-1)
            - type_proportion_similarity: Similarity of cell type proportions (0-1)
            - count_distribution_similarity: Similarity of count distributions (0-1)
        """
        # Ensure same number of genes
        n_genes = min(sim_counts.shape[1], ref_counts.shape[1])
        sim_counts = sim_counts[:, :n_genes]
        ref_counts = ref_counts[:, :n_genes]

        metrics = {}

        # 1. Mean expression similarity (per gene)
        sim_mean = sim_counts.mean(axis=0)
        ref_mean = ref_counts.mean(axis=0)

        # Use Wasserstein distance, then convert to similarity
        wass_dist = wasserstein_distance(
            np.arange(n_genes), np.arange(n_genes),
            sim_mean / sim_mean.sum(), ref_mean / ref_mean.sum()
        )
        # Normalize and invert (0 = different, 1 = same)
        metrics['expression_similarity'] = 1 / (1 + wass_dist)

        # 2. Variance similarity
        sim_var = sim_counts.var(axis=0)
        ref_var = ref_counts.var(axis=0)

        # Correlation of variances
        if sim_var.std() > 0 and ref_var.std() > 0:
            var_corr = np.corrcoef(sim_var, ref_var)[0, 1]
            metrics['variance_similarity'] = (var_corr + 1) / 2  # Transform to [0, 1]
        else:
            metrics['variance_similarity'] = 0.0

        # 3. Gene-gene correlation structure
        if sim_counts.shape[1] >= 3 and sim_counts.shape[0] >= 3:
            try:
                sim_corr = np.corrcoef(sim_counts.T)
                ref_corr = np.corrcoef(ref_counts.T)

                # Flatten upper triangles
                mask = np.triu_indices_from(sim_corr, k=1)
                sim_corr_flat = sim_corr[mask]
                ref_corr_flat = ref_corr[mask]

                # Correlation of correlations
                if len(sim_corr_flat) > 0:
                    corr_corr = np.corrcoef(sim_corr_flat, ref_corr_flat)[0, 1]
                    metrics['correlation_similarity'] = (corr_corr + 1) / 2
                else:
                    metrics['correlation_similarity'] = 0.5
            except:
                metrics['correlation_similarity'] = 0.5
        else:
            metrics['correlation_similarity'] = 0.5

        # 4. Cell type proportion similarity
        n_types = max(sim_types.max(), ref_types.max()) + 1
        sim_props = np.array([
            np.sum(sim_types == i) / len(sim_types) for i in range(n_types)
        ])
        ref_props = np.array([
            np.sum(ref_types == i) / len(ref_types) for i in range(n_types)
        ])

        # Use 1 - Jensen-Shannon divergence
        def kl_divergence(p, q):
            """KL divergence with smoothing."""
            p = p + 1e-10
            q = q + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            return np.sum(p * np.log(p / q))

        def js_divergence(p, q):
            """Jensen-Shannon divergence."""
            m = (p + q) / 2
            return (kl_divergence(p, m) + kl_divergence(q, m)) / 2

        js_div = js_divergence(sim_props, ref_props)
        metrics['type_proportion_similarity'] = 1 / (1 + js_div)

        # 5. Count distribution similarity (per cell total counts)
        sim_total = sim_counts.sum(axis=1)
        ref_total = ref_counts.sum(axis=1)

        # Use Wasserstein distance on empirical CDFs
        if len(sim_total) > 0 and len(ref_total) > 0:
            count_wass = wasserstein_distance(sim_total, ref_total)
            # Normalize by mean reference count
            ref_mean_count = ref_total.mean()
            if ref_mean_count > 0:
                normalized_wass = count_wass / ref_mean_count
                metrics['count_distribution_similarity'] = 1 / (1 + normalized_wass)
            else:
                metrics['count_distribution_similarity'] = 0.0
        else:
            metrics['count_distribution_similarity'] = 0.0

        return metrics


__all__ = [
    'DifficultyScorer',
    'ValidationMetrics',
]
