"""Validation metrics for comparing simulated and real data."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class ValidationMetrics:
    """Compare simulated data against real reference data.

    This class computes various metrics to assess how well simulated
    data matches real spatial transcriptomics characteristics.

    Parameters
    ----------
    seed : Optional[int], optional
        Random seed for reproducibility.

    Attributes
    ----------
    metrics : dict
        Dictionary of computed validation metrics.

    Examples
    --------
    >>> validator = ValidationMetrics()
    >>> metrics = validator.compare(
    ...     simulated_expression, real_expression,
    ...     simulated_types, real_types
    ... )
    >>> print(f"Expression similarity: {metrics['expression_similarity']:.2f}")
    """

    seed: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Initialize random state."""
        self.rng = np.random.default_rng(seed=self.seed)

    def compare(
        self,
        simulated_expression: NDArray[np.floating],
        real_expression: NDArray[np.floating],
        simulated_types: Optional[NDArray[np.integer]] = None,
        real_types: Optional[NDArray[np.integer]] = None,
        simulated_positions: Optional[NDArray[np.floating]] = None,
        real_positions: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, float]:
        """Compare simulated data to real reference data.

        Parameters
        ----------
        simulated_expression : np.ndarray
            Simulated expression matrix, shape (n_cells_sim, n_genes).
        real_expression : np.ndarray
            Real expression matrix, shape (n_cells_real, n_genes).
        simulated_types : np.ndarray, optional
            Simulated cell type labels.
        real_types : np.ndarray, optional
            Real cell type labels.
        simulated_positions : np.ndarray, optional
            Simulated cell positions, shape (n_cells_sim, 2).
        real_positions : np.ndarray, optional
            Real cell positions, shape (n_cells_real, 2).

        Returns
        -------
        dict
            Dictionary containing:
            - 'expression_similarity': Correlation of gene distributions
            - 'mean_expression_correlation': Correlation of mean expression per gene
            - 'variance_ratio': Ratio of expression variances
            - 'type_proportion_error': MAE of cell type proportions (if types provided)
            - 'spatial_autocorrelation_diff': Difference in Moran's I (if positions provided)
            - 'overall_quality': Combined quality score (0-1, higher = better)
        """
        self.metrics = {}

        # Expression distribution similarity
        self.metrics['expression_similarity'] = self._expression_distribution_similarity(
            simulated_expression, real_expression
        )

        # Mean expression correlation
        self.metrics['mean_expression_correlation'] = self._mean_expression_correlation(
            simulated_expression, real_expression
        )

        # Variance ratio
        self.metrics['variance_ratio'] = self._variance_ratio(
            simulated_expression, real_expression
        )

        # Gene-gene correlation preservation
        self.metrics['correlation_structure_similarity'] = self._correlation_structure_similarity(
            simulated_expression, real_expression
        )

        # Cell type proportion comparison
        if simulated_types is not None and real_types is not None:
            self.metrics['type_proportion_error'] = self._type_proportion_error(
                simulated_types, real_types
            )

        # Spatial statistics comparison
        if simulated_positions is not None and real_positions is not None:
            if simulated_types is not None and real_types is not None:
                self.metrics['spatial_autocorrelation_diff'] = self._spatial_autocorrelation_diff(
                    simulated_positions, real_positions,
                    simulated_types, real_types
                )

        # Compute overall quality score
        self.metrics['overall_quality'] = self._compute_overall_quality()

        return self.metrics

    def _expression_distribution_similarity(
        self,
        simulated: NDArray[np.floating],
        real: NDArray[np.floating],
    ) -> float:
        """Compare overall expression distributions.

        Parameters
        ----------
        simulated : np.ndarray
            Simulated expression.
        real : np.ndarray
            Real expression.

        Returns
        -------
        float
            Similarity score (0-1).
        """
        # Compare histograms of log-expression
        sim_flat = np.log1p(simulated.flatten())
        real_flat = np.log1p(real.flatten())

        # Compute histogram bins from combined data
        all_data = np.concatenate([sim_flat, real_flat])
        bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 50)

        sim_hist, _ = np.histogram(sim_flat, bins=bins, density=True)
        real_hist, _ = np.histogram(real_flat, bins=bins, density=True)

        # Normalize
        sim_hist = sim_hist / (sim_hist.sum() + 1e-10)
        real_hist = real_hist / (real_hist.sum() + 1e-10)

        # Jensen-Shannon divergence (bounded 0-1)
        m = 0.5 * (sim_hist + real_hist)
        with np.errstate(divide='ignore', invalid='ignore'):
            kl_sim = np.nansum(sim_hist * np.log(sim_hist / (m + 1e-10) + 1e-10))
            kl_real = np.nansum(real_hist * np.log(real_hist / (m + 1e-10) + 1e-10))
        js_div = 0.5 * (kl_sim + kl_real)

        # Convert to similarity (1 - normalized JS divergence)
        similarity = 1 - np.clip(js_div / np.log(2), 0, 1)

        return float(similarity)

    def _mean_expression_correlation(
        self,
        simulated: NDArray[np.floating],
        real: NDArray[np.floating],
    ) -> float:
        """Correlate mean expression per gene.

        Parameters
        ----------
        simulated : np.ndarray
            Simulated expression.
        real : np.ndarray
            Real expression.

        Returns
        -------
        float
            Pearson correlation coefficient.
        """
        sim_means = simulated.mean(axis=0)
        real_means = real.mean(axis=0)

        # Handle different number of genes
        n_genes = min(len(sim_means), len(real_means))
        sim_means = sim_means[:n_genes]
        real_means = real_means[:n_genes]

        corr = np.corrcoef(sim_means, real_means)[0, 1]
        return float(np.nan_to_num(corr, nan=0.0))

    def _variance_ratio(
        self,
        simulated: NDArray[np.floating],
        real: NDArray[np.floating],
    ) -> float:
        """Compare expression variance.

        Parameters
        ----------
        simulated : np.ndarray
            Simulated expression.
        real : np.ndarray
            Real expression.

        Returns
        -------
        float
            Ratio of variances (1.0 = perfect match).
        """
        sim_var = simulated.var()
        real_var = real.var()

        if real_var == 0:
            return 0.0

        ratio = sim_var / real_var
        return float(ratio)

    def _correlation_structure_similarity(
        self,
        simulated: NDArray[np.floating],
        real: NDArray[np.floating],
        n_genes_sample: int = 100,
    ) -> float:
        """Compare gene-gene correlation structure.

        Parameters
        ----------
        simulated : np.ndarray
            Simulated expression.
        real : np.ndarray
            Real expression.
        n_genes_sample : int
            Number of genes to sample for correlation matrix.

        Returns
        -------
        float
            Similarity of correlation matrices (0-1).
        """
        n_genes = min(simulated.shape[1], real.shape[1], n_genes_sample)

        # Sample genes if too many
        if simulated.shape[1] > n_genes:
            gene_idx = self.rng.choice(simulated.shape[1], n_genes, replace=False)
        else:
            gene_idx = np.arange(n_genes)

        sim_corr = np.corrcoef(simulated[:, gene_idx].T)
        real_corr = np.corrcoef(real[:, gene_idx].T)

        # Handle NaN
        sim_corr = np.nan_to_num(sim_corr, nan=0.0)
        real_corr = np.nan_to_num(real_corr, nan=0.0)

        # Correlation of upper triangle
        triu_idx = np.triu_indices(n_genes, k=1)
        corr = np.corrcoef(sim_corr[triu_idx], real_corr[triu_idx])[0, 1]

        return float(np.nan_to_num(corr, nan=0.0))

    def _type_proportion_error(
        self,
        simulated_types: NDArray[np.integer],
        real_types: NDArray[np.integer],
    ) -> float:
        """Compare cell type proportions.

        Parameters
        ----------
        simulated_types : np.ndarray
            Simulated cell type labels.
        real_types : np.ndarray
            Real cell type labels.

        Returns
        -------
        float
            Mean absolute error of proportions.
        """
        # Get unique types from both
        all_types = np.unique(np.concatenate([simulated_types, real_types]))
        n_types = len(all_types)

        sim_props = np.zeros(n_types)
        real_props = np.zeros(n_types)

        for i, t in enumerate(all_types):
            sim_props[i] = (simulated_types == t).mean()
            real_props[i] = (real_types == t).mean()

        mae = np.abs(sim_props - real_props).mean()
        return float(mae)

    def _spatial_autocorrelation_diff(
        self,
        sim_positions: NDArray[np.floating],
        real_positions: NDArray[np.floating],
        sim_types: NDArray[np.integer],
        real_types: NDArray[np.integer],
    ) -> float:
        """Compare spatial autocorrelation (Moran's I).

        Parameters
        ----------
        sim_positions : np.ndarray
            Simulated positions.
        real_positions : np.ndarray
            Real positions.
        sim_types : np.ndarray
            Simulated types.
        real_types : np.ndarray
            Real types.

        Returns
        -------
        float
            Absolute difference in Moran's I.
        """
        sim_moran = self._compute_morans_i(sim_positions, sim_types)
        real_moran = self._compute_morans_i(real_positions, real_types)

        return float(abs(sim_moran - real_moran))

    def _compute_morans_i(
        self,
        positions: NDArray[np.floating],
        types: NDArray[np.integer],
        k: int = 10,
    ) -> float:
        """Compute Moran's I for spatial autocorrelation.

        Parameters
        ----------
        positions : np.ndarray
            Cell positions.
        types : np.ndarray
            Cell type labels (converted to numeric).
        k : int
            Number of neighbors for weight matrix.

        Returns
        -------
        float
            Moran's I statistic.
        """
        from scipy.spatial import cKDTree

        n = len(positions)
        if n < k + 1:
            return 0.0

        # Convert types to numeric if needed
        x = types.astype(float)
        x = x - x.mean()

        # Build spatial weight matrix (k nearest neighbors)
        tree = cKDTree(positions)
        _, indices = tree.query(positions, k=k + 1)

        # Compute Moran's I
        numerator = 0.0
        denominator = np.sum(x ** 2)

        if denominator == 0:
            return 0.0

        for i in range(n):
            neighbors = indices[i, 1:]  # Exclude self
            for j in neighbors:
                numerator += x[i] * x[j]

        # Total weight (k neighbors per cell, undirected)
        w_sum = n * k

        morans_i = (n / w_sum) * (numerator / denominator)

        return float(morans_i)

    def _compute_overall_quality(self) -> float:
        """Compute overall quality score.

        Returns
        -------
        float
            Quality score (0-1, higher = better match).
        """
        weights = {
            'expression_similarity': 0.25,
            'mean_expression_correlation': 0.25,
            'correlation_structure_similarity': 0.20,
            'type_proportion_error': -0.15,  # Negative because error
            'spatial_autocorrelation_diff': -0.15,  # Negative because diff
        }

        quality = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in self.metrics:
                value = self.metrics[metric]
                if weight < 0:
                    # For error metrics, convert to quality (lower = better)
                    quality += abs(weight) * (1 - min(value, 1))
                else:
                    # Handle correlations (can be negative)
                    quality += weight * max(0, value)
                total_weight += abs(weight)

        if total_weight > 0:
            quality /= total_weight

        return float(np.clip(quality, 0, 1))

    def compute_per_gene_metrics(
        self,
        simulated: NDArray[np.floating],
        real: NDArray[np.floating],
        gene_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each gene.

        Parameters
        ----------
        simulated : np.ndarray
            Simulated expression.
        real : np.ndarray
            Real expression.
        gene_names : list, optional
            Gene names. If None, uses indices.

        Returns
        -------
        dict
            Per-gene metrics dictionary.
        """
        n_genes = min(simulated.shape[1], real.shape[1])

        if gene_names is None:
            gene_names = [str(i) for i in range(n_genes)]

        per_gene = {}
        for i in range(n_genes):
            sim_gene = simulated[:, i]
            real_gene = real[:, i]

            per_gene[gene_names[i]] = {
                'mean_ratio': float(sim_gene.mean() / (real_gene.mean() + 1e-6)),
                'variance_ratio': float(sim_gene.var() / (real_gene.var() + 1e-6)),
                'ks_statistic': self._ks_statistic(sim_gene, real_gene),
            }

        return per_gene

    def _ks_statistic(
        self,
        sample1: NDArray[np.floating],
        sample2: NDArray[np.floating],
    ) -> float:
        """Compute Kolmogorov-Smirnov statistic.

        Parameters
        ----------
        sample1 : np.ndarray
            First sample.
        sample2 : np.ndarray
            Second sample.

        Returns
        -------
        float
            KS statistic.
        """
        try:
            from scipy.stats import ks_2samp
            stat, _ = ks_2samp(sample1, sample2)
            return float(stat)
        except ImportError:
            # Fallback: compare CDFs manually
            all_values = np.sort(np.concatenate([sample1, sample2]))
            cdf1 = np.searchsorted(np.sort(sample1), all_values) / len(sample1)
            cdf2 = np.searchsorted(np.sort(sample2), all_values) / len(sample2)
            return float(np.max(np.abs(cdf1 - cdf2)))

    def summary(self) -> Dict[str, Any]:
        """Get summary of validation results.

        Returns
        -------
        dict
            Summary with interpretation.
        """
        quality = self.metrics.get('overall_quality', 0.5)

        if quality > 0.8:
            interpretation = "Excellent - simulated data closely matches real data"
        elif quality > 0.6:
            interpretation = "Good - simulated data reasonably matches real data"
        elif quality > 0.4:
            interpretation = "Fair - some discrepancies between simulated and real data"
        else:
            interpretation = "Poor - significant differences from real data"

        return {
            'metrics': self.metrics.copy(),
            'interpretation': interpretation,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'seed': self.seed,
            'metrics': self.metrics.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationMetrics":
        """Create from dictionary."""
        validator = cls(seed=data.get('seed'))
        validator.metrics = data.get('metrics', {})
        return validator
