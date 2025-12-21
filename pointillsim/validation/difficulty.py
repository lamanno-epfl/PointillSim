"""Difficulty scoring for simulated spatial transcriptomics data."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from numpy.typing import NDArray
import warnings


@dataclass
class DifficultyScorer:
    """Assess the difficulty of cell type classification in simulated data.

    This class estimates how challenging a classification task is based on
    the separability of cell types in expression space and spatial complexity.

    Parameters
    ----------
    use_classifier : bool, optional
        Whether to use a classifier-based difficulty estimate. Default is True.
        If True, requires scikit-learn. Falls back to statistical measures if not available.
    n_splits : int, optional
        Number of cross-validation splits for classifier-based scoring. Default is 5.
    seed : Optional[int], optional
        Random seed for reproducibility.

    Attributes
    ----------
    scores : dict
        Dictionary of computed difficulty scores.
    feature_importances : np.ndarray
        Gene importance scores from classifier (if available).

    Examples
    --------
    >>> scorer = DifficultyScorer()
    >>> scores = scorer.score(expression_matrix, cell_types)
    >>> print(f"Overall difficulty: {scores['overall']:.2f}")
    """

    use_classifier: bool = True
    n_splits: int = 5
    seed: Optional[int] = None

    scores: Dict[str, float] = field(default_factory=dict, repr=False)
    feature_importances: NDArray[np.floating] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize random state."""
        self.rng = np.random.default_rng(seed=self.seed)

    def score(
        self,
        expression: NDArray[np.floating],
        cell_types: NDArray[np.integer],
        positions: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, float]:
        """Compute difficulty scores for a dataset.

        Parameters
        ----------
        expression : np.ndarray
            Gene expression matrix, shape (n_cells, n_genes).
        cell_types : np.ndarray
            Cell type labels, shape (n_cells,).
        positions : np.ndarray, optional
            Cell positions, shape (n_cells, 2). If provided, computes
            spatial complexity metrics.

        Returns
        -------
        dict
            Dictionary containing:
            - 'separability': Class separability score (0-1, higher = easier)
            - 'classifier_accuracy': Cross-validated accuracy if use_classifier=True
            - 'spatial_complexity': Spatial pattern complexity if positions provided
            - 'class_imbalance': Imbalance ratio of cell types
            - 'overall': Combined difficulty score (0-1, higher = harder)
        """
        self.scores = {}

        # Basic statistics
        n_cells, n_genes = expression.shape
        unique_types = np.unique(cell_types)
        n_types = len(unique_types)

        if n_cells < 10 or n_types < 2:
            self.scores = {
                'separability': 0.0,
                'classifier_accuracy': 0.0,
                'spatial_complexity': 0.0,
                'class_imbalance': 1.0,
                'overall': 1.0,
            }
            return self.scores

        # Class imbalance
        type_counts = np.bincount(cell_types.astype(int), minlength=n_types)
        type_counts = type_counts[type_counts > 0]
        imbalance = 1 - (type_counts.min() / type_counts.max())
        self.scores['class_imbalance'] = imbalance

        # Separability (simplified Fisher's criterion)
        separability = self._compute_separability(expression, cell_types)
        self.scores['separability'] = separability

        # Classifier-based difficulty
        if self.use_classifier:
            try:
                accuracy = self._classifier_accuracy(expression, cell_types)
                self.scores['classifier_accuracy'] = accuracy
            except ImportError:
                warnings.warn("scikit-learn not available, skipping classifier-based scoring")
                self.scores['classifier_accuracy'] = separability

        # Spatial complexity
        if positions is not None:
            spatial_complexity = self._compute_spatial_complexity(positions, cell_types)
            self.scores['spatial_complexity'] = spatial_complexity
        else:
            self.scores['spatial_complexity'] = 0.5  # Default

        # Overall difficulty (0 = easy, 1 = hard)
        # Low separability, low accuracy, high imbalance, high spatial complexity = hard
        self.scores['overall'] = self._compute_overall_difficulty()

        return self.scores

    def _compute_separability(
        self,
        expression: NDArray[np.floating],
        cell_types: NDArray[np.integer],
    ) -> float:
        """Compute class separability using Fisher's criterion.

        Parameters
        ----------
        expression : np.ndarray
            Expression matrix.
        cell_types : np.ndarray
            Cell type labels.

        Returns
        -------
        float
            Separability score (0-1, higher = more separable).
        """
        unique_types = np.unique(cell_types)
        n_types = len(unique_types)

        if n_types < 2:
            return 0.0

        # Compute class means
        class_means = np.zeros((n_types, expression.shape[1]))
        class_vars = np.zeros((n_types, expression.shape[1]))

        for i, ct in enumerate(unique_types):
            mask = cell_types == ct
            if mask.sum() > 0:
                class_means[i] = expression[mask].mean(axis=0)
                class_vars[i] = expression[mask].var(axis=0) + 1e-6

        # Between-class variance
        global_mean = expression.mean(axis=0)
        between_var = np.mean((class_means - global_mean) ** 2)

        # Within-class variance
        within_var = np.mean(class_vars)

        # Fisher ratio
        fisher_ratio = between_var / (within_var + 1e-6)

        # Normalize to 0-1 range (using sigmoid-like transformation)
        separability = fisher_ratio / (fisher_ratio + 1)

        return float(separability)

    def _classifier_accuracy(
        self,
        expression: NDArray[np.floating],
        cell_types: NDArray[np.integer],
    ) -> float:
        """Estimate classification accuracy using cross-validation.

        Parameters
        ----------
        expression : np.ndarray
            Expression matrix.
        cell_types : np.ndarray
            Cell type labels.

        Returns
        -------
        float
            Cross-validated accuracy.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError:
            raise ImportError("scikit-learn required for classifier-based scoring")

        # Use a simple random forest
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=self.seed,
            n_jobs=-1,
        )

        # Cross-validated accuracy
        n_splits = min(self.n_splits, len(np.unique(cell_types)))
        scores = cross_val_score(clf, expression, cell_types, cv=n_splits)

        # Fit once more to get feature importances
        clf.fit(expression, cell_types)
        self.feature_importances = clf.feature_importances_

        return float(scores.mean())

    def _compute_spatial_complexity(
        self,
        positions: NDArray[np.floating],
        cell_types: NDArray[np.integer],
    ) -> float:
        """Compute spatial pattern complexity.

        Higher complexity = harder to predict from spatial context.

        Parameters
        ----------
        positions : np.ndarray
            Cell positions, shape (n_cells, 2).
        cell_types : np.ndarray
            Cell type labels.

        Returns
        -------
        float
            Spatial complexity score (0-1).
        """
        from scipy.spatial import cKDTree

        n_cells = len(positions)
        if n_cells < 10:
            return 0.5

        # Build KD-tree for neighbor lookups
        tree = cKDTree(positions)

        # For each cell, check how many neighbors are same type
        k = min(10, n_cells - 1)  # k nearest neighbors
        _, indices = tree.query(positions, k=k + 1)  # +1 because includes self

        # Compute neighbor type agreement
        same_type_ratio = np.zeros(n_cells)
        for i in range(n_cells):
            neighbors = indices[i, 1:]  # Exclude self
            neighbor_types = cell_types[neighbors]
            same_type_ratio[i] = (neighbor_types == cell_types[i]).mean()

        # Average spatial coherence
        coherence = same_type_ratio.mean()

        # Complexity is inverse of coherence (random = high complexity)
        # Random arrangement would give ~1/n_types coherence
        n_types = len(np.unique(cell_types))
        random_baseline = 1.0 / n_types
        complexity = 1 - (coherence - random_baseline) / (1 - random_baseline + 1e-6)
        complexity = np.clip(complexity, 0, 1)

        return float(complexity)

    def _compute_overall_difficulty(self) -> float:
        """Compute overall difficulty from component scores.

        Returns
        -------
        float
            Overall difficulty (0 = easy, 1 = hard).
        """
        # Invert separability (high separability = easy)
        sep_difficulty = 1 - self.scores.get('separability', 0.5)

        # Invert accuracy (high accuracy = easy)
        acc_difficulty = 1 - self.scores.get('classifier_accuracy', 0.5)

        # Imbalance directly contributes to difficulty
        imbalance = self.scores.get('class_imbalance', 0.5)

        # Spatial complexity directly contributes
        spatial = self.scores.get('spatial_complexity', 0.5)

        # Weighted average
        overall = (
            0.3 * sep_difficulty +
            0.3 * acc_difficulty +
            0.2 * imbalance +
            0.2 * spatial
        )

        return float(np.clip(overall, 0, 1))

    def get_gene_importance(self, gene_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get gene importance scores from classifier.

        Parameters
        ----------
        gene_names : list, optional
            Gene names. If None, uses indices.

        Returns
        -------
        dict
            Gene name/index to importance score mapping.
        """
        if self.feature_importances is None:
            return {}

        if gene_names is None:
            gene_names = [str(i) for i in range(len(self.feature_importances))]

        return dict(zip(gene_names, self.feature_importances))

    def summary(self) -> Dict[str, Any]:
        """Get summary of difficulty assessment.

        Returns
        -------
        dict
            Summary including scores and interpretation.
        """
        overall = self.scores.get('overall', 0.5)

        if overall < 0.3:
            interpretation = "Easy - cell types are well-separated"
        elif overall < 0.6:
            interpretation = "Moderate - some overlap between cell types"
        else:
            interpretation = "Hard - significant overlap or complexity"

        return {
            'scores': self.scores.copy(),
            'interpretation': interpretation,
            'n_informative_genes': (
                np.sum(self.feature_importances > 0.01)
                if self.feature_importances is not None else None
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'use_classifier': self.use_classifier,
            'n_splits': self.n_splits,
            'seed': self.seed,
            'scores': self.scores.copy(),
            'feature_importances': (
                self.feature_importances.tolist()
                if self.feature_importances is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DifficultyScorer":
        """Create from dictionary."""
        scorer = cls(
            use_classifier=data.get('use_classifier', True),
            n_splits=data.get('n_splits', 5),
            seed=data.get('seed'),
        )
        scorer.scores = data.get('scores', {})
        if data.get('feature_importances') is not None:
            scorer.feature_importances = np.array(data['feature_importances'])
        return scorer
