"""Batch effect modeling for spatial transcriptomics simulations."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from numpy.typing import NDArray


@dataclass
class BatchEffectModel:
    """Model for simulating technical batch effects in spatial transcriptomics.

    Batch effects are systematic technical variations that affect gene expression
    measurements. They can arise from differences in sample processing, imaging
    conditions, or other technical factors between FOVs or experimental batches.

    This class generates multiplicative and additive batch effects that can be
    applied to gene expression matrices to simulate realistic technical variation.

    Parameters
    ----------
    n_genes : int
        Number of genes in the expression matrix.
    n_batches : int, optional
        Number of distinct batches. Default is 1.
    global_effect_magnitude : float, optional
        Magnitude of global (all-gene) batch effects as log2 fold change.
        Default is 0.3.
    gene_effect_magnitude : float, optional
        Magnitude of gene-specific batch effects as log2 fold change.
        Default is 0.5.
    additive_noise_scale : float, optional
        Scale of additive noise component. Default is 0.1.
    correlation_within_batch : float, optional
        Correlation of effects within a batch (0-1). Higher values mean more
        genes are affected similarly. Default is 0.5.
    seed : Optional[int], optional
        Random seed for reproducibility.

    Attributes
    ----------
    multiplicative_effects : np.ndarray
        Multiplicative batch effects matrix, shape (n_batches, n_genes).
    additive_effects : np.ndarray
        Additive batch effects matrix, shape (n_batches, n_genes).
    global_effects : np.ndarray
        Global per-batch scaling factors, shape (n_batches,).

    Examples
    --------
    >>> batch_model = BatchEffectModel(n_genes=100, n_batches=3)
    >>> batch_model.generate_effects()
    >>> # Apply to expression matrix for batch 0
    >>> corrected = batch_model.apply(expression_matrix, batch_id=0)
    """

    n_genes: int
    n_batches: int = 1
    global_effect_magnitude: float = 0.3
    gene_effect_magnitude: float = 0.5
    additive_noise_scale: float = 0.1
    correlation_within_batch: float = 0.5
    seed: Optional[int] = None

    # Generated effects (set by generate_effects())
    multiplicative_effects: NDArray[np.floating] = field(default=None, repr=False)
    additive_effects: NDArray[np.floating] = field(default=None, repr=False)
    global_effects: NDArray[np.floating] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize random state and validate parameters."""
        self.rng = np.random.default_rng(seed=self.seed)

        if self.n_genes <= 0:
            raise ValueError("n_genes must be positive")
        if self.n_batches <= 0:
            raise ValueError("n_batches must be positive")
        if not 0 <= self.correlation_within_batch <= 1:
            raise ValueError("correlation_within_batch must be in [0, 1]")

    def generate_effects(self) -> "BatchEffectModel":
        """Generate batch effect parameters.

        This method must be called before apply() to generate the effect matrices.

        Returns
        -------
        BatchEffectModel
            Self, for method chaining.
        """
        # Global per-batch effects (affects all genes uniformly)
        self.global_effects = 2 ** self.rng.normal(
            0, self.global_effect_magnitude, size=self.n_batches
        )

        # Gene-specific multiplicative effects
        # Use a factor model to introduce correlation
        n_factors = max(1, int(self.n_genes * (1 - self.correlation_within_batch)))

        if self.correlation_within_batch > 0:
            # Generate latent factors
            factors = self.rng.normal(0, 1, size=(self.n_batches, n_factors))
            loadings = self.rng.normal(0, 1, size=(n_factors, self.n_genes))

            # Generate correlated effects
            correlated = factors @ loadings
            correlated = correlated / np.std(correlated) * self.gene_effect_magnitude

            # Add independent noise
            independent = self.rng.normal(
                0, self.gene_effect_magnitude * (1 - self.correlation_within_batch),
                size=(self.n_batches, self.n_genes)
            )

            log2_effects = correlated + independent
        else:
            log2_effects = self.rng.normal(
                0, self.gene_effect_magnitude,
                size=(self.n_batches, self.n_genes)
            )

        self.multiplicative_effects = 2 ** log2_effects

        # Additive effects (background-like noise)
        self.additive_effects = np.abs(self.rng.normal(
            0, self.additive_noise_scale,
            size=(self.n_batches, self.n_genes)
        ))

        return self

    def apply(
        self,
        expression: NDArray[np.floating],
        batch_id: int = 0,
        mode: str = "multiplicative"
    ) -> NDArray[np.floating]:
        """Apply batch effects to an expression matrix.

        Parameters
        ----------
        expression : np.ndarray
            Gene expression matrix, shape (n_cells, n_genes).
        batch_id : int, optional
            Batch index to apply. Default is 0.
        mode : str, optional
            Effect mode: 'multiplicative', 'additive', or 'both'. Default is 'multiplicative'.

        Returns
        -------
        np.ndarray
            Expression matrix with batch effects applied.

        Raises
        ------
        ValueError
            If effects haven't been generated or batch_id is invalid.
        """
        if self.multiplicative_effects is None:
            raise ValueError("Effects not generated. Call generate_effects() first.")

        if not 0 <= batch_id < self.n_batches:
            raise ValueError(f"batch_id must be in [0, {self.n_batches})")

        result = expression.copy().astype(float)

        if mode in ("multiplicative", "both"):
            result = result * self.global_effects[batch_id]
            result = result * self.multiplicative_effects[batch_id]

        if mode in ("additive", "both"):
            result = result + self.additive_effects[batch_id]

        return np.maximum(result, 0)

    def get_batch_effect_summary(self) -> Dict[str, Any]:
        """Get summary statistics of generated batch effects.

        Returns
        -------
        dict
            Dictionary with effect statistics.
        """
        if self.multiplicative_effects is None:
            return {"status": "effects not generated"}

        return {
            "n_batches": self.n_batches,
            "n_genes": self.n_genes,
            "global_effect_range": (
                float(np.min(self.global_effects)),
                float(np.max(self.global_effects))
            ),
            "multiplicative_effect_range": (
                float(np.min(self.multiplicative_effects)),
                float(np.max(self.multiplicative_effects))
            ),
            "additive_effect_mean": float(np.mean(self.additive_effects)),
            "log2_fold_change_std": float(np.std(np.log2(self.multiplicative_effects))),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize batch effect model to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the model.
        """
        return {
            "n_genes": self.n_genes,
            "n_batches": self.n_batches,
            "global_effect_magnitude": self.global_effect_magnitude,
            "gene_effect_magnitude": self.gene_effect_magnitude,
            "additive_noise_scale": self.additive_noise_scale,
            "correlation_within_batch": self.correlation_within_batch,
            "seed": self.seed,
            "multiplicative_effects": (
                self.multiplicative_effects.tolist()
                if self.multiplicative_effects is not None else None
            ),
            "additive_effects": (
                self.additive_effects.tolist()
                if self.additive_effects is not None else None
            ),
            "global_effects": (
                self.global_effects.tolist()
                if self.global_effects is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchEffectModel":
        """Create BatchEffectModel from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of the model.

        Returns
        -------
        BatchEffectModel
            Reconstructed model.
        """
        model = cls(
            n_genes=data["n_genes"],
            n_batches=data["n_batches"],
            global_effect_magnitude=data.get("global_effect_magnitude", 0.3),
            gene_effect_magnitude=data.get("gene_effect_magnitude", 0.5),
            additive_noise_scale=data.get("additive_noise_scale", 0.1),
            correlation_within_batch=data.get("correlation_within_batch", 0.5),
            seed=data.get("seed"),
        )

        if data.get("multiplicative_effects") is not None:
            model.multiplicative_effects = np.array(data["multiplicative_effects"])
        if data.get("additive_effects") is not None:
            model.additive_effects = np.array(data["additive_effects"])
        if data.get("global_effects") is not None:
            model.global_effects = np.array(data["global_effects"])

        return model
