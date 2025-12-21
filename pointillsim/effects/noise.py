"""Noise models for spatial transcriptomics simulations."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray


@dataclass
class TechnicalNoise:
    """Model for technical noise sources in spatial transcriptomics imaging.

    Simulates various sources of technical variation including:
    - Optical vignetting (non-uniform illumination)
    - Amplification efficiency variation
    - Focus-dependent detection efficiency

    Parameters
    ----------
    frame_size : int
        Size of the FOV in pixels.
    vignetting_strength : float, optional
        Strength of vignetting effect (0=none, 1=strong). Default is 0.2.
    amplification_cv : float, optional
        Coefficient of variation for amplification efficiency. Default is 0.15.
    focus_variation : float, optional
        Spatial scale of focus-related variation. Default is 0.1.
    seed : Optional[int], optional
        Random seed for reproducibility.

    Attributes
    ----------
    vignetting_map : np.ndarray
        2D map of vignetting effects, shape (frame_size, frame_size).
    amplification_factors : np.ndarray
        Per-cell amplification efficiency factors (generated on apply).

    Examples
    --------
    >>> noise_model = TechnicalNoise(frame_size=1000)
    >>> noise_model.generate_vignetting()
    >>> # Apply to expression matrix
    >>> noisy = noise_model.apply(expression_matrix, cell_positions)
    """

    frame_size: int
    vignetting_strength: float = 0.2
    amplification_cv: float = 0.15
    focus_variation: float = 0.1
    seed: Optional[int] = None

    # Generated maps
    vignetting_map: NDArray[np.floating] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize random state."""
        self.rng = np.random.default_rng(seed=self.seed)

    def generate_vignetting(self) -> "TechnicalNoise":
        """Generate vignetting map simulating optical field non-uniformity.

        Creates a radial falloff pattern typical of microscope optics.

        Returns
        -------
        TechnicalNoise
            Self, for method chaining.
        """
        # Create coordinate grid
        x = np.linspace(-1, 1, self.frame_size)
        y = np.linspace(-1, 1, self.frame_size)
        X, Y = np.meshgrid(x, y)

        # Radial distance from center
        R = np.sqrt(X**2 + Y**2)

        # Vignetting: falloff toward edges (cos^4 approximation)
        self.vignetting_map = 1 - self.vignetting_strength * (R / np.sqrt(2))**4

        return self

    def get_vignetting_at_positions(
        self,
        positions: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Get vignetting factors at specific cell positions.

        Parameters
        ----------
        positions : np.ndarray
            Cell centroid positions, shape (n_cells, 2).

        Returns
        -------
        np.ndarray
            Vignetting factors for each cell position.
        """
        if self.vignetting_map is None:
            self.generate_vignetting()

        # Convert positions to grid indices
        indices = np.clip(
            positions.astype(int),
            0, self.frame_size - 1
        )

        return self.vignetting_map[indices[:, 1], indices[:, 0]]

    def generate_amplification_factors(
        self,
        n_cells: int
    ) -> NDArray[np.floating]:
        """Generate per-cell amplification efficiency factors.

        Parameters
        ----------
        n_cells : int
            Number of cells.

        Returns
        -------
        np.ndarray
            Amplification factors for each cell.
        """
        # Log-normal distribution for multiplicative variation
        log_mean = 0
        log_std = np.sqrt(np.log(1 + self.amplification_cv**2))

        return self.rng.lognormal(log_mean, log_std, size=n_cells)

    def apply(
        self,
        expression: NDArray[np.floating],
        positions: NDArray[np.floating],
        apply_vignetting: bool = True,
        apply_amplification: bool = True,
    ) -> NDArray[np.floating]:
        """Apply technical noise to expression matrix.

        Parameters
        ----------
        expression : np.ndarray
            Gene expression matrix, shape (n_cells, n_genes).
        positions : np.ndarray
            Cell centroid positions, shape (n_cells, 2).
        apply_vignetting : bool, optional
            Whether to apply vignetting. Default is True.
        apply_amplification : bool, optional
            Whether to apply amplification variation. Default is True.

        Returns
        -------
        np.ndarray
            Expression matrix with technical noise applied.
        """
        result = expression.copy().astype(float)
        n_cells = len(expression)

        if apply_vignetting:
            vignetting_factors = self.get_vignetting_at_positions(positions)
            result = result * vignetting_factors[:, np.newaxis]

        if apply_amplification:
            amp_factors = self.generate_amplification_factors(n_cells)
            result = result * amp_factors[:, np.newaxis]

        return np.maximum(result, 0)


@dataclass
class BackgroundNoise:
    """Model for background transcript noise (false positives).

    Simulates random false positive transcript detections that are not
    associated with any cell. These can arise from autofluorescence,
    optical aberrations, or mislocalized transcripts.

    Parameters
    ----------
    frame_size : int
        Size of the FOV in pixels.
    background_rate : float, optional
        Mean number of background dots per unit area. Default is 0.001.
    gene_bias : Optional[np.ndarray], optional
        Per-gene probability weights for background. If None, uniform.
    spatial_pattern : str, optional
        Spatial pattern of background: 'uniform', 'clustered', or 'gradient'.
        Default is 'uniform'.
    seed : Optional[int], optional
        Random seed for reproducibility.

    Examples
    --------
    >>> bg_noise = BackgroundNoise(frame_size=1000, background_rate=0.01)
    >>> bg_dots_x, bg_dots_y, bg_genes = bg_noise.generate_background_dots(n_genes=100)
    """

    frame_size: int
    background_rate: float = 0.001
    gene_bias: Optional[NDArray[np.floating]] = None
    spatial_pattern: str = "uniform"
    seed: Optional[int] = None

    def __post_init__(self):
        """Initialize random state."""
        self.rng = np.random.default_rng(seed=self.seed)

    def generate_background_dots(
        self,
        n_genes: int,
        gene_names: Optional[list] = None,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.integer]]:
        """Generate random background transcript dots.

        Parameters
        ----------
        n_genes : int
            Number of genes in the panel.
        gene_names : Optional[list], optional
            Gene names for output. If None, uses indices.

        Returns
        -------
        tuple
            (x_positions, y_positions, gene_indices) arrays for background dots.
        """
        # Calculate expected number of background dots
        area = self.frame_size ** 2
        expected_dots = int(self.background_rate * area)
        n_dots = self.rng.poisson(expected_dots)

        if n_dots == 0:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=int)
            )

        # Generate positions based on spatial pattern
        if self.spatial_pattern == "uniform":
            x = self.rng.uniform(0, self.frame_size, n_dots)
            y = self.rng.uniform(0, self.frame_size, n_dots)

        elif self.spatial_pattern == "clustered":
            # Generate cluster centers
            n_clusters = max(1, n_dots // 10)
            cluster_x = self.rng.uniform(0, self.frame_size, n_clusters)
            cluster_y = self.rng.uniform(0, self.frame_size, n_clusters)
            cluster_assignments = self.rng.choice(n_clusters, n_dots)

            x = cluster_x[cluster_assignments] + self.rng.normal(0, self.frame_size / 20, n_dots)
            y = cluster_y[cluster_assignments] + self.rng.normal(0, self.frame_size / 20, n_dots)
            x = np.clip(x, 0, self.frame_size)
            y = np.clip(y, 0, self.frame_size)

        elif self.spatial_pattern == "gradient":
            # Higher background toward one edge
            x = self.rng.uniform(0, self.frame_size, n_dots)
            # Weight y toward bottom
            y = self.frame_size * np.sqrt(self.rng.uniform(0, 1, n_dots))

        else:
            raise ValueError(f"Unknown spatial_pattern: {self.spatial_pattern}")

        # Assign genes
        if self.gene_bias is not None:
            gene_probs = self.gene_bias / self.gene_bias.sum()
        else:
            gene_probs = np.ones(n_genes) / n_genes

        genes = self.rng.choice(n_genes, size=n_dots, p=gene_probs)

        return x, y, genes

    def add_to_dots_df(
        self,
        dots_x: NDArray[np.floating],
        dots_y: NDArray[np.floating],
        dots_gene: NDArray,
        n_genes: int,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray, NDArray[np.integer]]:
        """Add background dots to existing transcript data.

        Parameters
        ----------
        dots_x : np.ndarray
            Existing dot x positions.
        dots_y : np.ndarray
            Existing dot y positions.
        dots_gene : np.ndarray
            Existing dot gene assignments.
        n_genes : int
            Number of genes.

        Returns
        -------
        tuple
            (x, y, gene, is_background) arrays with background dots added.
        """
        bg_x, bg_y, bg_genes = self.generate_background_dots(n_genes)

        # Create is_background indicator
        is_bg = np.zeros(len(dots_x), dtype=bool)
        is_bg_new = np.ones(len(bg_x), dtype=bool)

        return (
            np.concatenate([dots_x, bg_x]),
            np.concatenate([dots_y, bg_y]),
            np.concatenate([dots_gene, bg_genes]),
            np.concatenate([is_bg, is_bg_new]),
        )


@dataclass
class DropoutModel:
    """Model for gene-specific detection dropout in spatial transcriptomics.

    Simulates the failure to detect transcripts, which can depend on:
    - Expression level (lower expression → higher dropout)
    - Gene-specific detection efficiency
    - Cell-specific factors

    Parameters
    ----------
    baseline_detection_rate : float, optional
        Baseline probability of detecting a transcript. Default is 0.9.
    expression_dependence : float, optional
        How much dropout depends on expression level (0=none, 1=strong).
        Default is 0.3.
    gene_variation_cv : float, optional
        Coefficient of variation for gene-specific detection. Default is 0.2.
    seed : Optional[int], optional
        Random seed for reproducibility.

    Attributes
    ----------
    gene_detection_rates : np.ndarray
        Per-gene detection rate factors (generated on apply).

    Examples
    --------
    >>> dropout = DropoutModel(baseline_detection_rate=0.85)
    >>> observed = dropout.apply(true_expression)
    """

    baseline_detection_rate: float = 0.9
    expression_dependence: float = 0.3
    gene_variation_cv: float = 0.2
    seed: Optional[int] = None

    gene_detection_rates: NDArray[np.floating] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize random state."""
        self.rng = np.random.default_rng(seed=self.seed)

        if not 0 <= self.baseline_detection_rate <= 1:
            raise ValueError("baseline_detection_rate must be in [0, 1]")
        if not 0 <= self.expression_dependence <= 1:
            raise ValueError("expression_dependence must be in [0, 1]")

    def generate_gene_detection_rates(self, n_genes: int) -> "DropoutModel":
        """Generate gene-specific detection rate factors.

        Parameters
        ----------
        n_genes : int
            Number of genes.

        Returns
        -------
        DropoutModel
            Self, for method chaining.
        """
        # Log-normal variation in detection rates
        log_std = np.sqrt(np.log(1 + self.gene_variation_cv**2))
        self.gene_detection_rates = np.clip(
            self.rng.lognormal(0, log_std, n_genes),
            0.5, 2.0  # Reasonable bounds
        )
        return self

    def compute_detection_probabilities(
        self,
        expression: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute detection probability for each transcript.

        Detection probability increases with expression level following
        a saturating curve.

        Parameters
        ----------
        expression : np.ndarray
            Gene expression matrix, shape (n_cells, n_genes).

        Returns
        -------
        np.ndarray
            Detection probability matrix, same shape as expression.
        """
        n_genes = expression.shape[1]

        if self.gene_detection_rates is None or len(self.gene_detection_rates) != n_genes:
            self.generate_gene_detection_rates(n_genes)

        # Expression-dependent term: sigmoid function
        # Higher expression → less dropout
        if self.expression_dependence > 0:
            # Normalize expression for sigmoid
            expr_normalized = expression / (expression.mean() + 1e-6)
            expr_factor = 1 / (1 + np.exp(-2 * (expr_normalized - 0.5)))
            expr_adjustment = 1 - self.expression_dependence * (1 - expr_factor)
        else:
            expr_adjustment = 1.0

        # Combine baseline, gene-specific, and expression-dependent factors
        detection_prob = (
            self.baseline_detection_rate *
            self.gene_detection_rates *
            expr_adjustment
        )

        return np.clip(detection_prob, 0, 1)

    def apply(
        self,
        expression: NDArray[np.floating],
        return_mask: bool = False
    ) -> NDArray[np.floating]:
        """Apply dropout to expression matrix.

        Parameters
        ----------
        expression : np.ndarray
            Gene expression matrix, shape (n_cells, n_genes).
        return_mask : bool, optional
            If True, also return the detection mask. Default is False.

        Returns
        -------
        np.ndarray or tuple
            Expression matrix with dropout applied. If return_mask=True,
            returns (expression, mask) where mask is boolean array of detections.
        """
        detection_probs = self.compute_detection_probabilities(expression)

        # For each count, sample whether it's detected
        # This applies dropout proportionally to counts
        detected = expression.copy().astype(float)

        # Apply as multiplicative factor (more robust for counts)
        mask = self.rng.random(expression.shape) < detection_probs
        detected = detected * mask

        if return_mask:
            return detected, mask
        return detected

    def to_dict(self) -> Dict[str, Any]:
        """Serialize dropout model to dictionary."""
        return {
            "baseline_detection_rate": self.baseline_detection_rate,
            "expression_dependence": self.expression_dependence,
            "gene_variation_cv": self.gene_variation_cv,
            "seed": self.seed,
            "gene_detection_rates": (
                self.gene_detection_rates.tolist()
                if self.gene_detection_rates is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DropoutModel":
        """Create DropoutModel from dictionary."""
        model = cls(
            baseline_detection_rate=data.get("baseline_detection_rate", 0.9),
            expression_dependence=data.get("expression_dependence", 0.3),
            gene_variation_cv=data.get("gene_variation_cv", 0.2),
            seed=data.get("seed"),
        )
        if data.get("gene_detection_rates") is not None:
            model.gene_detection_rates = np.array(data["gene_detection_rates"])
        return model
