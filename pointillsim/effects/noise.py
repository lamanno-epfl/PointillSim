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


@dataclass
class SpatialNoise:
    """Model for spatially varying noise levels across a FOV.

    Simulates the common observation that some regions of a FOV (or entire FOVs)
    have higher or lower noise levels due to technical issues such as:
    - Uneven focus across the field
    - Tissue folding or debris
    - Regional autofluorescence
    - Edge effects

    Creates a spatial noise field that can be used to modulate expression
    or detection efficiency across positions.

    Parameters
    ----------
    frame_size : int
        Size of the FOV in pixels.
    noise_pattern : str
        Type of spatial noise pattern:
        - 'gradient': Linear gradient across FOV
        - 'radial': Radial pattern from center
        - 'patches': Random patch-based variation
        - 'perlin': Perlin-like smooth noise
        Default is 'patches'.
    base_level : float
        Baseline noise/efficiency level (1.0 = no effect). Default 1.0.
    variation_strength : float
        Strength of spatial variation (0 = none, 1 = strong). Default 0.3.
    n_patches : int
        Number of noise patches (for 'patches' pattern). Default 5.
    gradient_direction : float
        Direction of gradient in radians (for 'gradient' pattern). Default 0.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    noise_field : np.ndarray
        2D spatial noise field, shape (frame_size, frame_size).

    Examples
    --------
    >>> spatial_noise = SpatialNoise(frame_size=1000, noise_pattern='patches')
    >>> spatial_noise.generate_field()
    >>> factors = spatial_noise.get_factors_at_positions(cell_positions)
    >>> noisy_expression = expression * factors[:, np.newaxis]
    """

    frame_size: int
    noise_pattern: str = "patches"
    base_level: float = 1.0
    variation_strength: float = 0.3
    n_patches: int = 5
    gradient_direction: float = 0.0
    seed: Optional[int] = None
    noise_field: NDArray[np.floating] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize random state and validate parameters."""
        self.rng = np.random.default_rng(seed=self.seed)

        valid_patterns = {'gradient', 'radial', 'patches', 'perlin'}
        if self.noise_pattern not in valid_patterns:
            raise ValueError(
                f"noise_pattern must be one of {valid_patterns}, "
                f"got '{self.noise_pattern}'"
            )

        if not 0 <= self.variation_strength <= 1:
            raise ValueError("variation_strength must be in [0, 1]")

    def generate_field(self) -> "SpatialNoise":
        """Generate the spatial noise field.

        Returns
        -------
        SpatialNoise
            Self, for method chaining.
        """
        if self.noise_pattern == "gradient":
            self.noise_field = self._generate_gradient()
        elif self.noise_pattern == "radial":
            self.noise_field = self._generate_radial()
        elif self.noise_pattern == "patches":
            self.noise_field = self._generate_patches()
        elif self.noise_pattern == "perlin":
            self.noise_field = self._generate_perlin_like()

        return self

    def _generate_gradient(self) -> NDArray[np.floating]:
        """Generate linear gradient pattern."""
        x = np.linspace(-1, 1, self.frame_size)
        y = np.linspace(-1, 1, self.frame_size)
        X, Y = np.meshgrid(x, y)

        # Gradient in specified direction
        gradient = (
            np.cos(self.gradient_direction) * X +
            np.sin(self.gradient_direction) * Y
        )
        gradient = (gradient + 1) / 2  # Normalize to [0, 1]

        return self.base_level + self.variation_strength * (gradient - 0.5)

    def _generate_radial(self) -> NDArray[np.floating]:
        """Generate radial pattern from center."""
        x = np.linspace(-1, 1, self.frame_size)
        y = np.linspace(-1, 1, self.frame_size)
        X, Y = np.meshgrid(x, y)

        R = np.sqrt(X**2 + Y**2) / np.sqrt(2)  # Normalize to [0, 1]

        return self.base_level - self.variation_strength * R

    def _generate_patches(self) -> NDArray[np.floating]:
        """Generate random patch-based variation."""
        field = np.ones((self.frame_size, self.frame_size)) * self.base_level

        for _ in range(self.n_patches):
            # Random patch center and size
            cx = self.rng.uniform(0, self.frame_size)
            cy = self.rng.uniform(0, self.frame_size)
            radius = self.rng.uniform(
                self.frame_size * 0.1, self.frame_size * 0.3
            )

            # Random effect (positive or negative)
            effect = self.rng.uniform(
                -self.variation_strength, self.variation_strength
            )

            # Apply Gaussian patch
            x = np.arange(self.frame_size)
            y = np.arange(self.frame_size)
            X, Y = np.meshgrid(x, y)

            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            patch = effect * np.exp(-dist**2 / (2 * radius**2))
            field += patch

        return field

    def _generate_perlin_like(self) -> NDArray[np.floating]:
        """Generate smooth Perlin-like noise pattern."""
        # Multi-scale noise approximation
        field = np.zeros((self.frame_size, self.frame_size))

        for scale in [4, 8, 16, 32]:
            # Generate random noise at lower resolution
            noise_size = max(4, self.frame_size // scale)
            noise = self.rng.normal(0, 1, (noise_size, noise_size))

            # Upsample to full resolution using bilinear interpolation
            from scipy.ndimage import zoom
            factor = self.frame_size / noise_size
            upsampled = zoom(noise, factor, order=1)

            # Trim to exact size if needed
            upsampled = upsampled[:self.frame_size, :self.frame_size]

            # Add with decreasing amplitude for higher frequencies
            field += upsampled / scale

        # Normalize
        field = (field - field.min()) / (field.max() - field.min() + 1e-10)

        return self.base_level + self.variation_strength * (field - 0.5)

    def get_factors_at_positions(
        self,
        positions: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Get noise factors at specific positions.

        Parameters
        ----------
        positions : np.ndarray
            Positions to query, shape (n_positions, 2).

        Returns
        -------
        np.ndarray
            Noise factors at each position.
        """
        if self.noise_field is None:
            self.generate_field()

        # Convert positions to grid indices
        indices = np.clip(
            positions.astype(int),
            0, self.frame_size - 1
        )

        return self.noise_field[indices[:, 1], indices[:, 0]]

    def apply(
        self,
        expression: NDArray[np.floating],
        positions: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Apply spatial noise to expression matrix.

        Parameters
        ----------
        expression : np.ndarray
            Gene expression matrix, shape (n_cells, n_genes).
        positions : np.ndarray
            Cell centroid positions, shape (n_cells, 2).

        Returns
        -------
        np.ndarray
            Expression matrix with spatial noise applied.
        """
        factors = self.get_factors_at_positions(positions)
        return np.maximum(expression * factors[:, np.newaxis], 0)

    def plot_field(self, ax=None, cmap: str = "RdBu_r"):
        """Visualize the spatial noise field.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates new figure if None.
        cmap : str
            Colormap for visualization.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if self.noise_field is None:
            self.generate_field()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(
            self.noise_field,
            cmap=cmap,
            origin="lower",
            extent=[0, self.frame_size, 0, self.frame_size],
            vmin=self.base_level - self.variation_strength,
            vmax=self.base_level + self.variation_strength,
        )
        ax.set_title(f"Spatial Noise Field ({self.noise_pattern})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, label="Noise Factor")

        return ax
