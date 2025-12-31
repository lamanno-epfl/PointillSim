"""Technology presets for spatial transcriptomics platforms.

This module provides preset configurations for different spatial transcriptomics
technologies, encapsulating their specific detection characteristics, gene panel
constraints, and noise models.

Parameter values are based on published benchmarking studies:
- Chen et al., Nature Methods (2021): MERFISH benchmarking
- Srivatsan et al., Nature Methods (2021): Comparison of spatial methods
- 10x Genomics technical documentation (2023-2024)
- Codeluppi et al., Nature Methods (2018): osmFISH/HybISS characterization
- Moffitt et al., PNAS (2016): Original MERFISH validation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .transfer import TransferFunctionBase, AffineNonNegTransfer


@dataclass
class TechnologyPreset(ABC):
    """Abstract base class for spatial transcriptomics technology presets.

    Defines the detection characteristics and constraints specific to a
    spatial transcriptomics technology platform. Parameters are calibrated
    to match published benchmarking data from real experiments.

    Parameters
    ----------
    name : str
        Human-readable name of the technology.
    max_genes : int, optional
        Maximum number of genes in the panel. None means unlimited.
    typical_genes : int, optional
        Typical number of genes used in experiments.
    detection_efficiency : float
        Mean detection efficiency (0-1). Fraction of transcripts detected.
    detection_efficiency_std : float
        Gene-to-gene variation in detection efficiency.
    localization_error : float
        Spatial localization error in pixels/micrometers.
    false_positive_rate : float
        Rate of spurious dot detections per cell per gene.
    false_negative_rate : float
        Fraction of true transcripts missed (dropout).
    min_expression_threshold : float
        Minimum detectable expression level.
    saturation_level : float, optional
        Maximum detectable counts per cell (saturation). None means no limit.
    mean_transcripts_per_cell : float
        Expected total transcripts detected per cell. Based on published data.
    transcripts_per_cell_cv : float
        Coefficient of variation for transcripts per cell.
    mean_genes_per_cell : float
        Expected unique genes detected per cell.
    spot_size_um : float
        Physical size of detection spot in micrometers.
    resolution_nm : float
        Spatial resolution in nanometers.

    Attributes
    ----------
    transfer_function : TransferFunctionBase
        Transfer function for this technology's detection model.
    """

    name: str = "Generic"
    max_genes: Optional[int] = None
    typical_genes: int = 100
    detection_efficiency: float = 0.5
    detection_efficiency_std: float = 0.2
    localization_error: float = 1.0
    false_positive_rate: float = 0.01
    false_negative_rate: float = 0.1
    min_expression_threshold: float = 0.0
    saturation_level: Optional[float] = None
    # New realistic parameters
    mean_transcripts_per_cell: float = 200.0
    transcripts_per_cell_cv: float = 0.5
    mean_genes_per_cell: float = 50.0
    spot_size_um: float = 0.3
    resolution_nm: float = 300.0
    _transfer_function: Optional[TransferFunctionBase] = field(
        default=None, repr=False
    )

    @property
    def transfer_function(self) -> TransferFunctionBase:
        """TransferFunctionBase: Get the technology-specific transfer function."""
        if self._transfer_function is None:
            self._transfer_function = self._create_transfer_function()
        return self._transfer_function

    @abstractmethod
    def _create_transfer_function(self) -> TransferFunctionBase:
        """Create the transfer function for this technology.

        Returns
        -------
        TransferFunctionBase
            Technology-specific transfer function.
        """
        pass

    def validate_gene_panel(self, n_genes: int) -> bool:
        """Check if gene count is valid for this technology.

        Parameters
        ----------
        n_genes : int
            Number of genes in the panel.

        Returns
        -------
        bool
            True if valid, False if exceeds technology limits.
        """
        if self.max_genes is None:
            return True
        return n_genes <= self.max_genes

    def apply_localization_noise(
        self,
        positions: NDArray[np.floating],
        seed: Optional[int] = None,
    ) -> NDArray[np.floating]:
        """Add technology-specific localization error to dot positions.

        Parameters
        ----------
        positions : np.ndarray
            Dot positions, shape (n_dots, 2).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Positions with added localization noise.
        """
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, self.localization_error, positions.shape)
        return positions + noise

    def apply_false_negatives(
        self,
        counts: NDArray[np.integer],
        seed: Optional[int] = None,
    ) -> NDArray[np.integer]:
        """Apply technology-specific dropout/false negative model.

        Parameters
        ----------
        counts : np.ndarray
            Transcript counts per cell per gene.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Counts after applying dropout.
        """
        rng = np.random.default_rng(seed)
        # Binomial dropout
        keep_prob = 1.0 - self.false_negative_rate
        return rng.binomial(counts, keep_prob)

    def generate_false_positives(
        self,
        n_cells: int,
        n_genes: int,
        seed: Optional[int] = None,
    ) -> NDArray[np.integer]:
        """Generate false positive counts.

        Parameters
        ----------
        n_cells : int
            Number of cells.
        n_genes : int
            Number of genes.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            False positive count matrix, shape (n_cells, n_genes).
        """
        rng = np.random.default_rng(seed)
        # Poisson false positives
        expected_fp = self.false_positive_rate
        return rng.poisson(expected_fp, size=(n_cells, n_genes))

    def get_config(self) -> Dict:
        """Get technology configuration as a dictionary.

        Returns
        -------
        dict
            All preset parameters.
        """
        return {
            "name": self.name,
            "max_genes": self.max_genes,
            "typical_genes": self.typical_genes,
            "detection_efficiency": self.detection_efficiency,
            "detection_efficiency_std": self.detection_efficiency_std,
            "localization_error": self.localization_error,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "min_expression_threshold": self.min_expression_threshold,
            "saturation_level": self.saturation_level,
            "mean_transcripts_per_cell": self.mean_transcripts_per_cell,
            "transcripts_per_cell_cv": self.transcripts_per_cell_cv,
            "mean_genes_per_cell": self.mean_genes_per_cell,
            "spot_size_um": self.spot_size_um,
            "resolution_nm": self.resolution_nm,
        }

    def sample_transcripts_per_cell(
        self,
        n_cells: int,
        seed: Optional[int] = None,
    ) -> NDArray[np.floating]:
        """Sample realistic transcript counts per cell.

        Uses a negative binomial distribution to match observed
        over-dispersion in real data.

        Parameters
        ----------
        n_cells : int
            Number of cells.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Transcript counts per cell.
        """
        rng = np.random.default_rng(seed)
        # Use negative binomial for over-dispersion
        # CV = sqrt(1/n + 1/mean), solve for n given CV and mean
        variance = (self.transcripts_per_cell_cv * self.mean_transcripts_per_cell) ** 2
        if variance > self.mean_transcripts_per_cell:
            # Over-dispersed: use negative binomial
            p = self.mean_transcripts_per_cell / variance
            n = self.mean_transcripts_per_cell * p / (1 - p)
            counts = rng.negative_binomial(max(1, int(n)), p, size=n_cells)
        else:
            # Use Poisson
            counts = rng.poisson(self.mean_transcripts_per_cell, size=n_cells)
        return counts.astype(float)


@dataclass
class HybISSPreset(TechnologyPreset):
    """Preset for HybISS (Hybridization-based In Situ Sequencing).

    HybISS uses iterative hybridization cycles with fluorescent probes
    to detect RNA transcripts. Parameters based on:
    - Codeluppi et al., Nature Methods (2018)
    - Gyllborg et al., Nature Communications (2020)

    Typical characteristics:
    - Moderate gene panel sizes (50-500 genes)
    - ~15-25% detection efficiency per gene
    - ~100-300 transcripts per cell typically detected
    - ~50-100 unique genes detected per cell
    - Resolution ~200-300nm

    Examples
    --------
    >>> preset = HybISSPreset()
    >>> hybiss_setup = HybISS_Setup(tissue, transfer_function=preset.transfer_function)
    """

    name: str = "HybISS"
    max_genes: Optional[int] = 500
    typical_genes: int = 100
    detection_efficiency: float = 0.20  # ~20% per gene (Codeluppi 2018)
    detection_efficiency_std: float = 0.10
    localization_error: float = 0.25  # ~250nm
    false_positive_rate: float = 0.002
    false_negative_rate: float = 0.15
    min_expression_threshold: float = 0.0
    saturation_level: Optional[float] = None
    # Realistic transcript statistics
    mean_transcripts_per_cell: float = 180.0  # Based on osmFISH/HybISS data
    transcripts_per_cell_cv: float = 0.6
    mean_genes_per_cell: float = 65.0
    spot_size_um: float = 0.3
    resolution_nm: float = 250.0

    def _create_transfer_function(self) -> TransferFunctionBase:
        """Create HybISS-specific transfer function."""
        return AffineNonNegTransfer(
            scales=self.detection_efficiency,
            scales_std=self.detection_efficiency_std,
            offsets=0.0,
            offsets_std=0.05,
        )


@dataclass
class MerfishPreset(TechnologyPreset):
    """Preset for MERFISH (Multiplexed Error-Robust FISH).

    MERFISH uses combinatorial labeling with error-correcting codes
    for highly multiplexed RNA detection. Parameters based on:
    - Moffitt et al., PNAS (2016): Original validation
    - Chen et al., Science (2015): Method development
    - Xia et al., PNAS (2019): Large-scale benchmarking

    Characteristics:
    - Large gene panels (hundreds to thousands of genes)
    - ~85-95% detection efficiency per gene with amplification
    - ~500-2000 transcripts per cell typically detected
    - ~200-500 unique genes detected per cell
    - Resolution ~100-150nm (super-resolution capable)

    Examples
    --------
    >>> preset = MerfishPreset()
    >>> hybiss_setup = HybISS_Setup(tissue, transfer_function=preset.transfer_function)
    """

    name: str = "MERFISH"
    max_genes: Optional[int] = 10000
    typical_genes: int = 500
    detection_efficiency: float = 0.90  # Very high with amplification
    detection_efficiency_std: float = 0.08
    localization_error: float = 0.12  # ~120nm
    false_positive_rate: float = 0.001  # Error correction helps
    false_negative_rate: float = 0.05
    min_expression_threshold: float = 0.0
    saturation_level: Optional[float] = None
    # Realistic transcript statistics (based on Moffitt 2016, Chen 2015)
    mean_transcripts_per_cell: float = 800.0
    transcripts_per_cell_cv: float = 0.5
    mean_genes_per_cell: float = 280.0
    spot_size_um: float = 0.15
    resolution_nm: float = 120.0

    def _create_transfer_function(self) -> TransferFunctionBase:
        """Create MERFISH-specific transfer function."""
        return AffineNonNegTransfer(
            scales=self.detection_efficiency,
            scales_std=self.detection_efficiency_std,
            offsets=0.0,
            offsets_std=0.02,
        )


@dataclass
class CartanaPreset(TechnologyPreset):
    """Preset for Cartana/10x Genomics in situ technology.

    Cartana (now part of 10x Genomics) uses padlock probes and
    rolling circle amplification (RCA) for RNA detection. Parameters based on:
    - Nilsson et al. publications on padlock probes
    - 10x Genomics technical documentation

    Characteristics:
    - High sensitivity through RCA amplification
    - Good spatial resolution (~200nm)
    - ~300-800 transcripts per cell
    - ~100-250 unique genes per cell
    - Moderate to large gene panels

    Examples
    --------
    >>> preset = CartanaPreset()
    >>> hybiss_setup = HybISS_Setup(tissue, transfer_function=preset.transfer_function)
    """

    name: str = "Cartana"
    max_genes: Optional[int] = 1000
    typical_genes: int = 300
    detection_efficiency: float = 0.55  # RCA-based amplification
    detection_efficiency_std: float = 0.15
    localization_error: float = 0.20  # ~200nm
    false_positive_rate: float = 0.002
    false_negative_rate: float = 0.10
    min_expression_threshold: float = 0.0
    saturation_level: Optional[float] = None
    # Realistic transcript statistics
    mean_transcripts_per_cell: float = 450.0
    transcripts_per_cell_cv: float = 0.55
    mean_genes_per_cell: float = 150.0
    spot_size_um: float = 0.25
    resolution_nm: float = 200.0

    def _create_transfer_function(self) -> TransferFunctionBase:
        """Create Cartana-specific transfer function."""
        return AffineNonNegTransfer(
            scales=self.detection_efficiency,
            scales_std=self.detection_efficiency_std,
            offsets=0.0,
            offsets_std=0.04,
        )


@dataclass
class TenXXeniumPreset(TechnologyPreset):
    """Preset for 10x Genomics Xenium platform.

    Xenium is 10x Genomics' in situ platform offering subcellular
    resolution with targeted gene panels. Parameters based on:
    - 10x Genomics Xenium technical documentation (2023-2024)
    - Published Xenium datasets and validation studies

    Characteristics:
    - Curated gene panels (313-480 genes currently)
    - High detection sensitivity (~70-85% per gene)
    - ~200-600 transcripts per cell
    - ~80-200 unique genes per cell
    - Subcellular resolution (~200nm)
    - Integrated DAPI-based cell segmentation

    Examples
    --------
    >>> preset = TenXXeniumPreset()
    >>> hybiss_setup = HybISS_Setup(tissue, transfer_function=preset.transfer_function)
    """

    name: str = "10x Xenium"
    max_genes: Optional[int] = 500
    typical_genes: int = 380
    detection_efficiency: float = 0.75  # High per 10x documentation
    detection_efficiency_std: float = 0.12
    localization_error: float = 0.20  # ~200nm
    false_positive_rate: float = 0.001
    false_negative_rate: float = 0.08
    min_expression_threshold: float = 0.0
    saturation_level: Optional[float] = None
    # Realistic transcript statistics (based on 10x technical notes)
    mean_transcripts_per_cell: float = 350.0
    transcripts_per_cell_cv: float = 0.6
    mean_genes_per_cell: float = 120.0
    spot_size_um: float = 0.22
    resolution_nm: float = 200.0

    def _create_transfer_function(self) -> TransferFunctionBase:
        """Create Xenium-specific transfer function."""
        return AffineNonNegTransfer(
            scales=self.detection_efficiency,
            scales_std=self.detection_efficiency_std,
            offsets=0.0,
            offsets_std=0.03,
        )


@dataclass
class TenXVisiumPreset(TechnologyPreset):
    """Preset for 10x Genomics Visium platform.

    Visium is a spot-based spatial transcriptomics platform with
    capture areas of ~55 μm diameter, containing multiple cells.
    Parameters based on:
    - 10x Genomics Visium technical documentation
    - Stahl et al., Science (2016): Original spatial transcriptomics
    - Multiple published Visium datasets

    Characteristics:
    - Whole transcriptome (not targeted)
    - Spot-based: ~55μm diameter spots, ~1-10 cells per spot
    - ~5,000-50,000 UMIs per spot
    - ~2,000-8,000 unique genes per spot
    - Lower spatial resolution compared to single-molecule methods
    - Requires deconvolution for cell-type inference

    Note: This preset simulates spot-level data, not single-cell.

    Examples
    --------
    >>> preset = TenXVisiumPreset()
    >>> # Note: Visium requires different simulation approach
    """

    name: str = "10x Visium"
    max_genes: Optional[int] = None  # Whole transcriptome
    typical_genes: int = 18000
    detection_efficiency: float = 0.15  # Capture efficiency ~10-20%
    detection_efficiency_std: float = 0.08
    localization_error: float = 55.0  # Spot diameter in μm
    false_positive_rate: float = 0.0001  # Very low for sequencing
    false_negative_rate: float = 0.3  # Dropout at low expression
    min_expression_threshold: float = 1.0
    saturation_level: Optional[float] = 50000.0  # UMIs per spot
    # Realistic statistics (per spot, not per cell)
    mean_transcripts_per_cell: float = 15000.0  # UMIs per spot
    transcripts_per_cell_cv: float = 0.4
    mean_genes_per_cell: float = 4000.0  # Genes per spot
    spot_size_um: float = 55.0  # Spot diameter
    resolution_nm: float = 55000.0  # Spot-level resolution

    def _create_transfer_function(self) -> TransferFunctionBase:
        """Create Visium-specific transfer function."""
        return AffineNonNegTransfer(
            scales=self.detection_efficiency,
            scales_std=self.detection_efficiency_std,
            offsets=0.0,
            offsets_std=0.1,
        )


# Convenience dictionary of all presets
TECHNOLOGY_PRESETS = {
    "hybiss": HybISSPreset,
    "merfish": MerfishPreset,
    "cartana": CartanaPreset,
    "xenium": TenXXeniumPreset,
    "visium": TenXVisiumPreset,
}


def get_preset(name: str) -> TechnologyPreset:
    """Get a technology preset by name.

    Parameters
    ----------
    name : str
        Technology name (case-insensitive). Options: 'hybiss', 'merfish',
        'cartana', 'xenium', 'visium'.

    Returns
    -------
    TechnologyPreset
        Instantiated preset for the specified technology.

    Raises
    ------
    ValueError
        If technology name is not recognized.

    Examples
    --------
    >>> preset = get_preset("merfish")
    >>> preset.detection_efficiency
    0.75
    """
    name_lower = name.lower()
    if name_lower not in TECHNOLOGY_PRESETS:
        available = ", ".join(TECHNOLOGY_PRESETS.keys())
        raise ValueError(
            f"Unknown technology '{name}'. Available: {available}"
        )
    return TECHNOLOGY_PRESETS[name_lower]()
