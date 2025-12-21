"""Technology presets for spatial transcriptomics platforms.

This module provides preset configurations for different spatial transcriptomics
technologies, encapsulating their specific detection characteristics, gene panel
constraints, and noise models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .transfer import TransferFunctionBase, IdentityTransfer, AffineNonNegTransfer


@dataclass
class TechnologyPreset(ABC):
    """Abstract base class for spatial transcriptomics technology presets.

    Defines the detection characteristics and constraints specific to a
    spatial transcriptomics technology platform.

    Parameters
    ----------
    name : str
        Human-readable name of the technology.
    max_genes : int, optional
        Maximum number of genes in the panel. None means unlimited.
    typical_genes : int, optional
        Typical number of genes used in experiments.
    detection_efficiency : float
        Mean detection efficiency (0-1). Default 0.5.
    detection_efficiency_std : float
        Gene-to-gene variation in detection efficiency. Default 0.2.
    localization_error : float
        Spatial localization error in pixels. Default 1.0.
    false_positive_rate : float
        Rate of spurious dot detections. Default 0.01.
    false_negative_rate : float
        Rate of missed true transcripts. Default 0.1.
    min_expression_threshold : float
        Minimum detectable expression level. Default 0.0.
    saturation_level : float, optional
        Maximum detectable counts per cell (saturation). None means no limit.

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
        }


@dataclass
class HybISSPreset(TechnologyPreset):
    """Preset for HybISS (Hybridization-based In Situ Sequencing).

    HybISS uses iterative hybridization cycles with fluorescent probes
    to detect RNA transcripts. Typical characteristics:
    - Moderate gene panel sizes (50-500 genes)
    - Good detection efficiency for targeted genes
    - Some localization error due to optical resolution

    Examples
    --------
    >>> preset = HybISSPreset()
    >>> hybiss_setup = HybISS_Setup(tissue, transfer_function=preset.transfer_function)
    """

    name: str = "HybISS"
    max_genes: Optional[int] = 500
    typical_genes: int = 100
    detection_efficiency: float = 0.6
    detection_efficiency_std: float = 0.25
    localization_error: float = 0.8
    false_positive_rate: float = 0.005
    false_negative_rate: float = 0.15
    min_expression_threshold: float = 0.0
    saturation_level: Optional[float] = None

    def _create_transfer_function(self) -> TransferFunctionBase:
        """Create HybISS-specific transfer function."""
        return AffineNonNegTransfer(
            scales=self.detection_efficiency,
            scales_std=self.detection_efficiency_std,
            offsets=0.0,
            offsets_std=0.1,
        )


@dataclass
class MerfishPreset(TechnologyPreset):
    """Preset for MERFISH (Multiplexed Error-Robust FISH).

    MERFISH uses combinatorial labeling with error-correcting codes
    for highly multiplexed RNA detection. Based on 2024 benchmarking studies.

    Characteristics:
    - Large gene panels (hundreds to thousands of genes)
    - High detection efficiency due to amplification
    - Excellent error correction reduces false positives
    - Subcellular resolution

    Examples
    --------
    >>> preset = MerfishPreset()
    >>> hybiss_setup = HybISS_Setup(tissue, transfer_function=preset.transfer_function)
    """

    name: str = "MERFISH"
    max_genes: Optional[int] = 10000
    typical_genes: int = 500
    detection_efficiency: float = 0.75
    detection_efficiency_std: float = 0.15
    localization_error: float = 0.3
    false_positive_rate: float = 0.001
    false_negative_rate: float = 0.08
    min_expression_threshold: float = 0.0
    saturation_level: Optional[float] = None

    def _create_transfer_function(self) -> TransferFunctionBase:
        """Create MERFISH-specific transfer function."""
        return AffineNonNegTransfer(
            scales=self.detection_efficiency,
            scales_std=self.detection_efficiency_std,
            offsets=0.0,
            offsets_std=0.05,
        )


@dataclass
class CartanaPreset(TechnologyPreset):
    """Preset for Cartana/10x Genomics in situ technology.

    Cartana (now part of 10x Genomics) uses padlock probes and
    rolling circle amplification for RNA detection.

    Characteristics:
    - High sensitivity through amplification
    - Good spatial resolution
    - Moderate to large gene panels

    Examples
    --------
    >>> preset = CartanaPreset()
    >>> hybiss_setup = HybISS_Setup(tissue, transfer_function=preset.transfer_function)
    """

    name: str = "Cartana"
    max_genes: Optional[int] = 1000
    typical_genes: int = 300
    detection_efficiency: float = 0.65
    detection_efficiency_std: float = 0.20
    localization_error: float = 0.5
    false_positive_rate: float = 0.003
    false_negative_rate: float = 0.12
    min_expression_threshold: float = 0.0
    saturation_level: Optional[float] = None

    def _create_transfer_function(self) -> TransferFunctionBase:
        """Create Cartana-specific transfer function."""
        return AffineNonNegTransfer(
            scales=self.detection_efficiency,
            scales_std=self.detection_efficiency_std,
            offsets=0.0,
            offsets_std=0.08,
        )


@dataclass
class TenXXeniumPreset(TechnologyPreset):
    """Preset for 10x Genomics Xenium platform.

    Xenium is 10x Genomics' in situ platform offering subcellular
    resolution with targeted gene panels.

    Characteristics:
    - Curated gene panels (typically 300-500 genes)
    - High detection sensitivity
    - Excellent subcellular resolution
    - Integrated cell segmentation

    Examples
    --------
    >>> preset = TenXXeniumPreset()
    >>> hybiss_setup = HybISS_Setup(tissue, transfer_function=preset.transfer_function)
    """

    name: str = "10x Xenium"
    max_genes: Optional[int] = 500
    typical_genes: int = 350
    detection_efficiency: float = 0.70
    detection_efficiency_std: float = 0.18
    localization_error: float = 0.4
    false_positive_rate: float = 0.002
    false_negative_rate: float = 0.10
    min_expression_threshold: float = 0.0
    saturation_level: Optional[float] = None

    def _create_transfer_function(self) -> TransferFunctionBase:
        """Create Xenium-specific transfer function."""
        return AffineNonNegTransfer(
            scales=self.detection_efficiency,
            scales_std=self.detection_efficiency_std,
            offsets=0.0,
            offsets_std=0.06,
        )


@dataclass
class TenXVisiumPreset(TechnologyPreset):
    """Preset for 10x Genomics Visium platform.

    Visium is a spot-based spatial transcriptomics platform with
    capture areas of ~55 μm diameter, containing multiple cells.

    Characteristics:
    - Whole transcriptome (not targeted)
    - Lower spatial resolution (spot-based, not single-cell)
    - High gene coverage
    - Requires deconvolution for cell-type inference

    Note: This preset simulates spot-level data, not single-cell.
    Parameters are adjusted for spot-based detection.

    Examples
    --------
    >>> preset = TenXVisiumPreset()
    >>> # Note: Visium requires different simulation approach
    """

    name: str = "10x Visium"
    max_genes: Optional[int] = None  # Whole transcriptome
    typical_genes: int = 20000
    detection_efficiency: float = 0.3  # Lower due to capture efficiency
    detection_efficiency_std: float = 0.3
    localization_error: float = 55.0  # Spot diameter
    false_positive_rate: float = 0.0  # Very low for sequencing
    false_negative_rate: float = 0.4  # High dropout in spatial
    min_expression_threshold: float = 1.0
    saturation_level: Optional[float] = 50000.0  # UMIs per spot

    def _create_transfer_function(self) -> TransferFunctionBase:
        """Create Visium-specific transfer function."""
        return AffineNonNegTransfer(
            scales=self.detection_efficiency,
            scales_std=self.detection_efficiency_std,
            offsets=0.0,
            offsets_std=0.5,
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
