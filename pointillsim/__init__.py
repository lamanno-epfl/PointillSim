"""PointillSim - Rule-based Simulation Engine for Imaging-based Spatial Transcriptomics.

PointillSim creates realistic synthetic Fields of View (FOVs) for imaging-based
spatial transcriptomics experiments, specifically HybISS (Hybridization-based
In Situ Sequencing). It generates ground truth cell-type annotations and spatial
gene expression data as dots within cells.

Quick Start
-----------
>>> from pointillsim import (
...     TissueCellTypes, CellTypesProperties, HybISS_Setup,
...     FOVDistribution, FrameWideElement, HistologicalElement,
...     RandomCellTypeRule, ProbabilityNodeFieldRule
... )
>>>
>>> # Define tissue with gene expression profiles
>>> tissue = TissueCellTypes()
>>> tissue.generate_types_and_markers(n_genes=50, n_cell_types=10)
>>>
>>> # Create cell type properties
>>> cell_props = CellTypesProperties(n_cell_types=10)
>>>
>>> # Define FOV distribution
>>> bg = lambda: FrameWideElement(frame_size=1000, rules=RandomCellTypeRule(10))
>>> fovd = FOVDistribution(frame_size=1000, background_element=bg)
>>>
>>> # Generate and observe
>>> fov = fovd.generate_fov()
>>> cell_props.apply(fov)
>>> hybiss = HybISS_Setup(tissue)
>>> hybiss.observe_dots(fov)
>>> dots_df = hybiss.make_pandas_df()
"""

__version__ = "0.1.0"

# Core classes
from .core.fov import FOV, FOVDistribution
from .core.tissue import TissueCellTypes, TissueSlice, RegionSpec

# Histological elements
from .elements.base import HistologicalElement
from .elements.frame import FrameWideElement, FrameWideUpdater
from .elements.structures import VacuolatedStructure

# Cell type rules
from .rules.base import CellTypeRuleBase, DummyRule
from .rules.random import RandomCellTypeRule, MixOfNCellTypesRule
from .rules.spatial import ProbabilityNodeFieldRule, SingleTypeRule
from .rules.neighbor import DeterministicNeighborAssignment
from .rules.composite import DistanceBasedRule, CompositeRule

# Experiment simulation
from .experiment.hybiss import HybISS_Setup
from .experiment.properties import CellTypesProperties
from .experiment.transfer import TransferFunctionBase, IdentityTransfer, AffineNonNegTransfer

# Utilities
from .utils.geometry import (
    generate_uniform_points_in_circle,
    generate_points_asin_cell,
    chaikin_smooth,
    smooth_polygon,
    add_edge_noise,
)
from .utils.math import lognorm_params_to_mean_std, intuitive_rand_lognormal
from .utils.interpolation import LinearNDInterpolatorExt
from .utils.encoding import one_hot_encode_array, unfold_int_matrix

# Visualization
from .viz.plotting import plot_fov, plot_expression_matrix

# I/O
from .io.dataset import generate_dataset, load_data

# Sample data
from .data import load_sample, list_samples, load_sample_metadata

# Configuration
from .config import (
    SimulationConfig,
    FOVConfig,
    TissueConfig,
    ExperimentConfig,
    ElementConfig,
    create_example_config,
    build_simulation_from_config,
)

# Effects (noise, batch, admixture)
from .effects import (
    BatchEffectModel,
    TechnicalNoise,
    BackgroundNoise,
    DropoutModel,
    SpatialNoise,
    Lateral2DAdmixture,
    ZAxisAdmixture,
    CompositeAdmixture,
    AdmixtureMetrics,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "FOV",
    "FOVDistribution",
    "TissueCellTypes",
    "TissueSlice",
    "RegionSpec",
    # Elements
    "HistologicalElement",
    "FrameWideElement",
    "FrameWideUpdater",
    "VacuolatedStructure",
    # Rules
    "CellTypeRuleBase",
    "DummyRule",
    "RandomCellTypeRule",
    "MixOfNCellTypesRule",
    "ProbabilityNodeFieldRule",
    "SingleTypeRule",
    "DeterministicNeighborAssignment",
    "DistanceBasedRule",
    "CompositeRule",
    # Experiment
    "HybISS_Setup",
    "CellTypesProperties",
    "TransferFunctionBase",
    "IdentityTransfer",
    "AffineNonNegTransfer",
    # Utilities
    "generate_uniform_points_in_circle",
    "generate_points_asin_cell",
    "chaikin_smooth",
    "smooth_polygon",
    "add_edge_noise",
    "lognorm_params_to_mean_std",
    "intuitive_rand_lognormal",
    "LinearNDInterpolatorExt",
    "one_hot_encode_array",
    "unfold_int_matrix",
    # Visualization
    "plot_fov",
    "plot_expression_matrix",
    # I/O
    "generate_dataset",
    "load_data",
    # Sample data
    "load_sample",
    "list_samples",
    "load_sample_metadata",
    # Configuration
    "SimulationConfig",
    "FOVConfig",
    "TissueConfig",
    "ExperimentConfig",
    "ElementConfig",
    "create_example_config",
    "build_simulation_from_config",
    # Effects
    "BatchEffectModel",
    "TechnicalNoise",
    "BackgroundNoise",
    "DropoutModel",
    "SpatialNoise",
    "Lateral2DAdmixture",
    "ZAxisAdmixture",
    "CompositeAdmixture",
    "AdmixtureMetrics",
]
