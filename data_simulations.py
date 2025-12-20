"""PointillSim - Legacy compatibility module.

This module re-exports all classes and functions from the new pointillsim package
for backward compatibility. New code should import directly from pointillsim.

Example (legacy, still works):
    from data_simulations import FOV, FOVDistribution, HybISS_Setup

Example (recommended):
    from pointillsim import FOV, FOVDistribution, HybISS_Setup
"""

# Re-export everything from the new package
from pointillsim import (
    # Version
    __version__,
    # Core
    FOV,
    FOVDistribution,
    TissueCellTypes,
    TissueSlice,
    RegionSpec,
    # Elements
    HistologicalElement,
    FrameWideElement,
    FrameWideUpdater,
    VacuolatedStructure,
    # Rules
    CellTypeRuleBase,
    DummyRule,
    RandomCellTypeRule,
    MixOfNCellTypesRule,
    ProbabilityNodeFieldRule,
    SingleTypeRule,
    DeterministicNeighborAssignment,
    # Experiment
    HybISS_Setup,
    CellTypesProperties,
    TransferFunctionBase,
    IdentityTransfer,
    AffineNonNegTransfer,
    # Utilities
    generate_uniform_points_in_circle,
    generate_points_asin_cell,
    lognorm_params_to_mean_std,
    intuitive_rand_lognormal,
    LinearNDInterpolatorExt,
    one_hot_encode_array,
    unfold_int_matrix,
    # I/O
    generate_dataset,
    load_data,
)

__all__ = [
    "generate_uniform_points_in_circle",
    "generate_points_asin_cell",
    "one_hot_encode_array",
    "lognorm_params_to_mean_std",
    "intuitive_rand_lognormal",
    "unfold_int_matrix",
    "generate_dataset",
    "load_data",
    "LinearNDInterpolatorExt",
    # Histological elements
    "HistologicalElement",
    "FrameWideElement",
    "FrameWideUpdater",
    "VacuolatedStructure",
    # Main Objects
    "FOVDistribution",
    "FOV",
    "CellTypesProperties",
    "TissueCellTypes",
    "TissueSlice",
    "RegionSpec",
    "HybISS_Setup",
    # Rules
    "CellTypeRuleBase",
    "RandomCellTypeRule",
    "SingleTypeRule",
    "ProbabilityNodeFieldRule",
    "DeterministicNeighborAssignment",
    "DummyRule",
    "MixOfNCellTypesRule",
    # Transfer functions
    "TransferFunctionBase",
    "AffineNonNegTransfer",
    "IdentityTransfer",
]
