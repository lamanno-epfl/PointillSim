"""Design module for controlled simulation experiments.

This module provides classes for:
- CovariateSystem: Define and manage simulation covariates
- EffectController: Enable/disable specific effects
- SimulationDesign: Specify multi-condition experiments
- DesignMatrix: Track covariate values across FOVs
"""

from .covariates import CovariateSystem, Covariate
from .controller import EffectController
from .simulation_design import SimulationDesign, DesignMatrix

__all__ = [
    "CovariateSystem",
    "Covariate",
    "EffectController",
    "SimulationDesign",
    "DesignMatrix",
]
