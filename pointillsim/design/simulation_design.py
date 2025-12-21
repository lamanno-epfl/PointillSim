"""Simulation design and experiment planning."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .covariates import CovariateSystem, Covariate
from .controller import EffectController


@dataclass
class DesignMatrix:
    """Track covariate values across generated FOVs.

    The DesignMatrix stores the covariate configuration for each FOV
    in a simulation experiment, enabling analysis of covariate effects.

    Parameters
    ----------
    covariate_names : List[str]
        Names of covariates being tracked.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame with covariate values for each FOV.

    Examples
    --------
    >>> matrix = DesignMatrix(["region", "batch"])
    >>> matrix.add_row({"region": "cortex", "batch": 0})
    >>> matrix.add_row({"region": "medulla", "batch": 1})
    >>> df = matrix.to_dataframe()
    """

    covariate_names: List[str]
    _rows: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def add_row(
        self,
        values: Dict[str, Any],
        fov_id: Optional[int] = None
    ) -> "DesignMatrix":
        """Add a row to the design matrix.

        Parameters
        ----------
        values : dict
            Dictionary of covariate values for this FOV.
        fov_id : int, optional
            FOV identifier. If None, auto-incremented.

        Returns
        -------
        DesignMatrix
            Self, for method chaining.
        """
        row = {"fov_id": fov_id if fov_id is not None else len(self._rows)}
        row.update(values)
        self._rows.append(row)
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convert design matrix to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with covariate values.
        """
        return pd.DataFrame(self._rows)

    @property
    def n_fovs(self) -> int:
        """int: Number of FOVs in the design."""
        return len(self._rows)

    def get_fov_covariates(self, fov_id: int) -> Dict[str, Any]:
        """Get covariate values for a specific FOV.

        Parameters
        ----------
        fov_id : int
            FOV identifier.

        Returns
        -------
        dict
            Covariate values for the FOV.
        """
        for row in self._rows:
            if row.get("fov_id") == fov_id:
                return {k: v for k, v in row.items() if k != "fov_id"}
        raise ValueError(f"FOV {fov_id} not found")

    def get_covariate_values(self, covariate: str) -> List[Any]:
        """Get all values of a specific covariate.

        Parameters
        ----------
        covariate : str
            Covariate name.

        Returns
        -------
        list
            Values across all FOVs.
        """
        return [row.get(covariate) for row in self._rows]

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics for the design.

        Returns
        -------
        dict
            Summary of covariate distributions.
        """
        df = self.to_dataframe()
        summary = {"n_fovs": len(df)}

        for col in self.covariate_names:
            if col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    summary[col] = {
                        "type": "continuous",
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                    }
                else:
                    summary[col] = {
                        "type": "categorical",
                        "unique_values": list(df[col].unique()),
                        "counts": df[col].value_counts().to_dict(),
                    }

        return summary


@dataclass
class SimulationDesign:
    """High-level class for specifying simulation experiments.

    SimulationDesign allows you to define multi-condition experiments
    with varying covariates, replicates, and effect settings. It generates
    a complete experimental design that can be executed.

    Parameters
    ----------
    name : str
        Name of the experiment.
    covariate_system : CovariateSystem, optional
        Covariate system defining the experimental factors.
    effect_controller : EffectController, optional
        Controller for effect settings.
    n_replicates : int, optional
        Number of replicates per condition. Default is 1.
    seed : Optional[int], optional
        Random seed for reproducibility.

    Examples
    --------
    >>> design = SimulationDesign("tissue_comparison")
    >>> design.add_covariate(Covariate("tissue", "categorical",
    ...                                 values=["brain", "liver"]))
    >>> design.set_replicates(3)
    >>> conditions = design.generate_conditions()
    """

    name: str
    covariate_system: CovariateSystem = field(default_factory=CovariateSystem)
    effect_controller: EffectController = field(default_factory=EffectController)
    n_replicates: int = 1
    seed: Optional[int] = None

    _conditions: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Initialize random state."""
        self.rng = np.random.default_rng(seed=self.seed)

    def add_covariate(self, covariate: Covariate) -> "SimulationDesign":
        """Add a covariate to the design.

        Parameters
        ----------
        covariate : Covariate
            Covariate to add.

        Returns
        -------
        SimulationDesign
            Self, for method chaining.
        """
        self.covariate_system.add_covariate(covariate)
        return self

    def set_replicates(self, n: int) -> "SimulationDesign":
        """Set the number of replicates per condition.

        Parameters
        ----------
        n : int
            Number of replicates.

        Returns
        -------
        SimulationDesign
            Self, for method chaining.
        """
        if n < 1:
            raise ValueError("n_replicates must be >= 1")
        self.n_replicates = n
        return self

    def set_effect(self, effect: str, enabled: bool = True, magnitude: float = 1.0) -> "SimulationDesign":
        """Configure an effect for this design.

        Parameters
        ----------
        effect : str
            Effect name.
        enabled : bool, optional
            Whether to enable the effect. Default is True.
        magnitude : float, optional
            Effect magnitude. Default is 1.0.

        Returns
        -------
        SimulationDesign
            Self, for method chaining.
        """
        if enabled:
            self.effect_controller.enable(effect)
        else:
            self.effect_controller.disable(effect)
        self.effect_controller.set_magnitude(effect, magnitude)
        return self

    def generate_conditions(
        self,
        mode: str = "grid",
        n_random: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate experimental conditions.

        Parameters
        ----------
        mode : str, optional
            Generation mode: 'grid' for full factorial, 'random' for random
            sampling, 'custom' for user-specified. Default is 'grid'.
        n_random : int, optional
            Number of random conditions if mode='random'. Default is 10.
        **kwargs
            For 'grid' mode: covariate names mapped to lists of values.

        Returns
        -------
        list
            List of condition dictionaries.
        """
        if mode == "grid":
            base_conditions = self.covariate_system.generate_grid(**kwargs)
        elif mode == "random":
            base_conditions = [
                self.covariate_system.sample_configuration(self.rng)
                for _ in range(n_random)
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Expand with replicates
        self._conditions = []
        condition_id = 0

        for base_cond in base_conditions:
            for rep in range(self.n_replicates):
                cond = base_cond.copy()
                cond["_condition_id"] = condition_id
                cond["_replicate"] = rep
                cond["_seed"] = self.rng.integers(0, 2**31)
                self._conditions.append(cond)
            condition_id += 1

        return self._conditions

    def get_design_matrix(self) -> DesignMatrix:
        """Get the design matrix for all conditions.

        Returns
        -------
        DesignMatrix
            Design matrix tracking all covariate values.
        """
        if not self._conditions:
            self.generate_conditions()

        matrix = DesignMatrix(covariate_names=list(self.covariate_system.covariates.keys()))

        for i, cond in enumerate(self._conditions):
            values = {k: v for k, v in cond.items() if not k.startswith("_")}
            values["condition_id"] = cond.get("_condition_id", i)
            values["replicate"] = cond.get("_replicate", 0)
            matrix.add_row(values, fov_id=i)

        return matrix

    @property
    def n_conditions(self) -> int:
        """int: Total number of conditions including replicates."""
        return len(self._conditions) if self._conditions else 0

    def summary(self) -> Dict[str, Any]:
        """Get summary of the experimental design.

        Returns
        -------
        dict
            Summary information.
        """
        return {
            "name": self.name,
            "n_covariates": self.covariate_system.n_covariates,
            "covariates": list(self.covariate_system.covariates.keys()),
            "n_replicates": self.n_replicates,
            "n_conditions": self.n_conditions,
            "effects": self.effect_controller.summary,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "covariate_system": self.covariate_system.to_dict(),
            "effect_controller": self.effect_controller.to_dict(),
            "n_replicates": self.n_replicates,
            "seed": self.seed,
            "conditions": self._conditions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationDesign":
        """Create from dictionary."""
        design = cls(
            name=data["name"],
            covariate_system=CovariateSystem.from_dict(data.get("covariate_system", {})),
            effect_controller=EffectController.from_dict(data.get("effect_controller", {})),
            n_replicates=data.get("n_replicates", 1),
            seed=data.get("seed"),
        )
        design._conditions = data.get("conditions", [])
        return design


def run_simulation_design(
    design: SimulationDesign,
    fov_generator: Callable,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Execute a simulation design and generate all FOVs.

    Parameters
    ----------
    design : SimulationDesign
        The experimental design to execute.
    fov_generator : callable
        Function that takes (condition_dict, seed) and returns an FOV.
    progress_callback : callable, optional
        Called with (current_index, total) for progress updates.

    Returns
    -------
    dict
        Dictionary with 'fovs' (list of FOV objects) and 'design_matrix'.

    Examples
    --------
    >>> def my_generator(condition, seed):
    ...     # Generate FOV based on condition
    ...     return fov
    >>> results = run_simulation_design(design, my_generator)
    """
    conditions = design._conditions if design._conditions else design.generate_conditions()

    fovs = []
    total = len(conditions)

    for i, condition in enumerate(conditions):
        seed = condition.get("_seed", i)
        fov = fov_generator(condition, seed)
        fovs.append(fov)

        if progress_callback:
            progress_callback(i + 1, total)

    return {
        "fovs": fovs,
        "design_matrix": design.get_design_matrix(),
        "conditions": conditions,
        "design": design,
    }
