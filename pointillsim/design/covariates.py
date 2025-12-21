"""Covariate system for simulation experiments."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable
import numpy as np
from numpy.typing import NDArray


@dataclass
class Covariate:
    """A single covariate that can influence simulation parameters.

    Covariates represent factors that can vary across FOVs or conditions,
    such as tissue region, distance from a landmark, or experimental batch.

    Parameters
    ----------
    name : str
        Unique identifier for the covariate.
    covariate_type : str
        Type of covariate: 'continuous', 'categorical', or 'ordinal'.
    values : Optional[list]
        For categorical/ordinal: list of possible values.
        For continuous: [min, max] range.
    description : str, optional
        Human-readable description of the covariate.
    default : Any, optional
        Default value if not specified.

    Examples
    --------
    >>> # Categorical covariate
    >>> tissue = Covariate("tissue_region", "categorical",
    ...                    values=["cortex", "white_matter", "meninges"])
    >>> # Continuous covariate
    >>> distance = Covariate("distance_from_edge", "continuous",
    ...                      values=[0, 1000], description="Distance in pixels")
    """

    name: str
    covariate_type: str
    values: Optional[list] = None
    description: str = ""
    default: Any = None

    def __post_init__(self):
        """Validate covariate parameters."""
        valid_types = {"continuous", "categorical", "ordinal"}
        if self.covariate_type not in valid_types:
            raise ValueError(f"covariate_type must be one of {valid_types}")

        if self.covariate_type in ("categorical", "ordinal") and not self.values:
            raise ValueError(f"{self.covariate_type} covariate requires values list")

        if self.covariate_type == "continuous":
            if self.values is None:
                self.values = [0, 1]  # Default range
            elif len(self.values) != 2:
                raise ValueError("continuous covariate values must be [min, max]")

    def sample(
        self,
        n: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> Union[list, NDArray]:
        """Sample random values from this covariate.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        list or np.ndarray
            Sampled values.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.covariate_type == "continuous":
            return rng.uniform(self.values[0], self.values[1], n)
        elif self.covariate_type in ("categorical", "ordinal"):
            return list(rng.choice(self.values, n))

    def validate(self, value: Any) -> bool:
        """Check if a value is valid for this covariate.

        Parameters
        ----------
        value : Any
            Value to validate.

        Returns
        -------
        bool
            True if valid.
        """
        if self.covariate_type == "continuous":
            return self.values[0] <= value <= self.values[1]
        else:
            return value in self.values

    def to_dict(self) -> Dict[str, Any]:
        """Serialize covariate to dictionary."""
        return {
            "name": self.name,
            "covariate_type": self.covariate_type,
            "values": self.values,
            "description": self.description,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Covariate":
        """Create Covariate from dictionary."""
        return cls(**data)


@dataclass
class CovariateSystem:
    """Framework for defining and managing simulation covariates.

    The CovariateSystem manages a collection of covariates that can influence
    simulation behavior. It tracks covariate values, validates assignments,
    and provides methods for experimental design.

    Parameters
    ----------
    name : str, optional
        Name for this covariate system.
    covariates : List[Covariate], optional
        Initial list of covariates.

    Attributes
    ----------
    covariates : dict
        Dictionary mapping covariate names to Covariate objects.
    current_values : dict
        Current assigned values for each covariate.

    Examples
    --------
    >>> system = CovariateSystem("spatial_experiment")
    >>> system.add_covariate(Covariate("region", "categorical",
    ...                                 values=["A", "B", "C"]))
    >>> system.add_covariate(Covariate("cell_density", "continuous",
    ...                                 values=[0.5, 2.0]))
    >>> system.set_value("region", "B")
    >>> system.set_value("cell_density", 1.2)
    """

    name: str = "default"
    covariates: Dict[str, Covariate] = field(default_factory=dict)
    current_values: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize from list if provided."""
        if isinstance(self.covariates, list):
            cov_dict = {}
            for cov in self.covariates:
                cov_dict[cov.name] = cov
            self.covariates = cov_dict

    def add_covariate(self, covariate: Covariate) -> "CovariateSystem":
        """Add a covariate to the system.

        Parameters
        ----------
        covariate : Covariate
            Covariate to add.

        Returns
        -------
        CovariateSystem
            Self, for method chaining.
        """
        self.covariates[covariate.name] = covariate
        if covariate.default is not None:
            self.current_values[covariate.name] = covariate.default
        return self

    def remove_covariate(self, name: str) -> "CovariateSystem":
        """Remove a covariate from the system.

        Parameters
        ----------
        name : str
            Name of covariate to remove.

        Returns
        -------
        CovariateSystem
            Self, for method chaining.
        """
        if name in self.covariates:
            del self.covariates[name]
        if name in self.current_values:
            del self.current_values[name]
        return self

    def set_value(self, name: str, value: Any) -> "CovariateSystem":
        """Set the current value of a covariate.

        Parameters
        ----------
        name : str
            Covariate name.
        value : Any
            Value to set.

        Returns
        -------
        CovariateSystem
            Self, for method chaining.

        Raises
        ------
        ValueError
            If covariate doesn't exist or value is invalid.
        """
        if name not in self.covariates:
            raise ValueError(f"Unknown covariate: {name}")

        if not self.covariates[name].validate(value):
            raise ValueError(f"Invalid value {value} for covariate {name}")

        self.current_values[name] = value
        return self

    def get_value(self, name: str) -> Any:
        """Get the current value of a covariate.

        Parameters
        ----------
        name : str
            Covariate name.

        Returns
        -------
        Any
            Current value.
        """
        if name not in self.covariates:
            raise ValueError(f"Unknown covariate: {name}")
        return self.current_values.get(name, self.covariates[name].default)

    def set_values(self, values: Dict[str, Any]) -> "CovariateSystem":
        """Set multiple covariate values at once.

        Parameters
        ----------
        values : dict
            Dictionary mapping covariate names to values.

        Returns
        -------
        CovariateSystem
            Self, for method chaining.
        """
        for name, value in values.items():
            self.set_value(name, value)
        return self

    def get_all_values(self) -> Dict[str, Any]:
        """Get all current covariate values.

        Returns
        -------
        dict
            Dictionary of all covariate values.
        """
        result = {}
        for name in self.covariates:
            result[name] = self.get_value(name)
        return result

    def sample_configuration(
        self,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, Any]:
        """Sample a random configuration of all covariates.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        dict
            Dictionary of sampled covariate values.
        """
        if rng is None:
            rng = np.random.default_rng()

        config = {}
        for name, cov in self.covariates.items():
            config[name] = cov.sample(1, rng)[0] if cov.covariate_type == "continuous" else cov.sample(1, rng)[0]
        return config

    def generate_grid(self, **kwargs) -> List[Dict[str, Any]]:
        """Generate a grid of covariate combinations.

        Parameters
        ----------
        **kwargs
            Covariate names mapped to lists of values to use.
            Covariates not specified use their default or first value.

        Returns
        -------
        list
            List of dictionaries, one per combination.
        """
        import itertools

        # Build value lists for each covariate
        value_lists = {}
        for name, cov in self.covariates.items():
            if name in kwargs:
                value_lists[name] = kwargs[name]
            elif cov.covariate_type in ("categorical", "ordinal"):
                value_lists[name] = [cov.values[0]]
            else:
                value_lists[name] = [cov.default if cov.default is not None else cov.values[0]]

        # Generate all combinations
        names = list(value_lists.keys())
        all_values = [value_lists[n] for n in names]

        grid = []
        for combo in itertools.product(*all_values):
            grid.append(dict(zip(names, combo)))

        return grid

    @property
    def n_covariates(self) -> int:
        """int: Number of covariates in the system."""
        return len(self.covariates)

    def __len__(self) -> int:
        return self.n_covariates

    def __contains__(self, name: str) -> bool:
        return name in self.covariates

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "covariates": {n: c.to_dict() for n, c in self.covariates.items()},
            "current_values": self.current_values.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CovariateSystem":
        """Create from dictionary."""
        covariates = {
            n: Covariate.from_dict(c)
            for n, c in data.get("covariates", {}).items()
        }
        system = cls(
            name=data.get("name", "default"),
            covariates=covariates,
        )
        system.current_values = data.get("current_values", {}).copy()
        return system
