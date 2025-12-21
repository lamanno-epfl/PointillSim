"""Configuration system for PointillSim simulations.

This module provides dataclasses and utilities for configuring simulations
via YAML or JSON files, enabling reproducible and shareable setups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class FOVConfig:
    """Configuration for a single Field of View.

    Parameters
    ----------
    frame_size : int
        Size of the FOV in pixels.
    seed : int, optional
        Random seed for reproducibility.
    """
    frame_size: int = 1000
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CellSpacingConfig:
    """Configuration for cell spacing and density.

    Parameters
    ----------
    background_spacing : float
        Cell spacing for background/stromal cells.
    element_spacing : float
        Cell spacing for histological elements.
    """
    background_spacing: float = 25.0
    element_spacing: float = 12.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ElementConfig:
    """Configuration for a histological element.

    Parameters
    ----------
    element_type : str
        Type of element: 'HistologicalElement', 'VacuolatedStructure',
        'LinearLumenStructure', 'LayeredElement', 'BranchingStructure'.
    scale : float
        Approximate size of the element.
    rule_type : str
        Type of rule to apply: 'RandomCellType', 'SingleType', 'Layer', etc.
    rule_params : dict
        Parameters for the rule.
    extra_params : dict
        Additional element-specific parameters.
    """
    element_type: str = "HistologicalElement"
    scale: float = 100.0
    rule_type: str = "RandomCellType"
    rule_params: Dict[str, Any] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TissueConfig:
    """Configuration for tissue cell types and expression.

    Parameters
    ----------
    n_cell_types : int
        Number of cell types.
    n_genes : int
        Number of genes.
    gene_names : list, optional
        Custom gene names.
    cell_type_names : list, optional
        Custom cell type names.
    expression_file : str, optional
        Path to CSV file with expression matrix.
    """
    n_cell_types: int = 5
    n_genes: int = 50
    gene_names: Optional[List[str]] = None
    cell_type_names: Optional[List[str]] = None
    expression_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Configuration for HybISS experiment parameters.

    Parameters
    ----------
    genes_sensitivities : float
        Mean detection sensitivity per gene.
    genes_sensitivities_variation : float
        Standard deviation of sensitivity (lognormal).
    transfer_function : str
        Transfer function type: 'identity', 'log', 'sqrt'.
    """
    genes_sensitivities: float = 1.0
    genes_sensitivities_variation: float = 0.3
    transfer_function: str = "identity"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SimulationConfig:
    """Complete configuration for a PointillSim simulation.

    This is the main configuration class that combines all sub-configurations.
    Can be loaded from and saved to YAML or JSON files.

    Parameters
    ----------
    name : str
        Name of the simulation configuration.
    description : str
        Description of what this simulation represents.
    fov : FOVConfig
        FOV configuration.
    cell_spacing : CellSpacingConfig
        Cell spacing configuration.
    tissue : TissueConfig
        Tissue cell type configuration.
    experiment : ExperimentConfig
        Experiment parameters.
    elements : list of ElementConfig
        List of histological elements to place.
    n_fovs : int
        Number of FOVs to generate.
    output_dir : str
        Directory for output files.

    Examples
    --------
    >>> config = SimulationConfig.load('simulation.yaml')
    >>> print(config.name)
    'My Simulation'
    >>> config.save('updated_config.yaml')
    """
    name: str = "Untitled Simulation"
    description: str = ""
    fov: FOVConfig = field(default_factory=FOVConfig)
    cell_spacing: CellSpacingConfig = field(default_factory=CellSpacingConfig)
    tissue: TissueConfig = field(default_factory=TissueConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    elements: List[ElementConfig] = field(default_factory=list)
    n_fovs: int = 1
    output_dir: str = "output"

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'fov': self.fov.to_dict(),
            'cell_spacing': self.cell_spacing.to_dict(),
            'tissue': self.tissue.to_dict(),
            'experiment': self.experiment.to_dict(),
            'elements': [e.to_dict() for e in self.elements],
            'n_fovs': self.n_fovs,
            'output_dir': self.output_dir,
        }

    def save(self, filepath: Union[str, Path], format: str = 'auto') -> None:
        """Save configuration to file.

        Parameters
        ----------
        filepath : str or Path
            Path to save the configuration.
        format : str
            File format: 'json', 'yaml', or 'auto' (from extension).
        """
        filepath = Path(filepath)

        if format == 'auto':
            if filepath.suffix in ('.yaml', '.yml'):
                format = 'yaml'
            else:
                format = 'json'

        data = self.to_dict()

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'yaml':
            import yaml
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def load(cls, filepath: Union[str, Path], format: str = 'auto') -> 'SimulationConfig':
        """Load configuration from file.

        Parameters
        ----------
        filepath : str or Path
            Path to the configuration file.
        format : str
            File format: 'json', 'yaml', or 'auto' (from extension).

        Returns
        -------
        SimulationConfig
            Loaded configuration.
        """
        filepath = Path(filepath)

        if format == 'auto':
            if filepath.suffix in ('.yaml', '.yml'):
                format = 'yaml'
            else:
                format = 'json'

        if format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif format == 'yaml':
            import yaml
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown format: {format}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create configuration from dictionary.

        Parameters
        ----------
        data : dict
            Configuration dictionary.

        Returns
        -------
        SimulationConfig
            Configuration object.
        """
        # Parse sub-configurations
        fov = FOVConfig(**data.get('fov', {}))
        cell_spacing = CellSpacingConfig(**data.get('cell_spacing', {}))
        tissue = TissueConfig(**data.get('tissue', {}))
        experiment = ExperimentConfig(**data.get('experiment', {}))

        elements = []
        for elem_data in data.get('elements', []):
            elements.append(ElementConfig(**elem_data))

        return cls(
            name=data.get('name', 'Untitled Simulation'),
            description=data.get('description', ''),
            fov=fov,
            cell_spacing=cell_spacing,
            tissue=tissue,
            experiment=experiment,
            elements=elements,
            n_fovs=data.get('n_fovs', 1),
            output_dir=data.get('output_dir', 'output'),
        )

    def validate(self) -> List[str]:
        """Validate the configuration.

        Returns
        -------
        list of str
            List of validation error messages. Empty if valid.
        """
        errors = []

        if self.fov.frame_size <= 0:
            errors.append("frame_size must be positive")

        if self.tissue.n_cell_types <= 0:
            errors.append("n_cell_types must be positive")

        if self.tissue.n_genes <= 0:
            errors.append("n_genes must be positive")

        if self.n_fovs <= 0:
            errors.append("n_fovs must be positive")

        for i, elem in enumerate(self.elements):
            if elem.scale <= 0:
                errors.append(f"elements[{i}].scale must be positive")

        return errors


def create_example_config() -> SimulationConfig:
    """Create an example simulation configuration.

    Returns
    -------
    SimulationConfig
        Example configuration for demonstration.

    Examples
    --------
    >>> config = create_example_config()
    >>> config.save('example_config.yaml')
    """
    return SimulationConfig(
        name="Example Tissue Simulation",
        description="A simple example with gland-like structures",
        fov=FOVConfig(frame_size=1000, seed=42),
        cell_spacing=CellSpacingConfig(
            background_spacing=25.0,
            element_spacing=12.0,
        ),
        tissue=TissueConfig(
            n_cell_types=5,
            n_genes=50,
            cell_type_names=["Epithelial", "Stromal", "Immune", "Endothelial", "Other"],
        ),
        experiment=ExperimentConfig(
            genes_sensitivities=1.0,
            genes_sensitivities_variation=0.3,
            transfer_function="identity",
        ),
        elements=[
            ElementConfig(
                element_type="VacuolatedStructure",
                scale=80.0,
                rule_type="Layer",
                rule_params={"layer_types": [0, 1], "layer_boundaries": [0.7]},
            ),
        ],
        n_fovs=10,
        output_dir="output/example",
    )
