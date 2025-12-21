"""Pre-generated sample datasets for PointillSim.

This module provides easy access to sample datasets that can be used for
testing, tutorials, and benchmarking without needing to generate data from scratch.

Available datasets:
- simple_fov: A simple FOV with random cell type distribution
- cortex_like: A cortex-like tissue with layered structure
- gland_fov: A glandular structure with vacuolated elements
- mixed_tissue: A complex tissue with multiple structure types

Example usage:
>>> from pointillsim.data import load_sample
>>> fov, tissue, dots_df = load_sample("simple_fov")
>>> print(f"Cells: {len(fov.cell_centroids)}, Dots: {len(dots_df)}")
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np

# Import core classes for reconstruction
from ..core.fov import FOV
from ..core.tissue import TissueCellTypes


def _get_data_dir() -> Path:
    """Get the path to the data directory."""
    return Path(__file__).parent


def list_samples() -> list:
    """List all available sample datasets.

    Returns
    -------
    list
        List of available sample dataset names.
    """
    data_dir = _get_data_dir()
    samples = []
    for f in data_dir.glob("*.npz"):
        samples.append(f.stem)
    return sorted(samples)


def load_sample(name: str) -> Tuple[FOV, TissueCellTypes, "pd.DataFrame"]:
    """Load a pre-generated sample dataset.

    Parameters
    ----------
    name : str
        Name of the sample dataset to load. Use list_samples() to see available options.

    Returns
    -------
    fov : FOV
        The Field of View object with cell positions and type assignments.
    tissue : TissueCellTypes
        The tissue object with gene expression profiles.
    dots_df : pd.DataFrame
        DataFrame containing transcript dot positions with columns: x, y, gene, cell.

    Raises
    ------
    FileNotFoundError
        If the requested sample dataset does not exist.

    Examples
    --------
    >>> from pointillsim.data import load_sample, list_samples
    >>> print(list_samples())
    ['cortex_like', 'gland_fov', 'mixed_tissue', 'simple_fov']
    >>> fov, tissue, dots_df = load_sample("simple_fov")
    >>> print(f"Number of cells: {len(fov.cell_centroids)}")
    >>> print(f"Number of transcripts: {len(dots_df)}")
    """
    import pandas as pd

    data_dir = _get_data_dir()
    filepath = data_dir / f"{name}.npz"

    if not filepath.exists():
        available = list_samples()
        raise FileNotFoundError(
            f"Sample dataset '{name}' not found. "
            f"Available datasets: {available}"
        )

    # Load the compressed data
    with np.load(filepath, allow_pickle=True) as data:
        # Reconstruct FOV
        fov = FOV(
            cell_centroids=data["cell_centroids"],
            cell_probabilities=data["cell_probabilities"],
        )
        # Set realized cell types
        fov._class_instance_one_hot = data["class_instance_one_hot"]

        # Set morphology if available
        if "cell_minor_axis" in data:
            fov.cell_minor_axis = data["cell_minor_axis"]
            fov.cell_major_axis = data["cell_major_axis"]
            fov.cell_rotation = data["cell_rotation"]
            fov.cell_rna_concentration = data["cell_rna_concentration"]

        # Reconstruct TissueCellTypes
        tissue = TissueCellTypes()
        tissue.gene_expression_by_type = data["expression_matrix"]
        tissue._gene_names = list(data["gene_names"])
        tissue._cell_type_names = list(data["cell_type_names"])

        # Reconstruct dots DataFrame
        dots_df = pd.DataFrame({
            "x": data["dots_x"],
            "y": data["dots_y"],
            "gene": data["dots_gene"],
            "cell": data["dots_cell"],
        })

    return fov, tissue, dots_df


def load_sample_metadata(name: str) -> Dict[str, Any]:
    """Load metadata for a sample dataset without loading the full data.

    Parameters
    ----------
    name : str
        Name of the sample dataset.

    Returns
    -------
    dict
        Dictionary containing metadata about the dataset.
    """
    data_dir = _get_data_dir()
    filepath = data_dir / f"{name}.npz"

    if not filepath.exists():
        available = list_samples()
        raise FileNotFoundError(
            f"Sample dataset '{name}' not found. "
            f"Available datasets: {available}"
        )

    with np.load(filepath, allow_pickle=True) as data:
        metadata = {
            "n_cells": len(data["cell_centroids"]),
            "n_genes": len(data["gene_names"]),
            "n_cell_types": len(data["cell_type_names"]),
            "n_dots": len(data["dots_x"]),
            "gene_names": list(data["gene_names"]),
            "cell_type_names": list(data["cell_type_names"]),
        }
        if "description" in data:
            metadata["description"] = str(data["description"])

    return metadata


__all__ = ["load_sample", "list_samples", "load_sample_metadata"]
