"""Admixture models for simulating transcript misassignment in spatial transcriptomics.

Admixture simulates transcript misassignment due to imperfect segmentation and 3D
tissue complexity. Unlike random noise, admixture is **structured** - contaminating
dots reflect actual neighboring cell types.

Two main sources of admixture are modeled:
1. Lateral 2D admixture: Dots near cell boundaries reassigned to spatial neighbors
2. Z-axis admixture: Contamination from cells above/below the imaging plane
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import KDTree


class AdmixtureModel(ABC):
    """Abstract base class for admixture simulation.

    Admixture models modify dot-to-cell assignments to simulate transcript
    misassignment that occurs in real spatial transcriptomics experiments
    due to segmentation errors and 3D tissue complexity.

    Unlike random noise, admixture is structured - contaminating transcripts
    reflect actual cell type expression profiles of neighboring cells.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    admixture_record : pd.DataFrame or None
        Record of admixture events after apply() is called.
        Contains original_cell, new_cell, dot_index, source columns.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.admixture_record: Optional[pd.DataFrame] = None

    @abstractmethod
    def apply(
        self,
        dots_df: pd.DataFrame,
        cell_centroids: NDArray[np.floating],
        cell_types: NDArray[np.integer],
        cell_radii: Optional[NDArray[np.floating]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Apply admixture to transcript assignments.

        Parameters
        ----------
        dots_df : pd.DataFrame
            DataFrame with columns: x, y, gene, cell (cell index).
        cell_centroids : np.ndarray
            Cell centroid coordinates, shape (n_cells, 2).
        cell_types : np.ndarray
            Cell type indices for each cell, shape (n_cells,).
        cell_radii : np.ndarray, optional
            Estimated radius for each cell (for boundary calculations).

        Returns
        -------
        pd.DataFrame
            Modified dots_df with updated cell assignments.
        """
        pass

    def get_admixture_summary(self) -> Dict[str, Any]:
        """Get summary statistics of admixture events.

        Returns
        -------
        dict
            Dictionary containing:
            - n_total_dots: Total number of dots
            - n_reassigned: Number of dots reassigned
            - reassignment_rate: Fraction of dots reassigned
            - by_source: Count by admixture source type
        """
        if self.admixture_record is None or len(self.admixture_record) == 0:
            return {
                "n_total_dots": 0,
                "n_reassigned": 0,
                "reassignment_rate": 0.0,
                "by_source": {},
            }

        n_reassigned = len(self.admixture_record)
        n_total = self.admixture_record.attrs.get("n_total_dots", n_reassigned)

        by_source = {}
        if "source" in self.admixture_record.columns:
            by_source = self.admixture_record["source"].value_counts().to_dict()

        return {
            "n_total_dots": n_total,
            "n_reassigned": n_reassigned,
            "reassignment_rate": n_reassigned / max(n_total, 1),
            "by_source": by_source,
        }


@dataclass
class Lateral2DAdmixture(AdmixtureModel):
    """Simulate lateral admixture from boundary-based transcript misassignment.

    Models dots near cell boundaries being reassigned to spatial neighbors.
    This is a common segmentation artifact where transcripts near cell edges
    are assigned to the wrong cell.

    Probability of reassignment is highest near cell boundaries and decreases
    toward the cell center following an exponential decay.

    Parameters
    ----------
    boundary_width : float
        Width of the boundary zone (in pixels) where admixture can occur.
        Dots within this distance from the cell edge are candidates.
        Default is 5.0.
    transfer_rate : float
        Base probability of transferring a dot at the cell boundary.
        Default is 0.3 (30% of boundary dots transferred).
    distance_decay : float
        Exponential decay rate for transfer probability with distance from
        boundary. Higher values = sharper decay. Default is 0.5.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    admixture_record : pd.DataFrame
        Record of which dots were reassigned, from which cell, to which cell.

    Examples
    --------
    >>> admixture = Lateral2DAdmixture(boundary_width=10, transfer_rate=0.2)
    >>> dots_df_admixed = admixture.apply(dots_df, cell_centroids, cell_types, cell_radii)
    >>> print(admixture.get_admixture_summary())
    """

    boundary_width: float = 5.0
    transfer_rate: float = 0.3
    distance_decay: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self):
        """Initialize the base class."""
        super().__init__(seed=self.seed)

        if self.boundary_width <= 0:
            raise ValueError("boundary_width must be positive")
        if not 0 <= self.transfer_rate <= 1:
            raise ValueError("transfer_rate must be in [0, 1]")
        if self.distance_decay < 0:
            raise ValueError("distance_decay must be non-negative")

    def apply(
        self,
        dots_df: pd.DataFrame,
        cell_centroids: NDArray[np.floating],
        cell_types: NDArray[np.integer],
        cell_radii: Optional[NDArray[np.floating]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Apply lateral 2D admixture to transcript assignments.

        Parameters
        ----------
        dots_df : pd.DataFrame
            DataFrame with columns: x, y, gene, cell.
        cell_centroids : np.ndarray
            Cell centroid coordinates, shape (n_cells, 2).
        cell_types : np.ndarray
            Cell type indices, shape (n_cells,).
        cell_radii : np.ndarray, optional
            Radius for each cell. If None, estimates from typical cell size.

        Returns
        -------
        pd.DataFrame
            Modified DataFrame with updated cell assignments.
        """
        del kwargs  # unused

        if len(dots_df) == 0 or len(cell_centroids) == 0:
            self.admixture_record = pd.DataFrame(
                columns=["dot_index", "original_cell", "new_cell", "source"]
            )
            self.admixture_record.attrs["n_total_dots"] = len(dots_df)
            return dots_df.copy()

        dots_df = dots_df.copy()
        n_cells = len(cell_centroids)

        # Estimate cell radii if not provided
        if cell_radii is None:
            if n_cells > 1:
                tree = KDTree(cell_centroids)
                dists, _ = tree.query(cell_centroids, k=2)
                avg_spacing = np.mean(dists[:, 1])
                cell_radii = np.full(n_cells, avg_spacing / 2.5)
            else:
                cell_radii = np.full(n_cells, 10.0)

        cell_tree = KDTree(cell_centroids)

        # --- Vectorized distance computation ---
        dot_positions = dots_df[["x", "y"]].values
        dot_cells = dots_df["cell"].values.astype(int)

        valid_mask = (dot_cells >= 0) & (dot_cells < n_cells)
        safe_cells = np.clip(dot_cells, 0, n_cells - 1)

        assigned_centers = cell_centroids[safe_cells]
        dists_to_center = np.linalg.norm(dot_positions - assigned_centers, axis=1)
        assigned_radii = cell_radii[safe_cells]
        dist_from_boundary = assigned_radii - dists_to_center

        # Dots in boundary zone
        boundary_mask = (
            valid_mask
            & (dist_from_boundary > 0)
            & (dist_from_boundary <= self.boundary_width)
        )

        # Transfer probability for boundary dots
        transfer_probs = np.zeros(len(dots_df))
        transfer_probs[boundary_mask] = self.transfer_rate * np.exp(
            -self.distance_decay * dist_from_boundary[boundary_mask]
        )

        # Stochastic selection
        rand_vals = self.rng.random(len(dots_df))
        transfer_mask = boundary_mask & (rand_vals < transfer_probs)
        transfer_indices = np.where(transfer_mask)[0]

        if len(transfer_indices) > 0:
            transfer_positions = dot_positions[transfer_indices]
            transfer_cells = dot_cells[transfer_indices]

            # Batch KDTree query for all transferring dots at once
            _, all_neighbors = cell_tree.query(transfer_positions, k=3)

            # Pick first neighbor that differs from the current cell
            new_cells = np.full(len(transfer_indices), -1, dtype=int)
            for k_col in range(3):
                neighbor_col = all_neighbors[:, k_col]
                can_assign = (
                    (neighbor_col != transfer_cells)
                    & (neighbor_col < n_cells)
                    & (new_cells == -1)
                )
                new_cells[can_assign] = neighbor_col[can_assign]

            # Apply reassignments in bulk
            has_new = new_cells >= 0
            actual_transfer = transfer_indices[has_new]
            actual_new_cells = new_cells[has_new]

            cell_col = dots_df["cell"].values.copy()
            original_cells = cell_col[actual_transfer].copy()
            cell_col[actual_transfer] = actual_new_cells
            dots_df["cell"] = cell_col

            admixture_events = pd.DataFrame(
                {
                    "dot_index": actual_transfer,
                    "original_cell": original_cells,
                    "new_cell": actual_new_cells,
                    "original_type": cell_types[original_cells],
                    "new_type": cell_types[actual_new_cells],
                    "source": "lateral_2d",
                }
            )
        else:
            admixture_events = pd.DataFrame(
                columns=["dot_index", "original_cell", "new_cell", "source"]
            )

        self.admixture_record = admixture_events
        self.admixture_record.attrs["n_total_dots"] = len(dots_df)

        return dots_df


@dataclass
class ZAxisAdmixture(AdmixtureModel):
    """Simulate Z-axis admixture from out-of-plane cell contamination.

    Models contamination from cells above/below the imaging plane (tissue
    section thickness). These "virtual" z-neighbors are not visible in 2D
    but can contribute transcripts to detected cells.

    Key insight: Z-neighbors are NOT random. Due to 3D tissue continuity,
    the cell type composition along the z-axis is correlated with the local
    2D neighborhood composition.

    Parameters
    ----------
    z_contamination_rate : float
        Fraction of dots that come from z-axis contamination.
        Default is 0.1 (10% of dots are from z-neighbors).
    neighborhood_correlation : float
        How strongly z-neighbor types correlate with 2D neighborhood.
        0.0 = random sampling from global type distribution
        1.0 = sample from local 2D neighborhood composition
        Default is 0.7.
    neighborhood_radius : float
        Radius for defining local neighborhood (in pixels).
        Default is 50.0.
    section_thickness : float
        Tissue section thickness in micrometers (for documentation).
        Default is 10.0.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> z_admix = ZAxisAdmixture(z_contamination_rate=0.15, neighborhood_correlation=0.8)
    >>> dots_df_admixed = z_admix.apply(dots_df, cell_centroids, cell_types)
    """

    z_contamination_rate: float = 0.1
    neighborhood_correlation: float = 0.7
    neighborhood_radius: float = 50.0
    section_thickness: float = 10.0
    seed: Optional[int] = None

    def __post_init__(self):
        """Initialize the base class."""
        super().__init__(seed=self.seed)

        if not 0 <= self.z_contamination_rate <= 1:
            raise ValueError("z_contamination_rate must be in [0, 1]")
        if not 0 <= self.neighborhood_correlation <= 1:
            raise ValueError("neighborhood_correlation must be in [0, 1]")
        if self.neighborhood_radius <= 0:
            raise ValueError("neighborhood_radius must be positive")

    def _get_neighborhood_type_distribution(
        self,
        cell_idx: int,
        cell_centroids: NDArray[np.floating],
        cell_types: NDArray[np.integer],
        n_cell_types: int,
        cell_tree: KDTree,
    ) -> NDArray[np.floating]:
        """Get cell type distribution in a cell's local neighborhood.

        Parameters
        ----------
        cell_idx : int
            Index of the cell.
        cell_centroids : np.ndarray
            All cell centroids.
        cell_types : np.ndarray
            Cell type indices.
        n_cell_types : int
            Number of unique cell types.
        cell_tree : KDTree
            KD-tree built from cell centroids.

        Returns
        -------
        np.ndarray
            Probability distribution over cell types.
        """
        center = cell_centroids[cell_idx]

        # Find cells within neighborhood radius
        neighbor_indices = cell_tree.query_ball_point(center, self.neighborhood_radius)

        # Exclude the cell itself
        neighbor_indices = [i for i in neighbor_indices if i != cell_idx]

        if len(neighbor_indices) == 0:
            # No neighbors, return uniform
            return np.ones(n_cell_types) / n_cell_types

        # Count cell types in neighborhood
        neighbor_types = cell_types[neighbor_indices]
        type_counts = np.bincount(neighbor_types, minlength=n_cell_types)

        # Convert to probability
        return type_counts / type_counts.sum()

    def apply(
        self,
        dots_df: pd.DataFrame,
        cell_centroids: NDArray[np.floating],
        cell_types: NDArray[np.integer],
        cell_radii: Optional[NDArray[np.floating]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Apply Z-axis admixture to transcript assignments.

        Parameters
        ----------
        dots_df : pd.DataFrame
            DataFrame with columns: x, y, gene, cell.
        cell_centroids : np.ndarray
            Cell centroid coordinates, shape (n_cells, 2).
        cell_types : np.ndarray
            Cell type indices, shape (n_cells,).
        cell_radii : np.ndarray, optional
            Not used, kept for API consistency.

        Returns
        -------
        pd.DataFrame
            Modified DataFrame with some dots reassigned.
        """
        del kwargs  # unused
        _ = cell_radii  # unused

        if len(dots_df) == 0 or len(cell_centroids) == 0:
            self.admixture_record = pd.DataFrame(
                columns=["dot_index", "original_cell", "new_cell", "source"]
            )
            self.admixture_record.attrs["n_total_dots"] = len(dots_df)
            return dots_df.copy()

        dots_df = dots_df.copy()
        n_cells = len(cell_centroids)
        n_cell_types = int(cell_types.max()) + 1

        # Global cell type distribution
        global_type_counts = np.bincount(cell_types, minlength=n_cell_types)
        global_type_dist = global_type_counts / global_type_counts.sum()

        cell_tree = KDTree(cell_centroids)

        n_dots = len(dots_df)
        n_contaminated = int(n_dots * self.z_contamination_rate)

        if n_contaminated == 0:
            self.admixture_record = pd.DataFrame(
                columns=["dot_index", "original_cell", "new_cell", "source"]
            )
            self.admixture_record.attrs["n_total_dots"] = len(dots_df)
            return dots_df

        contaminated_indices = self.rng.choice(
            n_dots, size=n_contaminated, replace=False
        )

        # --- Precompute lookups ---
        # cells_by_type: precompute once instead of per-dot np.where
        cells_by_type = [
            np.where(cell_types == t)[0] for t in range(n_cell_types)
        ]

        # Extract cell assignments as numpy array (avoid DataFrame row access)
        all_dot_cells = dots_df["cell"].values.astype(int)
        original_cells = all_dot_cells[contaminated_indices].copy()

        # Identify unique cells among contaminated dots and precompute
        # their neighborhood distributions (expensive KDTree query done
        # once per unique cell instead of once per dot)
        unique_cells = np.unique(original_cells)
        unique_cells = unique_cells[
            (unique_cells >= 0) & (unique_cells < n_cells)
        ]
        neighborhood_dists = {}
        for cell_idx in unique_cells:
            neighborhood_dists[cell_idx] = (
                self._get_neighborhood_type_distribution(
                    cell_idx, cell_centroids, cell_types,
                    n_cell_types, cell_tree,
                )
            )

        # --- Process contaminated dots grouped by original cell ---
        new_cells = np.full(n_contaminated, -1, dtype=int)

        # Group dot indices by their original cell for batch processing
        sort_order = np.argsort(original_cells)
        sorted_orig = original_cells[sort_order]
        split_points = np.searchsorted(
            sorted_orig,
            unique_cells,
            side="left",
        )
        split_points_right = np.searchsorted(
            sorted_orig,
            unique_cells,
            side="right",
        )

        for g, cell_idx in enumerate(unique_cells):
            local_dist = neighborhood_dists.get(cell_idx)
            if local_dist is None:
                continue

            # Blended distribution for this cell
            blended = (
                self.neighborhood_correlation * local_dist
                + (1 - self.neighborhood_correlation) * global_type_dist
            )
            current_type = cell_types[cell_idx]
            blended[current_type] *= 0.1
            bsum = blended.sum()
            if bsum > 0:
                blended /= bsum
            else:
                blended = global_type_dist.copy()

            # All dots in this group share the same blended distribution
            grp_start = split_points[g]
            grp_end = split_points_right[g]
            grp_sorted_idx = sort_order[grp_start:grp_end]
            n_grp = len(grp_sorted_idx)
            if n_grp == 0:
                continue

            # Batch-sample z-neighbor types for all dots in group
            z_types = self.rng.choice(n_cell_types, size=n_grp, p=blended)

            # For each unique z_type in this group, compute source cells
            unique_z = np.unique(z_types)
            for zt in unique_z:
                candidates = cells_by_type[zt]
                if len(candidates) == 0:
                    continue
                zt_mask = z_types == zt
                zt_indices = grp_sorted_idx[zt_mask]
                n_zt = len(zt_indices)

                # Distance-weighted sampling from candidates
                dists = np.linalg.norm(
                    cell_centroids[candidates] - cell_centroids[cell_idx],
                    axis=1,
                )
                weights = 1.0 / (dists + 1.0)
                weights /= weights.sum()
                source_cells = self.rng.choice(
                    candidates, size=n_zt, p=weights
                )
                new_cells[zt_indices] = source_cells

        # --- Apply reassignments in bulk ---
        has_new = new_cells >= 0
        actual_contam_pos = np.where(has_new)[0]
        actual_dot_indices = contaminated_indices[actual_contam_pos]
        actual_new = new_cells[actual_contam_pos]
        actual_orig = original_cells[actual_contam_pos]

        cell_col = dots_df["cell"].values.copy()
        cell_col[actual_dot_indices] = actual_new
        dots_df["cell"] = cell_col

        admixture_events = pd.DataFrame(
            {
                "dot_index": actual_dot_indices,
                "original_cell": actual_orig,
                "new_cell": actual_new,
                "original_type": cell_types[actual_orig],
                "new_type": cell_types[actual_new],
                "source": "z_axis",
            }
        )

        self.admixture_record = admixture_events
        self.admixture_record.attrs["n_total_dots"] = len(dots_df)

        return dots_df


@dataclass
class CompositeAdmixture(AdmixtureModel):
    """Combine multiple admixture models.

    Applies multiple admixture effects sequentially. Useful for simulating
    both lateral and z-axis admixture together.

    Parameters
    ----------
    models : list of AdmixtureModel
        List of admixture models to apply in order.
    seed : int, optional
        Random seed (passed to component models if they don't have one).

    Examples
    --------
    >>> lateral = Lateral2DAdmixture(transfer_rate=0.2)
    >>> z_axis = ZAxisAdmixture(z_contamination_rate=0.1)
    >>> combined = CompositeAdmixture([lateral, z_axis])
    >>> dots_df_admixed = combined.apply(dots_df, cell_centroids, cell_types)
    """

    models: List[AdmixtureModel] = field(default_factory=list)
    seed: Optional[int] = None

    def __post_init__(self):
        """Initialize base class."""
        super().__init__(seed=self.seed)

    def apply(
        self,
        dots_df: pd.DataFrame,
        cell_centroids: NDArray[np.floating],
        cell_types: NDArray[np.integer],
        cell_radii: Optional[NDArray[np.floating]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Apply all admixture models sequentially.

        Parameters
        ----------
        dots_df : pd.DataFrame
            Input dots DataFrame.
        cell_centroids : np.ndarray
            Cell centroids.
        cell_types : np.ndarray
            Cell type indices.
        cell_radii : np.ndarray, optional
            Cell radii.

        Returns
        -------
        pd.DataFrame
            Dots DataFrame after all admixture models applied.
        """
        result = dots_df.copy()

        all_events = []
        for model in self.models:
            result = model.apply(
                result, cell_centroids, cell_types, cell_radii, **kwargs
            )
            if model.admixture_record is not None and len(model.admixture_record) > 0:
                all_events.append(model.admixture_record)

        # Combine records
        if all_events:
            self.admixture_record = pd.concat(all_events, ignore_index=True)
            self.admixture_record.attrs["n_total_dots"] = len(dots_df)
        else:
            self.admixture_record = pd.DataFrame(
                columns=["dot_index", "original_cell", "new_cell", "source"]
            )
            self.admixture_record.attrs["n_total_dots"] = len(dots_df)

        return result


@dataclass
class AdmixtureMetrics:
    """Compute and track admixture metrics from simulation data.

    Provides detailed analysis of admixture effects including per-cell
    contamination, source type distributions, and spatial patterns.

    Parameters
    ----------
    admixture_record : pd.DataFrame
        Record of admixture events from an AdmixtureModel.
    dots_df : pd.DataFrame
        Original or admixed dots DataFrame.
    cell_types : np.ndarray
        Cell type assignments.
    cell_centroids : np.ndarray
        Cell centroid positions.

    Attributes
    ----------
    per_cell_contamination : np.ndarray
        Fraction of each cell's assigned dots that came from other cells.
    contamination_matrix : np.ndarray
        Matrix of contamination flow between cell types.
        Shape (n_types, n_types), entry [i,j] = dots from type i assigned to type j.
    """

    admixture_record: pd.DataFrame
    dots_df: pd.DataFrame
    cell_types: NDArray[np.integer]
    cell_centroids: NDArray[np.floating]

    def __post_init__(self):
        """Compute metrics on initialization."""
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute all admixture metrics."""
        n_cells = len(self.cell_types)
        n_types = int(self.cell_types.max()) + 1 if len(self.cell_types) > 0 else 0

        # Per-cell contamination
        self.per_cell_contamination = np.zeros(n_cells)
        self.per_cell_dots_received = np.zeros(n_cells)
        self.per_cell_dots_lost = np.zeros(n_cells)

        if len(self.admixture_record) > 0:
            # Count dots lost by each cell
            for _, event in self.admixture_record.iterrows():
                orig = int(event["original_cell"])
                new = int(event["new_cell"])
                if 0 <= orig < n_cells:
                    self.per_cell_dots_lost[orig] += 1
                if 0 <= new < n_cells:
                    self.per_cell_dots_received[new] += 1

        # Count original dots per cell
        cell_dot_counts = np.zeros(n_cells)
        if "cell" in self.dots_df.columns:
            for cell_idx in self.dots_df["cell"]:
                if 0 <= cell_idx < n_cells:
                    cell_dot_counts[int(cell_idx)] += 1

        # Contamination fraction
        total_dots = cell_dot_counts + self.per_cell_dots_lost
        with np.errstate(divide="ignore", invalid="ignore"):
            self.per_cell_contamination = np.where(
                total_dots > 0,
                self.per_cell_dots_received / total_dots,
                0.0,
            )

        # Contamination matrix (flow between cell types)
        self.contamination_matrix = np.zeros((n_types, n_types))
        if len(self.admixture_record) > 0 and "original_type" in self.admixture_record.columns:
            for _, event in self.admixture_record.iterrows():
                orig_type = int(event["original_type"])
                new_type = int(event["new_type"])
                if 0 <= orig_type < n_types and 0 <= new_type < n_types:
                    self.contamination_matrix[orig_type, new_type] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of admixture effects.

        Returns
        -------
        dict
            Dictionary containing summary statistics.
        """
        return {
            "total_dots_reassigned": len(self.admixture_record),
            "mean_contamination_per_cell": float(np.mean(self.per_cell_contamination)),
            "max_contamination_per_cell": float(np.max(self.per_cell_contamination))
            if len(self.per_cell_contamination) > 0
            else 0.0,
            "cells_with_contamination": int(np.sum(self.per_cell_contamination > 0)),
            "total_cells": len(self.cell_types),
        }

    def get_contamination_by_type(self) -> pd.DataFrame:
        """Get average contamination rate per cell type.

        Returns
        -------
        pd.DataFrame
            DataFrame with contamination statistics per cell type.
        """
        n_types = len(self.contamination_matrix)
        type_stats = []

        for t in range(n_types):
            cells_of_type = np.where(self.cell_types == t)[0]
            if len(cells_of_type) > 0:
                mean_contam = np.mean(self.per_cell_contamination[cells_of_type])
                dots_lost = int(self.contamination_matrix[t, :].sum())
                dots_gained = int(self.contamination_matrix[:, t].sum())
            else:
                mean_contam = 0.0
                dots_lost = 0
                dots_gained = 0

            type_stats.append(
                {
                    "cell_type": t,
                    "n_cells": len(cells_of_type),
                    "mean_contamination": mean_contam,
                    "dots_lost": dots_lost,
                    "dots_gained": dots_gained,
                    "net_flow": dots_gained - dots_lost,
                }
            )

        return pd.DataFrame(type_stats)

    def get_spatial_contamination_pattern(
        self, grid_size: int = 10
    ) -> NDArray[np.floating]:
        """Compute spatial pattern of admixture intensity.

        Divides the FOV into a grid and computes average contamination
        in each grid cell.

        Parameters
        ----------
        grid_size : int
            Number of grid cells per dimension.

        Returns
        -------
        np.ndarray
            2D array of shape (grid_size, grid_size) with contamination values.
        """
        if len(self.cell_centroids) == 0:
            return np.zeros((grid_size, grid_size))

        # Get FOV bounds
        x_min, y_min = self.cell_centroids.min(axis=0)
        x_max, y_max = self.cell_centroids.max(axis=0)

        x_bins = np.linspace(x_min, x_max + 1e-6, grid_size + 1)
        y_bins = np.linspace(y_min, y_max + 1e-6, grid_size + 1)

        # Assign cells to grid bins
        x_idx = np.digitize(self.cell_centroids[:, 0], x_bins) - 1
        y_idx = np.digitize(self.cell_centroids[:, 1], y_bins) - 1

        # Clip to valid range
        x_idx = np.clip(x_idx, 0, grid_size - 1)
        y_idx = np.clip(y_idx, 0, grid_size - 1)

        # Compute average contamination per grid cell
        spatial_pattern = np.zeros((grid_size, grid_size))
        counts = np.zeros((grid_size, grid_size))

        for i, (xi, yi) in enumerate(zip(x_idx, y_idx)):
            spatial_pattern[yi, xi] += self.per_cell_contamination[i]
            counts[yi, xi] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            spatial_pattern = np.where(counts > 0, spatial_pattern / counts, 0.0)

        return spatial_pattern
