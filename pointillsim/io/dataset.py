"""Dataset generation and loading utilities."""

import os
import copy
import glob
from collections import defaultdict, OrderedDict
import pandas as pd
import tqdm


def generate_dataset(outfolder, fovd, hybISS, tissue, cell_props, n_samples=100):
    """Generate a dataset of FOVs with ground truth and data.

    The function generates a dataset of FOVs with ground truth and data
    and saves them in a folder with the following structure:

    outfolder
    ├── data
    ├── ground_truths
    └── parameters
        └── cell_types.csv

    Parameters
    ----------
    outfolder : str
        The folder where to save the dataset.
    fovd : FOVDistribution
        The distribution of FOVs to sample from.
    hybISS : HybISS_Setup
        The setup for the HybISS experiment.
    tissue : TissueCellTypes
        The tissue with cell types and gene expression patterns.
    cell_props : CellTypesProperties
        The properties of the cell types.
    n_samples : int
        The number of samples to generate.
    """
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    if not os.path.exists(outfolder + "/data"):
        os.makedirs(outfolder + "/data")
    if not os.path.exists(outfolder + "/ground_truths"):
        os.makedirs(outfolder + "/ground_truths")
    if not os.path.exists(outfolder + "/parameters"):
        os.makedirs(outfolder + "/parameters")

    cell_type_df = tissue.make_pandas_df()
    cell_type_df.to_csv(outfolder + "/parameters/cell_types.csv")

    for i in tqdm.tqdm(range(n_samples)):
        fov = fovd.generate_fov()
        cell_props.apply(fov)
        fov_name = f"fov_{i+1:04}"
        ground_truth = fov.make_pandas_df()
        cell_centers = ground_truth.loc[:, ["X", "Y"]]
        ground_truth.to_csv(outfolder + f"/ground_truths/cells_FOV{i+1:04}.csv")
        cell_centers.to_csv(outfolder + f"/data/cell_centroids_FOV{i+1:04}.csv")

        hybISS.measure_gene_expression(fov)
        hybISS.observe_dots(fov)
        ground_truth_dots = hybISS.make_pandas_df()
        data = ground_truth_dots.loc[:, ["x", "y", "gene"]]
        ground_truth_dots.to_csv(outfolder + f"/ground_truths/dots_FOV{i+1:04}.csv")
        data.to_csv(outfolder + f"/data/dots_FOV{i+1:04}.csv")

    tissue_hybISS = copy.deepcopy(tissue)
    tissue_hybISS.gene_expression_by_type = hybISS.M
    cell_type_df_hybISS = tissue_hybISS.make_pandas_df()
    cell_type_df_hybISS.to_csv(outfolder + "/parameters/cell_types_hybISS.csv")


def load_data(outfolder, return_transformed=False):
    """Load all the data in a folder.

    The function assumes the data was generated calling generate_dataset
    but does not assume any of the parameters so it needs to check for
    how many files are in the folder.

    Parameters
    ----------
    outfolder : str
        The folder where the dataset is saved.
    return_transformed : bool, optional
        If True, also returns the transformed expression matrix.

    Returns
    -------
    data : dict
        A dictionary with the data.
    cell_types_info : pd.DataFrame
        The cell types expression matrix.
    cell_types_info_hybISS : pd.DataFrame, optional
        The transformed expression matrix (if return_transformed=True).
    """
    data = defaultdict(OrderedDict)

    for file in sorted(glob.glob(outfolder + "/ground_truths/*.csv")):
        filename = os.path.splitext(os.path.basename(file))[0]
        fov_name = filename.split("_")[-1]
        if "cells" in file:
            data["ground_truth"][fov_name] = pd.read_csv(file, index_col=0)
        elif "dots" in file:
            data["ground_truth_dots"][fov_name] = pd.read_csv(file, index_col=0)

    for file in sorted(glob.glob(outfolder + "/data/*.csv")):
        filename = os.path.splitext(os.path.basename(file))[0]
        fov_name = filename.split("_")[-1]
        if "cell_centroids" in file:
            data["cell_centroids"][fov_name] = pd.read_csv(file, index_col=0)
        elif "dots" in file:
            data["dots"][fov_name] = pd.read_csv(file, index_col=0)

    cell_types_info = pd.read_csv(outfolder + "/parameters/cell_types.csv", index_col=0)

    if return_transformed:
        cell_types_info_hybISS = pd.read_csv(
            outfolder + "/parameters/cell_types_hybISS.csv", index_col=0
        )
        return data, cell_types_info, cell_types_info_hybISS
    else:
        return data, cell_types_info
