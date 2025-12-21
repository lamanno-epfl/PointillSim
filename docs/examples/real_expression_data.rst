Using Real Expression Data
==========================

This example shows how to use scRNA-seq data as input for simulations.

Loading scRNA-seq Data
----------------------

.. code-block:: python

   import anndata as ad
   import numpy as np
   import pandas as pd

   # Load your reference dataset
   adata = ad.read_h5ad("reference_atlas.h5ad")

   print(f"Cells: {adata.n_obs}")
   print(f"Genes: {adata.n_vars}")
   print(f"Cell types: {adata.obs['cell_type'].nunique()}")

Computing Expression Profiles
-----------------------------

.. code-block:: python

   def compute_expression_profiles(adata, cell_type_col='cell_type'):
       """Compute mean expression per cell type."""
       import scipy.sparse as sp

       X = adata.X
       if sp.issparse(X):
           X = X.toarray()

       cell_types = adata.obs[cell_type_col].values
       unique_types = sorted(adata.obs[cell_type_col].unique())

       mean_expr = np.zeros((adata.n_vars, len(unique_types)))
       for i, ct in enumerate(unique_types):
           mask = cell_types == ct
           mean_expr[:, i] = X[mask].mean(axis=0)

       return pd.DataFrame(
           mean_expr,
           index=adata.var_names,
           columns=unique_types
       )

   expr_df = compute_expression_profiles(adata)

Selecting Marker Genes
----------------------

.. code-block:: python

   def select_marker_genes(expr_df, n_per_type=10, min_expr=1.0):
       """Select top marker genes per cell type."""
       selected = []

       for ct in expr_df.columns:
           others = [c for c in expr_df.columns if c != ct]
           max_others = expr_df[others].max(axis=1) + 0.1
           marker_score = expr_df[ct] / max_others
           marker_score = marker_score[expr_df[ct] >= min_expr]
           selected.extend(marker_score.nlargest(n_per_type).index.tolist())

       return list(dict.fromkeys(selected))  # Remove duplicates

   panel_genes = select_marker_genes(expr_df, n_per_type=15)
   print(f"Selected {len(panel_genes)} genes")

Creating TissueCellTypes
------------------------

.. code-block:: python

   from pointillsim import TissueCellTypes

   # Subset to panel genes
   expr_panel = expr_df.loc[panel_genes]

   # Create tissue object
   tissue = TissueCellTypes()
   tissue.gene_expression_by_type = expr_panel.values.copy()
   tissue._gene_names = list(expr_panel.index)
   tissue._cell_type_names = list(expr_panel.columns)

   print(f"Tissue: {tissue.n_genes} genes, {tissue.n_cell_types} types")

Adding Transfer Function
------------------------

Model platform-specific detection effects:

.. code-block:: python

   from pointillsim import HybISS_Setup, AffineNonNegTransfer

   hybiss = HybISS_Setup(
       tissue=tissue,
       genes_sensitivities=1.0,
       genes_sensitivities_variation=0.3,
       transfer_function=AffineNonNegTransfer(
           scales=1.0, scales_std=0.4,
           offsets=0.0, offsets_std=0.05
       )
   )

Complete Example
----------------

.. code-block:: python

   from pointillsim import (
       CellTypesProperties, FOVDistribution, FrameWideElement,
       VacuolatedStructure, MixOfNCellTypesRule, DistanceBasedRule,
   )

   n_cell_types = tissue.n_cell_types
   frame_size = 600

   # Cell properties
   cell_props = CellTypesProperties(n_cell_types=n_cell_types)

   # FOV distribution
   fov_dist = FOVDistribution(
       frame_size=frame_size,
       background_element=lambda: FrameWideElement(
           frame_size=frame_size,
           tipical_cell_spacing=15,
           rules=MixOfNCellTypesRule(
               n_cell_types=n_cell_types,
               list_N=[0, 1],
               proportions=[0.6, 0.4]
           )
       ),
       other_elements=[
           lambda: VacuolatedStructure(
               frame_size=frame_size,
               scale=70,
               hole_scale_factor=0.5,
               tipical_cell_spacing=10,
               rules=DistanceBasedRule(
                   n_cell_types=n_cell_types,
                   inner_type=2,
                   outer_type=3,
               )
           )
       ],
       elements_frequency=[0.5],
       attempts_at_elements=3,
   )

   # Generate
   np.random.seed(42)
   fov = fov_dist.generate_fov()
   cell_props.apply(fov)
   hybiss.observe_dots(fov)

   # Export
   dots_df = hybiss.make_pandas_df()
   adata_out = fov.to_anndata(tissue)

   print(f"Generated {fov.n_cells} cells, {len(dots_df)} transcripts")
