Basic Simulation Example
========================

This example walks through a complete simulation from start to finish.

Setup and Generate Data
-----------------------

.. plot::
   :context: reset
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from pointillsim import (
       TissueCellTypes,
       CellTypesProperties,
       HybISS_Setup,
       FOVDistribution,
       FrameWideElement,
       VacuolatedStructure,
       RandomCellTypeRule,
       MixOfNCellTypesRule,
       DistanceBasedRule,
   )

   np.random.seed(42)

   # Step 1: Define Tissue
   tissue = TissueCellTypes()
   tissue.generate_types_and_markers(
       n_genes=50,
       n_cell_types=5,
       expected_level=10.0,
       concentration=0.85,
   )

   # Step 2: Define Cell Properties
   cell_props = CellTypesProperties(
       n_cell_types=5,
       sizes=[12, 14, 10, 15, 8],
       anisotropy=[0.9, 0.85, 0.95, 0.8, 0.95],
       relative_rna_concentration=[1.0, 1.2, 0.8, 1.5, 0.6],
   )

   # Step 3: Create FOV Distribution
   frame_size = 600

   background = lambda: FrameWideElement(
       frame_size=frame_size,
       tipical_cell_spacing=15,
       rules=MixOfNCellTypesRule(
           n_cell_types=5,
           list_N=[0, 1],
           proportions=[0.6, 0.4]
       )
   )

   gland = lambda: VacuolatedStructure(
       frame_size=frame_size,
       scale=70,
       hole_scale_factor=0.5,
       tipical_cell_spacing=10,
       rules=DistanceBasedRule(
           n_cell_types=5,
           inner_type=2,
           outer_type=3,
       )
   )

   fov_dist = FOVDistribution(
       frame_size=frame_size,
       background_element=background,
       other_elements=[gland],
       elements_frequency=[0.7],
       attempts_at_elements=4,
   )

   # Step 4: Generate FOV
   fov = fov_dist.generate_fov()
   cell_props.apply(fov)

   # Step 5: Generate Observations
   hybiss = HybISS_Setup(
       tissue=tissue,
       genes_sensitivities=1.0,
       genes_sensitivities_variation=0.2,
   )
   hybiss.observe_dots(fov)
   dots_df = hybiss.make_pandas_df()

   print(f"Generated {fov.n_cells} cells and {len(dots_df)} transcript dots")

Visualization: Cell Types
-------------------------

.. plot::
   :context:
   :include-source:

   fig, ax = plt.subplots(figsize=(8, 8))
   scatter = ax.scatter(
       fov.cell_centroids[:, 0],
       fov.cell_centroids[:, 1],
       c=fov.class_instance,
       cmap='Set1',
       s=30, alpha=0.8
   )
   ax.set_aspect('equal')
   ax.set_xlim(0, frame_size)
   ax.set_ylim(0, frame_size)
   ax.set_title(f'Cell Types ({fov.n_cells} cells)', fontsize=14)
   ax.set_xlabel('X position')
   ax.set_ylabel('Y position')
   cbar = plt.colorbar(scatter, ax=ax, label='Cell Type')
   plt.tight_layout()

Visualization: Transcript Dots
------------------------------

.. plot::
   :context:
   :include-source:

   fig, ax = plt.subplots(figsize=(8, 8))
   ax.scatter(dots_df['x'], dots_df['y'], s=1, alpha=0.3, c='crimson')
   ax.set_aspect('equal')
   ax.set_xlim(0, frame_size)
   ax.set_ylim(0, frame_size)
   ax.set_title(f'Transcript Locations ({len(dots_df)} dots)', fontsize=14)
   ax.set_xlabel('X position')
   ax.set_ylabel('Y position')
   plt.tight_layout()

Visualization: Count Matrix
---------------------------

.. plot::
   :context:
   :include-source:

   fig, ax = plt.subplots(figsize=(10, 6))
   counts = hybiss.cellxgene_counts
   im = ax.imshow(np.log1p(counts[:50]).T, aspect='auto', cmap='viridis')
   ax.set_title('Gene Expression (first 50 cells, log1p)', fontsize=14)
   ax.set_xlabel('Cell Index')
   ax.set_ylabel('Gene Index')
   plt.colorbar(im, ax=ax, label='log1p(counts)')
   plt.tight_layout()

Combined Overview
-----------------

.. plot::
   :context:
   :include-source:

   fig, axes = plt.subplots(1, 3, figsize=(15, 5))

   # Cell types
   ax = axes[0]
   ax.scatter(
       fov.cell_centroids[:, 0],
       fov.cell_centroids[:, 1],
       c=fov.class_instance,
       cmap='Set1',
       s=20, alpha=0.7
   )
   ax.set_aspect('equal')
   ax.set_xlim(0, frame_size)
   ax.set_ylim(0, frame_size)
   ax.set_title('Cell Types')

   # Transcripts
   ax = axes[1]
   ax.scatter(dots_df['x'], dots_df['y'], s=1, alpha=0.3, c='red')
   ax.set_aspect('equal')
   ax.set_xlim(0, frame_size)
   ax.set_ylim(0, frame_size)
   ax.set_title(f'Transcripts ({len(dots_df)})')

   # Expression heatmap
   ax = axes[2]
   counts = hybiss.cellxgene_counts
   ax.imshow(np.log1p(counts[:50]).T, aspect='auto', cmap='viridis')
   ax.set_title('Count Matrix (first 50 cells)')
   ax.set_xlabel('Cell Index')
   ax.set_ylabel('Gene Index')

   plt.tight_layout()

Export Options
--------------

.. code-block:: python

   # DataFrames
   cells_df = fov.make_pandas_df()
   dots_df = hybiss.make_pandas_df()

   # AnnData
   adata = fov.to_anndata(tissue)
   adata.write_h5ad('simulation.h5ad')

   print("Export complete!")
