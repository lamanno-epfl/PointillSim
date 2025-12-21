Basic Simulation Example
========================

This example walks through a complete simulation from start to finish.

Setup
-----

.. code-block:: python

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

Step 1: Define Tissue
---------------------

Create expression profiles for 5 cell types with 50 genes:

.. code-block:: python

   tissue = TissueCellTypes()
   tissue.generate_types_and_markers(
       n_genes=50,
       n_cell_types=5,
       expected_level=10.0,
       concentration=0.85,
   )

   print(f"Created tissue: {tissue.n_genes} genes, {tissue.n_cell_types} types")

Step 2: Define Cell Properties
------------------------------

.. code-block:: python

   cell_props = CellTypesProperties(
       n_cell_types=5,
       sizes=[12, 14, 10, 15, 8],
       anisotropy=[0.9, 0.85, 0.95, 0.8, 0.95],
       relative_rna_concentration=[1.0, 1.2, 0.8, 1.5, 0.6],
   )

Step 3: Create FOV Distribution
-------------------------------

.. code-block:: python

   frame_size = 600

   # Background: stromal mixture
   background = lambda: FrameWideElement(
       frame_size=frame_size,
       tipical_cell_spacing=15,
       rules=MixOfNCellTypesRule(
           n_cell_types=5,
           list_N=[0, 1],
           proportions=[0.6, 0.4]
       )
   )

   # Glandular structure
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

Step 4: Generate FOV
--------------------

.. code-block:: python

   fov = fov_dist.generate_fov()
   cell_props.apply(fov)

   print(f"Generated {fov.n_cells} cells")

Step 5: Generate Observations
-----------------------------

.. code-block:: python

   hybiss = HybISS_Setup(
       tissue=tissue,
       genes_sensitivities=1.0,
       genes_sensitivities_variation=0.2,
   )
   hybiss.observe_dots(fov)

   dots_df = hybiss.make_pandas_df()
   print(f"Generated {len(dots_df)} transcript dots")

Step 6: Visualize
-----------------

.. code-block:: python

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
   ax.set_title('Cell Types')

   # Transcripts
   ax = axes[1]
   ax.scatter(dots_df['x'], dots_df['y'], s=1, alpha=0.3, c='red')
   ax.set_aspect('equal')
   ax.set_title(f'Transcripts ({len(dots_df)})')

   # Expression heatmap
   ax = axes[2]
   counts = hybiss.cellxgene_counts
   ax.imshow(np.log1p(counts[:50]), aspect='auto', cmap='viridis')
   ax.set_title('Count Matrix (first 50 cells)')

   plt.tight_layout()
   plt.show()

Step 7: Export
--------------

.. code-block:: python

   # DataFrames
   cells_df = fov.make_pandas_df()
   dots_df = hybiss.make_pandas_df()

   # AnnData
   adata = fov.to_anndata(tissue)
   adata.write_h5ad('simulation.h5ad')

   print("Export complete!")
