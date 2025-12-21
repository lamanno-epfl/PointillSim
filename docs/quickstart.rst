Quickstart
==========

This guide gets you generating simulated spatial transcriptomics data in minutes.

Minimal Example
---------------

.. code-block:: python

   import numpy as np
   from pointillsim import (
       TissueCellTypes,
       CellTypesProperties,
       HybISS_Setup,
       FOVDistribution,
       FrameWideElement,
       RandomCellTypeRule,
   )

   np.random.seed(42)

   # 1. Define tissue expression profiles
   tissue = TissueCellTypes()
   tissue.generate_types_and_markers(
       n_genes=50,       # Number of genes in panel
       n_cell_types=5,   # Number of cell types
   )

   # 2. Define cell morphology
   cell_props = CellTypesProperties(n_cell_types=5)

   # 3. Create FOV generator
   fov_dist = FOVDistribution(
       frame_size=800,
       background_element=lambda: FrameWideElement(
           frame_size=800,
           tipical_cell_spacing=18,
           rules=RandomCellTypeRule(n_cell_types=5)
       ),
   )

   # 4. Generate FOV and apply morphology
   fov = fov_dist.generate_fov()
   cell_props.apply(fov)

   # 5. Generate transcript observations
   hybiss = HybISS_Setup(tissue)
   hybiss.observe_dots(fov)

   # 6. Export
   dots_df = hybiss.make_pandas_df()
   cells_df = fov.make_pandas_df()

   print(f"Generated {len(cells_df)} cells with {len(dots_df)} transcript dots")

Loading Sample Data
-------------------

The fastest way to explore PointillSim is with pre-generated samples:

.. code-block:: python

   from pointillsim import load_sample, list_samples

   # See available samples
   print(list_samples())  # ['simple_fov', 'cortex_like', 'gland_fov', 'mixed_tissue']

   # Load a sample
   fov, tissue, dots_df = load_sample("simple_fov")

   print(f"FOV: {len(fov.cell_centroids)} cells")
   print(f"Tissue: {tissue.n_genes} genes, {tissue.n_cell_types} cell types")
   print(f"Dots: {len(dots_df)} transcripts")

Adding Tissue Structures
------------------------

Real tissues have structures like glands, vessels, and layers:

.. code-block:: python

   from pointillsim import (
       VacuolatedStructure,
       DistanceBasedRule,
       MixOfNCellTypesRule,
   )

   n_cell_types = 5
   frame_size = 600

   # Background: stromal tissue
   background = lambda: FrameWideElement(
       frame_size=frame_size,
       tipical_cell_spacing=15,
       rules=MixOfNCellTypesRule(
           n_cell_types=n_cell_types,
           list_N=[0, 1],
           proportions=[0.7, 0.3]
       )
   )

   # Foreground: glandular structure with lumen
   gland = lambda: VacuolatedStructure(
       frame_size=frame_size,
       scale=80,
       hole_scale_factor=0.5,
       tipical_cell_spacing=10,
       rules=DistanceBasedRule(
           n_cell_types=n_cell_types,
           inner_type=2,  # Near lumen
           outer_type=3,  # At periphery
       )
   )

   fov_dist = FOVDistribution(
       frame_size=frame_size,
       background_element=background,
       other_elements=[gland],
       elements_frequency=[0.8],
       attempts_at_elements=5,
   )

   fov = fov_dist.generate_fov()

Visualization
-------------

.. code-block:: python

   from pointillsim import plot_fov

   fig, axes = plot_fov(fov, tissue, dots_df=dots_df, figsize=(16, 4))
   plt.show()

Or manually:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(8, 8))
   ax.scatter(
       fov.cell_centroids[:, 0],
       fov.cell_centroids[:, 1],
       c=fov.class_instance,
       cmap='Set1',
       s=20, alpha=0.7
   )
   ax.set_aspect('equal')
   ax.set_title(f'{fov.n_cells} cells')
   plt.show()

Export to AnnData
-----------------

For integration with scanpy and other single-cell tools:

.. code-block:: python

   # Export to AnnData
   adata = fov.to_anndata(tissue)

   print(adata)
   # AnnData object with n_obs × n_vars = 1500 × 50
   #     obs: 'cell_type', 'cell_type_prob', ...
   #     obsm: 'spatial'

   # Save
   adata.write_h5ad('simulated_fov.h5ad')

Next Steps
----------

- :doc:`concepts`: Understand the simulation philosophy
- :doc:`user_guide/cell_type_rules`: Learn about spatial patterns
- :doc:`examples/tissue_simulations`: Simulate specific tissues
