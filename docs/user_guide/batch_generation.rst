Batch Generation
================

This guide covers generating multiple FOVs for training datasets or benchmarking.

Basic Batch Loop
----------------

.. code-block:: python

   import numpy as np
   from pointillsim import (
       TissueCellTypes, CellTypesProperties, HybISS_Setup,
       FOVDistribution, FrameWideElement, RandomCellTypeRule,
   )

   # Shared components
   tissue = TissueCellTypes()
   tissue.generate_types_and_markers(n_genes=50, n_cell_types=5)
   cell_props = CellTypesProperties(n_cell_types=5)

   fov_dist = FOVDistribution(
       frame_size=600,
       background_element=lambda: FrameWideElement(
           frame_size=600,
           tipical_cell_spacing=15,
           rules=RandomCellTypeRule(n_cell_types=5)
       ),
   )

   # Generate batch
   n_fovs = 10
   fov_list = []
   dots_list = []

   for i in range(n_fovs):
       fov = fov_dist.generate_fov()
       cell_props.apply(fov)

       hybiss = HybISS_Setup(tissue)
       hybiss.observe_dots(fov)

       fov_list.append(fov)
       dots_list.append(hybiss.make_pandas_df())

Reproducibility
---------------

Control random seeds for exact replication:

.. code-block:: python

   def generate_reproducible_batch(n_fovs, base_seed=42):
       results = []

       for i in range(n_fovs):
           np.random.seed(base_seed + i)

           fov = fov_dist.generate_fov()
           cell_props.apply(fov)

           results.append({
               'fov_id': i,
               'seed': base_seed + i,
               'fov': fov,
           })

       return results

Combining DataFrames
--------------------

Add FOV identifiers and concatenate:

.. code-block:: python

   import pandas as pd

   all_cells = []
   all_dots = []

   for i, (fov, dots_df) in enumerate(zip(fov_list, dots_list)):
       cells_df = fov.make_pandas_df()
       cells_df['fov_id'] = i
       all_cells.append(cells_df)

       dots_df = dots_df.copy()
       dots_df['fov_id'] = i
       all_dots.append(dots_df)

   combined_cells = pd.concat(all_cells, ignore_index=True)
   combined_dots = pd.concat(all_dots, ignore_index=True)

Variation Strategies
--------------------

Biological Variation
^^^^^^^^^^^^^^^^^^^^

Vary tissue composition across FOVs:

.. code-block:: python

   from pointillsim import MixOfNCellTypesRule

   for i in range(n_fovs):
       # Random proportions
       proportions = np.random.dirichlet([1, 1, 1, 1, 1])

       fov_dist = FOVDistribution(
           frame_size=600,
           background_element=lambda p=proportions: FrameWideElement(
               frame_size=600,
               tipical_cell_spacing=15,
               rules=MixOfNCellTypesRule(
                   n_cell_types=5,
                   list_N=[0, 1, 2, 3, 4],
                   proportions=p.tolist()
               )
           ),
       )

Technical Variation
^^^^^^^^^^^^^^^^^^^

Vary detection sensitivity:

.. code-block:: python

   for i in range(n_fovs):
       sensitivity = np.random.uniform(0.7, 1.3)

       hybiss = HybISS_Setup(
           tissue=tissue,
           genes_sensitivities=sensitivity,
           genes_sensitivities_variation=0.2,
       )

Structural Variation
^^^^^^^^^^^^^^^^^^^^

Vary number of structures:

.. code-block:: python

   for i in range(n_fovs):
       n_structures = np.random.randint(0, 10)

       fov_dist = FOVDistribution(
           ...,
           attempts_at_elements=n_structures,
       )

Saving Datasets
---------------

NPZ Format
^^^^^^^^^^

.. code-block:: python

   from pathlib import Path
   import json

   def save_dataset(output_dir, fov_list, dots_list, tissue):
       output_dir = Path(output_dir)
       output_dir.mkdir(exist_ok=True)

       # Save tissue
       np.savez(
           output_dir / 'tissue.npz',
           gene_expression=tissue.gene_expression_by_type,
       )

       # Save each FOV
       for i, (fov, dots_df) in enumerate(zip(fov_list, dots_list)):
           np.savez(
               output_dir / f'fov_{i:04d}.npz',
               cell_centroids=fov.cell_centroids,
               cell_probabilities=fov.cell_probabilities,
               class_instance=fov.class_instance,
               dot_x=dots_df['x'].values,
               dot_y=dots_df['y'].values,
               dot_gene=dots_df['gene'].values,
               dot_cell=dots_df['cell'].values,
           )

       # Metadata
       with open(output_dir / 'metadata.json', 'w') as f:
           json.dump({'n_fovs': len(fov_list)}, f)

Parquet Format
^^^^^^^^^^^^^^

.. code-block:: python

   combined_cells.to_parquet('cells.parquet', index=False)
   combined_dots.to_parquet('dots.parquet', index=False)
