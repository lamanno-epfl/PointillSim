Export Formats
==============

PointillSim supports multiple export formats for different analysis workflows.

Pandas DataFrames
-----------------

The most basic export format.

Cells DataFrame
^^^^^^^^^^^^^^^

.. code-block:: python

   cells_df = fov.make_pandas_df()

Columns:

- ``X``, ``Y``: Cell centroid coordinates
- ``Class ID``: Realized cell type index
- ``Class 0``, ``Class 1``, ...: One-hot encoded cell type
- ``ProbClass0``, ``ProbClass1``, ...: Soft probability ground truth
- ``Minor Axis``, ``Major Axis``, ``Rotation``: Cell morphology
- ``RNA Concentration``: Relative RNA content

Dots DataFrame
^^^^^^^^^^^^^^

.. code-block:: python

   dots_df = hybiss.make_pandas_df()

Columns:

- ``x``, ``y``: Transcript coordinates
- ``gene``: Gene name
- ``cell``: Index of the containing cell

Saving
^^^^^^

.. code-block:: python

   # CSV
   cells_df.to_csv('cells.csv', index=False)
   dots_df.to_csv('dots.csv', index=False)

   # Parquet (more efficient)
   cells_df.to_parquet('cells.parquet', index=False)
   dots_df.to_parquet('dots.parquet', index=False)

AnnData
-------

For integration with scanpy and the single-cell ecosystem.

.. code-block:: python

   # Export to AnnData
   adata = fov.to_anndata(tissue)

   print(adata)
   # AnnData object with n_obs × n_vars = 1500 × 50
   #     obs: 'cell_type', 'cell_type_prob_0', ...
   #     var: 'gene_names'
   #     obsm: 'spatial'
   #     layers: 'counts'

Structure:

- ``adata.X``: Count matrix (cells × genes)
- ``adata.obs``: Cell metadata (type, probabilities, morphology)
- ``adata.var``: Gene metadata
- ``adata.obsm['spatial']``: Spatial coordinates
- ``adata.layers['counts']``: Raw counts

Saving:

.. code-block:: python

   adata.write_h5ad('fov.h5ad')

SpatialData
-----------

For integration with the spatialdata ecosystem.

.. code-block:: python

   from pointillsim import fov_to_spatialdata

   sdata = fov_to_spatialdata(fov, tissue, dots_df=dots_df)

   print(sdata)
   # SpatialData object
   #   Tables: 'cells'
   #   Points: 'transcripts'
   #   Shapes: 'cell_boundaries'

Saving:

.. code-block:: python

   sdata.write('fov.zarr')

NumPy Arrays (NPZ)
------------------

Lightweight format for ML pipelines.

.. code-block:: python

   import numpy as np

   np.savez(
       'fov.npz',
       cell_centroids=fov.cell_centroids,
       cell_probabilities=fov.cell_probabilities,
       class_instance=fov.class_instance,
       cell_major_axis=fov.cell_major_axis,
       cell_minor_axis=fov.cell_minor_axis,
       dot_x=dots_df['x'].values,
       dot_y=dots_df['y'].values,
       dot_gene=dots_df['gene'].values,
       dot_cell=dots_df['cell'].values,
   )

   # Load
   data = np.load('fov.npz')
   centroids = data['cell_centroids']

Choosing a Format
-----------------

.. list-table::
   :header-rows: 1

   * - Format
     - Pros
     - Cons
     - Best For
   * - CSV
     - Universal
     - Large files
     - Sharing, Excel
   * - Parquet
     - Fast, compact
     - Less universal
     - Python pipelines
   * - AnnData
     - scanpy compatible
     - Requires anndata
     - Single-cell analysis
   * - SpatialData
     - Full spatial support
     - Requires spatialdata
     - Spatial analysis
   * - NPZ
     - Fast, simple
     - Arrays only
     - ML training
