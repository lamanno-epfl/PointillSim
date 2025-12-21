Observation Model
=================

The observation model simulates how transcripts are detected in spatial transcriptomics experiments.

HybISS_Setup
------------

``HybISS_Setup`` models Poisson-distributed transcript detection:

.. code-block:: python

   from pointillsim import HybISS_Setup, TissueCellTypes

   tissue = TissueCellTypes()
   tissue.generate_types_and_markers(n_genes=50, n_cell_types=5)

   hybiss = HybISS_Setup(
       tissue=tissue,
       genes_sensitivities=1.0,           # Mean detection efficiency
       genes_sensitivities_variation=0.3,  # Variation across genes
   )

Generating Observations
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # After generating FOV and applying cell properties
   hybiss.observe_dots(fov)

   # Get transcript DataFrame
   dots_df = hybiss.make_pandas_df()
   print(dots_df.head())
   #    x      y       gene    cell
   # 0  12.5   45.2   Gene_0    0
   # 1  15.8   42.1   Gene_5    0
   # ...

The Observation Model
^^^^^^^^^^^^^^^^^^^^^

For each cell, the number of transcripts detected per gene follows:

.. math::

   N_{c,g} \\sim \\text{Poisson}(\\lambda_{c,g})

where:

.. math::

   \\lambda_{c,g} = \\text{RNA}_c \\times E_g \\times S_g

- :math:`\\text{RNA}_c`: Cell's RNA concentration
- :math:`E_g`: Expression level of gene g in cell's type
- :math:`S_g`: Gene-specific detection sensitivity

Transfer Functions
------------------

Transfer functions model systematic differences between scRNA-seq references and spatial detection.

IdentityTransfer
^^^^^^^^^^^^^^^^

No transformation (ideal case):

.. code-block:: python

   from pointillsim import IdentityTransfer

   tf = IdentityTransfer()
   # output = input (no change)

AffineNonNegTransfer
^^^^^^^^^^^^^^^^^^^^

Linear transformation with gene-specific effects:

.. code-block:: python

   from pointillsim import AffineNonNegTransfer

   tf = AffineNonNegTransfer(
       scales=1.0,       # Mean scale factor
       scales_std=0.3,   # Variation in scale (some genes detected better)
       offsets=0.0,      # Mean offset (background)
       offsets_std=0.1,  # Variation in offset
   )

   # output = max(0, scale * input + offset)

The transformation models:

- **Gene-specific efficiency**: Different probes work differently
- **Background signal**: Ambient RNA, autofluorescence

Using Transfer Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   hybiss = HybISS_Setup(
       tissue=tissue,
       genes_sensitivities=1.0,
       genes_sensitivities_variation=0.2,
       transfer_function=AffineNonNegTransfer(
           scales=1.0, scales_std=0.4,
           offsets=0.0, offsets_std=0.05
       )
   )

Count Matrix
------------

Access the cell × gene count matrix:

.. code-block:: python

   counts = hybiss.cellxgene_counts  # (n_cells, n_genes) array

   # Statistics
   total_per_cell = counts.sum(axis=1)
   total_per_gene = counts.sum(axis=0)

Transcript Placement
--------------------

Transcripts are placed within cell boundaries using a 2D Gaussian distribution centered on the cell centroid:

- Major axis along ``cell_major_axis``
- Minor axis along ``cell_minor_axis``
- Rotated by ``cell_rotation``

This creates realistic dot patterns that follow cell morphology.
