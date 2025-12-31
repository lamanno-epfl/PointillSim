PointillSim Documentation
=========================

**PointillSim** is a rule-based simulation engine for imaging-based spatial transcriptomics data.

It generates synthetic fields of view (FOVs) with:

- Realistic cell type distributions
- Spatial tissue architecture
- Transcript dot observations
- Ground truth labels for benchmarking

Key Features
------------

- **Compositional architecture**: Build tissues from reusable histological elements
- **Flexible cell type assignment**: Rules for random, spatial, layered, and composite patterns
- **Realistic technical effects**: Noise, dropout, batch effects, and admixture simulation
- **Technology presets**: Pre-configured settings for HybISS, MERFISH, Cartana, Xenium, Visium
- **Multiple export formats**: AnnData, SpatialData, pandas DataFrames

Quick Example
-------------

.. code-block:: python

   from pointillsim import (
       TissueCellTypes, CellTypesProperties, HybISS_Setup,
       FOVDistribution, FrameWideElement, RandomCellTypeRule
   )

   # Define tissue with gene expression profiles
   tissue = TissueCellTypes()
   tissue.generate_types_and_markers(n_genes=50, n_cell_types=10)

   # Create cell type properties
   cell_props = CellTypesProperties(n_cell_types=10)

   # Define FOV distribution
   bg = lambda: FrameWideElement(frame_size=1000, rules=RandomCellTypeRule(10))
   fovd = FOVDistribution(frame_size=1000, background_element=bg)

   # Generate and observe
   fov = fovd.generate_fov()
   cell_props.apply(fov)
   hybiss = HybISS_Setup(tissue)
   hybiss.observe_dots(fov)
   dots_df = hybiss.make_pandas_df()

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   philosophy
   concepts
   use_cases

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/tissue_types
   user_guide/histological_elements
   user_guide/cell_type_rules
   user_guide/observation_model
   user_guide/batch_generation
   user_guide/export_formats

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/elements
   api/rules
   api/observation
   api/effects
   api/config
   api/visualization
   api/io

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_simulation
   examples/tissue_simulations
   examples/real_expression_data

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
