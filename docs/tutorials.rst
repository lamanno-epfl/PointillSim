Tutorials
=========

Interactive tutorials are provided as Jupyter notebooks. These cover the full range of PointillSim's capabilities, from basic usage to advanced simulations.

Getting Started
---------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **00_getting_started**
     - Quick introduction to PointillSim. Covers installation verification, basic FOV generation, and first simulation.

   * - **01_simulation_framework**
     - Deep dive into the simulation pipeline. Understand TissueCellTypes, FOVDistribution, and HybISS_Setup.

Core Concepts
-------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **02_elements_and_rules**
     - Histological elements (FrameWideElement, VacuolatedStructure, etc.) and cell type rules (random, single type, spatial).

   * - **05_cell_type_rules**
     - Comprehensive guide to all cell type rules: RandomCellTypeRule, SingleTypeRule, MixOfNCellTypesRule, DistanceBasedRule, LayerRule, GradientRule, ProbabilityNodeFieldRule, CompositeRule.

Working with Data
-----------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **03_batch_generation**
     - Generate multiple FOVs efficiently. Covers seeding for reproducibility, parallel generation, and dataset export.

   * - **04_real_expression_data**
     - Using real gene expression profiles from scRNA-seq. Loading expression matrices, marker gene selection.

   * - **06_tissue_simulations**
     - Creating tissue slices with multiple regions. TissueSlice, RegionSpec, and multi-region simulations.

Advanced Topics
---------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **07_effects_and_realism**
     - Adding technical variation: BatchEffectModel, TechnicalNoise, BackgroundNoise, DropoutModel, SpatialNoise.

   * - **08_simulation_design**
     - Designing simulations for specific benchmarking tasks. Experimental design, parameter sweeps.

   * - **09_validation_and_difficulty**
     - Estimating simulation difficulty. DifficultyEstimator for segmentation, classification, and spatial analysis.

   * - **10_advanced_structures**
     - Complex tissue architectures: LayeredElement, BranchingStructure, LinearLumenStructure, nested elements.

   * - **11_technology_presets**
     - Technology-specific simulations using presets: HybISS, MERFISH, Cartana, Xenium, Visium.

   * - **13_admixture_simulation**
     - Simulating transcript misassignment from segmentation errors (Lateral2DAdmixture) and z-axis effects (ZAxisAdmixture).

Gallery
-------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **12_beautiful_gallery**
     - Visual gallery of tissue architectures. Inspiration for creating realistic tissue simulations.

Running the Tutorials
---------------------

The notebooks are located in the ``notebooks/`` directory of the repository:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/lamanno-epfl/pointillsim.git
   cd pointillsim

   # Install with dev dependencies
   pip install -e ".[dev]"

   # Launch Jupyter
   jupyter notebook notebooks/

Or run them directly in VS Code, JupyterLab, or your preferred notebook environment.

Order of Study
--------------

For a comprehensive understanding, we recommend this order:

1. **00_getting_started** - Basic setup
2. **01_simulation_framework** - Core pipeline
3. **02_elements_and_rules** - Building blocks
4. **05_cell_type_rules** - Cell type assignment
5. **04_real_expression_data** - Using real profiles
6. **03_batch_generation** - Dataset creation
7. **07_effects_and_realism** - Technical variation
8. **11_technology_presets** - Platform simulation
9. **13_admixture_simulation** - Segmentation errors
10. **06_tissue_simulations** - Tissue slices
11. **10_advanced_structures** - Complex architectures
12. **08_simulation_design** - Experimental design
13. **09_validation_and_difficulty** - Difficulty estimation
14. **12_beautiful_gallery** - Inspiration
