Tutorials
=========

Interactive tutorials are provided as Jupyter notebooks that are rendered directly in this documentation.

Getting Started
---------------

.. toctree::
   :maxdepth: 1

   notebooks/00_getting_started
   notebooks/01_simulation_framework

Core Concepts
-------------

.. toctree::
   :maxdepth: 1

   notebooks/02_elements_and_rules
   notebooks/05_cell_type_rules

Working with Data
-----------------

.. toctree::
   :maxdepth: 1

   notebooks/03_batch_generation
   notebooks/04_real_expression_data
   notebooks/06_tissue_simulations

Advanced Topics
---------------

.. toctree::
   :maxdepth: 1

   notebooks/07_effects_and_realism
   notebooks/08_simulation_design
   notebooks/09_validation_and_difficulty
   notebooks/10_advanced_structures
   notebooks/11_technology_presets
   notebooks/13_admixture_simulation

Gallery
-------

.. toctree::
   :maxdepth: 1

   notebooks/12_beautiful_gallery

Running Locally
---------------

The notebooks are also available in the ``notebooks/`` directory of the repository:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/lamanno-epfl/pointillsim.git
   cd pointillsim

   # Install with dev dependencies
   pip install -e ".[dev]"

   # Launch Jupyter
   jupyter notebook notebooks/

Recommended Order
-----------------

For a comprehensive understanding, we recommend this order:

1. **00_getting_started** - Basic setup and first simulation
2. **01_simulation_framework** - Core pipeline understanding
3. **02_elements_and_rules** - Building blocks of tissue
4. **05_cell_type_rules** - Cell type assignment strategies
5. **04_real_expression_data** - Using real gene profiles
6. **03_batch_generation** - Creating datasets
7. **07_effects_and_realism** - Adding technical variation
8. **11_technology_presets** - Platform-specific simulation
9. **13_admixture_simulation** - Segmentation error modeling
10. **06_tissue_simulations** - Tissue slices
11. **10_advanced_structures** - Complex architectures
12. **08_simulation_design** - Experimental design
13. **09_validation_and_difficulty** - Difficulty estimation
14. **12_beautiful_gallery** - Visual inspiration
