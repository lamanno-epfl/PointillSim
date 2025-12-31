Configuration Module
====================

The configuration module provides dataclasses for configuring simulations via YAML or JSON files, enabling reproducible and shareable setups.

.. module:: pointillsim.config

Overview
--------

PointillSim configurations can be:

- Saved to and loaded from YAML/JSON files
- Validated before running simulations
- Used to build complete simulation pipelines

Configuration Classes
---------------------

SimulationConfig
^^^^^^^^^^^^^^^^

.. autoclass:: pointillsim.SimulationConfig
   :members:
   :undoc-members:
   :show-inheritance:

FOVConfig
^^^^^^^^^

.. autoclass:: pointillsim.FOVConfig
   :members:
   :undoc-members:
   :show-inheritance:

TissueConfig
^^^^^^^^^^^^

.. autoclass:: pointillsim.TissueConfig
   :members:
   :undoc-members:
   :show-inheritance:

ExperimentConfig
^^^^^^^^^^^^^^^^

.. autoclass:: pointillsim.ExperimentConfig
   :members:
   :undoc-members:
   :show-inheritance:

ElementConfig
^^^^^^^^^^^^^

.. autoclass:: pointillsim.ElementConfig
   :members:
   :undoc-members:
   :show-inheritance:

Factory Functions
-----------------

build_simulation_from_config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pointillsim.build_simulation_from_config

create_example_config
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pointillsim.create_example_config

Usage Examples
--------------

Creating a Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pointillsim import (
       SimulationConfig, FOVConfig, TissueConfig,
       ExperimentConfig, ElementConfig
   )

   config = SimulationConfig(
       name="My Tissue Simulation",
       description="Simulating intestinal crypts",
       fov=FOVConfig(frame_size=1000, seed=42),
       tissue=TissueConfig(
           n_cell_types=5,
           n_genes=100,
           cell_type_names=["Epithelial", "Goblet", "Enteroendocrine", "Stromal", "Immune"]
       ),
       experiment=ExperimentConfig(
           genes_sensitivities=1.0,
           genes_sensitivities_variation=0.3,
           transfer_function="identity"
       ),
       elements=[
           ElementConfig(
               element_type="VacuolatedStructure",
               scale=80.0,
               rule_type="Layer",
               rule_params={"layer_types": [0, 1, 2], "layer_boundaries": [0.6, 0.8]}
           )
       ],
       n_fovs=100,
       output_dir="output/intestine"
   )

Saving and Loading
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Save to YAML
   config.save("my_config.yaml")

   # Save to JSON
   config.save("my_config.json")

   # Load from file
   loaded_config = SimulationConfig.load("my_config.yaml")

Validation
^^^^^^^^^^

.. code-block:: python

   errors = config.validate()
   if errors:
       print("Configuration errors:")
       for error in errors:
           print(f"  - {error}")
   else:
       print("Configuration is valid")

Building Simulations
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pointillsim import build_simulation_from_config

   # Build all simulation components
   sim = build_simulation_from_config(config)

   # Access components
   tissue = sim['tissue']
   cell_props = sim['cell_props']
   fov_distribution = sim['fov_distribution']
   hybiss = sim['hybiss']

   # Generate data
   fov = fov_distribution.generate_fov()
   cell_props.apply(fov)
   hybiss.observe_dots(fov)
   dots_df = hybiss.make_pandas_df()

YAML Configuration Format
-------------------------

.. code-block:: yaml

   name: "Example Simulation"
   description: "Description of what this simulation represents"

   fov:
     frame_size: 1000
     seed: 42

   cell_spacing:
     background_spacing: 25.0
     element_spacing: 12.0

   tissue:
     n_cell_types: 5
     n_genes: 50
     cell_type_names:
       - Epithelial
       - Stromal
       - Immune
       - Endothelial
       - Other
     # Optional: expression_file: "path/to/expression.csv"

   experiment:
     genes_sensitivities: 1.0
     genes_sensitivities_variation: 0.3
     transfer_function: identity  # or 'log', 'sqrt'

   elements:
     - element_type: VacuolatedStructure
       scale: 80.0
       rule_type: Layer
       rule_params:
         layer_types: [0, 1]
         layer_boundaries: [0.7]
       extra_params:
         frequency: 0.5

     - element_type: LinearLumenStructure
       scale: 40.0
       rule_type: SingleType
       rule_params:
         cell_type_ix: 3
       extra_params:
         frequency: 0.3

   n_fovs: 100
   output_dir: "output/my_simulation"

Supported Element Types
-----------------------

- ``HistologicalElement``: Basic convex polygon element
- ``VacuolatedStructure``: Ring-shaped structure (glands, acini)
- ``LinearLumenStructure``: Tube-shaped structure (ducts, vessels)
- ``LayeredElement``: Concentric layer structure
- ``BranchingStructure``: Branching tube network

Supported Rule Types
--------------------

- ``RandomCellType`` / ``random``: Independent random probabilities
- ``SingleType`` / ``single``: All cells one type
- ``Mix``: Fixed proportions of types
- ``Layer``: Concentric layers by distance from center
