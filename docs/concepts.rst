Core Concepts
=============

This page explains the key concepts and design philosophy behind PointillSim.

Philosophy
----------

PointillSim follows three core principles:

Compositional Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^

Tissues are built from **histological elements** - discrete regions with defined boundaries.
Like building blocks, these elements can be combined to create complex tissue architectures.

Each element has:

- A **polygon boundary** defining its spatial extent
- A **cell population** placed within the polygon
- **Cell type rules** that assign probabilities to each cell

Separation of Concerns
^^^^^^^^^^^^^^^^^^^^^^

The framework cleanly separates:

- **Where** cells are placed (spatial layout via histological elements)
- **What** types they become (cell type assignment via rules)
- **How** they are observed (measurement process via HybISS_Setup)

This modular design allows mixing and matching components.

Probabilistic Ground Truth
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each cell carries a **probability vector** over all cell types, not a hard assignment.

This reflects biological reality where:

- Cell identity exists on a continuum
- Transition states and mixed phenotypes are common
- Uncertainty is intrinsic to the system

The **realization** step then samples a concrete cell type from this distribution.

Key Components
--------------

TissueCellTypes
^^^^^^^^^^^^^^^

Stores the gene expression profiles for each cell type:

.. code-block:: python

   tissue = TissueCellTypes()
   tissue.generate_types_and_markers(
       n_genes=50,
       n_cell_types=5,
       expected_level=10.0,      # Mean expression
       concentration=0.85,       # Marker specificity
   )

The ``concentration`` parameter controls marker gene specificity:

- High (0.9+): Strong markers, type-specific expression
- Low (0.3-0.5): Ubiquitous expression across types

FOV (Field of View)
^^^^^^^^^^^^^^^^^^^

The main data container holding:

- ``cell_centroids``: (N, 2) array of x, y positions
- ``cell_probabilities``: (N, K) array of probabilities over K cell types
- ``class_instance``: (N,) array of realized cell type indices
- Cell morphology: major/minor axis, rotation, RNA concentration

Histological Elements
^^^^^^^^^^^^^^^^^^^^^

Spatial structures that define where cells are placed:

- ``FrameWideElement``: Fills the entire FOV (background)
- ``HistologicalElement``: Bounded convex polygon
- ``VacuolatedStructure``: Ring-shaped (glands, vessels)
- ``LinearLumenStructure``: Tube-shaped (ducts, vessels)

Cell Type Rules
^^^^^^^^^^^^^^^

Algorithms that assign cell type probabilities:

- ``RandomCellTypeRule``: Independent random probabilities
- ``SingleTypeRule``: All cells one type
- ``MixOfNCellTypesRule``: Fixed proportions
- ``DistanceBasedRule``: Radial gradients
- ``LayerRule``: Concentric layers
- ``GradientRule``: Linear gradients
- ``ProbabilityNodeFieldRule``: Spatial interpolation
- ``CompositeRule``: Combine multiple rules

FOVDistribution
^^^^^^^^^^^^^^^

Factory that generates FOVs by composing elements:

.. code-block:: python

   fov_dist = FOVDistribution(
       frame_size=600,
       background_element=background_factory,
       other_elements=[gland_factory, vessel_factory],
       elements_frequency=[0.5, 0.3],
       attempts_at_elements=[3, 2],
   )

   fov = fov_dist.generate_fov()

The Simulation Pipeline
-----------------------

.. code-block:: text

   ┌─────────────────┐
   │  TissueCellTypes │ ← Gene expression profiles
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │ FOVDistribution │ ← Spatial layout + rules
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │      FOV        │ ← Cell positions + probabilities
   └────────┬────────┘
            │
   ┌────────▼────────────┐
   │ CellTypesProperties │ ← Cell morphology
   └────────┬────────────┘
            │
   ┌────────▼────────┐
   │   HybISS_Setup  │ ← Observation model
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │    dots_df      │ ← Transcript observations
   └─────────────────┘

1. **Define expression**: Create ``TissueCellTypes`` with gene profiles
2. **Define layout**: Create ``FOVDistribution`` with elements and rules
3. **Generate FOV**: Call ``generate_fov()`` to place cells
4. **Apply morphology**: Use ``CellTypesProperties.apply(fov)``
5. **Observe transcripts**: Use ``HybISS_Setup.observe_dots(fov)``
6. **Export**: Get DataFrames, AnnData, or SpatialData

Probability Vector Field
------------------------

Each cell has a probability vector over all cell types:

.. code-block:: python

   # Example: 3 cell types
   fov.cell_probabilities[0]  # [0.7, 0.2, 0.1] - 70% type 0, etc.

The realized cell type is sampled from this distribution:

.. code-block:: python

   fov.class_instance[0]  # 0 (sampled from [0.7, 0.2, 0.1])

This provides:

- **Soft ground truth** for benchmarking
- **Uncertainty quantification** (entropy of probability vector)
- **Flexible modeling** of transition states

Element Composition
-------------------

When elements overlap, **foreground punches through background**:

.. code-block:: text

   Background fills FOV
   ┌────────────────────────┐
   │  ░░░░░░░░░░░░░░░░░░░░  │
   │  ░░░░░░░░░░░░░░░░░░░░  │
   │  ░░░░░╔════╗░░░░░░░░░  │  ← Foreground element
   │  ░░░░░║    ║░░░░░░░░░  │    replaces background
   │  ░░░░░╚════╝░░░░░░░░░  │    cells in its region
   │  ░░░░░░░░░░░░░░░░░░░░  │
   └────────────────────────┘

Background cells within foreground boundaries are removed and replaced.
