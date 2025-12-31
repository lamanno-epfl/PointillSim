Philosophy and Design
=====================

Why Simulation?
---------------

Spatial transcriptomics methods produce rich, complex datasets where the ground truth is fundamentally unknowable. When benchmarking cell segmentation, cell type assignment, or spatial analysis methods, we need datasets where we *know* the correct answer.

PointillSim creates synthetic spatial transcriptomics data with **complete ground truth**:

- Exact cell boundaries
- True cell type assignments (and probability distributions)
- Noise-free transcript counts per cell
- Original transcript locations before any technical effects

This enables rigorous benchmarking of computational methods under controlled conditions.

Design Philosophy
-----------------

Compositional Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^

Tissues are built from **histological elements** - discrete regions with defined boundaries. Like building blocks, these elements can be combined to create complex tissue architectures.

.. code-block:: text

   ┌─────────────────────────────────────────┐
   │  Background (stromal cells)             │
   │    ┌───────────┐                        │
   │    │ Gland     │     ┌────────────┐     │
   │    │ (epithelial)    │ Vessel     │     │
   │    │  ○        │     │ (endothelial)    │
   │    └───────────┘     └────────────┘     │
   │                                         │
   │         ┌─────────────────┐             │
   │         │ Immune cluster  │             │
   │         └─────────────────┘             │
   └─────────────────────────────────────────┘

This compositional approach allows:

- **Reusable components**: Define a gland structure once, place it multiple times
- **Hierarchical composition**: Elements can contain sub-elements
- **Flexible layouts**: Same elements, different arrangements

Separation of Concerns
^^^^^^^^^^^^^^^^^^^^^^

The framework cleanly separates:

1. **Spatial Layout** (where cells are placed)

   - Histological elements define cell positions
   - Cell spacing and density are configurable
   - Overlap handling is automatic

2. **Cell Type Assignment** (what types they become)

   - Rules assign probability vectors to cells
   - Rules are composable and chainable
   - Same layout, different cell type patterns

3. **Observation Model** (how they are measured)

   - HybISS_Setup simulates transcript detection
   - Transfer functions model detection efficiency
   - Effects add realistic technical variation

This separation allows mixing and matching components independently.

Probabilistic Ground Truth
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each cell carries a **probability vector** over all cell types, not a hard assignment:

.. code-block:: python

   # Cell might be 70% epithelial, 20% transitional, 10% stromal
   cell_probabilities = [0.7, 0.2, 0.1]

This reflects biological reality where:

- Cell identity exists on a continuum
- Transition states and mixed phenotypes are common
- Uncertainty is intrinsic to biological systems

The framework provides both:

- **Soft labels**: The full probability distribution
- **Hard labels**: Sampled realization from the distribution

Controlled Realism
^^^^^^^^^^^^^^^^^^

PointillSim balances **simplicity** with **realism**:

- **Simple by default**: Minimal configuration produces usable data
- **Realistic when needed**: Effects modules add technical variation
- **Controllable complexity**: Each source of variation is independent

This allows:

- Quick prototyping with idealized data
- Rigorous benchmarking with realistic noise
- Systematic ablation studies varying one factor at a time

Core Abstractions
-----------------

The Rule System
^^^^^^^^^^^^^^^

Rules are the heart of cell type assignment. Each rule implements one strategy:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Rule
     - Description
   * - ``RandomCellTypeRule``
     - Independent random probabilities for each cell
   * - ``SingleTypeRule``
     - All cells assigned to one type
   * - ``MixOfNCellTypesRule``
     - Fixed proportions of cell types
   * - ``DistanceBasedRule``
     - Radial gradient from center
   * - ``LayerRule``
     - Concentric layers (like cortex)
   * - ``GradientRule``
     - Linear gradient across element
   * - ``ProbabilityNodeFieldRule``
     - Spatial interpolation between control points
   * - ``CompositeRule``
     - Combine multiple rules

Rules are **composable** - combine them for complex patterns:

.. code-block:: python

   from pointillsim import CompositeRule, LayerRule, RandomCellTypeRule

   # Layers with added randomness
   rule = CompositeRule(
       rules=[
           LayerRule(n_cell_types=5, layer_types=[0, 1, 2]),
           RandomCellTypeRule(n_cell_types=5),
       ],
       weights=[0.8, 0.2]  # 80% layer structure, 20% noise
   )

The Effects System
^^^^^^^^^^^^^^^^^^

Effects add technical variation to idealized simulations:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Effect
     - Description
   * - ``BatchEffectModel``
     - Systematic variation between FOVs/batches
   * - ``TechnicalNoise``
     - Detection efficiency variation
   * - ``BackgroundNoise``
     - False positive transcript detections
   * - ``DropoutModel``
     - Gene-specific detection failures
   * - ``SpatialNoise``
     - Regional variation across the FOV
   * - ``Lateral2DAdmixture``
     - Transcript misassignment from segmentation errors
   * - ``ZAxisAdmixture``
     - Signal from cells above/below the imaging plane

Effects are **modular** - apply them independently or combined:

.. code-block:: python

   from pointillsim import TechnicalNoise, BackgroundNoise, Lateral2DAdmixture

   # Apply multiple effects
   dots_df = hybiss.make_pandas_df()
   dots_df = TechnicalNoise(cv=0.3).apply(dots_df)
   dots_df = BackgroundNoise(rate=0.05).apply(dots_df)
   dots_df = Lateral2DAdmixture(mean_displacement=2.0).apply(dots_df, fov)

Technology Presets
^^^^^^^^^^^^^^^^^^

Pre-configured settings for common platforms:

- ``HybISSPreset``: Hybridization-based in situ sequencing
- ``MerfishPreset``: Multiplexed error-robust FISH
- ``CartanaPreset``: ISS-based spatial transcriptomics
- ``TenXXeniumPreset``: 10x Genomics Xenium
- ``TenXVisiumPreset``: 10x Genomics Visium

Presets encode realistic parameters calibrated from literature:

.. code-block:: python

   from pointillsim.experiment.presets import MerfishPreset

   preset = MerfishPreset()
   print(f"Detection efficiency: {preset.detection_efficiency}")
   print(f"Transcripts/cell: {preset.mean_transcripts_per_cell}")

When NOT to Use PointillSim
---------------------------

PointillSim is designed for **benchmarking** and **method development**, not for:

- **Biological discovery**: Simulated data cannot reveal new biology
- **Parameter estimation**: Use real data for calibrating biological parameters
- **Exact replication**: Simulations approximate, not replicate, real data

The goal is synthetic data that is **realistic enough** to challenge algorithms while maintaining **complete ground truth** for evaluation.
