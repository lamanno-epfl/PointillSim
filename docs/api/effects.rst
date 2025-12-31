Effects Module
==============

The effects module provides classes for adding realistic technical variation to simulated data.

.. module:: pointillsim.effects

Overview
--------

Technical effects in spatial transcriptomics include:

- **Batch effects**: Systematic variation between experiments
- **Detection noise**: Variability in transcript detection efficiency
- **Background**: False positive transcript detections
- **Dropout**: Gene-specific detection failures
- **Spatial noise**: Regional variation across the field of view
- **Admixture**: Transcript misassignment from imperfect segmentation

Batch Effects
-------------

.. autoclass:: pointillsim.effects.BatchEffectModel
   :members:
   :undoc-members:
   :show-inheritance:

Technical Noise
---------------

.. autoclass:: pointillsim.effects.TechnicalNoise
   :members:
   :undoc-members:
   :show-inheritance:

Background Noise
----------------

.. autoclass:: pointillsim.effects.BackgroundNoise
   :members:
   :undoc-members:
   :show-inheritance:

Dropout Model
-------------

.. autoclass:: pointillsim.effects.DropoutModel
   :members:
   :undoc-members:
   :show-inheritance:

Spatial Noise
-------------

.. autoclass:: pointillsim.effects.SpatialNoise
   :members:
   :undoc-members:
   :show-inheritance:

Example usage:

.. code-block:: python

   from pointillsim import SpatialNoise

   # Create gradient noise (e.g., edge-of-FOV effects)
   noise = SpatialNoise(
       frame_size=1000,
       noise_pattern='gradient',
       base_level=1.0,
       variation_strength=0.3,
       gradient_direction='diagonal'
   )

   # Apply to dots
   dots_df = noise.apply(dots_df)

   # Visualize the noise field
   noise.plot_field()

Available noise patterns:

- ``'gradient'``: Linear gradient across the FOV
- ``'radial'``: Center-to-edge radial pattern
- ``'patches'``: Random local patches
- ``'perlin'``: Smooth Perlin-like noise

Admixture Models
----------------

Admixture models simulate transcript misassignment due to imperfect cell segmentation or 3D tissue effects.

Lateral 2D Admixture
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pointillsim.effects.Lateral2DAdmixture
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from pointillsim import Lateral2DAdmixture

   admix = Lateral2DAdmixture(
       mean_displacement=3.0,  # pixels
       contamination_prob=0.1,  # fraction affected
       seed=42
   )

   # Apply to dots (needs FOV for cell positions)
   dots_df = admix.apply(dots_df, fov)

Z-Axis Admixture
^^^^^^^^^^^^^^^^

.. autoclass:: pointillsim.effects.ZAxisAdmixture
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from pointillsim import ZAxisAdmixture

   z_admix = ZAxisAdmixture(
       contamination_rate=0.05,  # fraction from other z-planes
       neighbor_bias=0.8,  # preference for nearby cells
       seed=42
   )

   # Apply to dots
   dots_df = z_admix.apply(dots_df, fov)

Composite Admixture
^^^^^^^^^^^^^^^^^^^

.. autoclass:: pointillsim.effects.CompositeAdmixture
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from pointillsim import Lateral2DAdmixture, ZAxisAdmixture, CompositeAdmixture

   composite = CompositeAdmixture([
       Lateral2DAdmixture(mean_displacement=2.0),
       ZAxisAdmixture(contamination_rate=0.03),
   ])

   dots_df = composite.apply(dots_df, fov)

Admixture Metrics
^^^^^^^^^^^^^^^^^

.. autoclass:: pointillsim.effects.AdmixtureMetrics
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from pointillsim import AdmixtureMetrics

   metrics = AdmixtureMetrics(dots_df_with_admixture)

   summary = metrics.get_summary()
   print(f"Contamination rate: {summary['contamination_rate']:.2%}")
   print(f"Mean displacement: {summary['mean_displacement']:.2f}")

   # Per-cell-type breakdown
   by_type = metrics.contamination_by_type()

Combined Effects Example
------------------------

.. code-block:: python

   from pointillsim import (
       TechnicalNoise, BackgroundNoise, DropoutModel,
       SpatialNoise, Lateral2DAdmixture
   )

   # Start with clean data
   dots_df = hybiss.make_pandas_df()

   # Add technical noise (detection efficiency variation)
   dots_df = TechnicalNoise(cv=0.3).apply(dots_df)

   # Add background noise (false positives)
   dots_df = BackgroundNoise(rate=0.05, n_genes=50).apply(dots_df)

   # Add dropout (gene-specific failures)
   dots_df = DropoutModel(dropout_rate=0.1).apply(dots_df)

   # Add spatial variation
   dots_df = SpatialNoise(
       frame_size=1000,
       noise_pattern='radial',
       variation_strength=0.2
   ).apply(dots_df)

   # Add segmentation error artifacts
   dots_df = Lateral2DAdmixture(
       mean_displacement=3.0,
       contamination_prob=0.1
   ).apply(dots_df, fov)
