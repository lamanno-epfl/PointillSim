Tissue-Specific Simulations
===========================

This page shows how to simulate specific real tissues.

See also: ``notebooks/06_tissue_simulations.ipynb`` for runnable examples.

Colon Crypts
------------

Colon epithelium contains crypt structures with stem cells at the base.

.. code-block:: python

   from pointillsim import (
       FOVDistribution, FrameWideElement, VacuolatedStructure,
       MixOfNCellTypesRule, DistanceBasedRule,
   )
   from pointillsim.rules.composite import LayerRule

   n_cell_types = 6
   # 0: Stromal, 1: Goblet, 2: Enterocyte, 3: Stem, 4: Transit-amplifying, 5: Immune

   # Background: lamina propria
   background = lambda: FrameWideElement(
       frame_size=800,
       tipical_cell_spacing=20,
       rules=MixOfNCellTypesRule(
           n_cell_types=n_cell_types,
           list_N=[0, 5],
           proportions=[0.8, 0.2]
       )
   )

   # Crypt structure
   crypt = lambda: VacuolatedStructure(
       frame_size=800,
       scale=60,
       hole_scale_factor=0.3,
       tipical_cell_spacing=8,
       rules=LayerRule(
           n_cell_types=n_cell_types,
           layer_types=[3, 4, 2, 1],  # Stem → TA → Enterocyte → Goblet
           layer_boundaries=[0.3, 0.5, 0.8],
           transition_width=5,
       )
   )

   fov_dist = FOVDistribution(
       frame_size=800,
       background_element=background,
       other_elements=[crypt],
       elements_frequency=[1.0],
       attempts_at_elements=12,
   )

Cortex Layers
-------------

Cerebral cortex has 6 distinct layers with different neuronal types.

.. code-block:: python

   from pointillsim import ProbabilityNodeFieldRule
   import numpy as np

   n_cell_types = 8
   # 0: L1 neurons, 1: L2/3, 2: L4, 3: L5, 4: L6, 5: Astrocyte, 6: Microglia, 7: Oligo

   frame_size = 1000

   # Create vertical gradient with layer-specific cell types
   n_points = 12
   y_positions = np.linspace(50, 950, n_points)
   reference_points = np.array([
       [x, y] for y in y_positions for x in [200, 500, 800]
   ]).reshape(-1, 2)

   # Define probabilities at each reference point
   def get_layer_probs(y):
       probs = np.zeros(n_cell_types)
       # Add glial cells everywhere
       probs[5] = 0.1  # Astrocyte
       probs[6] = 0.05  # Microglia
       probs[7] = 0.05  # Oligo

       # Layer-specific neurons
       if y < 100:
           probs[0] = 0.8  # L1
       elif y < 300:
           probs[1] = 0.8  # L2/3
       elif y < 450:
           probs[2] = 0.8  # L4
       elif y < 650:
           probs[3] = 0.8  # L5
       else:
           probs[4] = 0.8  # L6

       return probs / probs.sum()

   ref_probs = np.array([get_layer_probs(y) for x, y in reference_points])

   rule = ProbabilityNodeFieldRule(
       n_cell_types=n_cell_types,
       n_ref_points=len(reference_points),
       ref_probs=ref_probs,
       reference_points=reference_points,
   )

   fov_dist = FOVDistribution(
       frame_size=frame_size,
       background_element=lambda: FrameWideElement(
           frame_size=frame_size,
           tipical_cell_spacing=15,
           rules=rule
       ),
   )

Mammary Gland
-------------

Mammary tissue contains TDLU (terminal ductal lobular units) with acini.

.. code-block:: python

   n_cell_types = 7
   # 0: Luminal, 1: Myoepithelial, 2: Fibroblast, 3: Adipocyte, 4: Endothelial, 5: Immune, 6: Basal

   # Stromal background
   background = lambda: FrameWideElement(
       frame_size=1000,
       tipical_cell_spacing=25,
       rules=MixOfNCellTypesRule(
           n_cell_types=n_cell_types,
           list_N=[2, 3, 4, 5],
           proportions=[0.4, 0.35, 0.15, 0.1]
       )
   )

   # Acinar structure (bilayer epithelium)
   acinus = lambda: VacuolatedStructure(
       frame_size=1000,
       scale=60,
       hole_scale_factor=0.65,
       tipical_cell_spacing=10,
       rules=LayerRule(
           n_cell_types=n_cell_types,
           layer_types=[0, 1],  # Luminal inner, myoepithelial outer
           layer_boundaries=[0.7],
           transition_width=5,
       )
   )

   fov_dist = FOVDistribution(
       frame_size=1000,
       background_element=background,
       other_elements=[acinus],
       elements_frequency=[1.0],
       attempts_at_elements=15,
   )

Tips for Tissue Simulation
--------------------------

1. **Research the anatomy**: Understand cell types and their spatial organization
2. **Start simple**: Begin with 2-3 cell types, add complexity gradually
3. **Use LayerRule**: For concentric or stratified structures
4. **Use ProbabilityNodeFieldRule**: For smooth spatial gradients
5. **Combine rules**: Use CompositeRule to add noise to patterns
6. **Validate visually**: Compare to histology images
