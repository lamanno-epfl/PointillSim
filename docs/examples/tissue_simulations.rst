Tissue-Specific Simulations
===========================

This page shows how to simulate specific real tissues with visual outputs.

See also the tutorial notebook ``notebooks/06_tissue_simulations.ipynb`` for more examples.

Colon Crypts
------------

Colon epithelium contains crypt structures with stem cells at the base.

.. plot::
   :context: reset
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from pointillsim import (
       TissueCellTypes, CellTypesProperties,
       FOVDistribution, FrameWideElement, VacuolatedStructure,
       MixOfNCellTypesRule,
   )
   from pointillsim.rules.composite import LayerRule

   np.random.seed(42)

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
           layer_types=[3, 4, 2, 1],  # Stem -> TA -> Enterocyte -> Goblet
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

   # Generate and visualize
   fov = fov_dist.generate_fov()
   cell_props = CellTypesProperties(n_cell_types=n_cell_types)
   cell_props.apply(fov)

   cell_type_names = ['Stromal', 'Goblet', 'Enterocyte', 'Stem', 'Transit-amp', 'Immune']

   fig, ax = plt.subplots(figsize=(8, 8))
   scatter = ax.scatter(
       fov.cell_centroids[:, 0],
       fov.cell_centroids[:, 1],
       c=fov.class_instance,
       cmap='tab10',
       s=15, alpha=0.8
   )
   ax.set_aspect('equal')
   ax.set_xlim(0, 800)
   ax.set_ylim(0, 800)
   ax.set_title('Colon Crypts Simulation', fontsize=14)
   cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_cell_types))
   cbar.set_ticklabels(cell_type_names)
   plt.tight_layout()

Cortex Layers
-------------

Cerebral cortex has 6 distinct layers with different neuronal types.

.. plot::
   :context: reset
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from pointillsim import (
       TissueCellTypes, CellTypesProperties,
       FOVDistribution, FrameWideElement, ProbabilityNodeFieldRule,
   )

   np.random.seed(123)

   n_cell_types = 8
   frame_size = 600
   # 0: L1, 1: L2/3, 2: L4, 3: L5, 4: L6, 5: Astrocyte, 6: Microglia, 7: Oligo

   # Create vertical gradient with layer-specific cell types
   n_points = 8
   y_positions = np.linspace(50, 550, n_points)
   reference_points = np.array([
       [x, y] for y in y_positions for x in [150, 300, 450]
   ]).reshape(-1, 2)

   def get_layer_probs(y):
       probs = np.zeros(n_cell_types)
       probs[5] = 0.1  # Astrocyte
       probs[6] = 0.05  # Microglia
       probs[7] = 0.05  # Oligo

       if y < 80:
           probs[0] = 0.8  # L1
       elif y < 180:
           probs[1] = 0.8  # L2/3
       elif y < 280:
           probs[2] = 0.8  # L4
       elif y < 400:
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
           tipical_cell_spacing=12,
           rules=rule
       ),
   )

   fov = fov_dist.generate_fov()
   cell_props = CellTypesProperties(n_cell_types=n_cell_types)
   cell_props.apply(fov)

   layer_names = ['L1', 'L2/3', 'L4', 'L5', 'L6', 'Astro', 'Micro', 'Oligo']

   fig, ax = plt.subplots(figsize=(8, 8))
   scatter = ax.scatter(
       fov.cell_centroids[:, 0],
       fov.cell_centroids[:, 1],
       c=fov.class_instance,
       cmap='Set1',
       s=15, alpha=0.8
   )
   ax.set_aspect('equal')
   ax.set_xlim(0, frame_size)
   ax.set_ylim(0, frame_size)
   ax.set_title('Cortex Layers Simulation', fontsize=14)
   ax.set_xlabel('X position')
   ax.set_ylabel('Y position (Cortical Depth)')

   # Add layer annotations
   layer_bounds = [0, 80, 180, 280, 400, frame_size]
   layer_labels = ['L1', 'L2/3', 'L4', 'L5', 'L6']
   for i, label in enumerate(layer_labels):
       y_mid = (layer_bounds[i] + layer_bounds[i+1]) / 2
       ax.axhline(layer_bounds[i+1], color='gray', linestyle='--', alpha=0.5)
       ax.text(frame_size + 10, y_mid, label, fontsize=10, va='center')

   plt.tight_layout()

Mammary Gland
-------------

Mammary tissue contains TDLU (terminal ductal lobular units) with acini.

.. plot::
   :context: reset
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from pointillsim import (
       TissueCellTypes, CellTypesProperties,
       FOVDistribution, FrameWideElement, VacuolatedStructure,
       MixOfNCellTypesRule,
   )
   from pointillsim.rules.composite import LayerRule

   np.random.seed(456)

   n_cell_types = 7
   frame_size = 800
   # 0: Luminal, 1: Myoepithelial, 2: Fibroblast, 3: Adipocyte, 4: Endothelial, 5: Immune, 6: Basal

   # Stromal background
   background = lambda: FrameWideElement(
       frame_size=frame_size,
       tipical_cell_spacing=25,
       rules=MixOfNCellTypesRule(
           n_cell_types=n_cell_types,
           list_N=[2, 3, 4, 5],
           proportions=[0.4, 0.35, 0.15, 0.1]
       )
   )

   # Acinar structure (bilayer epithelium)
   acinus = lambda: VacuolatedStructure(
       frame_size=frame_size,
       scale=55,
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
       frame_size=frame_size,
       background_element=background,
       other_elements=[acinus],
       elements_frequency=[1.0],
       attempts_at_elements=15,
   )

   fov = fov_dist.generate_fov()
   cell_props = CellTypesProperties(n_cell_types=n_cell_types)
   cell_props.apply(fov)

   type_names = ['Luminal', 'Myoepi', 'Fibro', 'Adipo', 'Endo', 'Immune', 'Basal']

   fig, ax = plt.subplots(figsize=(8, 8))
   scatter = ax.scatter(
       fov.cell_centroids[:, 0],
       fov.cell_centroids[:, 1],
       c=fov.class_instance,
       cmap='tab10',
       s=15, alpha=0.8
   )
   ax.set_aspect('equal')
   ax.set_xlim(0, frame_size)
   ax.set_ylim(0, frame_size)
   ax.set_title('Mammary Gland (TDLU) Simulation', fontsize=14)
   cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_cell_types))
   cbar.set_ticklabels(type_names)
   plt.tight_layout()

Tips for Tissue Simulation
--------------------------

1. **Research the anatomy**: Understand cell types and their spatial organization
2. **Start simple**: Begin with 2-3 cell types, add complexity gradually
3. **Use LayerRule**: For concentric or stratified structures
4. **Use ProbabilityNodeFieldRule**: For smooth spatial gradients
5. **Combine rules**: Use CompositeRule to add noise to patterns
6. **Validate visually**: Compare to histology images
