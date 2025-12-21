Histological Elements
=====================

Histological elements define the spatial structures within which cells are placed.

Overview
--------

PointillSim provides several element types:

.. list-table::
   :header-rows: 1

   * - Element
     - Description
     - Use Case
   * - ``FrameWideElement``
     - Fills entire FOV
     - Background tissue
   * - ``HistologicalElement``
     - Convex polygon
     - Discrete structures
   * - ``VacuolatedStructure``
     - Polygon with hole
     - Glands, vessels
   * - ``LinearLumenStructure``
     - Tube-shaped
     - Ducts, vessels

FrameWideElement
----------------

Fills the entire field of view. Used for background tissue.

.. code-block:: python

   from pointillsim import FrameWideElement, RandomCellTypeRule

   background = FrameWideElement(
       frame_size=600,
       tipical_cell_spacing=15,  # Controls cell density
       rules=RandomCellTypeRule(n_cell_types=5)
   )

   generated = background.generate()
   print(f"Cells: {len(generated.cell_centroids)}")

HistologicalElement
-------------------

Bounded convex polygon with cells placed inside.

.. code-block:: python

   from pointillsim import HistologicalElement, SingleTypeRule
   import numpy as np

   element = HistologicalElement(
       frame_size=500,
       n_vertices=(8, 12),         # Random vertex count in range
       scale=100,                   # Approximate radius
       fixed_center=np.array([[250, 250]]),  # Center position
       tipical_cell_spacing=10,
       rules=SingleTypeRule(n_cell_types=5, cell_type_ix=2)
   )

   generated = element.generate()

VacuolatedStructure
-------------------

Ring-shaped structure with a central hole (lumen). Perfect for glands.

.. code-block:: python

   from pointillsim import VacuolatedStructure, DistanceBasedRule
   import numpy as np

   gland = VacuolatedStructure(
       frame_size=500,
       scale=80,                    # Outer radius
       hole_scale_factor=0.5,       # Hole is 50% of outer size
       fixed_center=np.array([[250, 250]]),
       tipical_cell_spacing=8,
       rules=DistanceBasedRule(
           n_cell_types=5,
           inner_type=2,            # Near lumen
           outer_type=3,            # At periphery
       )
   )

   generated = gland.generate()

The ``hole_scale_factor`` controls lumen size:

- 0.0: No hole (solid structure)
- 0.5: Hole is half the outer diameter
- 0.8: Large lumen, thin wall

LinearLumenStructure
--------------------

Tube-shaped structure with parallel walls.

.. code-block:: python

   from pointillsim import LinearLumenStructure
   import numpy as np

   duct = LinearLumenStructure(
       frame_size=600,
       start_point=np.array([100, 300]),
       end_point=np.array([500, 350]),
       outer_radius=30,
       wall_thickness=12,
       tipical_cell_spacing=8,
       rules=DistanceBasedRule(
           n_cell_types=5,
           inner_type=2,
           outer_type=3,
       )
   )

Combining Elements
------------------

Use ``FOVDistribution`` to combine background and foreground elements:

.. code-block:: python

   from pointillsim import FOVDistribution

   fov_dist = FOVDistribution(
       frame_size=600,
       background_element=lambda: FrameWideElement(...),
       other_elements=[
           lambda: VacuolatedStructure(...),
           lambda: HistologicalElement(...),
       ],
       elements_frequency=[0.5, 0.3],     # Probability of including
       attempts_at_elements=[3, 2],       # How many to try placing
   )

   fov = fov_dist.generate_fov()

Foreground elements "punch through" the background - background cells within foreground boundaries are removed.
