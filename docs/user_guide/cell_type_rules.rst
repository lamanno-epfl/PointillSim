Cell Type Rules
===============

Rules determine how cell type probabilities are assigned to cells within histological elements.

Overview
--------

.. list-table::
   :header-rows: 1

   * - Rule
     - Pattern
     - Use Case
   * - ``RandomCellTypeRule``
     - Random mixture
     - Heterogeneous tissue
   * - ``SingleTypeRule``
     - All same type
     - Homogeneous regions
   * - ``MixOfNCellTypesRule``
     - Fixed proportions
     - Known composition
   * - ``DistanceBasedRule``
     - Radial gradient
     - Zonal structures
   * - ``LayerRule``
     - Concentric rings
     - Cortical layers
   * - ``GradientRule``
     - Linear gradient
     - Directional variation
   * - ``ProbabilityNodeFieldRule``
     - Spatial interpolation
     - Complex gradients
   * - ``CompositeRule``
     - Combined rules
     - Complex patterns

Random Rules
------------

RandomCellTypeRule
^^^^^^^^^^^^^^^^^^

Each cell gets independent random probabilities from a Dirichlet distribution.

.. code-block:: python

   from pointillsim import RandomCellTypeRule

   rule = RandomCellTypeRule(n_cell_types=5)

SingleTypeRule
^^^^^^^^^^^^^^

All cells are assigned to a single specific type.

.. code-block:: python

   from pointillsim import SingleTypeRule

   rule = SingleTypeRule(
       n_cell_types=5,
       cell_type_ix=2  # All cells are type 2
   )

MixOfNCellTypesRule
^^^^^^^^^^^^^^^^^^^

All cells receive the same fixed mixture of selected types.

.. code-block:: python

   from pointillsim import MixOfNCellTypesRule

   rule = MixOfNCellTypesRule(
       n_cell_types=5,
       list_N=[0, 2, 4],         # Only these types present
       proportions=[0.5, 0.3, 0.2]  # In these proportions
   )

Spatial Rules
-------------

DistanceBasedRule
^^^^^^^^^^^^^^^^^

Creates radial patterns based on distance from center or boundary.

.. code-block:: python

   from pointillsim import DistanceBasedRule

   # Center-to-edge gradient
   rule = DistanceBasedRule(
       n_cell_types=5,
       reference='center',      # or 'boundary'
       inner_type=0,            # Type at center
       outer_type=3,            # Type at edge
       transition_width=30,     # Width of transition zone
       sharpness=1.0,           # Higher = sharper transition
   )

LayerRule
^^^^^^^^^

Creates concentric layers of different cell types.

.. code-block:: python

   from pointillsim.rules.composite import LayerRule

   rule = LayerRule(
       n_cell_types=6,
       layer_types=[0, 2, 4, 5],        # Types from center outward
       layer_boundaries=[0.25, 0.5, 0.75],  # Boundaries at 25%, 50%, 75% of radius
       transition_width=10,
   )

GradientRule
^^^^^^^^^^^^

Creates linear gradients across a structure.

.. code-block:: python

   from pointillsim.rules.composite import GradientRule

   rule = GradientRule(
       n_cell_types=5,
       start_type=0,
       end_type=3,
       direction='horizontal',  # or 'vertical', 'diagonal', or (dx, dy)
       transition_width=0.4,
   )

ProbabilityNodeFieldRule
^^^^^^^^^^^^^^^^^^^^^^^^

Creates smooth spatial probability gradients by interpolating between reference points.

.. code-block:: python

   from pointillsim import ProbabilityNodeFieldRule
   import numpy as np

   # Define reference points and their probabilities
   reference_points = np.array([
       [100, 100],  # Top-left
       [400, 100],  # Top-right
       [100, 400],  # Bottom-left
       [400, 400],  # Bottom-right
   ])

   ref_probs = np.array([
       [0.8, 0.1, 0.1, 0.0, 0.0],  # Top-left: mostly type 0
       [0.1, 0.8, 0.1, 0.0, 0.0],  # Top-right: mostly type 1
       [0.1, 0.1, 0.8, 0.0, 0.0],  # Bottom-left: mostly type 2
       [0.0, 0.0, 0.0, 0.5, 0.5],  # Bottom-right: mix of 3&4
   ])

   rule = ProbabilityNodeFieldRule(
       n_cell_types=5,
       n_ref_points=4,
       ref_probs=ref_probs,
       reference_points=reference_points,
   )

Composite Rules
---------------

CompositeRule
^^^^^^^^^^^^^

Combines multiple rules with configurable weights.

.. code-block:: python

   from pointillsim import CompositeRule, DistanceBasedRule, RandomCellTypeRule

   # 80% radial pattern, 20% random noise
   rule = CompositeRule(
       rules=[
           DistanceBasedRule(n_cell_types=5, inner_type=0, outer_type=2),
           RandomCellTypeRule(n_cell_types=5),
       ],
       weights=[0.8, 0.2],
       mode='average',  # Weighted average of probabilities
   )

Custom Rules
------------

Create custom rules by subclassing ``CellTypeRuleBase``:

.. code-block:: python

   from pointillsim.rules import CellTypeRuleBase
   import numpy as np

   class MyCustomRule(CellTypeRuleBase):
       def __init__(self, n_cell_types, my_parameter):
           super().__init__(n_cell_types)
           self.my_parameter = my_parameter

       def assign_probabilities(self, cell_centroids, polygon, element):
           """
           Assign cell type probabilities to each cell.

           Parameters
           ----------
           cell_centroids : np.ndarray
               (N, 2) array of cell positions
           polygon : shapely.Polygon
               Element boundary
           element : HistologicalElement
               The element containing these cells

           Returns
           -------
           probabilities : np.ndarray
               (N, n_cell_types) array of probabilities
           """
           n_cells = len(cell_centroids)
           probs = np.zeros((n_cells, self.n_cell_types))

           # Your custom logic here
           for i in range(n_cells):
               x, y = cell_centroids[i]
               # ... compute probabilities based on position
               probs[i] = [...]

           return probs
