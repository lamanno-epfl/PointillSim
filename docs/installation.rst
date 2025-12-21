Installation
============

Requirements
------------

PointillSim requires Python 3.9 or later.

Core dependencies:

- numpy >= 1.20
- scipy >= 1.7
- pandas >= 1.3
- matplotlib >= 3.4
- shapely >= 2.0

Optional dependencies for additional features:

- anndata: AnnData export
- spatialdata: SpatialData export
- tqdm: Progress bars for batch generation

Installation from PyPI
----------------------

.. code-block:: bash

   pip install pointillsim

Installation from Source
------------------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/lamanno-epfl/PointillSim.git
   cd PointillSim

Install in development mode:

.. code-block:: bash

   pip install -e ".[dev]"

This installs the package with development dependencies (pytest, sphinx, etc.).

Verifying Installation
----------------------

.. code-block:: python

   import pointillsim as ps

   # Check version
   print(f"PointillSim version: {ps.__version__}")

   # Quick test
   from pointillsim import TissueCellTypes, FOVDistribution, FrameWideElement, RandomCellTypeRule

   tissue = TissueCellTypes()
   tissue.generate_types_and_markers(n_genes=20, n_cell_types=3)

   fov_dist = FOVDistribution(
       frame_size=500,
       background_element=lambda: FrameWideElement(
           frame_size=500,
           tipical_cell_spacing=15,
           rules=RandomCellTypeRule(n_cell_types=3)
       ),
   )
   fov = fov_dist.generate_fov()
   print(f"Generated FOV with {fov.n_cells} cells")

Optional: Install Extras
------------------------

For AnnData/SpatialData export:

.. code-block:: bash

   pip install pointillsim[io]

For all optional dependencies:

.. code-block:: bash

   pip install pointillsim[all]
