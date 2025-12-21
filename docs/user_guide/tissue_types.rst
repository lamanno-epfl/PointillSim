Tissue Cell Types
=================

The ``TissueCellTypes`` class defines the gene expression profiles for each cell type in your simulation.

Creating Expression Profiles
----------------------------

Synthetic Profiles
^^^^^^^^^^^^^^^^^^

Generate random expression profiles with marker genes:

.. code-block:: python

   from pointillsim import TissueCellTypes

   tissue = TissueCellTypes()
   tissue.generate_types_and_markers(
       n_genes=50,           # Number of genes in panel
       n_cell_types=5,       # Number of distinct cell types
       expected_level=10.0,  # Average expression level
       concentration=0.85,   # Marker specificity (0-1)
   )

The ``concentration`` parameter controls marker gene strength:

- **High (0.9+)**: Strong markers, clear type separation
- **Medium (0.5-0.8)**: Moderate markers, some overlap
- **Low (0.2-0.4)**: Weak markers, high overlap

From Real scRNA-seq Data
^^^^^^^^^^^^^^^^^^^^^^^^

Use real expression profiles from single-cell data:

.. code-block:: python

   import anndata as ad
   import pandas as pd

   # Load your scRNA-seq reference
   adata = ad.read_h5ad("reference.h5ad")

   # Compute mean expression per cell type
   cell_types = adata.obs['cell_type'].unique()
   expr_matrix = pd.DataFrame(
       index=adata.var_names,
       columns=cell_types
   )

   for ct in cell_types:
       mask = adata.obs['cell_type'] == ct
       expr_matrix[ct] = adata[mask].X.mean(axis=0).A1

   # Create TissueCellTypes
   tissue = TissueCellTypes()
   tissue.gene_expression_by_type = expr_matrix.values
   tissue._gene_names = list(expr_matrix.index)
   tissue._cell_type_names = list(expr_matrix.columns)

Properties
----------

The tissue object has these key properties:

.. code-block:: python

   tissue.n_genes          # Number of genes
   tissue.n_cell_types     # Number of cell types
   tissue.gene_names       # List of gene names
   tissue.cell_type_names  # List of cell type names
   tissue.gene_expression_by_type  # (n_genes, n_types) array

Export
------

Export to pandas DataFrame:

.. code-block:: python

   df = tissue.make_pandas_df()
   # DataFrame with genes as rows, cell types as columns

Visualize expression matrix:

.. code-block:: python

   from pointillsim import plot_expression_matrix

   fig, ax = plot_expression_matrix(tissue, log_scale=True)
   plt.show()
