Use Cases and Capabilities
==========================

PointillSim is designed for several key use cases in spatial transcriptomics research.

Benchmarking Cell Segmentation
------------------------------

**Problem**: Cell segmentation algorithms need ground truth boundaries for evaluation, but manual annotation is tedious and subjective.

**Solution**: PointillSim provides exact cell boundaries and transcript assignments.

.. code-block:: python

   from pointillsim import (
       TissueCellTypes, CellTypesProperties, FOVDistribution,
       FrameWideElement, RandomCellTypeRule, HybISS_Setup
   )

   # Generate FOV with known cell boundaries
   tissue = TissueCellTypes()
   tissue.generate_types_and_markers(n_genes=50, n_cell_types=5)

   cell_props = CellTypesProperties(n_cell_types=5)

   def background():
       return FrameWideElement(
           frame_size=1000,
           tipical_cell_spacing=25,
           rules=RandomCellTypeRule(5)
       )

   fov_dist = FOVDistribution(frame_size=1000, background_element=background)
   fov = fov_dist.generate_fov()
   cell_props.apply(fov)

   hybiss = HybISS_Setup(tissue)
   hybiss.observe_dots(fov)

   # Get ground truth
   cells_df = fov.to_pandas_df()  # True cell boundaries
   dots_df = hybiss.make_pandas_df()  # Transcripts with true cell assignments

**Evaluation metrics**:

- Intersection over Union (IoU) with ground truth polygons
- Fraction of correctly assigned transcripts
- Cell count accuracy

Benchmarking Cell Type Assignment
---------------------------------

**Problem**: Cell typing methods need datasets with known cell type labels for accuracy assessment.

**Solution**: PointillSim provides probability-based soft labels and sampled hard labels.

.. code-block:: python

   # Get both soft and hard labels
   soft_labels = fov.cell_probabilities  # (N, K) probability matrix
   hard_labels = fov.class_instance  # (N,) sampled cell types

   # Soft labels enable nuanced evaluation
   # - Entropy: uncertainty in cell identity
   # - KL divergence: compare predicted vs true distributions

**Controllable difficulty**:

- High marker specificity = easy classification
- Low marker specificity = challenging mixed expression patterns
- Transition zones = continuous cell type gradients

Studying Spatial Patterns
-------------------------

**Problem**: Spatial analysis methods need validation data with known spatial organization.

**Solution**: PointillSim creates controlled spatial patterns using rules.

.. code-block:: python

   from pointillsim import (
       VacuolatedStructure, LayerRule, DistanceBasedRule,
       ProbabilityNodeFieldRule
   )

   # Layered structure (like cortex)
   cortex = HistologicalElement(
       frame_size=1000,
       scale=400,
       rules=LayerRule(
           n_cell_types=6,
           layer_types=[0, 1, 2, 3, 4, 5],
           layer_boundaries=[0.2, 0.35, 0.5, 0.7, 0.85]
       )
   )

   # Radial gradient (like tumor microenvironment)
   tumor = VacuolatedStructure(
       frame_size=1000,
       scale=150,
       rules=DistanceBasedRule(
           n_cell_types=3,
           inner_types=[0],  # Tumor core
           outer_types=[1, 2],  # Stroma, immune
       )
   )

   # Smooth spatial fields
   field = ProbabilityNodeFieldRule(
       n_cell_types=4,
       n_nodes=10,
       smoothness=0.5
   )

**Validation metrics**:

- Spatial autocorrelation recovery (Moran's I)
- Neighborhood enrichment accuracy
- Ligand-receptor interaction detection

Testing Effect of Technical Variation
-------------------------------------

**Problem**: Methods may be sensitive to technical artifacts; need systematic evaluation.

**Solution**: PointillSim's effects module enables controlled addition of noise.

.. code-block:: python

   from pointillsim import (
       BatchEffectModel, TechnicalNoise, BackgroundNoise,
       DropoutModel, SpatialNoise, Lateral2DAdmixture
   )

   # Generate idealized data
   dots_df_clean = hybiss.make_pandas_df()

   # Add technical noise
   noise = TechnicalNoise(cv=0.3, seed=42)
   dots_df = noise.apply(dots_df_clean)

   # Add background
   background = BackgroundNoise(rate=0.05, n_genes=50)
   dots_df = background.apply(dots_df)

   # Add dropout
   dropout = DropoutModel(dropout_rate=0.1)
   dots_df = dropout.apply(dots_df)

   # Add spatial variation
   spatial = SpatialNoise(
       frame_size=1000,
       noise_pattern='gradient',
       variation_strength=0.3
   )
   dots_df = spatial.apply(dots_df)

   # Add admixture (segmentation errors)
   admix = Lateral2DAdmixture(mean_displacement=3.0, contamination_prob=0.1)
   dots_df = admix.apply(dots_df, fov)

**Ablation studies**:

- Vary one effect while holding others constant
- Determine method robustness to specific noise sources
- Find failure modes

Simulating Specific Technologies
--------------------------------

**Problem**: Different platforms have different characteristics (sensitivity, resolution, gene panels).

**Solution**: Technology presets provide platform-specific parameters.

.. code-block:: python

   from pointillsim.experiment.presets import (
       HybISSPreset, MerfishPreset, CartanaPreset,
       TenXXeniumPreset, TenXVisiumPreset
   )

   # MERFISH simulation
   merfish = MerfishPreset()
   hybiss = HybISS_Setup(
       tissue,
       genes_sensitivities=merfish.detection_efficiency,
       genes_sensitivities_variation=merfish.sensitivity_cv,
   )

   # Sample realistic transcript counts
   n_transcripts = merfish.sample_transcripts_per_cell(n_cells=1000)

**Platform comparison**:

- Same tissue, different technologies
- Evaluate how platform choice affects downstream analysis

Dataset Generation for Machine Learning
---------------------------------------

**Problem**: ML methods need large training datasets with diverse tissue architectures.

**Solution**: PointillSim's batch generation creates diverse datasets efficiently.

.. code-block:: python

   from pointillsim import generate_dataset

   # Generate 1000 FOVs with variation
   generate_dataset(
       output_dir='training_data',
       n_fovs=1000,
       fov_distribution=fov_dist,
       tissue=tissue,
       hybiss=hybiss,
       cell_props=cell_props,
       format='anndata',
       seed=42,
   )

**Dataset characteristics**:

- Reproducible via seed
- Parallelized generation
- Multiple export formats (AnnData, SpatialData, CSV)

Configuration-Based Simulation
------------------------------

**Problem**: Need reproducible, shareable simulation setups.

**Solution**: YAML/JSON configuration files.

.. code-block:: python

   from pointillsim import SimulationConfig, build_simulation_from_config

   # Load configuration
   config = SimulationConfig.load('my_experiment.yaml')

   # Validate
   errors = config.validate()
   if errors:
       print(f"Configuration errors: {errors}")

   # Build simulation components
   sim = build_simulation_from_config(config)

   # Generate data
   fov = sim['fov_distribution'].generate_fov()
   sim['cell_props'].apply(fov)
   sim['hybiss'].observe_dots(fov)

**Example YAML configuration**:

.. code-block:: yaml

   name: "Colon Tissue Simulation"
   description: "Simulating colonic crypts with immune infiltration"

   fov:
     frame_size: 1000
     seed: 42

   tissue:
     n_cell_types: 6
     n_genes: 100
     cell_type_names:
       - Epithelial
       - Goblet
       - Enteroendocrine
       - Stromal
       - Immune
       - Endothelial

   elements:
     - element_type: VacuolatedStructure
       scale: 80
       rule_type: Layer
       rule_params:
         layer_types: [0, 1, 2]
         layer_boundaries: [0.6, 0.8]

   n_fovs: 100
   output_dir: "colon_dataset"

Creating Validation Datasets
----------------------------

**Problem**: Methods need validation on data with known difficulty levels.

**Solution**: PointillSim's validation module creates difficulty-graded datasets.

.. code-block:: python

   from pointillsim.validation import estimate_difficulty, DifficultyEstimator

   # Estimate task difficulty
   estimator = DifficultyEstimator()
   difficulty = estimator.estimate(fov, tissue)

   print(f"Segmentation difficulty: {difficulty['segmentation']:.2f}")
   print(f"Classification difficulty: {difficulty['classification']:.2f}")
   print(f"Spatial difficulty: {difficulty['spatial']:.2f}")

**Difficulty factors**:

- Cell density and size variation
- Expression matrix condition number
- Spatial pattern complexity
- Noise levels

Summary of Capabilities
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Capability
     - Key Features
   * - **Tissue Architecture**
     - Compositional elements, layered structures, vessels, glands
   * - **Cell Type Patterns**
     - Random, spatial gradients, layers, fields, custom rules
   * - **Expression Profiles**
     - Generated or loaded from real data, controllable specificity
   * - **Technical Effects**
     - Batch effects, noise, dropout, background, admixture
   * - **Technology Simulation**
     - Presets for HybISS, MERFISH, Cartana, Xenium, Visium
   * - **Data Export**
     - AnnData, SpatialData, pandas DataFrames, CSV
   * - **Reproducibility**
     - Seeds, configuration files, validation
   * - **Scalability**
     - Batch generation, parallel processing
