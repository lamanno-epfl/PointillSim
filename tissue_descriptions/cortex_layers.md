# Cerebral Cortex Layers

## Tissue Overview

The cerebral cortex is organized into 6 distinct layers (laminae) with different neuronal types and densities. This layered structure is visible in coronal or sagittal sections.

## Cell Types

| Type | Primary Layer(s) | Description |
|------|-----------------|-------------|
| Layer 1 neurons | Layer I | Sparse, mostly interneurons |
| Pyramidal (L2/3) | Layers II-III | Small pyramidal neurons, local connections |
| Pyramidal (L4) | Layer IV | Granular cells, receive thalamic input |
| Pyramidal (L5) | Layer V | Large pyramidal, project to subcortical |
| Pyramidal (L6) | Layer VI | Multiform, project to thalamus |
| Interneurons (PV) | All layers | Parvalbumin+ inhibitory |
| Interneurons (SST) | Layers II-VI | Somatostatin+ inhibitory |
| Interneurons (VIP) | Layers II-III | VIP+ inhibitory |
| Astrocytes | All layers | Glial support |
| Oligodendrocytes | Layers V-VI | Myelinating glia |
| Microglia | All layers | Immune cells |

## Spatial Organization

```
Pial surface
═══════════════════════════════════
Layer I    - Molecular layer (sparse cells, mostly neuropil)
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
Layer II   - External granular (small pyramidal)
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
Layer III  - External pyramidal (medium pyramidal)
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
Layer IV   - Internal granular (granule cells)
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
Layer V    - Internal pyramidal (large pyramidal)
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
Layer VI   - Multiform (heterogeneous)
═══════════════════════════════════
White matter
```

## Procedural Steps

### 1. Set up the frame
- Frame size: 1000-1500 pixels (Y-axis represents depth)
- The Y-axis represents laminar depth (0 = pial surface, max = white matter)
- Cell spacing: 18-20 pixels

### 2. Define layer boundaries
Approximate layer boundaries (as fraction of total depth):
```
Layer I:   0.00 - 0.10
Layer II:  0.10 - 0.20
Layer III: 0.20 - 0.40
Layer IV:  0.40 - 0.55
Layer V:   0.55 - 0.75
Layer VI:  0.75 - 1.00
```

### 3. Create background with layered probabilities
Use `ProbabilityNodeFieldRule` with reference points at each layer:

```python
# Reference points along Y-axis at multiple X positions
reference_points = [
    # Y=0.05 (Layer I)
    [0, 50], [500, 50], [1000, 50],
    # Y=0.15 (Layer II)
    [0, 150], [500, 150], [1000, 150],
    # ... etc for each layer boundary
]

# Probabilities at each reference point
# Each row: [L1_neuron, L2/3_pyr, L4_gran, L5_pyr, L6_pyr, PV, SST, Astro, Oligo]
```

### 4. Layer-specific probability profiles

**Layer I** (Y: 0-10%):
- 10% Layer I neurons
- 5% interneurons
- 30% astrocytes
- 55% "empty" or low-density marker

**Layer II-III** (Y: 10-40%):
- 60% L2/3 pyramidal
- 15% PV interneurons
- 10% SST interneurons
- 5% VIP interneurons
- 10% astrocytes

**Layer IV** (Y: 40-55%):
- 70% L4 granular cells
- 15% PV interneurons
- 10% SST interneurons
- 5% astrocytes

**Layer V** (Y: 55-75%):
- 65% L5 pyramidal
- 10% PV interneurons
- 10% SST interneurons
- 10% astrocytes
- 5% oligodendrocytes

**Layer VI** (Y: 75-100%):
- 55% L6 pyramidal
- 10% PV interneurons
- 10% SST interneurons
- 10% astrocytes
- 15% oligodendrocytes

### 5. Add cell density variation
- Layer I is sparse (larger tipical_cell_spacing or subsample)
- Layers II-IV have higher density
- Layers V-VI moderate density

### 6. Expression profiles
- Excitatory markers: SLC17A7 (VGLUT1), SATB2
- L5 markers: FEZF2, BCL11B (CTIP2)
- L6 markers: TBR1, FOXP2
- PV markers: PVALB, GAD1
- SST markers: SST, NPY
- Astrocyte markers: AQP4, GFAP
- Oligodendrocyte markers: MBP, MOG

## PointillSim Primitives Used

- `FrameWideElement`: Entire cortex section
- `ProbabilityNodeFieldRule`: Primary rule for layered composition
- `MixOfNCellTypesRule`: Can be used per-layer if needed
- No foreground structures needed (cortex is essentially one layered structure)

## Notes

- The cortex varies by region (e.g., Layer IV is prominent in sensory areas, reduced in motor areas)
- Consider creating area-specific variants (V1, M1, etc.)
- Interneurons are distributed across layers but with different densities
- Could add blood vessels using `LinearLumenStructure` running perpendicular to layers
- Layer boundaries are not perfectly sharp - use soft transitions
