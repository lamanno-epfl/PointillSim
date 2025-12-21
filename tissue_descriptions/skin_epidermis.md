# Skin Epidermis

## Tissue Overview

The epidermis is the outermost layer of skin, composed of stratified squamous epithelium. Keratinocytes differentiate as they move from the basal layer to the surface, forming distinct layers.

## Cell Types

| Type | Layer | Description |
|------|-------|-------------|
| Basal keratinocytes | Stratum basale | Stem cells, cuboidal shape |
| Spinous keratinocytes | Stratum spinosum | Polygonal, connected by desmosomes |
| Granular keratinocytes | Stratum granulosum | Flattened, keratohyalin granules |
| Cornified cells | Stratum corneum | Dead, flattened, no nuclei |
| Melanocytes | Stratum basale | Pigment-producing cells |
| Langerhans cells | Stratum spinosum | Dendritic immune cells |
| Merkel cells | Stratum basale | Mechanoreceptors (rare) |
| Fibroblasts | Dermis | Connective tissue |
| Endothelial cells | Dermis | Blood vessel lining |

## Spatial Organization

```
Skin surface
═══════════════════════════════════ (stratum corneum - dead cells)
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
═══════════════════════════════════
Stratum granulosum (2-3 cell layers)
───────────────────────────────────
Stratum spinosum (5-10 cell layers)
    Langerhans cells scattered here
───────────────────────────────────
Stratum basale (1 cell layer)
    Melanocytes and Merkel cells here
═══════════════════════════════════
Basement membrane
───────────────────────────────────
DERMIS
    Fibroblasts
    Blood vessels (endothelial)
```

## Procedural Steps

### 1. Set up the frame
- Frame size: 800-1200 pixels
- Y-axis represents depth (0 = dermis, max = surface)
- Cell spacing: 12-15 pixels (epidermis is tightly packed)

### 2. Define layer boundaries
Approximate layer positions (as fraction of epidermis thickness):
```
Dermis:           0.00 - 0.30  (below epidermis)
Stratum basale:   0.30 - 0.35  (single cell layer)
Stratum spinosum: 0.35 - 0.65  (thick layer)
Stratum granulosum: 0.65 - 0.80
Stratum corneum:  0.80 - 1.00  (dead cells, may be sparse/different)
```

### 3. Create dermal background
Use `FrameWideElement` for the lower portion (Y: 0-30%):
- Apply `MixOfNCellTypesRule`:
  - 80% fibroblasts
  - 15% endothelial (blood vessels)
  - 5% other (immune cells, etc.)
- Use sparser cell spacing (20-25 pixels)

### 4. Create epidermal layers
Use `ProbabilityNodeFieldRule` for the upper portion (Y: 30-100%):

**Reference point setup:**
- Create horizontal rows of reference points at each layer boundary
- Each row should have 3-5 points across the X-axis

**Layer probabilities:**

*Stratum basale (Y: 30-35%):*
- 85% basal keratinocytes
- 10% melanocytes
- 5% Merkel cells

*Stratum spinosum (Y: 35-65%):*
- 90% spinous keratinocytes
- 5% Langerhans cells
- 5% transition cells

*Stratum granulosum (Y: 65-80%):*
- 95% granular keratinocytes
- 5% transition cells

*Stratum corneum (Y: 80-100%):*
- 100% cornified cells (or reduce cell density significantly)

### 5. Cell morphology variation
Use `CellTypesProperties` with layer-specific parameters:
- Basal cells: cuboidal (anisotropy ~0.9)
- Spinous cells: polygonal (anisotropy ~0.85)
- Granular/Cornified: flattened (anisotropy ~0.4-0.6)

### 6. Add melanocyte distribution
Melanocytes are specifically in the basal layer:
- Use `CompositeRule` to ensure melanocytes only appear at the right depth
- Ratio: approximately 1 melanocyte per 10 basal keratinocytes

### 7. Expression profiles
- Basal markers: KRT5, KRT14, TP63
- Suprabasal markers: KRT1, KRT10
- Differentiation: IVL (involucrin), FLG (filaggrin)
- Melanocyte markers: TYR, MLANA, DCT
- Langerhans markers: CD1A, CD207

## PointillSim Primitives Used

- `FrameWideElement`: Both dermis and epidermis backgrounds
- `ProbabilityNodeFieldRule`: Layered keratinocyte differentiation
- `MixOfNCellTypesRule`: Dermal composition
- `CompositeRule`: Combine layer-specific rules
- `CellTypesProperties`: Layer-specific morphology

## Alternative Approach: Separate Elements

Instead of one background, use two overlapping elements:

1. **Dermis element**: `FrameWideElement` covering lower 30%
2. **Epidermis element**: Custom `LayeredElement` (when implemented)

## Notes

- The stratum corneum consists of dead cells - may want to represent differently
- Melanocyte dendrites extend between keratinocytes (not currently modelable)
- Rete ridges (undulating basement membrane) could be added with custom polygon shapes
- Hair follicles and sweat glands could be added as `VacuolatedStructure` or `LinearLumenStructure`
- Consider varying melanocyte density for different skin types
