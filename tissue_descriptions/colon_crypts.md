# Colon Crypts (Colonic Epithelium)

## Tissue Overview

The colon epithelium consists of finger-like invaginations called **crypts of Lieberkühn**. Each crypt is a test-tube shaped structure with:
- Stem cells at the base
- Transit-amplifying cells in the middle
- Differentiated cells (goblet, colonocytes) toward the surface
- Surrounding lamina propria (stromal tissue) between crypts

## Cell Types

| Type | Location | Description |
|------|----------|-------------|
| Stem cells | Crypt base | LGR5+ intestinal stem cells |
| Transit-amplifying | Lower-mid crypt | Proliferating progenitors |
| Enterocytes | Upper crypt / surface | Absorptive columnar cells |
| Goblet cells | Throughout crypt | Mucus-secreting cells |
| Enteroendocrine | Scattered | Hormone-secreting cells (rare) |
| Paneth cells | Crypt base | Antimicrobial peptide secretion |
| Stromal fibroblasts | Lamina propria | Connective tissue |
| Immune cells | Lamina propria | Lymphocytes, macrophages |

## Spatial Organization

```
Surface epithelium (enterocytes, goblet cells)
    |
    |  ← Crypt opening
    |
   [ ]  ← Upper crypt (differentiated)
   [ ]
   [ ]  ← Mid crypt (transit-amplifying)
   [ ]
   [*]  ← Crypt base (stem cells, Paneth)

Lamina propria fills space between crypts
```

## Procedural Steps

### 1. Set up the frame
- Frame size: 1000-2000 pixels (represents ~0.5-1mm of tissue)
- Cell spacing: 15-20 pixels (crypts have dense epithelium)

### 2. Create background (lamina propria)
- Use `FrameWideElement` for the entire frame
- Apply `MixOfNCellTypesRule`:
  - 70% stromal fibroblasts
  - 20% immune cells
  - 10% scattered other cells
- Use sparser cell spacing (25-30 pixels) for stroma

### 3. Place crypt structures
- Use `VacuolatedStructure` for each crypt (cross-section view)
- Crypt parameters:
  - Scale (radius): 40-60 pixels
  - Hole scale factor: 0.5-0.7 (lumen in center)
  - Place 15-25 crypts in a grid-like pattern with some jitter

### 4. Apply crypt cell type rules
For each crypt, use `LayerRule` or `DistanceBasedRule`:

**Option A: LayerRule (3 concentric zones)**
```
- Inner zone (base): Stem cells + Paneth cells
- Middle zone: Transit-amplifying cells
- Outer zone (surface): Enterocytes + Goblet cells
```

**Option B: DistanceBasedRule (2 zones)**
```
- Center (base): Stem cells (inner_type)
- Edge (surface): Differentiated cells (outer_type)
```

### 5. Add goblet cells
- Use `CompositeRule` to blend:
  - 80% zone-specific rule (above)
  - 20% `MixOfNCellTypesRule` for goblet cells scattered throughout

### 6. Expression profiles
- Stem cell markers: LGR5, ASCL2, OLFM4
- Goblet markers: MUC2, TFF3
- Enterocyte markers: FABP1, CA1
- Paneth markers: DEFA5, LYZ

## PointillSim Primitives Used

- `FrameWideElement`: Background lamina propria
- `VacuolatedStructure`: Individual crypts
- `LayerRule` or `DistanceBasedRule`: Stem-to-differentiated gradient
- `MixOfNCellTypesRule`: Stromal composition, scattered goblet cells
- `CompositeRule`: Combine zone-specific with scattered cells

## Notes

- Crypts in cross-section appear as ring-shaped structures
- Real crypts are ~50 cells deep, ~20 cells in circumference
- Consider adding `LinearLumenStructure` for longitudinal crypt views
- Crypt density: ~15-20 crypts per mm² in human colon
