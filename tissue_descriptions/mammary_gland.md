# Mammary Gland (Breast Tissue)

## Tissue Overview

The mammary gland consists of **terminal ductal lobular units (TDLUs)** embedded in fibrous and adipose stroma. Each TDLU contains multiple acini (secretory units) connected by ducts. This is the functional unit where milk is produced.

## Cell Types

| Type | Location | Description |
|------|----------|-------------|
| Luminal epithelial | Acini/duct inner layer | Secretory cells, form inner lining |
| Myoepithelial | Acini/duct outer layer | Contractile, surround luminal cells |
| Basal/stem cells | Basal layer | Progenitor cells |
| Fibroblasts | Stroma | Connective tissue |
| Adipocytes | Stroma | Fat cells |
| Endothelial | Blood vessels | Vessel lining |
| Immune cells | Stroma | Macrophages, lymphocytes |

## Spatial Organization

```
TDLU (Terminal Ductal Lobular Unit)
┌─────────────────────────────────────┐
│                                     │
│    ╭───╮     ╭───╮     ╭───╮       │
│   (     )   (     )   (     )       │  ← Acini
│    ╰───╯     ╰───╯     ╰───╯       │
│       \        |        /           │
│        \       |       /            │
│         ╲      |      ╱             │
│          ╲     |     ╱              │
│           ╲────┴────╱               │  ← Collecting duct
│            |        |               │
└────────────┴────────┴───────────────┘

Each acinus structure:
┌─────────────────────┐
│ Lumen (empty)       │
├─────────────────────┤  ← Luminal epithelial
├─────────────────────┤  ← Myoepithelial (flattened)
└─────────────────────┘
    Basement membrane
```

## Procedural Steps

### 1. Set up the frame
- Frame size: 1200-1500 pixels
- Cell spacing: 12-15 pixels (epithelium), 25-30 pixels (stroma)

### 2. Create stromal background
Use `FrameWideElement` with `MixOfNCellTypesRule`:
```
- 40% fibroblasts
- 35% adipocytes
- 15% immune cells
- 10% endothelial
```
Use larger cell spacing (25-30 pixels) for sparse stroma.

### 3. Place acinar structures
Use `VacuolatedStructure` for each acinus:
- Number: 10-20 acini per FOV
- Radius (scale): 50-80 pixels
- Hole scale factor: 0.6-0.7 (larger lumen)
- Arrange in loose clusters (TDLUs)

### 4. Apply acinar cell type rules
For each acinus, use `LayerRule` with 2 layers:

```python
LayerRule(
    n_cell_types=n_types,
    layer_types=[luminal_idx, myoepithelial_idx],
    layer_boundaries=[0.7],  # Inner 70% luminal, outer 30% myoepithelial
    transition_width=5,
)
```

### 5. Add ductal structures (optional)
Use `LinearLumenStructure` to connect acini:
- Outer radius: 30-40 pixels
- Wall thickness: 15-20 pixels
- Apply same `LayerRule` as acini

### 6. Cluster acini into lobules
Group 5-8 acini close together with connecting ducts:
- Use placement callbacks to ensure proximity
- Or manually specify `fixed_center` positions

### 7. Expression profiles
- Luminal markers: KRT8, KRT18, ESR1, PGR, GATA3
- Myoepithelial markers: KRT5, KRT14, ACTA2 (SMA), TP63
- Progenitor markers: KRT6, KIT
- Adipocyte markers: ADIPOQ, LEP
- Fibroblast markers: VIM, COL1A1

## PointillSim Primitives Used

- `FrameWideElement`: Stromal background
- `VacuolatedStructure`: Individual acini
- `LinearLumenStructure`: Ducts connecting acini
- `LayerRule`: Luminal-myoepithelial arrangement
- `MixOfNCellTypesRule`: Stromal composition
- `FOVDistribution` with `foreground_elements`: Place multiple acini

## Code Sketch

```python
n_cell_types = 7
# Types: 0=luminal, 1=myoepithelial, 2=fibroblast, 3=adipocyte,
#        4=endothelial, 5=immune, 6=basal

def background_factory():
    return FrameWideElement(
        frame_size=1200,
        tipical_cell_spacing=28,
        rules=MixOfNCellTypesRule(
            n_cell_types=n_cell_types,
            list_N=[2, 3, 4, 5],  # stroma types
            proportions=[0.4, 0.35, 0.15, 0.1]
        )
    )

def acinus_factory():
    rule = LayerRule(
        n_cell_types=n_cell_types,
        layer_types=[0, 1],  # luminal, myoepithelial
        layer_boundaries=[0.7],
        transition_width=5,
    )
    return VacuolatedStructure(
        frame_size=1200,
        scale=60 + np.random.uniform(-15, 15),
        hole_scale_factor=0.65,
        rules=rule,
        tipical_cell_spacing=12,
    )

fov_dist = FOVDistribution(
    frame_size=1200,
    background_element=background_factory,
    other_elements=[acinus_factory],
    elements_frequency=[1.0],
    attempts_at_elements=15,  # ~15 acini
)
```

## Notes

- In lactating tissue, luminal cells are more prominent and secretory
- Adipose content varies significantly between individuals
- Consider adding blood vessels as `LinearLumenStructure`
- Lobules should cluster together - may need custom placement logic
- For cancer simulations, could add invasive fronts or DCIS (duct-confined)
- The bilayer (luminal + myoepithelial) is key for breast biology
