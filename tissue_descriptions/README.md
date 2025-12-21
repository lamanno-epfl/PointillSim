# Tissue Descriptions

This folder contains plain-text procedural descriptions of how to create FOVs for specific real tissues. These descriptions serve two purposes:

1. **Verification**: Check that PointillSim has all necessary primitives to simulate detailed tissue architectures
2. **Reference**: Provide a blueprint for implementing working simulations (see corresponding notebooks)

## Available Tissue Descriptions

| File | Tissue Type | Status |
|------|-------------|--------|
| [colon_crypts.md](colon_crypts.md) | Colon epithelium with crypts | Description complete |
| [cortex_layers.md](cortex_layers.md) | Cerebral cortex layers | Description complete |
| [skin_epidermis.md](skin_epidermis.md) | Skin epidermis layers | Description complete |
| [mammary_gland.md](mammary_gland.md) | Mammary gland acini | Description complete |

## Format

Each description follows this structure:

1. **Tissue Overview**: What tissue this represents
2. **Cell Types**: List of cell types present
3. **Spatial Organization**: How cells are arranged
4. **Procedural Steps**: Step-by-step instructions for building the FOV
5. **PointillSim Primitives Used**: Which classes/rules are needed
6. **Notes**: Additional considerations
