#!/usr/bin/env python3
"""
Visualize each cell type individually with others in light gray background.
Adapted for the 20-cell-type advanced skin tissue simulation.
Skips unused/placeholder cell types (0 cells).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FOLDER = "advanced_skin_output"
FRAME_WIDTH = 800
FRAME_HEIGHT = 1000

# Cell type names (from generate_advanced_skin.py)
CELL_TYPE_NAMES = [
    'Keratinocyte_Cornified',    # 0
    'Keratinocyte_Granular',     # 1
    'Keratinocyte_Spinous',      # 2
    'Keratinocyte_Basal',        # 3
    'Melanocyte',                # 4
    'Fibroblast_Papillary',      # 5
    'Fibroblast_Reticular',      # 6
    'Endothelial',               # 7
    'Pericyte',                  # 8
    'Sebocyte',                  # 9
    'Sweat_Eccrine',             # 10
    'Sweat_Apocrine',            # 11
    'Hair',                      # 12
    'Hair_Unused',               # 13 (placeholder, not assigned)
    'Adipocyte',                 # 14
    'Langerhans',                # 15
    'Macrophage',                # 16
    'T_Cell',                    # 17
    'Mast_Cell',                 # 18
    'Sensory_Corpuscle',         # 19
]

# Cell type colors (from generate_advanced_skin.py)
CELL_TYPE_COLORS = {
    0:  '#FF1493',  # Keratinocyte_Cornified - Deep Pink
    1:  '#FF6347',  # Keratinocyte_Granular - Tomato
    2:  '#FF8C00',  # Keratinocyte_Spinous - Dark Orange
    3:  '#FFD700',  # Keratinocyte_Basal - Gold
    4:  '#8B4513',  # Melanocyte - Saddle Brown
    5:  '#00FF00',  # Fibroblast_Papillary - Lime
    6:  '#006400',  # Fibroblast_Reticular - Dark Green
    7:  '#FF0000',  # Endothelial - Red
    8:  '#DC143C',  # Pericyte - Crimson
    9:  '#FFFF00',  # Sebocyte - Yellow
    10: '#0000FF',  # Sweat_Eccrine - Blue
    11: '#00BFFF',  # Sweat_Apocrine - Deep Sky Blue
    12: '#FF00FF',  # Hair - Magenta
    13: '#FF1493',  # Hair_Unused (placeholder)
    14: '#FFA500',  # Adipocyte - Orange
    15: '#9370DB',  # Langerhans - Medium Purple
    16: '#BA55D3',  # Macrophage - Medium Orchid
    17: '#8A2BE2',  # T_Cell - Blue Violet
    18: '#DA70D6',  # Mast_Cell - Orchid
    19: '#00FFFF',  # Sensory_Corpuscle - Cyan
}

# Skin layer boundaries for reference lines
DEJ_DEPTH = 100
PAPILLARY_DERMIS_DEPTH = 200
RETICULAR_DERMIS_DEPTH = 800

# ============================================================================
# MAIN
# ============================================================================

# Read the cells data
cells_df = pd.read_csv(f'{INPUT_FOLDER}/cells.csv')

# Get active cell types (those with at least 1 cell), skip placeholders
all_types = sorted(cells_df['Class ID'].unique())
active_types = [ct for ct in all_types if cells_df['Class ID'].eq(ct).sum() > 0]
n_cell_types = len(active_types)

print(f"Generating individual cell type visualizations for {n_cell_types} active types...")
print(f"Skipped types (0 cells): {[CELL_TYPE_NAMES[t] for t in all_types if t not in active_types]}")

# Calculate grid layout
n_cols = int(np.ceil(np.sqrt(n_cell_types)))
n_rows = int(np.ceil(n_cell_types / n_cols))

# ============================================================================
# LIGHT BACKGROUND VERSION
# ============================================================================

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 4.5))
axes = axes.flatten() if n_cell_types > 1 else [axes]

for idx, cell_type in enumerate(active_types):
    ax = axes[idx]

    # Plot all cells in light gray (background)
    ax.scatter(cells_df['X'], cells_df['Y'],
              c='lightgray', s=0.5, alpha=0.2, rasterized=True)

    # Plot the specific cell type in its color
    mask = cells_df['Class ID'] == cell_type
    cell_count = mask.sum()

    if cell_count > 0:
        ax.scatter(cells_df.loc[mask, 'X'],
                  cells_df.loc[mask, 'Y'],
                  c=CELL_TYPE_COLORS[cell_type],
                  s=5, alpha=0.95, rasterized=True,
                  edgecolors='none')

    # Layer reference lines
    for depth, lw in [(DEJ_DEPTH, 0.8), (PAPILLARY_DERMIS_DEPTH, 0.5),
                      (RETICULAR_DERMIS_DEPTH, 0.5)]:
        ax.axhline(y=depth, color='gray', linestyle=':', alpha=0.3, linewidth=lw)

    # Set title with cell type name and count
    title_text = f'{CELL_TYPE_NAMES[cell_type]}\n({cell_count:,} cells)'
    ax.set_title(title_text, fontsize=8, fontweight='bold', pad=6)

    ax.set_xlim(0, FRAME_WIDTH)
    ax.set_ylim(FRAME_HEIGHT, 0)  # Inverted: 0=top (skin surface), bottom=deep
    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)', fontsize=6)
    ax.set_ylabel('Depth (µm)', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.grid(True, alpha=0.1, linewidth=0.3)
    ax.set_facecolor('#F8F8F8')

# Hide unused subplots
for idx in range(n_cell_types, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Skin Tissue - Individual Cell Types', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
output_path = f'{INPUT_FOLDER}/skin_cell_types_individual.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  Saved {output_path}")

# ============================================================================
# DARK BACKGROUND VERSION (high contrast)
# ============================================================================

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 4.5))
axes = axes.flatten() if n_cell_types > 1 else [axes]

for idx, cell_type in enumerate(active_types):
    ax = axes[idx]

    # Plot all cells in dark gray (background)
    ax.scatter(cells_df['X'], cells_df['Y'],
              c='#404040', s=0.4, alpha=0.15, rasterized=True)

    # Plot the specific cell type in its color
    mask = cells_df['Class ID'] == cell_type
    cell_count = mask.sum()

    if cell_count > 0:
        ax.scatter(cells_df.loc[mask, 'X'],
                  cells_df.loc[mask, 'Y'],
                  c=CELL_TYPE_COLORS[cell_type],
                  s=5, alpha=1.0, rasterized=True,
                  edgecolors='none')

    # Layer reference lines
    for depth, lw in [(DEJ_DEPTH, 0.8), (PAPILLARY_DERMIS_DEPTH, 0.5),
                      (RETICULAR_DERMIS_DEPTH, 0.5)]:
        ax.axhline(y=depth, color='#555555', linestyle=':', alpha=0.4, linewidth=lw)

    # Set title with cell type name and count
    title_text = f'{CELL_TYPE_NAMES[cell_type]}\n({cell_count:,} cells)'
    ax.set_title(title_text, fontsize=8, fontweight='bold', color='white', pad=6)

    ax.set_xlim(0, FRAME_WIDTH)
    ax.set_ylim(FRAME_HEIGHT, 0)  # Inverted: 0=top (skin surface), bottom=deep
    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)', fontsize=6, color='white')
    ax.set_ylabel('Depth (µm)', fontsize=6, color='white')
    ax.tick_params(labelsize=5, colors='white')
    ax.grid(True, alpha=0.08, linewidth=0.3, color='white')
    ax.set_facecolor('black')

    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
        spine.set_linewidth(0.5)

# Hide unused subplots
for idx in range(n_cell_types, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Skin Tissue - Individual Cell Types', fontsize=14,
             fontweight='bold', color='white', y=1.01)
plt.tight_layout()
output_path_dark = f'{INPUT_FOLDER}/skin_cell_types_individual_dark.png'
plt.savefig(output_path_dark, dpi=300, bbox_inches='tight',
           facecolor='black', edgecolor='white')
print(f"  Saved {output_path_dark}")

plt.close('all')

print("\n" + "=" * 70)
print("INDIVIDUAL CELL TYPE VISUALIZATION COMPLETE!")
print("=" * 70)
print(f"\nGenerated 2 visualizations:")
print(f"  1. skin_cell_types_individual.png (light background)")
print(f"  2. skin_cell_types_individual_dark.png (dark background)")
print(f"\nActive cell types: {n_cell_types}")
print(f"Grid layout: {n_rows} rows x {n_cols} columns")
print("=" * 70)
