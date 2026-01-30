#!/usr/bin/env python3
"""
Visualize the generated colon tissue simulation
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
print("Loading data...")
cells_df = pd.read_csv("colon_tissue_output/cells.csv")
dots_df = pd.read_csv("colon_tissue_output/dots.csv")

print(f"Loaded {len(cells_df)} cells and {len(dots_df):,} transcripts")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Cell types
ax = axes[0]
scatter = ax.scatter(
    cells_df['X'],
    cells_df['Y'],
    c=cells_df['Class ID'],
    cmap='Set1',
    s=10,
    alpha=0.7
)
ax.set_aspect('equal')
ax.set_title(f'Cell Types\n({len(cells_df)} cells, 6 types)', fontsize=12, fontweight='bold')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
plt.colorbar(scatter, ax=ax, label='Cell Type ID')

# Add cell type counts
cell_type_names = ['Epi_Crypt', 'Epi_Surface', 'Stromal', 'Immune', 'Endothelial', 'Muscle']
counts = cells_df['Class ID'].value_counts().sort_index()
text_y = 50
for i, (ct, name) in enumerate(zip(counts.index, cell_type_names)):
    color = plt.cm.Set1(i / 6)
    ax.text(1050, text_y + i*50, f'{name}: {counts[ct]}',
            fontsize=9, color=color, fontweight='bold')

# Plot 2: Transcript dots
ax = axes[1]
ax.scatter(dots_df['x'], dots_df['y'], s=0.5, alpha=0.3, c='darkred', rasterized=True)
ax.set_aspect('equal')
ax.set_title(f'Transcript Dots\n({len(dots_df):,} total)', fontsize=12, fontweight='bold')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')

# Plot 3: Cell density heatmap
ax = axes[2]
from scipy.ndimage import gaussian_filter

# Create 2D histogram
H, xedges, yedges = np.histogram2d(
    cells_df['X'],
    cells_df['Y'],
    bins=50,
    range=[[0, 1000], [0, 1000]]
)

# Apply Gaussian smoothing
H_smooth = gaussian_filter(H.T, sigma=2)

# Plot
im = ax.imshow(H_smooth, extent=[0, 1000, 0, 1000], origin='lower',
               cmap='viridis', aspect='equal')
ax.set_title('Cell Density Heatmap', fontsize=12, fontweight='bold')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
plt.colorbar(im, ax=ax, label='Cell density')

plt.suptitle('Colon Tissue Simulation - PointillSim', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
output_file = 'colon_tissue_output/visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to {output_file}")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nCells by type:")
for i, (ct, name) in enumerate(zip(counts.index, cell_type_names)):
    pct = 100 * counts[ct] / len(cells_df)
    print(f"  {name:15s}: {counts[ct]:5d} ({pct:5.1f}%)")

print(f"\nTranscripts by gene (top 10):")
gene_counts = dots_df['gene'].value_counts().head(10)
for gene, count in gene_counts.items():
    pct = 100 * count / len(dots_df)
    print(f"  {gene:10s}: {count:6d} ({pct:5.1f}%)")

print(f"\nMorphology statistics:")
print(f"  Cell size (minor axis): {cells_df['Minor Axis'].mean():.2f} ± {cells_df['Minor Axis'].std():.2f}")
print(f"  Cell size (major axis): {cells_df['Major Axis'].mean():.2f} ± {cells_df['Major Axis'].std():.2f}")
print(f"  RNA concentration:      {cells_df['RNA Concentration'].mean():.3f} ± {cells_df['RNA Concentration'].std():.3f}")

print("\n" + "="*60)
