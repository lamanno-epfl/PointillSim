#!/usr/bin/env python3
"""
Detailed visualization of the generated colon tissue simulation
Includes marginal and cell-type-conditioned gene distributions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Load data
print("Loading data...")
cells_df = pd.read_csv("colon_tissue_output/cells.csv")
dots_df = pd.read_csv("colon_tissue_output/dots.csv")

print(f"Loaded {len(cells_df)} cells and {len(dots_df):,} transcripts")

# Cell type names
cell_type_names = ['Epi_Crypt', 'Epi_Surface', 'Stromal', 'Immune', 'Endothelial', 'Muscle']

# ============================================================================
# FIGURE 1: Basic spatial visualization
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Cell types
ax = axes[0]
scatter = ax.scatter(
    cells_df['X'],
    cells_df['Y'],
    c=cells_df['Class ID'],
    cmap='Set1',
    s=20,
    alpha=0.7
)
ax.set_aspect('equal')
ax.set_title(f'Cell Types\n({len(cells_df)} cells, 6 types)', fontsize=12, fontweight='bold')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
plt.colorbar(scatter, ax=ax, label='Cell Type ID')

# Add cell type counts
counts = cells_df['Class ID'].value_counts().sort_index()
text_y = 50
for i, (ct, name) in enumerate(zip(counts.index, cell_type_names)):
    color = plt.cm.Set1(i / 6)
    ax.text(1050, text_y + i*50, f'{name}: {counts[ct]}',
            fontsize=9, color=color, fontweight='bold')

# Plot 2: Transcript dots
ax = axes[1]
# Separate signal from background
signal_mask = dots_df['cell'] >= 0
ax.scatter(dots_df.loc[signal_mask, 'x'], dots_df.loc[signal_mask, 'y'],
           s=0.5, alpha=0.3, c='darkred', rasterized=True, label='Signal')
if (~signal_mask).sum() > 0:
    ax.scatter(dots_df.loc[~signal_mask, 'x'], dots_df.loc[~signal_mask, 'y'],
               s=1, alpha=0.5, c='orange', rasterized=True, label='Background')
    ax.legend(markerscale=5, loc='upper right')
ax.set_aspect('equal')
ax.set_title(f'Transcript Dots\n({len(dots_df):,} total, {(~signal_mask).sum()} background)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')

# Plot 3: Cell density heatmap
ax = axes[2]
H, xedges, yedges = np.histogram2d(
    cells_df['X'],
    cells_df['Y'],
    bins=50,
    range=[[0, 1000], [0, 1000]]
)
H_smooth = gaussian_filter(H.T, sigma=2)
im = ax.imshow(H_smooth, extent=[0, 1000, 0, 1000], origin='lower',
               cmap='viridis', aspect='equal')
ax.set_title('Cell Density Heatmap', fontsize=12, fontweight='bold')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
plt.colorbar(im, ax=ax, label='Cell density')

plt.suptitle('Colon Tissue Simulation - Spatial Overview', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('colon_tissue_output/spatial_overview.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved spatial_overview.png")
plt.show()


# ============================================================================
# FIGURE 2: Marginal gene distribution
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Overall gene counts
ax = axes[0, 0]
gene_counts = dots_df['gene'].value_counts().sort_values(ascending=False)
x_pos = np.arange(len(gene_counts))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gene_counts)))
ax.bar(x_pos, gene_counts.values, color=colors, alpha=0.8)
ax.set_xlabel('Gene (ranked by count)', fontsize=10)
ax.set_ylabel('Transcript Count', fontsize=10)
ax.set_title('Marginal Gene Distribution\n(All cells combined)', fontsize=11, fontweight='bold')
ax.set_xlim(-1, len(gene_counts))

# Top 10 genes
ax = axes[0, 1]
top10 = gene_counts.head(10)
colors_top10 = plt.cm.tab10(np.arange(10))
bars = ax.barh(range(10), top10.values[::-1], color=colors_top10[::-1], alpha=0.8)
ax.set_yticks(range(10))
ax.set_yticklabels(top10.index[::-1], fontsize=9)
ax.set_xlabel('Transcript Count', fontsize=10)
ax.set_title('Top 10 Most Abundant Genes', fontsize=11, fontweight='bold')
ax.invert_yaxis()
for i, (gene, count) in enumerate(zip(top10.index[::-1], top10.values[::-1])):
    pct = 100 * count / len(dots_df)
    ax.text(count + 10, i, f'{count} ({pct:.1f}%)', va='center', fontsize=8)

# Gene count distribution histogram
ax = axes[1, 0]
ax.hist(gene_counts.values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Transcript Count per Gene', fontsize=10)
ax.set_ylabel('Number of Genes', fontsize=10)
ax.set_title('Gene Count Distribution', fontsize=11, fontweight='bold')
ax.axvline(gene_counts.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {gene_counts.mean():.0f}')
ax.axvline(gene_counts.median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {gene_counts.median():.0f}')
ax.legend()

# Cumulative distribution
ax = axes[1, 1]
sorted_counts = np.sort(gene_counts.values)[::-1]
cumsum = np.cumsum(sorted_counts)
cumsum_pct = 100 * cumsum / cumsum[-1]
ax.plot(range(1, len(sorted_counts)+1), cumsum_pct, linewidth=2, color='darkgreen')
ax.fill_between(range(1, len(sorted_counts)+1), cumsum_pct, alpha=0.3, color='green')
ax.set_xlabel('Number of Top Genes', fontsize=10)
ax.set_ylabel('Cumulative % of Transcripts', fontsize=10)
ax.set_title('Cumulative Gene Contribution', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% mark')
ax.axhline(80, color='orange', linestyle='--', alpha=0.5, label='80% mark')
# Find genes contributing to 50% and 80%
n_genes_50 = np.argmax(cumsum_pct >= 50) + 1
n_genes_80 = np.argmax(cumsum_pct >= 80) + 1
ax.axvline(n_genes_50, color='red', linestyle=':', alpha=0.5)
ax.axvline(n_genes_80, color='orange', linestyle=':', alpha=0.5)
ax.text(n_genes_50 + 1, 52, f'{n_genes_50} genes', fontsize=8)
ax.text(n_genes_80 + 1, 82, f'{n_genes_80} genes', fontsize=8)
ax.legend()

plt.suptitle('Marginal Gene Expression Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('colon_tissue_output/marginal_gene_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved marginal_gene_distribution.png")
plt.show()


# ============================================================================
# FIGURE 3: Cell-type-conditioned gene distributions
# ============================================================================

# Merge dots with cell types (only for signal dots)
signal_dots = dots_df[dots_df['cell'] >= 0].copy()
signal_dots = signal_dots.merge(
    cells_df[['Class ID']].reset_index().rename(columns={'index': 'cell'}),
    on='cell',
    how='left'
)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for ct_id in range(6):
    ax = axes[ct_id]

    # Get dots for this cell type
    ct_dots = signal_dots[signal_dots['Class ID'] == ct_id]

    if len(ct_dots) > 0:
        gene_counts_ct = ct_dots['gene'].value_counts().sort_values(ascending=False)

        # Show top 20 genes
        top_n = min(20, len(gene_counts_ct))
        top_genes = gene_counts_ct.head(top_n)

        colors = plt.cm.Set1([ct_id] * top_n)
        bars = ax.barh(range(top_n), top_genes.values[::-1], color=colors, alpha=0.7)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_genes.index[::-1], fontsize=7)
        ax.set_xlabel('Transcript Count', fontsize=9)
        ax.set_title(f'{cell_type_names[ct_id]}\n({len(ct_dots):,} transcripts, {len(gene_counts_ct)} genes)',
                    fontsize=10, fontweight='bold')
        ax.invert_yaxis()

        # Add percentage labels
        for i, count in enumerate(top_genes.values[::-1]):
            pct = 100 * count / len(ct_dots)
            ax.text(count + max(top_genes.values)*0.02, i, f'{pct:.1f}%',
                   va='center', fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No transcripts', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{cell_type_names[ct_id]}\n(0 transcripts)',
                    fontsize=10, fontweight='bold')

plt.suptitle('Cell-Type-Conditioned Gene Distributions\n(Top 20 genes per cell type)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('colon_tissue_output/celltype_gene_distributions.png', dpi=150, bbox_inches='tight')
print("✓ Saved celltype_gene_distributions.png")
plt.show()


# ============================================================================
# FIGURE 4: Gene expression heatmap by cell type
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 10))

# Create gene × cell type count matrix
gene_by_celltype = signal_dots.groupby(['gene', 'Class ID']).size().unstack(fill_value=0)

# Normalize by total counts per cell type (to account for different numbers of cells)
n_cells_per_type = cells_df['Class ID'].value_counts().sort_index()
gene_by_celltype_norm = gene_by_celltype.div(n_cells_per_type, axis=1)

# Sort genes by variance across cell types
gene_var = gene_by_celltype_norm.var(axis=1)
top_var_genes = gene_var.nlargest(30).index
matrix_to_plot = gene_by_celltype_norm.loc[top_var_genes]

# Plot heatmap
im = ax.imshow(matrix_to_plot, aspect='auto', cmap='YlOrRd')
ax.set_yticks(range(len(top_var_genes)))
ax.set_yticklabels(top_var_genes, fontsize=8)
ax.set_xticks(range(6))
ax.set_xticklabels(cell_type_names, rotation=45, ha='right', fontsize=10)
ax.set_xlabel('Cell Type', fontsize=11, fontweight='bold')
ax.set_ylabel('Gene (top 30 by variance)', fontsize=11, fontweight='bold')
ax.set_title('Gene Expression by Cell Type\n(Normalized counts per cell)',
             fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='Transcripts per cell')

plt.tight_layout()
plt.savefig('colon_tissue_output/gene_celltype_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved gene_celltype_heatmap.png")
plt.show()


# ============================================================================
# Print summary statistics
# ============================================================================
print("\n" + "="*70)
print("DETAILED SUMMARY STATISTICS")
print("="*70)

print(f"\n1. CELLS")
print(f"   Total cells: {len(cells_df)}")
print(f"   Cells by type:")
for i, (ct, name) in enumerate(zip(counts.index, cell_type_names)):
    pct = 100 * counts[ct] / len(cells_df)
    print(f"     {name:15s}: {counts[ct]:5d} ({pct:5.1f}%)")

print(f"\n2. TRANSCRIPTS")
print(f"   Total transcripts: {len(dots_df):,}")
signal_count = (dots_df['cell'] >= 0).sum()
background_count = (dots_df['cell'] == -1).sum()
print(f"   Signal transcripts: {signal_count:,} ({100*signal_count/len(dots_df):.1f}%)")
print(f"   Background transcripts: {background_count:,} ({100*background_count/len(dots_df):.1f}%)")
print(f"   Average transcripts per cell: {signal_count/len(cells_df):.1f}")

print(f"\n3. GENES")
print(f"   Total genes: {len(gene_counts)}")
print(f"   Transcripts per gene: mean={gene_counts.mean():.1f}, std={gene_counts.std():.1f}")
print(f"   Top 5 most abundant:")
for i, (gene, count) in enumerate(gene_counts.head(5).items(), 1):
    pct = 100 * count / len(dots_df)
    print(f"     {i}. {gene:10s}: {count:6d} ({pct:5.1f}%)")

print(f"\n4. TRANSCRIPTS BY CELL TYPE")
for ct_id in range(6):
    ct_dots = signal_dots[signal_dots['Class ID'] == ct_id]
    ct_cells = (cells_df['Class ID'] == ct_id).sum()
    if ct_cells > 0:
        avg_per_cell = len(ct_dots) / ct_cells
        print(f"   {cell_type_names[ct_id]:15s}: {len(ct_dots):6d} total, {avg_per_cell:5.1f} per cell")
    else:
        print(f"   {cell_type_names[ct_id]:15s}:      0 total,   0.0 per cell")

print(f"\n5. MORPHOLOGY")
print(f"   Cell size (minor axis): {cells_df['Minor Axis'].mean():.2f} ± {cells_df['Minor Axis'].std():.2f}")
print(f"   Cell size (major axis): {cells_df['Major Axis'].mean():.2f} ± {cells_df['Major Axis'].std():.2f}")
print(f"   RNA concentration:      {cells_df['RNA Concentration'].mean():.3f} ± {cells_df['RNA Concentration'].std():.3f}")

print("\n" + "="*70)
print("✓ All visualizations complete!")
print("="*70)
