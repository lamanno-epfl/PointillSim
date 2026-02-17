#!/usr/bin/env python3
"""
Validation: Compare simulated technology datasets against real data.

Compares simulated BARseq/EELFISH outputs from generate_allen_brain_technology_SUBCLASS.py
against real datasets (wholecortex_BARseq.h5ad, EELFISH_Saggital_MouseBrain.h5ad).

Metrics computed:
  A. Per-gene marginal distribution comparison (KS statistic, Wasserstein distance)
  B. Sparsity analysis (zero fractions, dropout curve)
  C. Mean-variance relationship (overdispersion comparison)
  D. Global count distributions (counts/cell, genes/cell)

Usage:
  python validate_allen_brain_technology.py
"""

import os
import json
import numpy as np
import pandas as pd
import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from scipy import sparse


# ============================================================================
# CONFIGURATION
# ============================================================================

SIM_OUTPUT_FOLDER = "allen_brain_files/technology_output_SUBCLASS"
HVG_LIST_PATH = "allen_brain_files/genes_variance_genes1000.txt"

TECHNOLOGIES = {
    "BARseq": {
        "sim_folder": f"{SIM_OUTPUT_FOLDER}/TECH_BARseq",
        "real_h5ad": "allen_brain_files/wholecortex_BARseq.h5ad",
        "n_genes_panel": 100,
    },
    "EELFISH": {
        "sim_folder": f"{SIM_OUTPUT_FOLDER}/TECH_EELFISH",
        "real_h5ad": "allen_brain_files/EELFISH_Saggital_MouseBrain.h5ad",
        "n_genes_panel": 400,
    },
}

# Subsample real data for efficiency (set None to use all)
MAX_REAL_CELLS = 200_000
# Number of top overlapping genes to show in per-gene histograms
N_GENES_PLOT = 12

VALIDATION_OUTPUT_FOLDER = "allen_brain_files/validation_output"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_hvg_list(path):
    """Load the pre-ranked HVG gene list."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_real_data(h5ad_path, max_cells=None):
    """Load real h5ad and return (count_matrix, gene_names)."""
    adata = anndata.read_h5ad(h5ad_path)
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)

    gene_names = list(adata.var_names)

    if max_cells is not None and X.shape[0] > max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], max_cells, replace=False)
        X = X[idx]

    return X, gene_names


def load_simulated_data(sim_folder):
    """Load simulated dots.csv and convert to cell x gene count matrix."""
    dots_path = os.path.join(sim_folder, "dots.csv")
    dots_df = pd.read_csv(dots_path)

    # Build count matrix from dots
    gene_names = sorted(dots_df["gene"].unique())
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    cell_ids = sorted(dots_df["cell"].unique())
    cell_to_idx = {c: i for i, c in enumerate(cell_ids)}

    n_cells = len(cell_ids)
    n_genes = len(gene_names)
    X = np.zeros((n_cells, n_genes), dtype=np.float64)

    for _, row in dots_df.iterrows():
        ci = cell_to_idx[row["cell"]]
        gi = gene_to_idx[row["gene"]]
        X[ci, gi] += 1.0

    return X, gene_names


def load_simulated_data_fast(sim_folder):
    """Load simulated dots.csv using groupby for speed.

    Reads hyperparameters.json to get n_total_cells; if cells were dropped
    (cell-level dropout), zero-count rows are added so the count matrix
    reflects all cells including those with no detected transcripts.
    """
    dots_path = os.path.join(sim_folder, "dots.csv")
    dots_df = pd.read_csv(dots_path)

    # Pivot to count matrix
    counts = dots_df.groupby(["cell", "gene"]).size().reset_index(name="count")
    pivot = counts.pivot(index="cell", columns="gene", values="count").fillna(0)

    gene_names = list(pivot.columns)
    X = pivot.values.astype(np.float64)

    # Check for total cell count (includes dropped cells)
    hp_path = os.path.join(sim_folder, "hyperparameters.json")
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            hp = json.load(f)
        n_total = hp.get("n_total_cells", None)
        if n_total is not None and n_total > X.shape[0]:
            # Add zero-count rows for cells that were dropped
            n_missing = n_total - X.shape[0]
            zero_rows = np.zeros((n_missing, X.shape[1]), dtype=np.float64)
            X = np.vstack([X, zero_rows])
            print(f"   Added {n_missing} zero-count cells "
                  f"(total: {n_total}, with dots: {n_total - n_missing})")

    return X, gene_names


def find_overlapping_genes(sim_genes, real_genes, hvg_list=None):
    """Find genes present in both simulated and real data, ordered by HVG rank."""
    overlap = set(sim_genes) & set(real_genes)
    if hvg_list is not None:
        # Order by HVG rank
        ordered = [g for g in hvg_list if g in overlap]
        # Add any remaining overlap genes not in HVG list
        remaining = sorted(overlap - set(ordered))
        ordered.extend(remaining)
        return ordered
    return sorted(overlap)


def subset_to_genes(X, gene_names, target_genes):
    """Subset count matrix to specific genes, preserving target_genes order."""
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    indices = [gene_to_idx[g] for g in target_genes if g in gene_to_idx]
    found_genes = [g for g in target_genes if g in gene_to_idx]
    return X[:, indices], found_genes


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def compute_per_gene_ks(sim_X, real_X, gene_names):
    """Compute KS statistic per gene between simulated and real."""
    results = {}
    for i, gene in enumerate(gene_names):
        stat, pval = ks_2samp(sim_X[:, i], real_X[:, i])
        results[gene] = {"ks_statistic": stat, "ks_pvalue": pval}
    return results


def compute_per_gene_wasserstein(sim_X, real_X, gene_names):
    """Compute Wasserstein distance per gene."""
    results = {}
    for i, gene in enumerate(gene_names):
        w = wasserstein_distance(sim_X[:, i], real_X[:, i])
        results[gene] = {"wasserstein": w}
    return results


def compute_sparsity_metrics(sim_X, real_X):
    """Compute sparsity-related metrics."""
    # Per-cell zero fraction
    sim_zero_per_cell = (sim_X == 0).mean(axis=1)
    real_zero_per_cell = (real_X == 0).mean(axis=1)

    # Per-gene zero fraction
    sim_zero_per_gene = (sim_X == 0).mean(axis=0)
    real_zero_per_gene = (real_X == 0).mean(axis=0)

    # Overall sparsity
    sim_sparsity = (sim_X == 0).mean()
    real_sparsity = (real_X == 0).mean()

    # KS test on per-cell zero fractions
    ks_cell_zeros, p_cell_zeros = ks_2samp(sim_zero_per_cell, real_zero_per_cell)

    return {
        "sim_overall_sparsity": float(sim_sparsity),
        "real_overall_sparsity": float(real_sparsity),
        "sparsity_difference": float(abs(sim_sparsity - real_sparsity)),
        "per_cell_zero_fraction_ks": float(ks_cell_zeros),
        "per_cell_zero_fraction_pval": float(p_cell_zeros),
        "sim_zero_per_cell": sim_zero_per_cell,
        "real_zero_per_cell": real_zero_per_cell,
        "sim_zero_per_gene": sim_zero_per_gene,
        "real_zero_per_gene": real_zero_per_gene,
    }


def compute_mean_variance(sim_X, real_X, gene_names):
    """Compute mean-variance relationship for each gene."""
    sim_means = sim_X.mean(axis=0)
    sim_vars = sim_X.var(axis=0)
    real_means = real_X.mean(axis=0)
    real_vars = real_X.var(axis=0)

    # Correlation of log-variance at matched genes (only for genes with nonzero mean)
    mask = (sim_means > 0) & (real_means > 0) & (sim_vars > 0) & (real_vars > 0)
    if mask.sum() > 2:
        log_sim_var = np.log1p(sim_vars[mask])
        log_real_var = np.log1p(real_vars[mask])
        var_corr = float(np.corrcoef(log_sim_var, log_real_var)[0, 1])
    else:
        var_corr = 0.0

    # Mean expression correlation
    if mask.sum() > 2:
        log_sim_mean = np.log1p(sim_means[mask])
        log_real_mean = np.log1p(real_means[mask])
        mean_corr = float(np.corrcoef(log_sim_mean, log_real_mean)[0, 1])
    else:
        mean_corr = 0.0

    return {
        "sim_means": sim_means,
        "sim_vars": sim_vars,
        "real_means": real_means,
        "real_vars": real_vars,
        "log_variance_correlation": var_corr,
        "log_mean_correlation": mean_corr,
    }


def compute_global_distributions(sim_X, real_X):
    """Compute global count distribution metrics."""
    sim_counts_per_cell = sim_X.sum(axis=1)
    real_counts_per_cell = real_X.sum(axis=1)
    sim_genes_per_cell = (sim_X > 0).sum(axis=1)
    real_genes_per_cell = (real_X > 0).sum(axis=1)

    ks_counts, p_counts = ks_2samp(sim_counts_per_cell, real_counts_per_cell)
    ks_genes, p_genes = ks_2samp(sim_genes_per_cell, real_genes_per_cell)

    return {
        "sim_counts_per_cell": sim_counts_per_cell,
        "real_counts_per_cell": real_counts_per_cell,
        "sim_genes_per_cell": sim_genes_per_cell,
        "real_genes_per_cell": real_genes_per_cell,
        "counts_per_cell_ks": float(ks_counts),
        "counts_per_cell_pval": float(p_counts),
        "genes_per_cell_ks": float(ks_genes),
        "genes_per_cell_pval": float(p_genes),
        "sim_mean_counts": float(sim_counts_per_cell.mean()),
        "real_mean_counts": float(real_counts_per_cell.mean()),
        "sim_mean_genes": float(sim_genes_per_cell.mean()),
        "real_mean_genes": float(real_genes_per_cell.mean()),
    }


def compute_full_panel_metrics(sim_X_full, real_X_full):
    """Compute global metrics across entire gene panels (not just overlap).

    Even though simulated and real panels may contain different genes,
    the global statistics (counts/cell, genes/cell, sparsity) characterize
    the technology's detection behavior and should be comparable.
    """
    # Counts per cell
    sim_counts = sim_X_full.sum(axis=1)
    real_counts = real_X_full.sum(axis=1)
    ks_counts, p_counts = ks_2samp(sim_counts, real_counts)

    # Genes per cell
    sim_genes = (sim_X_full > 0).sum(axis=1).astype(float)
    real_genes = (real_X_full > 0).sum(axis=1).astype(float)
    ks_genes, p_genes = ks_2samp(sim_genes, real_genes)

    # Overall sparsity
    sim_sparsity = float((sim_X_full == 0).mean())
    real_sparsity = float((real_X_full == 0).mean())

    return {
        "sim_counts_per_cell": sim_counts,
        "real_counts_per_cell": real_counts,
        "counts_per_cell_ks": float(ks_counts),
        "counts_per_cell_pval": float(p_counts),
        "sim_mean_counts": float(sim_counts.mean()),
        "real_mean_counts": float(real_counts.mean()),
        "sim_median_counts": float(np.median(sim_counts)),
        "real_median_counts": float(np.median(real_counts)),
        "sim_genes_per_cell": sim_genes,
        "real_genes_per_cell": real_genes,
        "genes_per_cell_ks": float(ks_genes),
        "genes_per_cell_pval": float(p_genes),
        "sim_mean_genes": float(sim_genes.mean()),
        "real_mean_genes": float(real_genes.mean()),
        "sim_sparsity": sim_sparsity,
        "real_sparsity": real_sparsity,
        "sim_n_genes_panel": int(sim_X_full.shape[1]),
        "real_n_genes_panel": int(real_X_full.shape[1]),
    }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_validation(tech_name, overlap_genes, sim_X, real_X,
                    sparsity, mean_var, global_dist, per_gene_ks,
                    full_panel, output_dir):
    """Generate multi-panel validation figure."""
    from matplotlib.gridspec import GridSpec

    n_hist_genes = min(N_GENES_PLOT, len(overlap_genes))
    n_hist_rows = (n_hist_genes + 5) // 6  # ceil(n/6)
    n_total_rows = n_hist_rows + 3  # +sparsity +overlap_global +full_panel

    fig = plt.figure(figsize=(24, 5 * n_total_rows + 2))
    fig.suptitle(f"{tech_name}: Simulated vs Real Data Comparison\n"
                 f"({len(overlap_genes)} overlapping genes, "
                 f"sim {full_panel['sim_n_genes_panel']} vs "
                 f"real {full_panel['real_n_genes_panel']} total genes)",
                 fontsize=16, fontweight="bold", y=0.99)

    gs = GridSpec(n_total_rows, 6, figure=fig, hspace=0.50, wspace=0.35)

    # --- Rows 0-(n_hist_rows-1): Per-gene marginal distributions ---
    for idx in range(n_hist_genes):
        row = idx // 6
        col = idx % 6
        ax = fig.add_subplot(gs[row, col])
        gene = overlap_genes[idx]
        gene_idx = overlap_genes.index(gene)

        sim_vals = sim_X[:, gene_idx]
        real_vals = real_X[:, gene_idx]

        max_val = max(np.percentile(sim_vals, 99), np.percentile(real_vals, 99), 1)
        bins = np.arange(0, max_val + 2) - 0.5

        ax.hist(real_vals, bins=bins, density=True, alpha=0.5, color="steelblue",
                label="Real", edgecolor="none")
        ax.hist(sim_vals, bins=bins, density=True, alpha=0.5, color="coral",
                label="Sim", edgecolor="none")

        ks_val = per_gene_ks[gene]["ks_statistic"]
        p_val = per_gene_ks[gene]["ks_pvalue"]
        ax.set_title(f"{gene}\nKS={ks_val:.3f} p={p_val:.1e}", fontsize=8)
        ax.set_xlabel("Count", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # --- Row n_hist_rows: Sparsity panels (overlap genes) ---
    r = n_hist_rows

    # Panel 1: Per-cell zero fraction
    ax_zc = fig.add_subplot(gs[r, 0:2])
    bins_z = np.linspace(0, 1, 50)
    ax_zc.hist(sparsity["real_zero_per_cell"], bins=bins_z, density=True,
               alpha=0.5, color="steelblue", label="Real")
    ax_zc.hist(sparsity["sim_zero_per_cell"], bins=bins_z, density=True,
               alpha=0.5, color="coral", label="Sim")
    ax_zc.set_xlabel("Zero fraction per cell")
    ax_zc.set_ylabel("Density")
    ax_zc.set_title(f"Per-cell zero fraction (overlap)\n"
                     f"KS={sparsity['per_cell_zero_fraction_ks']:.3f} "
                     f"p={sparsity['per_cell_zero_fraction_pval']:.1e}")
    ax_zc.legend(fontsize=8)

    # Panel 2: Per-gene zero fraction scatter
    ax_zg = fig.add_subplot(gs[r, 2:4])
    ax_zg.scatter(sparsity["real_zero_per_gene"], sparsity["sim_zero_per_gene"],
                  alpha=0.6, s=20, c="teal", edgecolors="none")
    ax_zg.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_zg.set_xlabel("Real zero fraction (per gene)")
    ax_zg.set_ylabel("Sim zero fraction (per gene)")
    ax_zg.set_title("Per-gene zero fraction (overlap)")
    ax_zg.set_xlim(0, 1.05)
    ax_zg.set_ylim(0, 1.05)

    # Panel 3: Dropout curve
    ax_dc = fig.add_subplot(gs[r, 4:6])
    real_means = mean_var["real_means"]
    sim_means = mean_var["sim_means"]
    real_zero_frac = sparsity["real_zero_per_gene"]
    sim_zero_frac = sparsity["sim_zero_per_gene"]

    mask_r = real_means > 0
    mask_s = sim_means > 0
    ax_dc.scatter(np.log1p(real_means[mask_r]), real_zero_frac[mask_r],
                  alpha=0.5, s=20, c="steelblue", label="Real", edgecolors="none")
    ax_dc.scatter(np.log1p(sim_means[mask_s]), sim_zero_frac[mask_s],
                  alpha=0.5, s=20, c="coral", label="Sim", edgecolors="none")
    ax_dc.set_xlabel("log(1 + mean expression)")
    ax_dc.set_ylabel("Zero fraction")
    ax_dc.set_title("Dropout curve (overlap)")
    ax_dc.legend(fontsize=8)

    # --- Row n_hist_rows+1: Mean-var, overlap counts/cell, overlap genes/cell, summary ---
    r2 = n_hist_rows + 1

    # Panel 1: Mean-variance relationship
    ax_mv = fig.add_subplot(gs[r2, 0:2])
    mask = (mean_var["real_means"] > 0) & (mean_var["sim_means"] > 0)
    if mask.sum() > 0:
        ax_mv.scatter(np.log1p(mean_var["real_means"][mask]),
                      np.log1p(mean_var["real_vars"][mask]),
                      alpha=0.5, s=20, c="steelblue", label="Real", edgecolors="none")
        ax_mv.scatter(np.log1p(mean_var["sim_means"][mask]),
                      np.log1p(mean_var["sim_vars"][mask]),
                      alpha=0.5, s=20, c="coral", label="Sim", edgecolors="none")
        x_range = np.linspace(0, max(np.log1p(mean_var["real_means"][mask]).max(),
                                      np.log1p(mean_var["sim_means"][mask]).max()), 100)
        ax_mv.plot(x_range, x_range, "k--", lw=1, alpha=0.4, label="Poisson (var=mean)")
    ax_mv.set_xlabel("log(1 + mean)")
    ax_mv.set_ylabel("log(1 + variance)")
    ax_mv.set_title(f"Mean-variance (overlap)\nr(log-var)={mean_var['log_variance_correlation']:.3f}")
    ax_mv.legend(fontsize=7)

    # Panel 2: Overlap counts per cell
    ax_cc = fig.add_subplot(gs[r2, 2:3])
    max_count = max(np.percentile(global_dist["real_counts_per_cell"], 99),
                    np.percentile(global_dist["sim_counts_per_cell"], 99))
    bins_c = np.linspace(0, max_count, 60)
    ax_cc.hist(global_dist["real_counts_per_cell"], bins=bins_c, density=True,
               alpha=0.5, color="steelblue", label="Real")
    ax_cc.hist(global_dist["sim_counts_per_cell"], bins=bins_c, density=True,
               alpha=0.5, color="coral", label="Sim")
    ax_cc.set_xlabel("Total counts per cell")
    ax_cc.set_ylabel("Density")
    ax_cc.set_title(f"Counts/cell (overlap)\nKS={global_dist['counts_per_cell_ks']:.3f} "
                     f"p={global_dist['counts_per_cell_pval']:.1e}")
    ax_cc.legend(fontsize=7)

    # Panel 3: Overlap genes per cell
    ax_gc = fig.add_subplot(gs[r2, 3:4])
    max_genes = max(np.percentile(global_dist["real_genes_per_cell"], 99),
                    np.percentile(global_dist["sim_genes_per_cell"], 99))
    bins_g = np.arange(0, max_genes + 2) - 0.5
    ax_gc.hist(global_dist["real_genes_per_cell"], bins=bins_g, density=True,
               alpha=0.5, color="steelblue", label="Real")
    ax_gc.hist(global_dist["sim_genes_per_cell"], bins=bins_g, density=True,
               alpha=0.5, color="coral", label="Sim")
    ax_gc.set_xlabel("Genes detected per cell")
    ax_gc.set_ylabel("Density")
    ax_gc.set_title(f"Genes/cell (overlap)\nKS={global_dist['genes_per_cell_ks']:.3f} "
                     f"p={global_dist['genes_per_cell_pval']:.1e}")
    ax_gc.legend(fontsize=7)

    # Panel 4: Overlap summary
    ax_sum = fig.add_subplot(gs[r2, 4:6])
    ax_sum.axis("off")
    ks_values = [per_gene_ks[g]["ks_statistic"] for g in overlap_genes]
    summary_text = (
        f"OVERLAP GENES ({len(overlap_genes)})\n"
        f"{'=' * 32}\n"
        f"Per-gene KS (mean):  {np.mean(ks_values):.3f}\n"
        f"Per-gene KS (median):{np.median(ks_values):.3f}\n\n"
        f"Sparsity sim/real:   {sparsity['sim_overall_sparsity']:.3f}/{sparsity['real_overall_sparsity']:.3f}\n"
        f"MV corr:   {mean_var['log_variance_correlation']:.3f}\n"
        f"Mean corr: {mean_var['log_mean_correlation']:.3f}\n\n"
        f"Counts/cell: {global_dist['sim_mean_counts']:.1f} / {global_dist['real_mean_counts']:.1f}\n"
        f"  KS={global_dist['counts_per_cell_ks']:.3f} p={global_dist['counts_per_cell_pval']:.1e}\n"
        f"Genes/cell:  {global_dist['sim_mean_genes']:.1f} / {global_dist['real_mean_genes']:.1f}\n"
        f"  KS={global_dist['genes_per_cell_ks']:.3f} p={global_dist['genes_per_cell_pval']:.1e}\n"
    )
    ax_sum.text(0.05, 0.95, summary_text, transform=ax_sum.transAxes,
                fontsize=9, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # --- Row n_hist_rows+2: Full-panel global metrics ---
    r3 = n_hist_rows + 2

    # Full-panel counts per cell
    ax_fcc = fig.add_subplot(gs[r3, 0:2])
    max_count_f = max(np.percentile(full_panel["real_counts_per_cell"], 99),
                      np.percentile(full_panel["sim_counts_per_cell"], 99))
    bins_cf = np.linspace(0, max_count_f, 80)
    ax_fcc.hist(full_panel["real_counts_per_cell"], bins=bins_cf, density=True,
                alpha=0.5, color="steelblue", label=f"Real ({full_panel['real_n_genes_panel']}g)")
    ax_fcc.hist(full_panel["sim_counts_per_cell"], bins=bins_cf, density=True,
                alpha=0.5, color="coral", label=f"Sim ({full_panel['sim_n_genes_panel']}g)")
    ax_fcc.set_xlabel("Total counts per cell")
    ax_fcc.set_ylabel("Density")
    ax_fcc.set_title(f"FULL PANEL: Counts/cell\nKS={full_panel['counts_per_cell_ks']:.3f} "
                      f"p={full_panel['counts_per_cell_pval']:.1e}")
    ax_fcc.legend(fontsize=8)

    # Full-panel genes per cell
    ax_fgc = fig.add_subplot(gs[r3, 2:4])
    max_genes_f = max(np.percentile(full_panel["real_genes_per_cell"], 99),
                      np.percentile(full_panel["sim_genes_per_cell"], 99))
    bins_gf = np.arange(0, max_genes_f + 2) - 0.5
    ax_fgc.hist(full_panel["real_genes_per_cell"], bins=bins_gf, density=True,
                alpha=0.5, color="steelblue", label=f"Real ({full_panel['real_n_genes_panel']}g)")
    ax_fgc.hist(full_panel["sim_genes_per_cell"], bins=bins_gf, density=True,
                alpha=0.5, color="coral", label=f"Sim ({full_panel['sim_n_genes_panel']}g)")
    ax_fgc.set_xlabel("Genes detected per cell")
    ax_fgc.set_ylabel("Density")
    ax_fgc.set_title(f"FULL PANEL: Genes/cell\nKS={full_panel['genes_per_cell_ks']:.3f} "
                      f"p={full_panel['genes_per_cell_pval']:.1e}")
    ax_fgc.legend(fontsize=8)

    # Full-panel summary
    ax_fsum = fig.add_subplot(gs[r3, 4:6])
    ax_fsum.axis("off")
    fp_text = (
        f"FULL PANEL SUMMARY\n"
        f"{'=' * 32}\n"
        f"Sim panel:  {full_panel['sim_n_genes_panel']} genes\n"
        f"Real panel: {full_panel['real_n_genes_panel']} genes\n\n"
        f"Counts/cell (sim):  {full_panel['sim_mean_counts']:.1f} "
        f"(med {full_panel['sim_median_counts']:.1f})\n"
        f"Counts/cell (real): {full_panel['real_mean_counts']:.1f} "
        f"(med {full_panel['real_median_counts']:.1f})\n"
        f"  KS={full_panel['counts_per_cell_ks']:.3f} "
        f"p={full_panel['counts_per_cell_pval']:.1e}\n\n"
        f"Genes/cell (sim):  {full_panel['sim_mean_genes']:.1f}\n"
        f"Genes/cell (real): {full_panel['real_mean_genes']:.1f}\n"
        f"  KS={full_panel['genes_per_cell_ks']:.3f} "
        f"p={full_panel['genes_per_cell_pval']:.1e}\n\n"
        f"Sparsity sim/real: "
        f"{full_panel['sim_sparsity']:.3f}/{full_panel['real_sparsity']:.3f}\n"
    )
    ax_fsum.text(0.05, 0.95, fp_text, transform=ax_fsum.transAxes,
                 fontsize=9, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))

    fig_path = os.path.join(output_dir, f"validation_{tech_name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Figure saved: {fig_path}")
    return fig_path


# ============================================================================
# MAIN
# ============================================================================

def validate_technology(tech_name, tech_cfg, hvg_list):
    """Run full validation for one technology."""
    print(f"\n{'=' * 70}")
    print(f"  VALIDATING: {tech_name}")
    print(f"{'=' * 70}")

    # Load real data
    print(f"   Loading real data: {tech_cfg['real_h5ad']}")
    real_X, real_genes = load_real_data(tech_cfg["real_h5ad"], MAX_REAL_CELLS)
    print(f"   Real data: {real_X.shape[0]} cells x {real_X.shape[1]} genes")

    # Load simulated data
    print(f"   Loading simulated data: {tech_cfg['sim_folder']}")
    sim_X, sim_genes = load_simulated_data_fast(tech_cfg["sim_folder"])
    print(f"   Simulated data: {sim_X.shape[0]} cells x {sim_X.shape[1]} genes")

    # Find overlapping genes (ordered by HVG rank)
    overlap_genes = find_overlapping_genes(sim_genes, real_genes, hvg_list)
    print(f"   Overlapping genes: {len(overlap_genes)}")
    if len(overlap_genes) == 0:
        print("   ERROR: No overlapping genes found. Skipping.")
        return None
    print(f"   Top overlapping: {overlap_genes[:10]}")

    # Subset both matrices to overlapping genes
    sim_sub, sim_found = subset_to_genes(sim_X, sim_genes, overlap_genes)
    real_sub, real_found = subset_to_genes(real_X, real_genes, overlap_genes)
    # Ensure same gene order
    assert sim_found == real_found, "Gene order mismatch"
    overlap_genes = sim_found
    print(f"   Subsetted to {sim_sub.shape[1]} overlapping genes")

    # --- Compute full-panel metrics (before subsetting) ---
    print("   Computing full-panel global metrics...")
    full_panel = compute_full_panel_metrics(sim_X, real_X)

    # --- Compute overlap-gene metrics ---
    print("   Computing per-gene KS statistics...")
    per_gene_ks = compute_per_gene_ks(sim_sub, real_sub, overlap_genes)

    print("   Computing per-gene Wasserstein distances...")
    per_gene_w = compute_per_gene_wasserstein(sim_sub, real_sub, overlap_genes)

    print("   Computing sparsity metrics...")
    sparsity = compute_sparsity_metrics(sim_sub, real_sub)

    print("   Computing mean-variance relationship...")
    mean_var = compute_mean_variance(sim_sub, real_sub, overlap_genes)

    print("   Computing global distributions (overlap)...")
    global_dist = compute_global_distributions(sim_sub, real_sub)

    # --- Summary ---
    ks_values = [per_gene_ks[g]["ks_statistic"] for g in overlap_genes]
    w_values = [per_gene_w[g]["wasserstein"] for g in overlap_genes]

    print(f"\n   --- RESULTS: {tech_name} ---")
    print(f"   OVERLAP GENES ({len(overlap_genes)}):")
    print(f"   Per-gene KS:  mean={np.mean(ks_values):.3f}, "
          f"median={np.median(ks_values):.3f}")
    print(f"   Per-gene Wasserstein:  mean={np.mean(w_values):.3f}, "
          f"median={np.median(w_values):.3f}")
    print(f"   Sparsity:  sim={sparsity['sim_overall_sparsity']:.3f}, "
          f"real={sparsity['real_overall_sparsity']:.3f}, "
          f"diff={sparsity['sparsity_difference']:.3f}")
    print(f"   Mean-var correlation: {mean_var['log_variance_correlation']:.3f}")
    print(f"   Mean expression correlation: {mean_var['log_mean_correlation']:.3f}")
    print(f"   Counts/cell:  sim={global_dist['sim_mean_counts']:.1f}, "
          f"real={global_dist['real_mean_counts']:.1f}, "
          f"KS={global_dist['counts_per_cell_ks']:.3f} "
          f"p={global_dist['counts_per_cell_pval']:.1e}")
    print(f"   Genes/cell:   sim={global_dist['sim_mean_genes']:.1f}, "
          f"real={global_dist['real_mean_genes']:.1f}, "
          f"KS={global_dist['genes_per_cell_ks']:.3f} "
          f"p={global_dist['genes_per_cell_pval']:.1e}")
    print(f"\n   FULL PANEL (sim {full_panel['sim_n_genes_panel']}g / "
          f"real {full_panel['real_n_genes_panel']}g):")
    print(f"   Counts/cell:  sim={full_panel['sim_mean_counts']:.1f}, "
          f"real={full_panel['real_mean_counts']:.1f}, "
          f"KS={full_panel['counts_per_cell_ks']:.3f} "
          f"p={full_panel['counts_per_cell_pval']:.1e}")
    print(f"   Genes/cell:   sim={full_panel['sim_mean_genes']:.1f}, "
          f"real={full_panel['real_mean_genes']:.1f}, "
          f"KS={full_panel['genes_per_cell_ks']:.3f} "
          f"p={full_panel['genes_per_cell_pval']:.1e}")
    print(f"   Sparsity:     sim={full_panel['sim_sparsity']:.3f}, "
          f"real={full_panel['real_sparsity']:.3f}")

    # --- Plot ---
    print("   Generating validation figure...")
    output_dir = VALIDATION_OUTPUT_FOLDER
    os.makedirs(output_dir, exist_ok=True)
    fig_path = plot_validation(
        tech_name, overlap_genes, sim_sub, real_sub,
        sparsity, mean_var, global_dist, per_gene_ks,
        full_panel, output_dir
    )

    # --- Save results JSON ---
    results = {
        "technology": tech_name,
        "n_overlapping_genes": len(overlap_genes),
        "overlapping_genes": overlap_genes,
        # Overlap-gene metrics
        "overlap": {
            "per_gene_ks_mean": round(float(np.mean(ks_values)), 4),
            "per_gene_ks_median": round(float(np.median(ks_values)), 4),
            "per_gene_wasserstein_mean": round(float(np.mean(w_values)), 4),
            "per_gene_wasserstein_median": round(float(np.median(w_values)), 4),
            "sparsity_sim": round(sparsity["sim_overall_sparsity"], 4),
            "sparsity_real": round(sparsity["real_overall_sparsity"], 4),
            "sparsity_difference": round(sparsity["sparsity_difference"], 4),
            "per_cell_zero_fraction_ks": round(sparsity["per_cell_zero_fraction_ks"], 4),
            "per_cell_zero_fraction_pval": round(sparsity["per_cell_zero_fraction_pval"], 4),
            "mean_variance_correlation": round(mean_var["log_variance_correlation"], 4),
            "mean_expression_correlation": round(mean_var["log_mean_correlation"], 4),
            "counts_per_cell_ks": round(global_dist["counts_per_cell_ks"], 4),
            "counts_per_cell_pval": round(global_dist["counts_per_cell_pval"], 4),
            "counts_per_cell_sim_mean": round(global_dist["sim_mean_counts"], 1),
            "counts_per_cell_real_mean": round(global_dist["real_mean_counts"], 1),
            "genes_per_cell_ks": round(global_dist["genes_per_cell_ks"], 4),
            "genes_per_cell_pval": round(global_dist["genes_per_cell_pval"], 4),
            "genes_per_cell_sim_mean": round(global_dist["sim_mean_genes"], 1),
            "genes_per_cell_real_mean": round(global_dist["real_mean_genes"], 1),
        },
        # Full-panel metrics
        "full_panel": {
            "sim_n_genes": full_panel["sim_n_genes_panel"],
            "real_n_genes": full_panel["real_n_genes_panel"],
            "counts_per_cell_ks": round(full_panel["counts_per_cell_ks"], 4),
            "counts_per_cell_pval": round(full_panel["counts_per_cell_pval"], 4),
            "counts_per_cell_sim_mean": round(full_panel["sim_mean_counts"], 1),
            "counts_per_cell_real_mean": round(full_panel["real_mean_counts"], 1),
            "counts_per_cell_sim_median": round(full_panel["sim_median_counts"], 1),
            "counts_per_cell_real_median": round(full_panel["real_median_counts"], 1),
            "genes_per_cell_ks": round(full_panel["genes_per_cell_ks"], 4),
            "genes_per_cell_pval": round(full_panel["genes_per_cell_pval"], 4),
            "genes_per_cell_sim_mean": round(full_panel["sim_mean_genes"], 1),
            "genes_per_cell_real_mean": round(full_panel["real_mean_genes"], 1),
            "sparsity_sim": round(full_panel["sim_sparsity"], 4),
            "sparsity_real": round(full_panel["real_sparsity"], 4),
        },
        # Per-gene details with p-values
        "per_gene_details": {
            gene: {
                "ks_statistic": round(per_gene_ks[gene]["ks_statistic"], 4),
                "ks_pvalue": round(per_gene_ks[gene]["ks_pvalue"], 6),
                "wasserstein": round(per_gene_w[gene]["wasserstein"], 4),
            }
            for gene in overlap_genes
        },
        "figure_path": fig_path,
    }

    json_path = os.path.join(output_dir, f"validation_{tech_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved: {json_path}")

    return results


def main():
    print("=" * 70)
    print("VALIDATION: Simulated vs Real Technology Comparison")
    print("=" * 70)

    hvg_list = load_hvg_list(HVG_LIST_PATH)
    print(f"Loaded HVG list: {len(hvg_list)} genes")

    all_results = {}
    for tech_name, tech_cfg in TECHNOLOGIES.items():
        if not os.path.exists(tech_cfg["sim_folder"]):
            print(f"\n   WARNING: Simulated data not found for {tech_name} "
                  f"({tech_cfg['sim_folder']}). Skipping.")
            continue
        if not os.path.exists(tech_cfg["real_h5ad"]):
            print(f"\n   WARNING: Real data not found for {tech_name} "
                  f"({tech_cfg['real_h5ad']}). Skipping.")
            continue

        results = validate_technology(tech_name, tech_cfg, hvg_list)
        if results is not None:
            all_results[tech_name] = results

    # --- Cross-technology summary ---
    if all_results:
        print(f"\n{'=' * 70}")
        print("CROSS-TECHNOLOGY SUMMARY")
        print(f"{'=' * 70}")

        print(f"\n  OVERLAP-GENE METRICS:")
        print(f"  {'Tech':>10s}  {'#Genes':>6s}  {'KS(med)':>8s}  {'W(med)':>8s}  "
              f"{'Spars':>8s}  {'MV-corr':>7s}  {'Cnt/cell':>10s}  {'Gen/cell':>10s}")
        print("  " + "-" * 82)
        for tech, r in all_results.items():
            o = r["overlap"]
            print(f"  {tech:>10s}  {r['n_overlapping_genes']:6d}  "
                  f"{o['per_gene_ks_median']:8.3f}  {o['per_gene_wasserstein_median']:8.3f}  "
                  f"{o['sparsity_difference']:8.3f}  {o['mean_variance_correlation']:7.3f}  "
                  f"{o['counts_per_cell_sim_mean']:5.1f}/{o['counts_per_cell_real_mean']:<4.1f}  "
                  f"{o['genes_per_cell_sim_mean']:5.1f}/{o['genes_per_cell_real_mean']:<4.1f}")

        print(f"\n  FULL-PANEL METRICS:")
        print(f"  {'Tech':>10s}  {'SimG':>5s}  {'RealG':>5s}  "
              f"{'Cnt/cell(sim)':>13s}  {'Cnt/cell(real)':>14s}  {'KS':>6s}  "
              f"{'Gen/cell(sim)':>13s}  {'Gen/cell(real)':>14s}  {'KS':>6s}")
        print("  " + "-" * 100)
        for tech, r in all_results.items():
            fp = r["full_panel"]
            print(f"  {tech:>10s}  {fp['sim_n_genes']:5d}  {fp['real_n_genes']:5d}  "
                  f"{fp['counts_per_cell_sim_mean']:13.1f}  {fp['counts_per_cell_real_mean']:14.1f}  "
                  f"{fp['counts_per_cell_ks']:6.3f}  "
                  f"{fp['genes_per_cell_sim_mean']:13.1f}  {fp['genes_per_cell_real_mean']:14.1f}  "
                  f"{fp['genes_per_cell_ks']:6.3f}")

    print(f"\nOutput directory: {VALIDATION_OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()
