#!/usr/bin/env python3
"""Evaluate TACCO on all sweep datasets and report accuracy."""
import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import tacco as tc
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = 'skin_sweep_output_new'
RESULTS = {}

def build_reference_from_csv(ref_csv_path, label_name="cell_type"):
    df = pd.read_csv(ref_csv_path, index_col=0)
    eps = 1e-6
    col_sums = df.sum(axis=0)
    zero_cols = col_sums.index[col_sums == 0]
    df.loc[:, zero_cols] = eps
    X = df.T.values.astype("float32")
    ref_adata = sc.AnnData(X=X)
    ref_adata.var_names = df.index.astype(str)
    ref_adata.obs_names = df.columns.astype(str)
    ref_adata.obs[label_name] = df.columns.astype(str)
    return ref_adata

def build_spatial_from_csv(cell_csv_path, dots_csv_path, reference_genes,
                           gt_column="Class ID", x_col="X", y_col="Y",
                           cell_id_col="cell"):
    cell_df = pd.read_csv(cell_csv_path)
    genes_df = pd.read_csv(dots_csv_path).dropna()
    genes_df["cell_id"] = genes_df[cell_id_col]
    assigned = genes_df.dropna(subset=["cell_id"])
    count_matrix = (
        assigned.groupby(["cell_id", "gene"])
        .size()
        .unstack(fill_value=0)
    )
    count_matrix = count_matrix.reindex(index=cell_df.index, fill_value=0)
    missing = [g for g in reference_genes if g not in count_matrix.columns]
    for g in missing:
        count_matrix[g] = 0
    count_matrix = count_matrix[reference_genes]
    expr_array = count_matrix.values.astype("float32")
    st_adata = sc.AnnData(X=expr_array)
    st_adata.var_names = reference_genes
    st_adata.obs_names = cell_df.index.astype(str)
    locations = cell_df[[x_col, y_col]].values
    st_adata.obsm["spatial"] = locations
    if gt_column in cell_df.columns:
        st_adata.obs["gt_label"] = cell_df[gt_column].astype(str).values
    st_adata.layers["counts"] = st_adata.X.copy()
    st_adata.raw = st_adata
    return st_adata

# Sort directories for consistent ordering
dirs = sorted([d for d in os.listdir(BASE_DIR)
               if os.path.isdir(os.path.join(BASE_DIR, d))])

for directory in dirs:
    setting = directory
    ref_path = f"{BASE_DIR}/{setting}/cell_type_expression.csv"
    cell_path = f"{BASE_DIR}/{setting}/cells.csv"
    dots_path = f"{BASE_DIR}/{setting}/dots.csv"

    if not all(os.path.exists(p) for p in [ref_path, cell_path, dots_path]):
        print(f"  SKIP {setting} (missing files)")
        continue

    print(f"  Evaluating {setting}...", end=" ", flush=True)

    ref_adata = build_reference_from_csv(ref_path, label_name="cluster")
    reference_genes = ref_adata.var_names.tolist()

    st_adata = build_spatial_from_csv(
        cell_path, dots_path,
        reference_genes=reference_genes,
        gt_column="Class ID",
    )

    tc.tl.annotate(
        adata=st_adata,
        reference=ref_adata,
        method="OT",
        annotation_key="cluster",
        result_key="tacco_scores",
    )

    scores = st_adata.obsm["tacco_scores"]
    pred = scores.idxmax(axis=1).astype(str).values

    gt_series = st_adata.obs["gt_label"].astype(str)
    ref_labels = list(map(str, ref_adata.obs_names.tolist()))
    gt_mapped = gt_series.map(
        lambda x: ref_labels[int(x)] if x.isdigit() and int(x) < len(ref_labels) else x
    )

    acc = accuracy_score(gt_mapped, pred)
    f1_m = f1_score(gt_mapped, pred, average="macro", zero_division=0)
    f1_w = f1_score(gt_mapped, pred, average="weighted", zero_division=0)

    RESULTS[setting] = {
        'accuracy': round(acc, 4),
        'f1_macro': round(f1_m, 4),
        'f1_weighted': round(f1_w, 4),
        'n_cells': len(gt_mapped),
    }
    print(f"acc={acc:.3f}  f1_macro={f1_m:.3f}  f1_weighted={f1_w:.3f}")

# Save results
with open('tacco_sweep_results.json', 'w') as f:
    json.dump(RESULTS, f, indent=2)

# Print summary by case
print("\n" + "=" * 70)
print("SUMMARY BY CASE")
print("=" * 70)

for case_prefix in ['CASE1', 'CASE2', 'CASE3', 'CASE4', 'CASE5a', 'CASE5b']:
    case_results = {k: v for k, v in RESULTS.items() if k.startswith(case_prefix)}
    if case_results:
        accs = [v['accuracy'] for v in case_results.values()]
        print(f"\n{case_prefix}:")
        for name, res in case_results.items():
            print(f"  {name:40s}  acc={res['accuracy']:.3f}")
        print(f"  Range: {min(accs):.3f} → {max(accs):.3f}")
