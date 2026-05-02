# main_random_forest.py
"""
Replication of RQ1 from the paper:
    "A Dead End for Classical Machine Learning in Merge Conflict Resolution?"
 
This script compares two data-partitioning strategies:
    - S1: Random chunk-level splitting (WITH data leakage) --> expected accuracy ~0.83
    - S3: Merge-level grouping (WITHOUT data leakage) --> expected accuracy ~0.66
 
Both use 5-fold cross-validation and a Random Forest classifier with
the hyperparameters from Elias et al. [3] (n_estimators=400).
 
Reference: Paper Table II, Section IV-A.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
 
# Allow running from any directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
 
from src.data.preprocess_dataset import load_dataset
from src.data.feature_builder import build_features
from src.models.random_forest import train_rf, predict_rf
from src.metrics.evaluation import (
    compute_all_metrics,
    print_classification_report,
    print_confusion_matrix,
)

N_SPLITS = 5 


def run_s1_random_splitting(X, y, label_encoder=None):
    """
    S1: Random chunk-level splitting WITHOUT merge-level grouping.
    This replicates the data-leakage scenario that inflates accuracy to ~0.83.
    Chunks from the same merge can appear in both train and test.
 
    Args:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Encoded labels.
        label_encoder (LabelEncoder, optional): For printing class names.
 
    Returns:
        dict: Mean metrics across all folds.
    """
    print("\n" + "="*70)
    print("S1: Random Chunk-Level Splitting (WITH data leakage)")
    print("    Expected accuracy: ~0.83  (paper Table II, left side)")
    print("="*70)
 
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
 
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
 
        model = train_rf(X_train, y_train)
        y_pred = predict_rf(model, X_test)
        metrics = compute_all_metrics(y_test, y_pred, y_train)
        fold_metrics.append(metrics)
 
        print(f"  Fold {fold}: acc={metrics['accuracy']:.4f}  "
              f"zeror={metrics['zeror']:.4f}  "
              f"f1={metrics['f1_weighted']:.4f}  "
              f"NI={metrics['normalized_improvement']:.4f}")
 
    mean_metrics = _mean_metrics(fold_metrics, prefix="S1")
 
    print(f"\n  MEAN across {N_SPLITS} folds:")
    print(f"    Accuracy:               {mean_metrics['S1_accuracy']:.4f}")
    print(f"    ZeroR:                  {mean_metrics['S1_zeror']:.4f}")
    print(f"    Weighted F1:            {mean_metrics['S1_f1_weighted']:.4f}")
    print(f"    Weighted Precision:     {mean_metrics['S1_precision_weighted']:.4f}")
    print(f"    Weighted Recall:        {mean_metrics['S1_recall_weighted']:.4f}")
    print(f"    Normalized Improvement: {mean_metrics['S1_normalized_improvement']:.4f}")
 
    # Print full classification report on last fold for reference
    if label_encoder:
        print(f"\n  Classification Report (fold {N_SPLITS}):")
        print_classification_report(y_test, y_pred, label_encoder)
 
    return mean_metrics
 
 
def run_s3_merge_level_grouping(X, y, merge_ids, label_encoder=None):
    """
    S3: Merge-level grouping WITHOUT data leakage.
    All chunks from the same merge commit go entirely to train OR test.
    This is the correct evaluation setting. Expected accuracy ~0.66.
 
    Args:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Encoded labels.
        merge_ids (pd.Series): Merge commit IDs for grouping.
        label_encoder (LabelEncoder, optional): For printing class names.
 
    Returns:
        dict: Mean metrics across all folds.
    """
    print("\n" + "="*70)
    print("S3: Merge-Level Grouping (WITHOUT data leakage)")
    print("    Expected accuracy: ~0.66  (paper Table II, right side)")
    print("="*70)
 
    # Shuffle merge groups before splitting to avoid ordering bias
    unique_merges = merge_ids.unique()
    rng = np.random.default_rng(seed=42)
    shuffled_merges = rng.permutation(unique_merges)
 
    # Build a mapping from merge_id to fold index
    fold_assignment = {
        mid: i % N_SPLITS
        for i, mid in enumerate(shuffled_merges)
    }
    fold_labels = merge_ids.map(fold_assignment).values
 
    fold_metrics = []
    for fold in range(N_SPLITS):
        test_mask = fold_labels == fold
        train_mask = ~test_mask
 
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
 
        model = train_rf(X_train, y_train)
        y_pred = predict_rf(model, X_test)
        metrics = compute_all_metrics(y_test, y_pred, y_train)
        fold_metrics.append(metrics)
 
        print(f"  Fold {fold+1}: acc={metrics['accuracy']:.4f}  "
              f"zeror={metrics['zeror']:.4f}  "
              f"f1={metrics['f1_weighted']:.4f}  "
              f"NI={metrics['normalized_improvement']:.4f}")
 
    mean_metrics = _mean_metrics(fold_metrics, prefix="S3")
 
    print(f"\n  MEAN across {N_SPLITS} folds:")
    print(f"    Accuracy:               {mean_metrics['S3_accuracy']:.4f}")
    print(f"    ZeroR:                  {mean_metrics['S3_zeror']:.4f}")
    print(f"    Weighted F1:            {mean_metrics['S3_f1_weighted']:.4f}")
    print(f"    Weighted Precision:     {mean_metrics['S3_precision_weighted']:.4f}")
    print(f"    Weighted Recall:        {mean_metrics['S3_recall_weighted']:.4f}")
    print(f"    Normalized Improvement: {mean_metrics['S3_normalized_improvement']:.4f}")
 
    if label_encoder:
        print(f"\n  Classification Report (fold {N_SPLITS}):")
        print_classification_report(y_test, y_pred, label_encoder)
 
    return mean_metrics
 
 
def run_per_project(X, y, merge_ids, project_names, strategy="S3"):
    """
    Run S1 or S3 per project to replicate the per-project breakdown in Table II.
 
    Args:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Encoded labels.
        merge_ids (pd.Series): Merge commit IDs.
        project_names (pd.Series): Project name for each sample.
        strategy (str): "S1" or "S3".
 
    Returns:
        pd.DataFrame: Per-project results.
    """
    print(f"\n{'='*70}")
    print(f"Per-Project Results — {strategy}")
    print(f"{'='*70}")
    print(f"  {'Project':<35} {'Merges':>7} {'Chunks':>7} {'Acc':>6} "
          f"{'ZeroR':>6} {'F1':>6} {'NI':>6}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
 
    rows = []
    for proj in sorted(project_names.unique()):
        mask = (project_names == proj).values
 
        X_proj = X[mask]
        y_proj = y[mask]
        mid_proj = merge_ids[mask]
        n_chunks = mask.sum()
        n_merges = mid_proj.nunique()
 
        if n_merges < N_SPLITS:
            print(f"  {proj:<35} — skipped (only {n_merges} merges, need {N_SPLITS})")
            continue
 
        if strategy == "S1":
            # Random stratified split per project
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
            splits = list(skf.split(X_proj, y_proj))
        else:
            # Merge-level grouping per project
            unique_merges = mid_proj.unique()
            rng = np.random.default_rng(seed=42)
            shuffled_merges = rng.permutation(unique_merges)
            fold_assignment = {mid: i % N_SPLITS for i, mid in enumerate(shuffled_merges)}
            fold_labels_proj = mid_proj.map(fold_assignment).values
 
            splits = []
            X_proj_reset = X_proj.reset_index(drop=True)
            for fold in range(N_SPLITS):
                test_idx = np.where(fold_labels_proj == fold)[0]
                train_idx = np.where(fold_labels_proj != fold)[0]
                splits.append((train_idx, test_idx))
 
        fold_metrics = []
        for train_idx, test_idx in splits:
            if strategy == "S1":
                Xtr, Xte = X_proj.iloc[train_idx], X_proj.iloc[test_idx]
                ytr, yte = y_proj[train_idx], y_proj[test_idx]
            else:
                Xtr = X_proj.iloc[train_idx] if hasattr(X_proj, 'iloc') else X_proj[train_idx]
                Xte = X_proj.iloc[test_idx]  if hasattr(X_proj, 'iloc') else X_proj[test_idx]
                ytr, yte = y_proj[train_idx], y_proj[test_idx]
 
            if len(np.unique(ytr)) < 2:
                continue
 
            model  = train_rf(Xtr, ytr)
            y_pred = predict_rf(model, Xte)
            fold_metrics.append(compute_all_metrics(yte, y_pred, ytr))
 
        if not fold_metrics:
            continue
 
        m = _mean_metrics(fold_metrics, prefix="")
        row = {
            "project": proj,
            "merges": n_merges,
            "chunks": n_chunks,
            "accuracy": m["accuracy"],
            "zeror": m["zeror"],
            "f1": m["f1_weighted"],
            "precision": m["precision_weighted"],
            "recall": m["recall_weighted"],
            "NI": m["normalized_improvement"],
            "strategy": strategy,
        }
        rows.append(row)
        print(f"  {proj:<35} {n_merges:>7} {n_chunks:>7} "
              f"{row['accuracy']:>6.2f} {row['zeror']:>6.2f} "
              f"{row['f1']:>6.2f} {row['NI']:>6.2f}")
 
    if rows:
        df_proj = pd.DataFrame(rows)
        mean_row = df_proj[["accuracy","zeror","f1","precision","recall","NI"]].mean()
        print(f"\n  {'MEAN':<35} {'':>7} {'':>7} "
              f"{mean_row['accuracy']:>6.2f} {mean_row['zeror']:>6.2f} "
              f"{mean_row['f1']:>6.2f} {mean_row['NI']:>6.2f}")
        return df_proj
 
    return pd.DataFrame()
 
 
def _mean_metrics(fold_metrics_list, prefix=""):
    """Compute mean of all metrics across folds, with optional key prefix."""
    keys = fold_metrics_list[0].keys()
    mean = {}
    for k in keys:
        key = f"{prefix}_{k}" if prefix else k
        mean[key] = float(np.mean([m[k] for m in fold_metrics_list]))
    return mean
 
 
def main():
    # Paths
    data_path = os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
 
    # Load data
    df = load_dataset(data_path)
 
    # Recover label encoder from the preprocessed dataset
    le = LabelEncoder()
    le.fit(df["conflictResolutionResult"])
 
    X, y, merge_ids, project_names = build_features(df)
 
    print(f"\nDataset loaded:")
    print(f"  Samples:        {len(df)}")
    print(f"  Features:       {X.shape[1]}")
    print(f"  Projects:       {project_names.nunique()}")
    print(f"  Merge commits:  {merge_ids.nunique()}")
    print(f"  Classes ({len(le.classes_)}):    {list(le.classes_)}")
 
    # S1: Random splitting (WITH data leakage)  → expected ~0.83
    s1_results = run_s1_random_splitting(X, y, label_encoder=le)
 
    # S3: Merge-level grouping (WITHOUT data leakage) → expected ~0.66
    s3_results = run_s3_merge_level_grouping(X, y, merge_ids, label_encoder=le)
 
    # Per-project breakdown (replicates Table II)
    df_s1_proj = run_per_project(X, y, merge_ids, project_names, strategy="S1")
    df_s3_proj = run_per_project(X, y, merge_ids, project_names, strategy="S3")
 
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY — RQ1 Replication")
    print("="*70)
    print(f"  {'Metric':<30} {'S1 (leakage)':>14} {'S3 (no leakage)':>16}")
    print(f"  {'-'*30} {'-'*14} {'-'*16}")
    metrics_to_show = ["accuracy", "zeror", "f1_weighted",
                       "precision_weighted", "recall_weighted",
                       "normalized_improvement"]
    for m in metrics_to_show:
        v1 = s1_results.get(f"S1_{m}", "—")
        v3 = s3_results.get(f"S3_{m}", "—")
        print(f"  {m:<30} {v1:>14.4f} {v3:>16.4f}")
 
    print(f"\n  Paper reference:")
    print(f"    S1 accuracy: ~0.83  |  S3 accuracy: ~0.66")
    print(f"    Delta: {s1_results['S1_accuracy'] - s3_results['S3_accuracy']:.4f}  "
          f"(paper reports ~0.168)")
 
    # Save results
    summary_rows = [
        {"strategy": "S1_random_splitting",    **{k.replace("S1_",""):v for k,v in s1_results.items()}},
        {"strategy": "S3_merge_level_grouping", **{k.replace("S3_",""):v for k,v in s3_results.items()}},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, "rf_rq1_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved → {summary_path}")
 
    if not df_s1_proj.empty:
        p1 = os.path.join(results_dir, "rf_rq1_per_project_S1.csv")
        df_s1_proj.to_csv(p1, index=False)
        print(f"  Per-project S1 → {p1}")
 
    if not df_s3_proj.empty:
        p3 = os.path.join(results_dir, "rf_rq1_per_project_S3.csv")
        df_s3_proj.to_csv(p3, index=False)
        print(f"  Per-project S3 → {p3}")
 
 
if __name__ == "__main__":
    main()
 