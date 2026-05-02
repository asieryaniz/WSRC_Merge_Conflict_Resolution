# main_wsrc.py
"""
Phase 2: WSRC vs. Random Forest comparison — RQ1 setting.

Evaluates WSRC per project using S3 (merge-level grouping, no data leakage),
matching the evaluation setup used for Random Forest in main_random_forest.py.

Key design decisions to make WSRC computationally feasible:

1. Per-project evaluation: each project is its own experiment (same as paper).
   Project sizes range from ~1k to ~15k chunks; the 84k aosp-mirror project
   is handled with dictionary subsampling.

2. Stratified dictionary subsampling: instead of using the full training set
   as the WSRC dictionary (which makes Lasso intractable for large projects),
   we sample at most `max_dict_per_class` training examples per class.
   This keeps the dictionary size bounded while preserving class balance.

3. alpha tuning: alpha=0.05 offers a good speed/sparsity tradeoff
   (~17ms/sample vs ~68ms at alpha=0.01) while maintaining meaningful
   sparse representations.

4. Weight method: "similarity" (inverse Euclidean distance) — locally
   adaptive, no class-frequency assumptions.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.data.preprocess_dataset import load_dataset
from src.data.feature_builder import build_features
from src.models.wsrc import wsrc_predict, compute_weights
from src.models.random_forest import train_rf, predict_rf
from src.metrics.evaluation import compute_all_metrics


# Configuration
N_SPLITS = 5 # 5-fold CV (same as paper)
ALPHA = 0.05 # Lasso regularization — balances speed and sparsity
WEIGHT_METHOD = "similarity" # "similarity" | "class" | "uniform"
TOP_K = 1  # Only for weight_method="class"
MAX_DICT_PER_CLASS = 200 # Max training samples per class in the dictionary keeps dict ≤ 1200 samples (6 classes × 200)
RANDOM_STATE = 42


def build_merge_level_folds(merge_ids, n_splits, random_state=42):
    """
    Build fold indices using merge-level grouping (S3).
    All chunks from the same merge commit go entirely into one fold.

    Args:
        merge_ids (pd.Series): Merge ID for each sample.
        n_splits (int): Number of folds.
        random_state (int): Seed for reproducibility.

    Returns:
        np.ndarray: Fold assignment (0..n_splits-1) for each sample.
    """
    unique_merges = merge_ids.unique()
    rng = np.random.default_rng(seed=random_state)
    shuffled_merges = rng.permutation(unique_merges)
    assignment = {mid: i % n_splits for i, mid in enumerate(shuffled_merges)}
    return merge_ids.map(assignment).values


def subsample_dictionary(X_train, y_train, max_per_class, random_state=42):
    """
    Stratified subsampling of the training set to build a bounded dictionary.
    Samples at most `max_per_class` examples per class, preserving balance.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        max_per_class (int): Maximum samples per class.
        random_state (int): Seed.

    Returns:
        tuple: (X_dict, y_dict) — subsampled dictionary.
    """
    rng = np.random.default_rng(seed=random_state)
    selected = []
    for c in np.unique(y_train):
        idx = np.where(y_train == c)[0]
        k = min(max_per_class, len(idx))
        selected.extend(rng.choice(idx, k, replace=False))
    selected = np.array(selected)
    return X_train[selected], y_train[selected]


def run_wsrc_per_project(X, y, merge_ids, project_names):
    """
    Run WSRC with S3 (merge-level grouping) per project.

    Returns:
        pd.DataFrame: Per-project results with all metrics.
    """
    print("\n" + "="*72)
    print("WSRC — Per-Project Evaluation (S3: merge-level grouping)")
    print(f"  alpha={ALPHA}, weight={WEIGHT_METHOD}, "
          f"max_dict_per_class={MAX_DICT_PER_CLASS}")
    print("="*72)
    print(f"  {'Project':<38} {'Chunks':>6} {'Acc':>6} "
          f"{'ZeroR':>6} {'F1':>6} {'NI':>6} {'Time':>7}")
    print(f"  {'-'*38} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    rows = []
    projects = sorted(project_names.unique())

    for proj in projects:
        mask = (project_names == proj).values
        X_proj = X[mask]
        y_proj = y[mask]
        mid_proj = merge_ids[mask]
        n_chunks = mask.sum()
        n_merges = mid_proj.nunique()

        if n_merges < N_SPLITS:
            print(f"  {proj:<38} — skipped ({n_merges} merges < {N_SPLITS})")
            continue

        fold_labels = build_merge_level_folds(mid_proj, N_SPLITS, RANDOM_STATE)
        fold_metrics = []
        proj_start = time.time()

        for fold in range(N_SPLITS):
            train_idx = np.where(fold_labels != fold)[0]
            test_idx  = np.where(fold_labels == fold)[0]

            X_tr_raw = X_proj.iloc[train_idx].values
            X_te_raw = X_proj.iloc[test_idx].values
            y_tr = y_proj[train_idx]
            y_te = y_proj[test_idx]

            # Normalize (essential for SRC/WSRC — distances are meaningful only
            # when features are on the same scale)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            # Build bounded dictionary (stratified subsample)
            X_dict, y_dict = subsample_dictionary(
                X_tr, y_tr, MAX_DICT_PER_CLASS, RANDOM_STATE
            )

            # WSRC: predict each test sample individually
            y_pred = []
            for x_te in X_te:
                weights = compute_weights(
                    X_dict, y_dict, x_te,
                    method=WEIGHT_METHOD, top_k=TOP_K
                )
                pred = wsrc_predict(
                    X_dict, y_dict,
                    x_te.reshape(1, -1),
                    weights=weights,
                    alpha=ALPHA
                )
                y_pred.append(pred[0])

            fold_metrics.append(compute_all_metrics(y_te, np.array(y_pred), y_tr))

        elapsed = time.time() - proj_start

        # Average metrics across folds
        acc = np.mean([m["accuracy"] for m in fold_metrics])
        zeror = np.mean([m["zeror"] for m in fold_metrics])
        f1 = np.mean([m["f1_weighted"] for m in fold_metrics])
        prec = np.mean([m["precision_weighted"] for m in fold_metrics])
        rec = np.mean([m["recall_weighted"] for m in fold_metrics])
        ni = np.mean([m["normalized_improvement"] for m in fold_metrics])

        row = {
            "project": proj,
            "chunks": n_chunks,
            "merges": n_merges,
            "accuracy": round(acc,  4),
            "zeror": round(zeror,4),
            "f1": round(f1,   4),
            "precision": round(prec, 4),
            "recall": round(rec,  4),
            "NI": round(ni,   4),
            "time_s": round(elapsed, 1),
        }
        rows.append(row)
        print(f"  {proj:<38} {n_chunks:>6} {acc:>6.2f} "
              f"{zeror:>6.2f} {f1:>6.2f} {ni:>6.2f} {elapsed:>6.0f}s")

    df = pd.DataFrame(rows)
    if not df.empty:
        mean = df[["accuracy","zeror","f1","NI"]].mean()
        print(f"\n  {'MEAN':<38} {'':>6} {mean['accuracy']:>6.2f} "
              f"{mean['zeror']:>6.2f} {mean['f1']:>6.2f} {mean['NI']:>6.2f}")
    return df


def run_rf_per_project(X, y, merge_ids, project_names):
    """
    Run Random Forest with S3 (merge-level grouping) per project.
    This is the direct comparison baseline for WSRC.

    Returns:
        pd.DataFrame: Per-project results.
    """
    print("\n" + "="*72)
    print("Random Forest — Per-Project Evaluation (S3: merge-level grouping)")
    print("="*72)
    print(f"  {'Project':<38} {'Chunks':>6} {'Acc':>6} "
          f"{'ZeroR':>6} {'F1':>6} {'NI':>6}")
    print(f"  {'-'*38} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    rows = []
    for proj in sorted(project_names.unique()):
        mask = (project_names == proj).values
        X_proj = X[mask]
        y_proj = y[mask]
        mid_proj = merge_ids[mask]
        n_chunks = mask.sum()
        n_merges = mid_proj.nunique()

        if n_merges < N_SPLITS:
            continue

        fold_labels  = build_merge_level_folds(mid_proj, N_SPLITS, RANDOM_STATE)
        fold_metrics = []

        for fold in range(N_SPLITS):
            train_idx = np.where(fold_labels != fold)[0]
            test_idx  = np.where(fold_labels == fold)[0]

            X_tr = X_proj.iloc[train_idx]
            X_te = X_proj.iloc[test_idx]
            y_tr = y_proj[train_idx]
            y_te = y_proj[test_idx]

            model  = train_rf(X_tr, y_tr)
            y_pred = predict_rf(model, X_te)
            fold_metrics.append(compute_all_metrics(y_te, y_pred, y_tr))

        acc = np.mean([m["accuracy"] for m in fold_metrics])
        zeror = np.mean([m["zeror"] for m in fold_metrics])
        f1 = np.mean([m["f1_weighted"] for m in fold_metrics])
        prec = np.mean([m["precision_weighted"] for m in fold_metrics])
        rec = np.mean([m["recall_weighted"] for m in fold_metrics])
        ni = np.mean([m["normalized_improvement"] for m in fold_metrics])

        row = {
            "project": proj,
            "chunks": n_chunks,
            "merges": n_merges,
            "accuracy": round(acc,  4),
            "zeror": round(zeror,4),
            "f1": round(f1,   4),
            "precision": round(prec, 4),
            "recall": round(rec,  4),
            "NI": round(ni,   4),
        }
        rows.append(row)
        print(f"  {proj:<38} {n_chunks:>6} {acc:>6.2f} "
              f"{zeror:>6.2f} {f1:>6.2f} {ni:>6.2f}")

    df = pd.DataFrame(rows)
    if not df.empty:
        mean = df[["accuracy","zeror","f1","NI"]].mean()
        print(f"\n  {'MEAN':<38} {'':>6} {mean['accuracy']:>6.2f} "
              f"{mean['zeror']:>6.2f} {mean['f1']:>6.2f} {mean['NI']:>6.2f}")
    return df


def print_comparison(df_wsrc, df_rf):
    """Print side-by-side comparison of WSRC vs RF per project."""
    print("\n" + "="*72)
    print("COMPARISON: WSRC vs. Random Forest (S3, per project)")
    print("="*72)
    print(f"  {'Project':<38} {'WSRC Acc':>9} {'RF Acc':>7} "
          f"{'WSRC F1':>8} {'RF F1':>6} {'Winner':>8}")
    print(f"  {'-'*38} {'-'*9} {'-'*7} {'-'*8} {'-'*6} {'-'*8}")

    merged = df_wsrc.merge(df_rf, on="project", suffixes=("_wsrc", "_rf"))
    wsrc_wins = 0

    for _, row in merged.iterrows():
        winner = "WSRC" if row["accuracy_wsrc"] > row["accuracy_rf"] else "RF"
        if winner == "WSRC":
            wsrc_wins += 1
        print(f"  {row['project']:<38} "
              f"{row['accuracy_wsrc']:>9.4f} {row['accuracy_rf']:>7.4f} "
              f"{row['f1_wsrc']:>8.4f} {row['f1_rf']:>6.4f} "
              f"{winner:>8}")

    print(f"\n  MEAN WSRC acc: {merged['accuracy_wsrc'].mean():.4f}  |  "
          f"MEAN RF acc: {merged['accuracy_rf'].mean():.4f}")
    print(f"  MEAN WSRC F1:  {merged['f1_wsrc'].mean():.4f}  |  "
          f"MEAN RF F1:  {merged['f1_rf'].mean():.4f}")
    print(f"\n  WSRC wins in {wsrc_wins}/{len(merged)} projects")


def main():
    data_path = os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    df = load_dataset(data_path)
    le = LabelEncoder()
    le.fit(df["conflictResolutionResult"])
    X, y, merge_ids, project_names = build_features(df)

    print(f"\nDataset: {len(df):,} samples | {X.shape[1]} features | "
          f"{project_names.nunique()} projects | {merge_ids.nunique():,} merges")
    print(f"Classes: {list(le.classes_)}")
    print(f"\nConfig: alpha={ALPHA}, weight={WEIGHT_METHOD}, "
          f"max_dict_per_class={MAX_DICT_PER_CLASS}, n_splits={N_SPLITS}")

    total_start = time.time()

    # Run both models
    df_wsrc = run_wsrc_per_project(X, y, merge_ids, project_names)
    df_rf = run_rf_per_project(X, y, merge_ids, project_names)

    print(f"\nTotal time: {(time.time()-total_start)/60:.1f} min")

    # Comparison
    if not df_wsrc.empty and not df_rf.empty:
        print_comparison(df_wsrc, df_rf)

    # Save results
    wsrc_path = os.path.join(results_dir, "wsrc_per_project.csv")
    rf_path = os.path.join(results_dir, "rf_per_project_phase2.csv")
    df_wsrc.to_csv(wsrc_path, index=False)
    df_rf.to_csv(rf_path,     index=False)
    print(f"\n  Results saved:")
    print(f"    WSRC → {wsrc_path}")
    print(f"    RF   → {rf_path}")


if __name__ == "__main__":
    main()