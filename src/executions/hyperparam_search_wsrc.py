# hyperparam_search_wsrc.py
"""
Hyperparameter search for WSRC over 3 representative projects:
  - Small:  getrailo/railo (~1k chunks,  156 merges)
  - Medium: apache/accumulo (~15k chunks, 1788 merges)
  - Large:  zkoss/zk (~7k chunks,  675 merges)

We skip aosp-mirror to keep total runtime under ~2 hours.

Grid:
  alpha : [0.001, 0.01, 0.05, 0.1]
  max_dict_per_class : [100, 200, 500]
  weight_method : ["similarity", "class", "uniform"]

Results are saved to results/wsrc_hyperparam_search.csv
"""

import os
import sys
import time
import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.data.preprocess_dataset import load_dataset
from src.data.feature_builder import build_features
from src.models.wsrc import wsrc_predict, compute_weights
from src.metrics.evaluation import compute_all_metrics

# Config
N_SPLITS = 5
RANDOM_STATE = 42

# Projects used for search (small / medium / large — no aosp)
SEARCH_PROJECTS = [
    "getrailo/railo", # small  ~1k 
    "apache/accumulo", # medium ~15k 
    "zkoss/zk", # large  ~7k 
]

# Hyperparameter grid
ALPHAS = [0.001, 0.01, 0.05, 0.1]
DICT_SIZES = [100, 200, 500, 1000, 2000] # max samples per class  
WEIGHT_METHODS = ["similarity", "class", "uniform"] 


# Helpers (same as main_wsrc.py)
def build_merge_level_folds(merge_ids, n_splits, random_state=42):
    unique_merges = merge_ids.unique()
    rng = np.random.default_rng(seed=random_state)
    shuffled_merges = rng.permutation(unique_merges)
    assignment = {mid: i % n_splits for i, mid in enumerate(shuffled_merges)}
    return merge_ids.map(assignment).values


def subsample_dictionary(X_train, y_train, max_per_class, random_state=42):
    rng = np.random.default_rng(seed=random_state)
    selected = []
    for c in np.unique(y_train):
        idx = np.where(y_train == c)[0]
        k = min(max_per_class, len(idx))
        selected.extend(rng.choice(idx, k, replace=False))
    selected = np.array(selected)
    return X_train[selected], y_train[selected]


def evaluate_wsrc_project(X_proj, y_proj, mid_proj, alpha, max_per_class, weight_method):
    """Run 5-fold S3 WSRC on a single project with given hyperparams."""
    fold_labels = build_merge_level_folds(mid_proj, N_SPLITS, RANDOM_STATE)
    fold_metrics = []

    for fold in range(N_SPLITS):
        train_idx = np.where(fold_labels != fold)[0]
        test_idx = np.where(fold_labels == fold)[0]

        X_tr_raw = X_proj.iloc[train_idx].values
        X_te_raw = X_proj.iloc[test_idx].values
        y_tr = y_proj[train_idx]
        y_te = y_proj[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        X_dict, y_dict = subsample_dictionary(X_tr, y_tr, max_per_class, RANDOM_STATE)

        y_pred = []
        for x_te in X_te:
            weights = compute_weights(
                X_dict, y_dict, x_te,
                method=weight_method, top_k=1
            )
            pred = wsrc_predict(
                X_dict, y_dict,
                x_te.reshape(1, -1),
                weights=weights,
                alpha=alpha
            )
            y_pred.append(pred[0])

        fold_metrics.append(compute_all_metrics(y_te, np.array(y_pred), y_tr))

    return {
        "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "zeror": float(np.mean([m["zeror"] for m in fold_metrics])),
        "f1": float(np.mean([m["f1_weighted"] for m in fold_metrics])),
        "NI": float(np.mean([m["normalized_improvement"] for m in fold_metrics])),
    }


def main():
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "wsrc_hyperparam_search.csv")

    # Load data
    df = load_dataset(os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv"))
    X, y, merge_ids, project_names = build_features(df)

    # Total combinations
    grid = list(itertools.product(ALPHAS, DICT_SIZES, WEIGHT_METHODS))
    total = len(grid) * len(SEARCH_PROJECTS)
    print(f"\nHyperparameter search: {len(grid)} configs × {len(SEARCH_PROJECTS)} projects = {total} runs\n")

    all_rows = []
    run = 0

    for proj in SEARCH_PROJECTS:
        mask = (project_names == proj).values
        X_proj = X[mask]
        y_proj = y[mask]
        mid_proj = merge_ids[mask]
        n_chunks = mask.sum()
        n_merges = mid_proj.nunique()

        print(f"\n{'='*72}")
        print(f"Project: {proj}  ({n_chunks:,} chunks, {n_merges} merges)")
        print(f"{'='*72}")
        print(f"  {'alpha':>6}  {'dict/cls':>8}  {'weight':>12}  "
              f"{'Acc':>6}  {'ZeroR':>6}  {'F1':>6}  {'NI':>7}  {'Time':>6}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*12}  "
              f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}")

        best_acc = -1
        best_cfg = None

        for alpha, max_per_class, weight_method in grid:
            run += 1
            t0 = time.time()

            metrics = evaluate_wsrc_project(
                X_proj, y_proj, mid_proj,
                alpha, max_per_class, weight_method
            )

            elapsed = time.time() - t0
            acc = metrics["accuracy"]
            zeror = metrics["zeror"]
            f1 = metrics["f1"]
            ni = metrics["NI"]

            marker = " ◄ best" if acc > best_acc else ""
            if acc > best_acc:
                best_acc = acc
                best_cfg = (alpha, max_per_class, weight_method)

            print(f"  {alpha:>6}  {max_per_class:>8}  {weight_method:>12}  "
                  f"{acc:>6.4f}  {zeror:>6.4f}  {f1:>6.4f}  {ni:>7.4f}  "
                  f"{elapsed:>5.0f}s{marker}")

            row = {
                "project": proj,
                "alpha": alpha,
                "max_per_class": max_per_class,
                "weight_method": weight_method,
                "accuracy": round(acc, 4),
                "zeror": round(zeror, 4),
                "f1": round(f1, 4),
                "NI": round(ni, 4),
                "time_s": round(elapsed, 1),
            }
            all_rows.append(row)

            # Save incrementally (safe if interrupted)
            pd.DataFrame(all_rows).to_csv(output_path, index=False)

        print(f"\n  Best config for {proj}: "
              f"alpha={best_cfg[0]}, dict={best_cfg[1]}, weight={best_cfg[2]}  "
              f"→ acc={best_acc:.4f}")

    # Final summary: best config per project + overall best
    df_res = pd.DataFrame(all_rows)
    print(f"\n{'='*72}")
    print("SUMMARY — Best config per project")
    print(f"{'='*72}")
    for proj in SEARCH_PROJECTS:
        sub = df_res[df_res["project"] == proj]
        best = sub.loc[sub["accuracy"].idxmax()]
        print(f"  {proj}")
        print(f"    alpha={best['alpha']}, dict/cls={best['max_per_class']}, "
              f"weight={best['weight_method']}")
        print(f"    acc={best['accuracy']:.4f}, f1={best['f1']:.4f}, NI={best['NI']:.4f}")

    # Overall best (highest mean accuracy across projects)
    mean_acc = df_res.groupby(["alpha","max_per_class","weight_method"])["accuracy"].mean()
    best_overall = mean_acc.idxmax()
    print(f"\n  Best overall config (mean acc across {len(SEARCH_PROJECTS)} projects):")
    print(f"    alpha={best_overall[0]}, dict/cls={best_overall[1]}, "
          f"weight={best_overall[2]}  → mean acc={mean_acc.max():.4f}")

    print(f"\n  Full results saved → {output_path}")


if __name__ == "__main__":
    main()