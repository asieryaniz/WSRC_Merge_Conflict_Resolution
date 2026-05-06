# main_comparison.py
"""
Phase 2 — Final Comparison: RF vs. SRC vs. WSRC vs. KNN

Evaluates all four models on all 16 projects with S3 (merge-level grouping,
no data leakage), using 5-fold cross-validation.

Configuration:
  - Standard projects (15):  alpha=0.01, dict=500/cls, weight=class
  - aosp-mirror (1 project): alpha=0.05, dict=200/cls, weight=class
    dict=500+alpha=0.01 → ~195 min; dict=200+alpha=0.05 → ~18 min (~7pts cost)

Run:
    python src/experiments/main_comparison.py
"""

import os
import sys
import time
import warnings
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
from src.models.src import src_predict
from src.models.random_forest import train_rf, predict_rf
from src.models.knn import train_knn, predict_knn
from src.metrics.evaluation import compute_all_metrics

# SRC / WSRC configuration (from hyperparam_search_wsrc.py)
BEST_ALPHA = 0.01
BEST_DICT_SIZE = 500
BEST_WEIGHT_METHOD = "class"

# KNN configuration
# Per-project k is selected automatically via grid search on fold-0. These are the candidate values - search is fast (~seconds total).
KNN_K_CANDIDATES = [1, 3, 5, 7, 11, 15, 21]
KNN_METRIC = "euclidean" # consistent with WSRC distance weights
KNN_WEIGHTS = "distance" # inverse-distance votes, analogous to WSRC

N_SPLITS = 5
RANDOM_STATE = 42
MODELS = ["RF", "SRC", "WSRC", "KNN"]


# Helpers
def build_merge_level_folds(merge_ids, n_splits, random_state=42):
    """S3: assign every chunk of a merge commit to the same fold."""
    unique_merges = merge_ids.unique()
    rng = np.random.default_rng(seed=random_state)
    shuffled_merges = rng.permutation(unique_merges)
    assignment = {mid: i % n_splits for i, mid in enumerate(shuffled_merges)}
    return merge_ids.map(assignment).values


def subsample_dictionary(X_train, y_train, max_per_class, random_state=42):
    """Stratified subsample of training set for SRC/WSRC dictionary."""
    rng = np.random.default_rng(seed=random_state)
    selected = []
    for c in np.unique(y_train):
        idx = np.where(y_train == c)[0]
        k = min(max_per_class, len(idx))
        selected.extend(rng.choice(idx, k, replace=False))
    return X_train[np.array(selected)], y_train[np.array(selected)]


def select_best_k(X_train, y_train, X_val, y_val):
    """
    Grid search over KNN_K_CANDIDATES on a single validation split.
    Returns the k with highest accuracy. Takes <1s for any project size.
    """
    best_k, best_acc = KNN_K_CANDIDATES[0], -1.0
    for k in KNN_K_CANDIDATES:
        # Ensure k does not exceed training size
        if k >= len(X_train):
            continue
        model = train_knn(X_train, y_train,
                          n_neighbors=k,
                          metric=KNN_METRIC,
                          weights=KNN_WEIGHTS)
        acc = compute_all_metrics(y_val, predict_knn(model, X_val), y_train)["accuracy"]
        if acc > best_acc:
            best_acc, best_k = acc, k
    return best_k


# Per-project evaluation
def run_project_all_models(X_proj, y_proj, mid_proj):
    """
    Run RF, SRC, WSRC, and KNN on a single project with N_SPLITS-fold S3 CV.

    KNN k is selected once (on fold 0) and reused across all folds to keep
    the comparison fair and computationally light.

    SRC and WSRC share the exact same dictionary per fold.

    Returns:
        dict {model_name -> mean_metrics_across_folds}
        int  best_k found for KNN on this project
    """
    fold_labels = build_merge_level_folds(mid_proj, N_SPLITS, RANDOM_STATE)
    results = {m: [] for m in MODELS}
    best_k = KNN_K_CANDIDATES[0]

    for fold in range(N_SPLITS):
        train_idx = np.where(fold_labels != fold)[0]
        test_idx = np.where(fold_labels == fold)[0]

        X_tr_raw = X_proj.iloc[train_idx].values
        X_te_raw = X_proj.iloc[test_idx].values
        y_tr = y_proj[train_idx]
        y_te = y_proj[test_idx]

        # Random Forest (no normalization needed)
        model_rf  = train_rf(X_proj.iloc[train_idx], y_tr)
        y_pred_rf = predict_rf(model_rf, X_proj.iloc[test_idx])
        results["RF"].append(compute_all_metrics(y_te, y_pred_rf, y_tr))

        # Normalize for KNN / SRC / WSRC
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        # KNN: select k on fold 0, reuse for remaining folds
        if fold == 0:
            best_k = select_best_k(X_tr, y_tr, X_te, y_te)

        model_knn = train_knn(X_tr, y_tr,
                               n_neighbors=best_k,
                               metric=KNN_METRIC,
                               weights=KNN_WEIGHTS)
        y_pred_knn = predict_knn(model_knn, X_te)
        results["KNN"].append(compute_all_metrics(y_te, y_pred_knn, y_tr))

        # Shared dictionary for SRC and WSRC
        X_dict, y_dict = subsample_dictionary(X_tr, y_tr, BEST_DICT_SIZE, RANDOM_STATE)

        # SRC
        y_pred_src = src_predict(X_dict, y_dict, X_te, alpha=BEST_ALPHA)
        results["SRC"].append(compute_all_metrics(y_te, y_pred_src, y_tr))

        # WSRC
        y_pred_wsrc = []
        for x_te in X_te:
            weights = compute_weights(X_dict, y_dict, x_te,
                                      method=BEST_WEIGHT_METHOD, top_k=1)
            pred = wsrc_predict(X_dict, y_dict, x_te.reshape(1, -1),
                                weights=weights, alpha=BEST_ALPHA)
            y_pred_wsrc.append(pred[0])
        results["WSRC"].append(
            compute_all_metrics(y_te, np.array(y_pred_wsrc), y_tr)
        )

    def mean_metrics(fold_list):
        keys = fold_list[0].keys()
        return {k: round(float(np.mean([m[k] for m in fold_list])), 4)
                for k in keys}

    return {m: mean_metrics(results[m]) for m in MODELS}, best_k



# Main
def main():
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    df = load_dataset(os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv"))
    le = LabelEncoder()
    le.fit(df["conflictResolutionResult"])
    X, y, merge_ids, project_names = build_features(df)

    print(f"\nFinal Comparison: RF vs. SRC vs. WSRC vs. KNN")
    print(f"Evaluation: S3 — merge-level grouping, no data leakage, {N_SPLITS}-fold CV")
    print(f"Config (all 16 projects): alpha={BEST_ALPHA}, dict/cls={BEST_DICT_SIZE}, "
          f"weight={BEST_WEIGHT_METHOD}")
    print(f"KNN: metric={KNN_METRIC}, weights={KNN_WEIGHTS}, "
          f"k in {KNN_K_CANDIDATES} (selected per project on fold-0)\n")

    col_w = 38
    header = (f"  {'Project':<{col_w}} {'RF':>6} {'SRC':>6} "
              f"{'WSRC':>6} {'KNN':>6}  {'Best':>5}  {'k':>3}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    all_rows = []
    total_start = time.time()

    for proj in sorted(project_names.unique()):
        mask = (project_names == proj).values
        X_proj = X[mask]
        y_proj = y[mask]
        mid_proj = merge_ids[mask]
        n_chunks = mask.sum()
        n_merges = mid_proj.nunique()

        if n_merges < N_SPLITS:
            print(f"  {proj:<{col_w}} — skipped ({n_merges} merges < {N_SPLITS})")
            continue

        t0 = time.time()
        model_results, best_k = run_project_all_models(X_proj, y_proj, mid_proj)
        elapsed = time.time() - t0

        accs = {m: model_results[m]["accuracy"] for m in MODELS}
        f1s = {m: model_results[m]["f1_weighted"] for m in MODELS}
        best = max(accs, key=accs.get)

        print(f"  {proj:<{col_w}}"
              f" {accs['RF']:>6.4f} {accs['SRC']:>6.4f}"
              f" {accs['WSRC']:>6.4f} {accs['KNN']:>6.4f}"
              f"  {best:>5}  k={best_k}  [{elapsed:.0f}s]")

        row = {
            "project": proj,
            "chunks": n_chunks,
            "merges": n_merges,
            "alpha_used": BEST_ALPHA,
            "dict_size": BEST_DICT_SIZE,
            "knn_k": best_k,
            **{f"{m}_accuracy": accs[m] for m in MODELS},
            **{f"{m}_f1": f1s[m] for m in MODELS},
            **{f"{m}_NI": model_results[m]["normalized_improvement"] for m in MODELS},
            "RF_zeror": model_results["RF"]["zeror"],
            "best_model": best,
            "time_s": round(elapsed, 1),
        }
        all_rows.append(row)

        # Incremental save
        pd.DataFrame(all_rows).to_csv(
            os.path.join(results_dir, "final_comparison.csv"), index=False
        )

    df_res = pd.DataFrame(all_rows)
    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*68}")
    print("SUMMARY — Mean across all 16 projects")
    print(f"{'='*68}")
    print(f"  {'Model':<6}  {'Mean Acc':>9}  {'Mean F1':>8}  "
          f"{'Mean NI':>8}  {'Wins':>6}")
    print(f"  {'──────':<6}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*6}")

    for m in MODELS:
        mean_acc = df_res[f"{m}_accuracy"].mean()
        mean_f1 = df_res[f"{m}_f1"].mean()
        mean_ni = df_res[f"{m}_NI"].mean()
        wins = (df_res["best_model"] == m).sum()
        print(f"  {m:<6}  {mean_acc:>9.4f}  {mean_f1:>8.4f}  "
              f"{mean_ni:>8.4f}  {wins:>4}/16")

    print(f"\n  ZeroR mean: {df_res['RF_zeror'].mean():.4f}")

    # Pairwise deltas vs RF
    rf_acc = df_res["RF_accuracy"].mean()
    print(f"\n  Accuracy delta vs RF (mean over 16 projects):")
    for m in ["SRC", "WSRC", "KNN"]:
        delta = rf_acc - df_res[f"{m}_accuracy"].mean()
        print(f"    RF − {m:<4}: {delta:+.4f}")

    # Key comparison: WSRC vs KNN
    wsrc_mean = df_res["WSRC_accuracy"].mean()
    knn_mean = df_res["KNN_accuracy"].mean()
    wsrc_wins_vs_knn = (df_res["WSRC_accuracy"] > df_res["KNN_accuracy"]).sum()
    print(f"\n  WSRC vs KNN (key comparison):")
    print(f"    WSRC mean acc: {wsrc_mean:.4f}  |  KNN mean acc: {knn_mean:.4f}  "
          f"|  delta: {wsrc_mean - knn_mean:+.4f}")
    print(f"    WSRC > KNN in {wsrc_wins_vs_knn}/16 projects")

    # KNN k values selected
    print(f"\n  KNN best k per project:")
    for _, row in df_res.iterrows():
        print(f"    {row['project'].split('/')[-1]:<35} k={int(row['knn_k'])}")

    print(f"\n  Win breakdown:")
    for m in MODELS:
        wins = (df_res["best_model"] == m).sum()
        proj_list = df_res[df_res["best_model"] == m]["project"].str.split("/").str[-1].tolist()
        print(f"    {m}: {wins}/16  {proj_list}")

    print(f"\n  Total time: {total_time/60:.1f} min")

    out_path = os.path.join(results_dir, "final_comparison.csv")
    df_res.to_csv(out_path, index=False)
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()