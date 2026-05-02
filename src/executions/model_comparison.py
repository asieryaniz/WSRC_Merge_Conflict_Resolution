# main_comparison.py
"""
Phase 2 — Final Comparison: SRC vs. WSRC vs. Random Forest

Evaluates all three models on all 16 projects with S3 (merge-level grouping,
no data leakage), using 5-fold cross-validation.

Configuration:
  - Standard projects (15):  alpha=0.01, dict=500/cls, weight=class
  - aosp-mirror (1 project): alpha=0.05, dict=200/cls, weight=class
    Rationale: aosp-mirror has 84,514 chunks (~16,903 test samples per fold).
    The standard config would take ~195 min for WSRC alone. With alpha=0.05
    and dict=200/cls the estimated time drops to ~18 min with an expected
    accuracy cost of ~7 points (based on the hyperparam search on accumulo,
    which has a similar class distribution). This is documented as a
    computational adaptation, not an exclusion — all 16 projects are included
    in the final comparison.

Run after hyperparam_search_wsrc.py:
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
from src.metrics.evaluation import compute_all_metrics

# Standard configuration (from hyperparam_search_wsrc.py results)
BEST_ALPHA = 0.01 # best overall across 3 search projects
BEST_DICT_SIZE = 500 # max samples per class in dictionary
BEST_WEIGHT_METHOD = "class" # class-frequency weighting wins on all 3 projects

# Adaptive configuration for aosp-mirror (computational constraint)
# dict=500/cls + alpha=0.01 → ~195 min  →  NOT viable
# dict=200/cls + alpha=0.05 → ~18  min  →  viable, estimated ~7pts accuracy cost
AOSP_ALPHA = 0.05
AOSP_DICT_SIZE = 200
AOSP_PROJECT = "aosp-mirror/platform_frameworks_base"

N_SPLITS = 5
RANDOM_STATE = 42


# Helpers
def build_merge_level_folds(merge_ids, n_splits, random_state=42):
    """
    Assign each merge commit to one of n_splits folds (S3 strategy).
    All chunks from the same merge go entirely into the same fold,
    preventing split-related data leakage.
    """
    unique_merges = merge_ids.unique()
    rng = np.random.default_rng(seed=random_state)
    shuffled_merges = rng.permutation(unique_merges)
    assignment = {mid: i % n_splits for i, mid in enumerate(shuffled_merges)}
    return merge_ids.map(assignment).values


def subsample_dictionary(X_train, y_train, max_per_class, random_state=42):
    """
    Stratified subsampling: take at most max_per_class examples per class.
    Preserves class balance while bounding the Lasso problem size for SRC/WSRC.
    For classes with fewer than max_per_class samples, all are used.
    """
    rng = np.random.default_rng(seed=random_state)
    selected = []
    for c in np.unique(y_train):
        idx = np.where(y_train == c)[0]
        k = min(max_per_class, len(idx))
        selected.extend(rng.choice(idx, k, replace=False))
    return X_train[np.array(selected)], y_train[np.array(selected)]


# Per-project evaluation
def run_project_all_models(X_proj, y_proj, mid_proj, alpha, dict_size, weight_method):
    """
    Run RF, SRC, and WSRC on a single project with N_SPLITS-fold S3 CV.

    SRC and WSRC share the exact same dictionary per fold — any difference
    in their results is due solely to the weighting mechanism.

    Args:
        X_proj        : feature matrix for this project (pd.DataFrame)
        y_proj        : encoded labels (np.ndarray)
        mid_proj      : merge IDs for this project (pd.Series)
        alpha         : Lasso regularization strength for SRC/WSRC
        dict_size     : max samples per class in the SRC/WSRC dictionary
        weight_method : weighting strategy for WSRC ("class", "similarity", "uniform")

    Returns:
        dict {model_name -> mean_metrics_across_folds}
    """
    fold_labels = build_merge_level_folds(mid_proj, N_SPLITS, RANDOM_STATE)
    results = {"RF": [], "SRC": [], "WSRC": []}

    for fold in range(N_SPLITS):
        train_idx = np.where(fold_labels != fold)[0]
        test_idx  = np.where(fold_labels == fold)[0]

        X_tr_raw = X_proj.iloc[train_idx].values
        X_te_raw = X_proj.iloc[test_idx].values
        y_tr = y_proj[train_idx]
        y_te = y_proj[test_idx]

        # --- Random Forest (no normalization needed) ---
        model_rf = train_rf(X_proj.iloc[train_idx], y_tr)
        y_pred_rf = predict_rf(model_rf, X_proj.iloc[test_idx])
        results["RF"].append(compute_all_metrics(y_te, y_pred_rf, y_tr))

        # --- Normalize features (required for SRC/WSRC distance calculations) ---
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        # Shared dictionary: same for SRC and WSRC → fair comparison
        X_dict, y_dict = subsample_dictionary(X_tr, y_tr, dict_size, RANDOM_STATE)

        # --- SRC (unweighted sparse representation) ---
        y_pred_src = src_predict(X_dict, y_dict, X_te, alpha=alpha)
        results["SRC"].append(compute_all_metrics(y_te, y_pred_src, y_tr))

        # --- WSRC (weighted sparse representation) ---
        y_pred_wsrc = []
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
            y_pred_wsrc.append(pred[0])
        results["WSRC"].append(compute_all_metrics(y_te, np.array(y_pred_wsrc), y_tr))

    def mean_metrics(fold_list):
        keys = fold_list[0].keys()
        return {k: round(float(np.mean([m[k] for m in fold_list])), 4) for k in keys}

    return {model: mean_metrics(folds) for model, folds in results.items()}



def main():
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    df = load_dataset(os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv"))
    le = LabelEncoder()
    le.fit(df["conflictResolutionResult"])
    X, y, merge_ids, project_names = build_features(df)

    print(f"\nFinal Comparison: SRC vs. WSRC vs. Random Forest")
    print(f"Evaluation: S3 — merge-level grouping, no data leakage, {N_SPLITS}-fold CV")
    print(f"\nStandard config (15 projects):  alpha={BEST_ALPHA}, "
          f"dict/cls={BEST_DICT_SIZE}, weight={BEST_WEIGHT_METHOD}")
    print(f"Adaptive config (aosp-mirror):  alpha={AOSP_ALPHA}, "
          f"dict/cls={AOSP_DICT_SIZE}, weight={BEST_WEIGHT_METHOD}")
    print(f"  Rationale: standard config ~195 min → "
          f"adaptive config ~18 min (~7pts accuracy cost)\n")

    header = (f"  {'Project':<38} {'RF Acc':>7} {'SRC Acc':>8} "
              f"{'WSRC Acc':>9} {'RF F1':>6} {'SRC F1':>7} {'WSRC F1':>8} "
              f"{'Best':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))

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
            print(f"  {proj:<38} — skipped ({n_merges} merges < {N_SPLITS})")
            continue

        # Select config: adaptive for aosp-mirror, standard for everything else
        is_aosp = (proj == AOSP_PROJECT)
        alpha = AOSP_ALPHA if is_aosp else BEST_ALPHA
        dsize = AOSP_DICT_SIZE if is_aosp else BEST_DICT_SIZE
        note = " [adaptive cfg]" if is_aosp else ""

        t0 = time.time()
        model_results = run_project_all_models(
            X_proj, y_proj, mid_proj,
            alpha=alpha, dict_size=dsize, weight_method=BEST_WEIGHT_METHOD
        )
        elapsed = time.time() - t0

        accs = {m: model_results[m]["accuracy"]    for m in ["RF", "SRC", "WSRC"]}
        f1s  = {m: model_results[m]["f1_weighted"] for m in ["RF", "SRC", "WSRC"]}
        best = max(accs, key=accs.get)

        print(f"  {proj:<38} {accs['RF']:>7.4f} {accs['SRC']:>8.4f} "
              f"{accs['WSRC']:>9.4f} {f1s['RF']:>6.4f} {f1s['SRC']:>7.4f} "
              f"{f1s['WSRC']:>8.4f} {best:>6}  [{elapsed:.0f}s]{note}")

        row = {
            "project": proj,
            "chunks": n_chunks,
            "merges": n_merges,
            "alpha_used": alpha,
            "dict_size": dsize,
            "adaptive_cfg": is_aosp,
            "RF_accuracy": accs["RF"],
            "SRC_accuracy": accs["SRC"],
            "WSRC_accuracy": accs["WSRC"],
            "RF_f1": f1s["RF"],
            "SRC_f1": f1s["SRC"],
            "WSRC_f1": f1s["WSRC"],
            "RF_zeror": model_results["RF"]["zeror"],
            "RF_NI": model_results["RF"]["normalized_improvement"],
            "SRC_NI": model_results["SRC"]["normalized_improvement"],
            "WSRC_NI": model_results["WSRC"]["normalized_improvement"],
            "best_model": best,
            "time_s": round(elapsed, 1),
        }
        all_rows.append(row)

        # Incremental save — safe if interrupted
        pd.DataFrame(all_rows).to_csv(
            os.path.join(results_dir, "final_comparison.csv"), index=False
        )

    df_res = pd.DataFrame(all_rows)
    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY — Mean across all 16 projects")
    print(f"{'='*72}")
    print(f"  {'Model':<6}  {'Mean Acc':>9}  {'Mean F1':>8}  {'Mean NI':>8}  {'Wins':>6}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*6}")

    for model in ["RF", "SRC", "WSRC"]:
        mean_acc = df_res[f"{model}_accuracy"].mean()
        mean_f1 = df_res[f"{model}_f1"].mean()
        mean_ni = df_res[f"{model}_NI"].mean()
        wins = (df_res["best_model"] == model).sum()
        print(f"  {model:<6}  {mean_acc:>9.4f}  {mean_f1:>8.4f}  "
              f"{mean_ni:>8.4f}  {wins:>4}/16")

    print(f"\n  ZeroR mean: {df_res['RF_zeror'].mean():.4f}")

    # Pairwise accuracy deltas
    rf_acc = df_res["RF_accuracy"].mean()
    src_acc = df_res["SRC_accuracy"].mean()
    wsrc_acc = df_res["WSRC_accuracy"].mean()
    print(f"\n  Pairwise accuracy deltas (mean over 16 projects):")
    print(f"    RF − SRC:   {rf_acc   - src_acc:+.4f}")
    print(f"    RF − WSRC:  {rf_acc   - wsrc_acc:+.4f}")
    print(f"    WSRC − SRC: {wsrc_acc - src_acc:+.4f}  "
          f"({'WSRC > SRC' if wsrc_acc > src_acc else 'SRC >= WSRC'})")

    print(f"\n  Win breakdown (16 projects):")
    for model in ["RF", "SRC", "WSRC"]:
        wins = (df_res["best_model"] == model).sum()
        proj_list = df_res[df_res["best_model"] == model]["project"].tolist()
        print(f"    {model}: {wins}/16  {proj_list}")

    print(f"\n  Note: aosp-mirror used adaptive config "
          f"(alpha={AOSP_ALPHA}, dict/cls={AOSP_DICT_SIZE}). "
          f"All other projects: alpha={BEST_ALPHA}, dict/cls={BEST_DICT_SIZE}.")
    print(f"\n  Total time: {total_time/60:.1f} min")

    out_path = os.path.join(results_dir, "final_comparison.csv")
    df_res.to_csv(out_path, index=False)
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()