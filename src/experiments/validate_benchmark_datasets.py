# validate_benchmark_datasets.py
"""
Benchmark: RF vs. KNN vs. SRC vs. WSRC on Wine and Digits datasets.

Purpose: identify conditions under which WSRC outperforms RF and KNN,
to complement the merge-conflict results and characterise WSRC's strengths.

Datasets:
  - Wine:   178 samples, 13 features, 3 balanced classes. ZeroR=0.40.
            Small enough to use the full training set as WSRC dictionary.
  - Digits: 1797 samples, 64 features, 10 balanced classes. ZeroR=0.10.
            High-dimensional, closest to original SRC use case (images).
            Dictionary subsampled to 200/class for speed (~30s total).

Experiments per dataset:
  1. Full CV comparison (5-fold stratified) - headline numbers
  2. Sample-size sensitivity: accuracy vs. training set size (learning curve)
     Shows whether WSRC advantage appears at low data regimes.
  3. WSRC hyperparameter analysis: alpha x weight_method grid

Usage:
    python src/experiments/validate_benchmark_datasets.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_digits
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.models.random_forest import train_rf, predict_rf
from src.models.knn import train_knn, predict_knn
from src.models.src import src_predict
from src.models.wsrc import wsrc_predict, compute_weights
from src.metrics.evaluation import compute_all_metrics

# Configuration
N_SPLITS = 5
RANDOM_STATE = 42

# WSRC / SRC grid
ALPHA_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
WEIGHT_METHODS = ["similarity", "class", "uniform"]

# KNN grid
KNN_K_CANDIDATES = [1, 3, 5, 7, 11, 15]
KNN_METRIC = "euclidean"
KNN_WEIGHTS = "distance"

# Digits: subsample dictionary to bound computation
# Wine: small enough to use all training data (no subsampling needed)
DIGITS_MAX_PER_CLASS = 200 # ~30s for full 5CV

# Learning curve: fractions of training data to test
TRAIN_FRACTIONS = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]


# Helpers
def subsample_dict(X_train, y_train, max_per_class, seed=42):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y_train):
        ci = np.where(y_train == c)[0]
        idx.extend(rng.choice(ci, min(max_per_class, len(ci)), replace=False))
    return X_train[idx], y_train[idx]


def select_best_k(X_tr, y_tr, X_val, y_val):
    best_k, best_acc = KNN_K_CANDIDATES[0], -1.0
    for k in KNN_K_CANDIDATES:
        if k >= len(X_tr):
            continue
        m   = train_knn(X_tr, y_tr, n_neighbors=k,
                        metric=KNN_METRIC, weights=KNN_WEIGHTS)
        acc = (predict_knn(m, X_val) == y_val).mean()
        if acc > best_acc:
            best_acc, best_k = acc, k
    return best_k


def run_wsrc(X_dict, y_dict, X_te, alpha, weight_method):
    y_pred = []
    for x in X_te:
        w    = compute_weights(X_dict, y_dict, x,
                               method=weight_method, top_k=1)
        pred = wsrc_predict(X_dict, y_dict, x.reshape(1, -1),
                            weights=w, alpha=alpha)
        y_pred.append(pred[0])
    return np.array(y_pred)


def mean_acc(fold_list):
    return float(np.mean([m["accuracy"] for m in fold_list]))


# Experiment 1: full 5-fold CV comparison
def experiment_full_cv(X, y, dataset_name, max_dict_per_class=None):
    """Run all four models with best hyperparams, return summary dict."""
    print(f"\n  {'─'*52}")
    print(f"  Full 5-fold CV — {dataset_name}")
    print(f"  {'─'*52}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)

    rf_m, knn_m, src_m, wsrc_m = [], [], [], []
    best_k_list = []

    # --- find best alpha for SRC/WSRC via quick pre-search on fold 0 ---
    # For speed: fix weight=similarity (consistently best on balanced datasets
    # per Iris validation) and only search over alpha values
    tr0, te0 = next(iter(skf.split(X, y)))
    sc0 = StandardScaler()
    X0tr = sc0.fit_transform(X[tr0])
    X0te = sc0.transform(X[te0])
    dict0, ydict0 = (subsample_dict(X0tr, y[tr0], max_dict_per_class)
                     if max_dict_per_class else (X0tr, y[tr0]))

    best_alpha, best_wm, best_pre = ALPHA_VALUES[0], "similarity", -1.0
    for alpha in ALPHA_VALUES:
        yp = run_wsrc(dict0, ydict0, X0te, alpha, "similarity")
        acc = (yp == y[te0]).mean()
        if acc > best_pre:
            best_pre, best_alpha = acc, alpha
    # Keep weight=similarity fixed (validated on Iris and Wine)
    best_wm = "similarity"

    print(f"  Best WSRC config (pre-search on fold 0): "
          f"alpha={best_alpha}, weight={best_wm}")

    # --- full CV with best config ---
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr_raw, X_te_raw = X[tr], X[te]
        y_tr, y_te         = y[tr], y[te]

        # RF
        rf_pred = predict_rf(
            train_rf(pd.DataFrame(X_tr_raw), y_tr),
            pd.DataFrame(X_te_raw)
        )
        rf_m.append(compute_all_metrics(y_te, rf_pred, y_tr))

        # Normalize
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr_raw)
        X_te = sc.transform(X_te_raw)

        # KNN
        if fold == 0:
            best_k = select_best_k(X_tr, y_tr, X_te, y_te)
            best_k_list.append(best_k)
        knn_pred = predict_knn(
            train_knn(X_tr, y_tr, n_neighbors=best_k,
                      metric=KNN_METRIC, weights=KNN_WEIGHTS),
            X_te
        )
        knn_m.append(compute_all_metrics(y_te, knn_pred, y_tr))

        # SRC / WSRC dictionary
        X_dict, y_dict = (subsample_dict(X_tr, y_tr, max_dict_per_class)
                          if max_dict_per_class else (X_tr, y_tr))

        # SRC
        src_m.append(compute_all_metrics(
            y_te, src_predict(X_dict, y_dict, X_te, alpha=best_alpha), y_tr
        ))

        # WSRC
        wsrc_m.append(compute_all_metrics(
            y_te, run_wsrc(X_dict, y_dict, X_te, best_alpha, best_wm), y_tr
        ))

    results = {
        "RF": mean_acc(rf_m),
        "KNN": mean_acc(knn_m),
        "SRC": mean_acc(src_m),
        "WSRC": mean_acc(wsrc_m),
    }
    zeror = np.bincount(y).max() / len(y)

    print(f"  {'Model':<6}  {'Acc':>7}  {'vs ZeroR':>9}  {'Bar'}")
    print(f"  {'──────':<6}  {'───────':>7}  {'─────────':>9}")
    for m, acc in results.items():
        bar = "█" * int(acc * 30)
        print(f"  {m:<6}  {acc:>7.4f}  {acc-zeror:>+9.4f}  {bar}")
    print(f"  {'ZeroR':<6}  {zeror:>7.4f}")
    print(f"\n  KNN best k: {best_k}  |  "
          f"WSRC best: alpha={best_alpha}, weight={best_wm}")

    return results, best_alpha, best_wm, best_k

# Experiment 2: learning curve (accuracy vs. training set size)
def experiment_learning_curve(X, y, dataset_name,
                               best_alpha, best_wm, best_k,
                               max_dict_per_class=None):
    """
    Fix test set (20%), vary training size from 10% to 80%.
    Shows whether WSRC advantage appears at low-data regime.
    """
    print(f"\n  {'─'*52}")
    print(f"  Learning Curve — {dataset_name}")
    print(f"  (test set fixed at 20%, training size varies)")
    print(f"  {'─'*52}")
    print(f"  {'train%':>7}  {'n_tr':>5}  {'RF':>7}  {'KNN':>7}  "
          f"{'SRC':>7}  {'WSRC':>7}  {'WSRC best?':>10}")
    print(f"  {'───────':>7}  {'─────':>5}  {'───────':>7}  {'───────':>7}  "
          f"{'───────':>7}  {'───────':>7}  {'──────────':>10}")

    # Fixed test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20,
                                 random_state=RANDOM_STATE)
    tr_full, te_idx = next(sss.split(X, y))
    X_te_raw, y_te = X[te_idx], y[te_idx]

    rows = []
    for frac in TRAIN_FRACTIONS:
        n_tr = max(int(len(tr_full) * frac), len(np.unique(y)) * 2)

        # Stratified subsample of tr_full
        if frac < 1.0:
            sss2  = StratifiedShuffleSplit(n_splits=1,
                                           train_size=n_tr,
                                           random_state=RANDOM_STATE)
            tr_idx, _ = next(sss2.split(X[tr_full], y[tr_full]))
            tr_idx = tr_full[tr_idx]
        else:
            tr_idx = tr_full

        X_tr_raw, y_tr = X[tr_idx], y[tr_idx]

        # RF
        rf_acc = (predict_rf(
            train_rf(pd.DataFrame(X_tr_raw), y_tr),
            pd.DataFrame(X_te_raw)
        ) == y_te).mean()

        # Normalize
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr_raw)
        X_te = sc.transform(X_te_raw)

        # KNN
        k_use = min(best_k, len(y_tr) - 1)
        knn_acc = (predict_knn(
            train_knn(X_tr, y_tr, n_neighbors=k_use,
                      metric=KNN_METRIC, weights=KNN_WEIGHTS),
            X_te
        ) == y_te).mean()

        # SRC / WSRC
        X_dict, y_dict = (subsample_dict(X_tr, y_tr, max_dict_per_class)
                          if max_dict_per_class else (X_tr, y_tr))

        src_acc = (src_predict(X_dict, y_dict, X_te,
                                alpha=best_alpha) == y_te).mean()
        wsrc_acc = (run_wsrc(X_dict, y_dict, X_te,
                             best_alpha, best_wm) == y_te).mean()

        wsrc_best = "✓ WSRC" if wsrc_acc == max(rf_acc, knn_acc, src_acc, wsrc_acc) else ""
        print(f"  {frac*100:>6.0f}%  {n_tr:>5}  {rf_acc:>7.4f}  {knn_acc:>7.4f}  "
              f"{src_acc:>7.4f}  {wsrc_acc:>7.4f}  {wsrc_best:>10}")

        rows.append({"dataset": dataset_name, "train_frac": frac,
                     "n_train": n_tr, "RF": rf_acc, "KNN": knn_acc,
                     "SRC": src_acc, "WSRC": wsrc_acc})

    return pd.DataFrame(rows)


# Main
def main():
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    datasets = {
        "Wine": (load_wine(),   None), # full dict, ~0.1s
        "Digits": (load_digits(), DIGITS_MAX_PER_CLASS) # subsampled dict, ~30s
    }

    all_cv_rows   = []
    all_lc_frames = []

    for ds_name, (data, max_dict) in datasets.items():
        X, y = data.data, data.target
        print(f"\n{'='*58}")
        print(f"Dataset: {ds_name}")
        print(f"  {X.shape[0]} samples | {X.shape[1]} features | "
              f"{len(np.unique(y))} classes | "
              f"ZeroR={np.bincount(y).max()/len(y):.3f}")
        if max_dict:
            print(f"  WSRC/SRC dict: max {max_dict} samples/class")
        print(f"{'='*58}")

        # Exp 1: full CV
        t0 = time.time()
        cv_results, best_alpha, best_wm, best_k = experiment_full_cv(
            X, y, ds_name, max_dict
        )
        print(f"  [CV time: {time.time()-t0:.1f}s]")

        for m, acc in cv_results.items():
            all_cv_rows.append({"dataset": ds_name, "model": m, "accuracy": acc})

        # Exp 2: learning curve
        t0 = time.time()
        lc_df = experiment_learning_curve(
            X, y, ds_name, best_alpha, best_wm, best_k, max_dict
        )
        print(f"  [Learning curve time: {time.time()-t0:.1f}s]")
        all_lc_frames.append(lc_df)

    # Save results
    cv_df = pd.DataFrame(all_cv_rows)
    lc_df = pd.concat(all_lc_frames, ignore_index=True)

    cv_path = os.path.join(results_dir, "benchmark_cv_results.csv")
    lc_path = os.path.join(results_dir, "benchmark_learning_curve.csv")
    cv_df.to_csv(cv_path, index=False)
    lc_df.to_csv(lc_path, index=False)

    # Final summary
    print(f"\n{'='*58}")
    print("FINAL SUMMARY — Full CV Accuracy")
    print(f"{'='*58}")
    pivot = cv_df.pivot(index="model", columns="dataset", values="accuracy")
    print(pivot.to_string())

    print(f"\n  Key question: does WSRC outperform RF or KNN on any dataset?")
    for ds in ["Wine", "Digits"]:
        sub = cv_df[cv_df["dataset"] == ds].set_index("model")["accuracy"]
        winner = sub.idxmax()
        wsrc = sub.get("WSRC", 0)
        rf = sub.get("RF", 0)
        knn = sub.get("KNN", 0)
        print(f"  {ds:<8}: winner={winner}  "
              f"WSRC={wsrc:.4f}  RF={rf:.4f}  KNN={knn:.4f}  "
              f"WSRC>RF: {'yes' if wsrc>rf else 'no'}  "
              f"WSRC>KNN: {'yes' if wsrc>knn else 'no'}")

    print(f"\n  Results saved:")
    print(f"    CV results    → {cv_path}")
    print(f"    Learning curve → {lc_path}")


if __name__ == "__main__":
    main()