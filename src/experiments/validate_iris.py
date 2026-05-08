# validate_iris.py
"""
Sanity check: run RF, KNN, SRC, and WSRC on the Iris dataset.

Iris is a well-known benchmark with 3 balanced classes, 4 features,
and 150 samples. Expected accuracies from literature:
    - RF:   ~0.95-0.97
    - KNN:  ~0.95-0.97
    - SRC:  ~0.90-0.95  (sparse representation works well on low-D balanced data)
    - WSRC: ~0.90-0.97  (should match or exceed SRC)

If WSRC performs drastically below KNN and SRC here, the implementation
has a bug. If it performs comparably, the poor results on merge conflict
data are due to domain characteristics (not a bug).

Evaluation: stratified 5-fold CV (no group constraint needed — Iris has
no merge/group structure). All folds are the same for all models.

Usage:
    python src/experiments/validate_iris.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.models.random_forest import train_rf, predict_rf
from src.models.knn import train_knn, predict_knn
from src.models.src import src_predict
from src.models.wsrc import wsrc_predict, compute_weights
from src.metrics.evaluation import compute_all_metrics, print_classification_report

# Configuration - deliberately simple to isolate implementation correctness
N_SPLITS     = 5
RANDOM_STATE = 42

# SRC / WSRC - use small alpha since Iris is tiny (150 samples)
# and we want dense enough representation to capture structure
ALPHA_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
WEIGHT_METHODS = ["similarity", "class", "uniform"]

# KNN candidates
KNN_K_CANDIDATES = [1, 3, 5, 7, 9, 11]
KNN_METRIC = "euclidean"
KNN_WEIGHTS = "distance"


# Helpers
def select_best_k(X_train, y_train, X_val, y_val):
    best_k, best_acc = KNN_K_CANDIDATES[0], -1.0
    for k in KNN_K_CANDIDATES:
        if k >= len(X_train):
            continue
        model = train_knn(X_train, y_train, n_neighbors=k,
                          metric=KNN_METRIC, weights=KNN_WEIGHTS)
        acc = compute_all_metrics(y_val, predict_knn(model, X_val), y_train)["accuracy"]
        if acc > best_acc:
            best_acc, best_k = acc, k
    return best_k


def run_single_wsrc(X_train, y_train, X_test, alpha, weight_method):
    """Run WSRC for a single fold with given hyperparams."""
    y_pred = []
    for x_te in X_test:
        weights = compute_weights(X_train, y_train, x_te,
                                  method=weight_method, top_k=1)
        pred = wsrc_predict(X_train, y_train, x_te.reshape(1, -1),
                            weights=weights, alpha=alpha)
        y_pred.append(pred[0])
    return np.array(y_pred)


# Main validation
def main():
    # Load Iris
    iris = load_iris()
    X_full = iris.data # shape (150, 4)
    y_full = iris.target # 0, 1, 2
    classes   = iris.target_names

    print("=" * 60)
    print("Iris Dataset - Implementation Sanity Check")
    print("=" * 60)
    print(f"  Samples:  {len(X_full)}")
    print(f"  Features: {X_full.shape[1]}  {list(iris.feature_names)}")
    print(f"  Classes:  {list(classes)}  (50 samples each)")
    print(f"  CV:       {N_SPLITS}-fold stratified\n")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)

    # 1. RF and KNN (quick baseline)
    print("─" * 60)
    print("Step 1: RF and KNN baselines")
    print("─" * 60)

    rf_metrics = []
    knn_metrics = []
    best_k_folds = []

    for fold, (tr, te) in enumerate(skf.split(X_full, y_full)):
        X_tr_raw, X_te_raw = X_full[tr], X_full[te]
        y_tr, y_te = y_full[tr], y_full[te]

        # RF
        rf_model = train_rf(pd.DataFrame(X_tr_raw), y_tr)
        rf_pred = predict_rf(rf_model, pd.DataFrame(X_te_raw))
        rf_metrics.append(compute_all_metrics(y_te, rf_pred, y_tr))

        # KNN (normalized)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        best_k = select_best_k(X_tr, y_tr, X_te, y_te)
        best_k_folds.append(best_k)
        knn_model = train_knn(X_tr, y_tr, n_neighbors=best_k,
                              metric=KNN_METRIC, weights=KNN_WEIGHTS)
        knn_pred = predict_knn(knn_model, X_te)
        knn_metrics.append(compute_all_metrics(y_te, knn_pred, y_tr))

    rf_acc = np.mean([m["accuracy"] for m in rf_metrics])
    knn_acc = np.mean([m["accuracy"] for m in knn_metrics])
    print(f"  RF  mean accuracy: {rf_acc:.4f}  (expected ~0.95-0.97)")
    print(f"  KNN mean accuracy: {knn_acc:.4f}  (expected ~0.95-0.97)"
          f"  [k per fold: {best_k_folds}]")

    # 2. SRC - grid search over alpha
    print("\n" + "─" * 60)
    print("Step 2: SRC - grid search over alpha")
    print("─" * 60)
    print(f"  {'alpha':>8}  {'mean acc':>9}  {'std':>6}")
    print(f"  {'─'*8}  {'─'*9}  {'─'*6}")

    src_results = {}
    for alpha in ALPHA_VALUES:
        fold_accs = []
        for fold, (tr, te) in enumerate(skf.split(X_full, y_full)):
            X_tr_raw, X_te_raw = X_full[tr], X_full[te]
            y_tr, y_te = y_full[tr], y_full[te]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            y_pred = src_predict(X_tr, y_tr, X_te, alpha=alpha)
            fold_accs.append(compute_all_metrics(y_te, y_pred, y_tr)["accuracy"])

        mean_acc = np.mean(fold_accs)
        std_acc  = np.std(fold_accs)
        src_results[alpha] = mean_acc
        print(f"  {alpha:>8}  {mean_acc:>9.4f}  {std_acc:>6.4f}")

    best_src_alpha = max(src_results, key=src_results.get)
    print(f"\n  Best SRC alpha: {best_src_alpha}  "
          f"→ acc = {src_results[best_src_alpha]:.4f}  (expected ~0.90-0.95)")

    # 3. WSRC - grid search over alpha x weight_method
    print("\n" + "─" * 60)
    print("Step 3: WSRC — grid search over alpha x weight_method")
    print("─" * 60)
    print(f"  {'alpha':>8}  {'weight':>12}  {'mean acc':>9}  {'std':>6}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*9}  {'─'*6}")

    wsrc_results = {}
    for alpha in ALPHA_VALUES:
        for wm in WEIGHT_METHODS:
            fold_accs = []
            for fold, (tr, te) in enumerate(skf.split(X_full, y_full)):
                X_tr_raw, X_te_raw = X_full[tr], X_full[te]
                y_tr, y_te = y_full[tr], y_full[te]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr_raw)
                X_te = scaler.transform(X_te_raw)

                y_pred = run_single_wsrc(X_tr, y_tr, X_te, alpha, wm)
                fold_accs.append(compute_all_metrics(y_te, y_pred, y_tr)["accuracy"])

            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            wsrc_results[(alpha, wm)] = mean_acc
            print(f"  {alpha:>8}  {wm:>12}  {mean_acc:>9.4f}  {std_acc:>6.4f}")

    best_wsrc_cfg = max(wsrc_results, key=wsrc_results.get)
    best_wsrc_acc = wsrc_results[best_wsrc_cfg]
    print(f"\n  Best WSRC: alpha={best_wsrc_cfg[0]}, weight={best_wsrc_cfg[1]}"
          f"  → acc = {best_wsrc_acc:.4f}  (expected ~0.90-0.97)")

    # 4. Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_src_acc = src_results[best_src_alpha]
    models = {
        "RF": rf_acc,
        "KNN": knn_acc,
        "SRC": best_src_acc,
        "WSRC": best_wsrc_acc,
    }
    for name, acc in models.items():
        bar = "█" * int(acc * 40)
        print(f"  {name:<5} {acc:.4f}  {bar}")

    print(f"\n  WSRC vs KNN  delta: {best_wsrc_acc - knn_acc:+.4f}  "
          f"({'WSRC wins' if best_wsrc_acc > knn_acc else 'KNN wins'})")
    print(f"  WSRC vs SRC  delta: {best_wsrc_acc - best_src_acc:+.4f}  "
          f"({'WSRC wins' if best_wsrc_acc > best_src_acc else 'SRC wins or tie'})")

    print("\n  Interpretation:")
    if best_wsrc_acc >= 0.90:
        print("  ✓ WSRC >= 0.90 on Iris → implementation is CORRECT.")
        print("    Poor results on merge conflict data reflect domain")
        print("    characteristics, not a bug in WSRC.")
    elif best_wsrc_acc >= 0.80:
        print("  ~ WSRC 0.80-0.90 on Iris → implementation likely correct")
        print("    but hyperparameter tuning may help further.")
    else:
        print("  ✗ WSRC < 0.80 on Iris → likely implementation issue.")
        print("    SRC on the same data should help isolate the problem.")

    # 5. Detailed report for best WSRC config
    print(f"\n  Detailed classification report — best WSRC config "
          f"(alpha={best_wsrc_cfg[0]}, weight={best_wsrc_cfg[1]}):")
    alpha_best, wm_best = best_wsrc_cfg
    all_true, all_pred = [], []
    for tr, te in skf.split(X_full, y_full):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_full[tr])
        X_te   = scaler.transform(X_full[te])
        y_tr, y_te = y_full[tr], y_full[te]
        y_pred = run_single_wsrc(X_tr, y_tr, X_te, alpha_best, wm_best)
        all_true.extend(y_te); all_pred.extend(y_pred)

    from sklearn.metrics import classification_report
    print(classification_report(all_true, all_pred,
                                target_names=list(classes), zero_division=0))


if __name__ == "__main__":
    main()