# analyze_errors.py
"""
Error Analysis: per-class and per-project confusion matrices for all models.

Generates:
  - Confusion matrix for each model (aggregated across all projects)
  - Per-class F1, precision, recall for each model
  - Class-level error patterns: which classes does each model confuse most?
  - Per-project error breakdown: which projects have the worst per-class performance?
  - WSRC vs RF error overlap: do they make the same mistakes?

Reads:  results/final_comparison.csv  (produced by main_comparison.py)
        data/dataset_preprocessed.csv

Saves plots to: plots/error_analysis/
Saves tables to: results/error_analysis/

Usage:
    python src/analysis/analyze_errors.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.data.preprocess_dataset import load_dataset
from src.data.feature_builder import build_features
from src.models.random_forest import train_rf, predict_rf
from src.models.knn import train_knn, predict_knn
from src.models.src import src_predict
from src.models.wsrc import wsrc_predict, compute_weights
from src.metrics.evaluation import compute_all_metrics

PLOTS_DIR = os.path.join(BASE_DIR, "plots", "error_analysis")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "error_analysis")
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Same config as main_comparison.py
BEST_ALPHA = 0.01
BEST_DICT_SIZE = 500
BEST_WEIGHT_METHOD = "class"
KNN_K = 11 # typical best k - fixed here for speed
N_SPLITS = 5
RANDOM_STATE = 42

MODELS = ["RF", "KNN", "SRC", "WSRC"]
COLORS = {"RF": "#2E86AB", "KNN": "#3BB273", "SRC": "#F4A261", "WSRC": "#E84855"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 150,
})


# Helpers (same as main_comparison.py)
def build_merge_level_folds(merge_ids, n_splits, random_state=42):
    unique = merge_ids.unique()
    rng = np.random.default_rng(seed=random_state)
    shuffled = rng.permutation(unique)
    assign = {mid: i % n_splits for i, mid in enumerate(shuffled)}
    return merge_ids.map(assign).values


def subsample_dict(X_tr, y_tr, max_per_class, seed=42):
    rng = np.random.default_rng(seed)
    sel = []
    for c in np.unique(y_tr):
        idx = np.where(y_tr == c)[0]
        sel.extend(rng.choice(idx, min(max_per_class, len(idx)), replace=False))
    return X_tr[np.array(sel)], y_tr[np.array(sel)]


def collect_predictions(X_proj, y_proj, mid_proj):
    """
    Run all models on a project and collect (y_true, y_pred) pairs
    across all folds. Returns dict {model: (y_true_all, y_pred_all)}.
    """
    fold_labels = build_merge_level_folds(mid_proj, N_SPLITS, RANDOM_STATE)
    store = {m: {"true": [], "pred": []} for m in MODELS}

    for fold in range(N_SPLITS):
        tr = np.where(fold_labels != fold)[0]
        te = np.where(fold_labels == fold)[0]

        X_tr_raw, X_te_raw = X_proj.iloc[tr].values, X_proj.iloc[te].values
        y_tr, y_te = y_proj[tr], y_proj[te]

        # RF
        rf_pred = predict_rf(train_rf(X_proj.iloc[tr], y_tr),
                             X_proj.iloc[te])
        store["RF"]["true"].extend(y_te)
        store["RF"]["pred"].extend(rf_pred)

        # Normalize
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr_raw)
        X_te = sc.transform(X_te_raw)

        # KNN
        knn_pred = predict_knn(
            train_knn(X_tr, y_tr, n_neighbors=min(KNN_K, len(y_tr)-1),
                      metric="euclidean", weights="distance"), X_te)
        store["KNN"]["true"].extend(y_te)
        store["KNN"]["pred"].extend(knn_pred)

        # SRC / WSRC dictionary
        X_dict, y_dict = subsample_dict(X_tr, y_tr, BEST_DICT_SIZE)

        # SRC
        src_pred = src_predict(X_dict, y_dict, X_te, alpha=BEST_ALPHA)
        store["SRC"]["true"].extend(y_te)
        store["SRC"]["pred"].extend(src_pred)

        # WSRC
        wsrc_pred = []
        for x in X_te:
            w = compute_weights(X_dict, y_dict, x,
                                method=BEST_WEIGHT_METHOD, top_k=1)
            p = wsrc_predict(X_dict, y_dict, x.reshape(1, -1),
                             weights=w, alpha=BEST_ALPHA)
            wsrc_pred.append(p[0])
        store["WSRC"]["true"].extend(y_te)
        store["WSRC"]["pred"].extend(wsrc_pred)

    return {m: (np.array(store[m]["true"]), np.array(store[m]["pred"]))
            for m in MODELS}


# Plot 1: Aggregated confusion matrices (one per model)
def plot_confusion_matrices(all_true, all_pred, class_names):
    """4-panel figure: one normalised confusion matrix per model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    short_names = [c.replace("CHUNK_", "").replace("CANONICAL_", "")
                   .replace("SEMICANONICAL_", "SEMI_") for c in class_names]

    for ax, model in zip(axes, MODELS):
        cm = confusion_matrix(all_true[model], all_pred[model],
                              labels=list(range(len(class_names))))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        acc = (np.array(all_true[model]) == np.array(all_pred[model])).mean()
        ax.set_title(f"{model}  (acc={acc:.4f})", fontsize=12,
                     fontweight="bold", color=COLORS[model])

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_norm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if val > 0.5 else "black")

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Normalised Confusion Matrices — All Models\n"
                 "(aggregated across all 16 projects, S3 evaluation)",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.88, hspace=0.45, wspace=0.35)

    path = os.path.join(PLOTS_DIR, "1_confusion_matrices_all_models.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 2: Per-class F1 comparison across models
def plot_per_class_f1(all_true, all_pred, class_names):
    """Grouped bar chart: x=class, groups=models, y=F1."""
    from sklearn.metrics import f1_score

    short_names = [c.replace("CHUNK_CANONICAL_", "")
                    .replace("CHUNK_SEMICANONICAL_", "SEMI_")
                    .replace("CHUNK_", "") for c in class_names]
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, model in enumerate(MODELS):
        f1s = f1_score(all_true[model], all_pred[model],
                       average=None, labels=list(range(n_classes)),
                       zero_division=0)
        offset = (i - 1.5) * width
        ax.bar(x + offset, f1s, width, label=model,
               color=COLORS[model], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1-Score by Model\n"
                 "(aggregated across all 16 projects)",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, ncol=4)
    ax.axhline(0, color="black", linewidth=0.8)

    path = os.path.join(PLOTS_DIR, "2_per_class_f1_by_model.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 3: Error overlap — do RF and WSRC make the same mistakes?
def plot_error_overlap(all_true, all_pred, class_names):
    """
    For each test sample: categorise as
      - Both correct
      - Only RF correct
      - Only WSRC correct
      - Both wrong (same class)
      - Both wrong (different class)
    """
    y_true = np.array(all_true["RF"]) # same for all models
    rf_ok = (np.array(all_pred["RF"]) == y_true)
    ws_ok = (np.array(all_pred["WSRC"]) == y_true)

    both_correct = ( rf_ok &  ws_ok).sum()
    only_rf = ( rf_ok & ~ws_ok).sum()
    only_wsrc = (~rf_ok &  ws_ok).sum()
    both_wrong = (~rf_ok & ~ws_ok).sum()
    n = len(y_true)

    labels = [
        f"Both correct\n({both_correct/n:.1%})",
        f"Only RF correct\n({only_rf/n:.1%})",
        f"Only WSRC correct\n({only_wsrc/n:.1%})",
        f"Both wrong\n({both_wrong/n:.1%})",
    ]
    sizes = [both_correct, only_rf, only_wsrc, both_wrong]
    colors = ["#3BB273", COLORS["RF"], COLORS["WSRC"], "#AAAAAA"]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts = ax.pie(sizes, colors=colors, startangle=90,
                            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.legend(wedges, labels, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=10)
    ax.set_title("Error Overlap: RF vs. WSRC\n"
                 "(all predictions across all projects and folds)",
                 fontsize=12, fontweight="bold")

    path = os.path.join(PLOTS_DIR, "3_error_overlap_rf_vs_wsrc.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")

    print(f"\n  Error overlap summary (RF vs WSRC):")
    print(f"    Both correct:    {both_correct:>7,} ({both_correct/n:.1%})")
    print(f"    Only RF correct: {only_rf:>7,} ({only_rf/n:.1%})")
    print(f"    Only WSRC correct:{only_wsrc:>6,} ({only_wsrc/n:.1%})")
    print(f"    Both wrong:      {both_wrong:>7,} ({both_wrong/n:.1%})")


# Plot 4: Per-class recall heatmap across projects (RF only — main model)
def plot_per_project_class_recall(proj_results, class_names):
    """
    Heatmap: rows=projects, columns=classes, value=recall per class (RF).
    Highlights which classes and projects are hardest.
    """
    from sklearn.metrics import recall_score

    short_cls = [c.replace("CHUNK_CANONICAL_", "")
                   .replace("CHUNK_SEMICANONICAL_", "SEMI_")
                   .replace("CHUNK_", "") for c in class_names]
    short_proj = [p.split("/")[-1] for p in sorted(proj_results.keys())]
    n_proj = len(short_proj)
    n_cls = len(class_names)

    matrix = np.zeros((n_proj, n_cls))
    for i, proj in enumerate(sorted(proj_results.keys())):
        y_true, y_pred = proj_results[proj]["RF"]
        recalls = recall_score(y_true, y_pred,
                               labels=list(range(n_cls)),
                               average=None, zero_division=0)
        matrix[i] = recalls

    fig, ax = plt.subplots(figsize=(10, max(6, n_proj * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(short_cls, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_proj))
    ax.set_yticklabels(short_proj, fontsize=9)
    ax.set_title("Per-Class Recall per Project — Random Forest\n"
                 "(red = model fails on this class, green = good recall)",
                 fontsize=12, fontweight="bold")

    for i in range(n_proj):
        for j in range(n_cls):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if val < 0.4 or val > 0.75 else "black")

    plt.colorbar(im, ax=ax, shrink=0.6, label="Recall")
    fig.subplots_adjust(left=0.22, bottom=0.2)

    path = os.path.join(PLOTS_DIR, "4_per_project_class_recall_rf.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Save classification reports to CSV
def save_classification_reports(all_true, all_pred, class_names):
    rows = []
    for model in MODELS:
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f, s = precision_recall_fscore_support(
            all_true[model], all_pred[model],
            labels=list(range(len(class_names))),
            zero_division=0
        )
        for i, cls in enumerate(class_names):
            rows.append({"model": model, "class": cls,
                         "precision": round(p[i], 4),
                         "recall": round(r[i], 4),
                         "f1": round(f[i], 4),
                         "support": int(s[i])})
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "per_class_metrics_all_models.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: per_class_metrics_all_models.csv")
    return df


# Main
def main():
    print(f"\nError Analysis — collecting predictions across all projects")
    print(f"Config: alpha={BEST_ALPHA}, dict/cls={BEST_DICT_SIZE}, "
          f"weight={BEST_WEIGHT_METHOD}, KNN k={KNN_K}\n")

    df = load_dataset(os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv"))
    le = LabelEncoder()
    le.fit(df["conflictResolutionResult"])
    class_names = list(le.classes_)
    X, y, merge_ids, project_names = build_features(df)

    # Collect predictions per project
    all_true = {m: [] for m in MODELS}
    all_pred = {m: [] for m in MODELS}
    proj_results = {}

    projects = sorted(project_names.unique())
    for proj in projects:
        mask = (project_names == proj).values
        X_proj = X[mask]
        y_proj = y[mask]
        mid_proj = merge_ids[mask]
        n_merges = mid_proj.nunique()

        if n_merges < N_SPLITS:
            print(f"  Skipping {proj} ({n_merges} merges)")
            continue

        print(f"  Processing {proj} ({mask.sum():,} chunks)...", end=" ", flush=True)
        preds = collect_predictions(X_proj, y_proj, mid_proj)
        print("done")

        proj_results[proj] = {m: preds[m] for m in MODELS}
        for m in MODELS:
            all_true[m].extend(preds[m][0])
            all_pred[m].extend(preds[m][1])

    # Convert to arrays
    all_true = {m: np.array(all_true[m]) for m in MODELS}
    all_pred = {m: np.array(all_pred[m]) for m in MODELS}

    print(f"\n  Total predictions collected: {len(all_true['RF']):,}")
    print(f"  Classes: {class_names}")

    # Generate plots and tables
    print(f"\n[1/4] Confusion matrices...")
    plot_confusion_matrices(all_true, all_pred, class_names)

    print(f"\n[2/4] Per-class F1 by model...")
    plot_per_class_f1(all_true, all_pred, class_names)

    print(f"\n[3/4] Error overlap RF vs WSRC...")
    plot_error_overlap(all_true, all_pred, class_names)

    print(f"\n[4/4] Per-project class recall heatmap (RF)...")
    plot_per_project_class_recall(proj_results, class_names)

    print(f"\n[5/5] Saving classification reports...")
    df_report = save_classification_reports(all_true, all_pred, class_names)

    # Print summary table
    print(f"\n  Per-class F1 summary:")
    pivot = df_report.pivot(index="class", columns="model", values="f1")
    print(pivot[MODELS].to_string())

    print(f"\n  Plots → {PLOTS_DIR}")
    print(f"  Tables → {RESULTS_DIR}")


if __name__ == "__main__":
    main()