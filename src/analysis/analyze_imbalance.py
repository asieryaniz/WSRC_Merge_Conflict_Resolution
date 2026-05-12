# analyze_imbalance.py
"""
Sensitivity to class imbalance: does imbalance hurt WSRC more than RF?

Experiments:
  1. Correlation between ZeroR (imbalance proxy) and accuracy gap RF−WSRC
     across the 16 projects — does higher imbalance = bigger WSRC disadvantage?

  2. Balanced subsampling: re-run RF and WSRC on each project with a
     training set balanced to equal class sizes. Does balancing help WSRC
     more than RF?

  3. class_weight="balanced" in RF: does correcting for imbalance in RF
     change the ranking vs WSRC?

  4. WSRC with uniform dictionary (equal samples per class) vs class-weighted:
     which weight strategy works better at different imbalance levels?

Reads:  results/final_comparison.csv  (if available) OR re-runs S3 evaluation
        data/dataset_preprocessed.csv

Saves:  plots/imbalance/
        results/imbalance/

Usage:
    python src/analysis/analyze_imbalance.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.data.preprocess_dataset import load_dataset
from src.data.feature_builder import build_features
from src.models.random_forest import train_rf, predict_rf
from src.models.wsrc import wsrc_predict, compute_weights
from src.models.src import src_predict
from src.models.knn import train_knn, predict_knn
from src.metrics.evaluation import compute_all_metrics

PLOTS_DIR   = os.path.join(BASE_DIR, "plots",   "imbalance")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "imbalance")
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_ALPHA = 0.01
BEST_DICT_SIZE = 500
N_SPLITS = 5
RANDOM_STATE = 42

COLORS = {"RF": "#2E86AB", "RF_balanced": "#7BC8E2",
          "WSRC_class": "#E84855", "WSRC_uniform": "#F4A261",
          "KNN": "#3BB273"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150, "savefig.dpi": 150,
})


# Helpers
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


def balance_training_set(X_tr, y_tr, seed=42):
    """
    Undersample training set to equal class sizes.
    All classes reduced to the size of the smallest class.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y_tr)
    min_count = min((y_tr == c).sum() for c in classes)
    if min_count < 2:
        return X_tr, y_tr # too few samples to balance
    sel = []
    for c in classes:
        idx = np.where(y_tr == c)[0]
        sel.extend(rng.choice(idx, min_count, replace=False))
    sel = np.array(sel)
    return X_tr[sel], y_tr[sel]


def run_wsrc_fold(X_dict, y_dict, X_te, alpha, weight_method):
    y_pred = []
    for x in X_te:
        w = compute_weights(X_dict, y_dict, x, method=weight_method, top_k=1)
        p = wsrc_predict(X_dict, y_dict, x.reshape(1,-1), weights=w, alpha=alpha)
        y_pred.append(p[0])
    return np.array(y_pred)


def evaluate_project(X_proj, y_proj, mid_proj, balanced=False, csv_only=False):
    """
    Run models on one project. Returns mean accuracy per model across folds.

    If csv_only=True, skips RF and WSRC_class (values come from CSV instead),
    and only computes RF_balanced and WSRC_uniform — saving ~half the time.
    """
    fold_labels = build_merge_level_folds(mid_proj, N_SPLITS, RANDOM_STATE)
    results = {"RF": [], "RF_balanced": [],
               "WSRC_class": [], "WSRC_uniform": []}

    for fold in range(N_SPLITS):
        tr = np.where(fold_labels != fold)[0]
        te = np.where(fold_labels == fold)[0]

        X_tr_raw = X_proj.iloc[tr].values
        X_te_raw = X_proj.iloc[te].values
        y_tr, y_te = y_proj[tr], y_proj[te]

        if len(np.unique(y_tr)) < 2:
            continue

        X_tr_use, y_tr_use = (balance_training_set(X_tr_raw, y_tr)
                               if balanced else (X_tr_raw, y_tr))

        # RF standard — skip if values come from CSV
        if not csv_only:
            rf_pred = predict_rf(train_rf(pd.DataFrame(X_tr_use), y_tr_use),
                                 pd.DataFrame(X_te_raw))
            results["RF"].append(
                compute_all_metrics(y_te, rf_pred, y_tr)["accuracy"])

        # RF balanced class_weight — always computed (not in CSV)
        from sklearn.ensemble import RandomForestClassifier
        rf_bal = RandomForestClassifier(n_estimators=400, random_state=42,
                                        class_weight="balanced", n_jobs=-1)
        rf_bal.fit(X_tr_use, y_tr_use)
        results["RF_balanced"].append(
            compute_all_metrics(y_te, rf_bal.predict(X_te_raw), y_tr)["accuracy"])

        # Normalize for WSRC
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr_use)
        X_te = sc.transform(X_te_raw)

        # WSRC class — skip if values come from CSV
        if not csv_only:
            X_dict_c, y_dict_c = subsample_dict(X_tr, y_tr_use, BEST_DICT_SIZE)
            wsrc_c = run_wsrc_fold(X_dict_c, y_dict_c, X_te, BEST_ALPHA, "class")
            results["WSRC_class"].append(
                compute_all_metrics(y_te, wsrc_c, y_tr)["accuracy"])

        # WSRC uniform — always computed (not in CSV)
        X_dict_u, y_dict_u = subsample_dict(X_tr, y_tr_use, BEST_DICT_SIZE)
        wsrc_u = run_wsrc_fold(X_dict_u, y_dict_u, X_te, BEST_ALPHA, "uniform")
        results["WSRC_uniform"].append(
            compute_all_metrics(y_te, wsrc_u, y_tr)["accuracy"])

    return {m: np.mean(v) if v else np.nan for m, v in results.items()}


# Plot 1: Imbalance level vs accuracy gap (RF − WSRC) per project
def plot_imbalance_vs_gap(df_results):
    """
    Scatter: x=ZeroR (imbalance proxy), y=RF−WSRC accuracy gap.
    Each point is one project. Tests the hypothesis: higher imbalance = bigger gap.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    x = df_results["zeror"]
    y = df_results["RF"] - df_results["WSRC_class"]

    ax.scatter(x, y, color=COLORS["WSRC_class"], s=80,
               edgecolors="white", linewidths=0.8, zorder=5)

    for _, row in df_results.iterrows():
        ax.annotate(row["proj_short"], (row["zeror"], row["RF"] - row["WSRC_class"]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7.5, color="gray")

    # Trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, p(xline), color=COLORS["WSRC_class"],
            linestyle="--", linewidth=1.5, alpha=0.7)

    # Correlation
    corr = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.axhline(0, color="black", linewidth=1, linestyle="-")
    ax.set_xlabel("ZeroR baseline (class imbalance proxy)")
    ax.set_ylabel("Accuracy gap (RF − WSRC)")
    ax.set_title("Does Class Imbalance Explain the RF−WSRC Gap?\n"
                 "(each point = one project)",
                 fontsize=12, fontweight="bold")

    path = os.path.join(PLOTS_DIR, "1_imbalance_vs_accuracy_gap.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")
    print(f"  Pearson correlation (ZeroR vs RF−WSRC gap): r = {corr:.3f}")


# Plot 2: Standard vs balanced training — per project bar chart
def plot_balanced_vs_standard(df_std, df_bal):
    """
    Compare RF standard, RF balanced, WSRC_class, WSRC_uniform
    before and after training set balancing.
    """
    n = len(df_std)
    x = np.arange(n)
    width = 0.2
    labels = df_std["proj_short"].tolist()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for ax, df, title_suffix in zip(axes,
                                    [df_std, df_bal],
                                    ["Standard training set",
                                     "Balanced training set (undersampled)"]):
        for i, (model, color) in enumerate(
                [("RF", COLORS["RF"]),
                 ("RF_balanced", COLORS["RF_balanced"]),
                 ("WSRC_class", COLORS["WSRC_class"]),
                 ("WSRC_uniform", COLORS["WSRC_uniform"])]):
            offset = (i - 1.5) * width
            ax.bar(x + offset, df[model], width,
                   label=model.replace("_", " "), color=color,
                   alpha=0.85, edgecolor="white")

        ax.plot(x, df["zeror"], color="gray", marker="D",
                linewidth=1.5, markersize=5, linestyle="--",
                label="ZeroR", zorder=5)

        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(title_suffix, fontsize=11, fontweight="bold")
        ax.legend(frameon=False, ncol=5, fontsize=9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    fig.suptitle("Effect of Training Set Balancing on RF and WSRC\n"
                 "(S3 evaluation, 5-fold CV)",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.88, hspace=0.15)

    path = os.path.join(PLOTS_DIR, "2_balanced_vs_standard_training.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 3: Imbalance level groups — boxplot of accuracy by imbalance tier
def plot_accuracy_by_imbalance_tier(df_std):
    """
    Split projects into low/medium/high imbalance and compare model accuracy.
    Shows whether WSRC disadvantage is specific to highly imbalanced projects.
    """
    # Assign tiers based on ZeroR
    df = df_std.copy()
    df["tier"] = pd.cut(df["zeror"],
                        bins=[0, 0.55, 0.70, 1.0],
                        labels=["Low\n(ZeroR<0.55)",
                                "Medium\n(0.55-0.70)",
                                "High\n(ZeroR>0.70)"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    fig.suptitle("Model Accuracy by Class Imbalance Level\n"
                 "(projects grouped by ZeroR baseline)",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.85)

    plot_models = ["RF", "WSRC_class", "WSRC_uniform"]
    plot_colors = [COLORS["RF"], COLORS["WSRC_class"], COLORS["WSRC_uniform"]]

    for ax, tier in zip(axes, df["tier"].cat.categories):
        sub = df[df["tier"] == tier]
        if sub.empty:
            ax.set_title(f"{tier}\n(no projects)", fontsize=10)
            continue

        data = [sub[m].dropna().values for m in plot_models]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})

        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Add individual project points
        for i, m in enumerate(plot_models):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(sub))
            ax.scatter(np.ones(len(sub)) * (i+1) + jitter, sub[m],
                       color="black", s=25, alpha=0.6, zorder=5)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["RF", "WSRC\nclass", "WSRC\nuniform"],
                           fontsize=9)
        ax.set_title(f"{tier}\n({len(sub)} projects)", fontsize=10,
                     fontweight="bold")
        ax.set_ylabel("Accuracy" if ax == axes[0] else "")

        # ZeroR mean for this tier
        ax.axhline(sub["zeror"].mean(), color="gray", linestyle=":",
                   linewidth=1.5, label=f"ZeroR ({sub['zeror'].mean():.2f})")
        ax.legend(frameon=False, fontsize=8)

    path = os.path.join(PLOTS_DIR, "3_accuracy_by_imbalance_tier.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Main
def main():
    print(f"\nClass Imbalance Sensitivity Analysis")
    print(f"Config: alpha={BEST_ALPHA}, dict/cls={BEST_DICT_SIZE}\n")

    CSV_PATH = os.path.join(BASE_DIR, "results", "final_comparison.csv")
    csv_available = os.path.exists(CSV_PATH)

    if csv_available:
        print(f"  Found final_comparison.csv — loading RF, SRC, WSRC, KNN results.")
        print(f"  Only RF_balanced and WSRC_uniform will be re-computed.\n")
        df_csv = pd.read_csv(CSV_PATH)
        df_csv["proj_short"] = df_csv["project"].apply(lambda x: x.split("/")[-1])
    else:
        print(f"  No final_comparison.csv found — all models will be computed.\n")
        df_csv = None

    df_data = load_dataset(
        os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv"))
    le = LabelEncoder()
    le.fit(df_data["conflictResolutionResult"])
    X, y, merge_ids, project_names = build_features(df_data)

    # Run experiments
    print("Running experiments (RF_balanced + WSRC_uniform, balanced training)...")
    print(f"  {'Project':<38} {'ZeroR':>6} "
          f"{'RF':>6} {'RF_bal':>7} {'WSRC_c':>7} {'WSRC_u':>7} {'KNN':>6}")
    print(f"  {'─'*38} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*6}")

    std_rows, bal_rows = [], []

    for proj in sorted(project_names.unique()):
        mask = (project_names == proj).values
        X_proj = X[mask]
        y_proj = y[mask]
        mid_proj = merge_ids[mask]
        n_merges = mid_proj.nunique()

        if n_merges < N_SPLITS:
            continue

        zeror = np.bincount(y_proj).max() / len(y_proj)
        proj_short = proj.split("/")[-1]

        # Load RF/WSRC_class/SRC/KNN from CSV if available
        if csv_available:
            row_csv = df_csv[df_csv["project"] == proj]
            if len(row_csv) == 0:
                continue
            rf_std = float(row_csv["RF_accuracy"].iloc[0])
            wsrc_std = float(row_csv["WSRC_accuracy"].iloc[0])
            src_std = float(row_csv["SRC_accuracy"].iloc[0])
            knn_std = float(row_csv["KNN_accuracy"].iloc[0])
        else:
            full = evaluate_project(X_proj, y_proj, mid_proj, balanced=False)
            rf_std = full["RF"]
            wsrc_std = full["WSRC_class"]
            src_std = full.get("SRC", np.nan)
            knn_std = np.nan

        # Always compute RF_balanced and WSRC_uniform (not in CSV)
        # csv_only=True skips RF and WSRC_class computation (values come from CSV)
        extra_std = evaluate_project(X_proj, y_proj, mid_proj,
                                     balanced=False, csv_only=csv_available)
        extra_bal = evaluate_project(X_proj, y_proj, mid_proj,
                                     balanced=True,  csv_only=False)

        print(f"  {proj:<38} {zeror:>6.3f} "
              f"{rf_std:>6.4f} {extra_std['RF_balanced']:>7.4f} "
              f"{wsrc_std:>7.4f} {extra_std['WSRC_uniform']:>7.4f} "
              f"{knn_std:>6.4f}")

        std_rows.append({
            "project": proj,
            "proj_short": proj_short,
            "zeror": zeror,
            "RF": rf_std,
            "RF_balanced": extra_std["RF_balanced"],
            "WSRC_class": wsrc_std,
            "WSRC_uniform": extra_std["WSRC_uniform"],
            "SRC": src_std,
            "KNN": knn_std,
        })
        bal_rows.append({
            "project": proj,
            "proj_short": proj_short,
            "zeror": zeror,
            "RF": extra_bal["RF"],
            "RF_balanced": extra_bal["RF_balanced"],
            "WSRC_class": extra_bal["WSRC_class"],
            "WSRC_uniform": extra_bal["WSRC_uniform"],
            "SRC": np.nan,
            "KNN": np.nan,
        })

    df_std = pd.DataFrame(std_rows)
    df_bal = pd.DataFrame(bal_rows)

    # Summary
    print(f"\n  Mean accuracy (standard training):")
    for m in ["RF", "RF_balanced", "WSRC_class", "WSRC_uniform", "KNN"]:
        if m in df_std and df_std[m].notna().any():
            print(f"    {m:<14}: {df_std[m].mean():.4f}")

    print(f"\n  Mean accuracy (balanced training):")
    for m in ["RF", "RF_balanced", "WSRC_class", "WSRC_uniform"]:
        print(f"    {m:<14}: {df_bal[m].mean():.4f}")

    print(f"\n  Effect of balancing on WSRC_class (mean delta per project):")
    delta_wsrc = df_bal["WSRC_class"] - df_std["WSRC_class"]
    delta_rf   = df_bal["RF"]         - df_std["RF"]
    print(f"    WSRC_class: {delta_wsrc.mean():+.4f}  "
          f"(RF: {delta_rf.mean():+.4f}) — "
          f"{'balancing helps WSRC more' if delta_wsrc.mean() > delta_rf.mean() else 'balancing helps RF more or equally'}")

    # Plots
    print(f"\n[1/3] Imbalance level vs accuracy gap...")
    plot_imbalance_vs_gap(df_std)

    print(f"\n[2/3] Balanced vs standard training...")
    plot_balanced_vs_standard(df_std, df_bal)

    print(f"\n[3/3] Accuracy by imbalance tier...")
    plot_accuracy_by_imbalance_tier(df_std)

    # Save tables
    df_std.to_csv(os.path.join(RESULTS_DIR, "imbalance_std_results.csv"), index=False)
    df_bal.to_csv(os.path.join(RESULTS_DIR, "imbalance_bal_results.csv"), index=False)

    print(f"\n  Plots  → {PLOTS_DIR}")
    print(f"  Tables → {RESULTS_DIR}")


if __name__ == "__main__":
    main()