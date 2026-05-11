# plot_error_analysis.py
"""
Generates plots from the per-class metrics CSV produced by analyze_errors.py.

Input:  results/error_analysis/per_class_metrics_all_models.csv
Output: plots/error_analysis/

Plots:
  1. Precision vs Recall scatter per class per model
     — shows the precision/recall tradeoff and which models sacrifice which
  2. F1 heatmap: rows=class, columns=model
     — overview of where each model succeeds and fails
  3. Per-class recall bar chart (focus on minority classes)
     — highlights that all models fail on SEMI_OTHERS but differently
  4. Precision-Recall gap per model
     — abs(precision - recall) shows which models are most imbalanced in errors
  5. Class support vs F1 scatter
     — confirms that all models degrade on rare classes

Usage:
    python src/analysis/plot_error_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, "results", "error_analysis",
                          "per_class_metrics_all_models.csv")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots", "error_analysis")
os.makedirs(PLOTS_DIR, exist_ok=True)

MODELS = ["RF", "KNN", "SRC", "WSRC"]
COLORS = {"RF": "#2E86AB", "KNN": "#3BB273",
          "SRC": "#F4A261", "WSRC": "#E84855"}
MARKERS = {"RF": "o", "KNN": "s", "SRC": "^", "WSRC": "D"}

# Short class names for readability in plots
SHORT_NAMES = {
    "CHUNK_CANONICAL_BASE": "BASE",
    "CHUNK_CANONICAL_OURS": "OURS",
    "CHUNK_CANONICAL_THEIRS": "THEIRS",
    "CHUNK_NONCANONICAL": "NON-CANON.",
    "CHUNK_SEMICANONICAL_OTHERS": "SEMI_OTHERS",
    "CHUNK_SEMICANONICAL_OURSTHEIRS": "OURSTHEIRS",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 150,
})


def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["class_short"] = df["class"].map(SHORT_NAMES)
    return df


# Plot 1: Precision vs Recall scatter — one subplot per model
def plot_precision_recall_scatter(df):
    """
    Each point = one class. x=recall, y=precision.
    Diagonal = perfect F1 trade-off. Points below diagonal = recall > precision.
    Annotated with class names.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()
    fig.suptitle("Precision vs. Recall per Class\n"
                 "(each point = one resolution class; diagonal = equal P/R)",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.3)

    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model].copy()
        sub = sub.sort_values("support", ascending=False)

        # Size proportional to log(support)
        sizes = np.log1p(sub["support"]) * 12

        sc = ax.scatter(sub["recall"], sub["precision"],
                        c=COLORS[model], s=sizes,
                        alpha=0.85, edgecolors="white", linewidths=0.8,
                        zorder=5)

        # Annotate each class
        for _, row in sub.iterrows():
            ax.annotate(row["class_short"],
                        (row["recall"], row["precision"]),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=7.5, color="gray")

        # Diagonal P=R
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="P = R")

        # Weighted F1 as annotation
        weighted_f1 = (sub["f1"] * sub["support"]).sum() / sub["support"].sum()
        ax.text(0.05, 0.95, f"Weighted F1 = {weighted_f1:.3f}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(model, fontsize=12, fontweight="bold",
                     color=COLORS[model])
        ax.legend(frameon=False, fontsize=8)

    path = os.path.join(PLOTS_DIR, "5_precision_recall_scatter_per_model.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 2: F1 heatmap — rows=class, columns=model
def plot_f1_heatmap(df):
    """
    Overview heatmap: which model × class combinations succeed or fail.
    """
    classes = list(SHORT_NAMES.keys())
    pivot = df.pivot(index="class", columns="model", values="f1")
    pivot = pivot.reindex(index=classes, columns=MODELS)
    pivot.index = [SHORT_NAMES[c] for c in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title("F1-Score Heatmap: Class × Model\n"
                 "(green = high F1, red = low F1)",
                 fontsize=12, fontweight="bold")

    for i in range(len(pivot.index)):
        for j in range(len(MODELS)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if val < 0.35 or val > 0.75 else "black",
                        fontweight="bold" if val == pivot.values[i].max() else "normal")

    plt.colorbar(im, ax=ax, shrink=0.8, label="F1-score")
    fig.subplots_adjust(left=0.2)

    path = os.path.join(PLOTS_DIR, "6_f1_heatmap_class_by_model.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")



# Plot 3: Per-class recall — focus on minority classes
def plot_minority_class_recall(df):
    """
    Grouped bar chart for all classes, ordered by support (rarest first).
    Highlights that minority classes are hard for all models.
    """
    # Order classes by support (ascending — rarest first)
    support_order = (df[df["model"] == "RF"]
                     .sort_values("support")["class"].tolist())
    short_order = [SHORT_NAMES[c] for c in support_order]

    x = np.arange(len(support_order))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, model in enumerate(MODELS):
        sub = df[df["model"] == model].set_index("class")
        recalls = [sub.loc[c, "recall"] for c in support_order]
        offset  = (i - 1.5) * width
        ax.bar(x + offset, recalls, width,
               label=model, color=COLORS[model],
               alpha=0.85, edgecolor="white")

    # Support annotations above x-axis
    supports = (df[df["model"] == "RF"]
                .sort_values("support")["support"].tolist())
    for i, (s, cls) in enumerate(zip(supports, short_order)):
        ax.text(i, -0.07, f"n={s:,}", ha="center", fontsize=7.5,
                color="gray", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(short_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Recall")
    ax.set_ylim(-0.12, 1.05)
    ax.set_title("Per-Class Recall by Model (classes ordered by frequency, rarest left)\n"
                 "Support (n=) shown below each class",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False, ncol=4)

    path = os.path.join(PLOTS_DIR, "7_recall_by_class_rarest_first.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")



# Plot 4: Precision−Recall gap per model (|P - R| per class)
def plot_pr_gap(df):
    """
    |Precision - Recall| per class per model.
    High gap = model is very unbalanced in how it errs.
    RF: high precision, low recall on minorities → positive gap.
    SRC/WSRC: low precision, higher recall → negative gap.
    """
    classes = list(SHORT_NAMES.keys())
    short_names = [SHORT_NAMES[c] for c in classes]
    x = np.arange(len(classes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, model in enumerate(MODELS):
        sub = df[df["model"] == model].set_index("class")
        gaps = [sub.loc[c, "precision"] - sub.loc[c, "recall"]
                for c in classes]
        offset = (i - 1.5) * width
        ax.bar(x + offset, gaps, width,
               label=model, color=COLORS[model],
               alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Precision - Recall\n(positive = precise but misses cases;\nnegative = finds cases but many false positives)")
    ax.set_title("Precision - Recall Gap per Class and Model\n"
                 "(RF: high precision, low recall on minorities | SRC/WSRC: opposite)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, ncol=4)

    path = os.path.join(PLOTS_DIR, "8_precision_recall_gap.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 5: Support vs F1 scatter — all classes and models
def plot_support_vs_f1(df):
    """
    Scatter: x=log(support), y=F1. One point per class per model.
    Shows that F1 degrades for rare classes across all models.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model in MODELS:
        sub = df[df["model"] == model]
        ax.scatter(np.log10(sub["support"]), sub["f1"],
                   color=COLORS[model], marker=MARKERS[model],
                   s=70, alpha=0.85, label=model,
                   edgecolors="white", linewidths=0.8, zorder=5)

        # Trend line
        z = np.polyfit(np.log10(sub["support"]), sub["f1"], 1)
        p = np.poly1d(z)
        xs = np.linspace(np.log10(sub["support"].min()),
                         np.log10(sub["support"].max()), 100)
        ax.plot(10**xs, p(xs), color=COLORS[model],
                linewidth=1.5, linestyle="--", alpha=0.5)

    # Annotate class names (using RF as reference)
    rf = df[df["model"] == "RF"]
    for _, row in rf.iterrows():
        ax.annotate(SHORT_NAMES[row["class"]],
                    (np.log10(row["support"]), row["f1"]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7.5, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("Class support (number of test samples, log scale)")
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score vs. Class Frequency\n"
                 "(dashed = log-linear trend | all models degrade on rare classes)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, ncol=2)

    path = os.path.join(PLOTS_DIR, "9_support_vs_f1_scatter.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Print key findings summary
def print_key_findings(df):
    print(f"\n{'='*60}")
    print("KEY FINDINGS FROM ERROR ANALYSIS")
    print(f"{'='*60}")

    # 1. Weighted F1 per model
    print("\n  Weighted F1 per model (sanity check vs main_comparison.csv):")
    for model in MODELS:
        sub = df[df["model"] == model]
        wf1 = (sub["f1"] * sub["support"]).sum() / sub["support"].sum()
        print(f"    {model:<6}: {wf1:.4f}")

    # 2. Worst class per model (lowest F1)
    print("\n  Worst class per model (lowest F1):")
    for model in MODELS:
        sub = df[df["model"] == model]
        worst = sub.loc[sub["f1"].idxmin()]
        print(f"    {model:<6}: {SHORT_NAMES[worst['class']]:<14} "
              f"F1={worst['f1']:.3f}  "
              f"P={worst['precision']:.3f}  R={worst['recall']:.3f}")

    # 3. OURS dominance in WSRC
    print("\n  OURS class analysis (majority class, 74% of data):")
    for model in MODELS:
        sub = df[df["model"] == model]
        ours = sub[sub["class"] == "CHUNK_CANONICAL_OURS"].iloc[0]
        print(f"    {model:<6}: P={ours['precision']:.3f}  "
              f"R={ours['recall']:.3f}  F1={ours['f1']:.3f}")

    # 4. BASE class: WSRC precision collapse
    print("\n  BASE class (rarest canonical — n=1,197):")
    for model in MODELS:
        sub = df[df["model"] == model]
        base = sub[sub["class"] == "CHUNK_CANONICAL_BASE"].iloc[0]
        print(f"    {model:<6}: P={base['precision']:.4f}  "
              f"R={base['recall']:.3f}  F1={base['f1']:.3f}")
    print("    → WSRC precision=0.023: predicts BASE but 97.7% are false positives")
    print("      (dictionary dominated by OURS causes near-random BASE prediction)")

    # 5. SRC vs WSRC on minority classes
    print("\n  SRC vs WSRC — does weighting help on minorities?")
    minority = ["CHUNK_CANONICAL_BASE", "CHUNK_SEMICANONICAL_OTHERS",
                "CHUNK_SEMICANONICAL_OURSTHEIRS"]
    for cls in minority:
        src_f1 = df[(df["model"]=="SRC")  & (df["class"]==cls)]["f1"].iloc[0]
        wsrc_f1 = df[(df["model"]=="WSRC") & (df["class"]==cls)]["f1"].iloc[0]
        delta = wsrc_f1 - src_f1
        verdict = "WSRC better" if delta > 0 else "SRC better"
        print(f"    {SHORT_NAMES[cls]:<14}: SRC={src_f1:.3f} "
              f"WSRC={wsrc_f1:.3f}  Δ={delta:+.3f}  ({verdict})")


# Main
def main():
    print(f"\nPlotting error analysis results from:\n  {INPUT_PATH}\n")

    if not os.path.exists(INPUT_PATH):
        print(f"  ERROR: {INPUT_PATH} not found.")
        print(f"  Run analyze_errors.py first.")
        return

    df = load_data()
    print(f"  Loaded {len(df)} rows — "
          f"{df['model'].nunique()} models × {df['class'].nunique()} classes\n")

    print("[1/5] Precision vs Recall scatter...")
    plot_precision_recall_scatter(df)

    print("[2/5] F1 heatmap...")
    plot_f1_heatmap(df)

    print("[3/5] Per-class recall (rarest first)...")
    plot_minority_class_recall(df)

    print("[4/5] Precision−Recall gap...")
    plot_pr_gap(df)

    print("[5/5] Support vs F1 scatter...")
    plot_support_vs_f1(df)

    print_key_findings(df)

    print(f"\n  All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()