# plot_results_analysis.py
"""
Generates comprehensive analysis plots of the final comparison results
from main_comparison.py. Saves all figures to plots/results/.

Plots produced:
  1.  Per-project accuracy: RF vs SRC vs WSRC (grouped bar chart)
  2.  Per-project F1: RF vs SRC vs WSRC
  3.  Per-project Normalized Improvement (NI) comparison
  4.  Accuracy gap: RF − WSRC and RF − SRC per project
  5.  WSRC vs SRC scatter (which model wins per project)
  6.  Accuracy vs dataset size (chunks) — all three models
  7.  ZeroR vs model accuracy (how much each model beats the baseline)
  8.  Summary radar chart (mean metrics across all models)
  9.  Per-class F1 breakdown (from final fold classification report) — if available
  10. Win/loss matrix heatmap

Usage:
    python src/analysis/plot_results_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

INPUT_PATH = os.path.join(BASE_DIR, "results", "final_comparison.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aesthetics
COLORS = {
    "RF": "#2E86AB",
    "SRC": "#F4A261",
    "WSRC": "#E84855",
    "KNN": "#3BB273",
    "ZeroR": "#AAAAAA",
}
HATCHES = {"RF": "", "SRC": "///", "WSRC": "...", "KNN": "xxx"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.bbox_inches": "tight",
    "savefig.dpi": 150,
})

MODELS = ["RF", "SRC", "WSRC", "KNN"]


def load_data():
    df = pd.read_csv(INPUT_PATH)
    # Short project names for readability
    df["proj_short"] = df["project"].apply(lambda x: x.split("/")[-1])
    df = df.sort_values("RF_accuracy", ascending=False).reset_index(drop=True)
    print(f"Loaded {len(df)} projects from {INPUT_PATH}")
    return df


# Plot 1 & 2: Per-project grouped bar (Accuracy and F1)
def plot_grouped_bars(df, metric, ylabel, title_suffix, filename):
    """Grouped bar chart: one group per project, one bar per model."""
    n = len(df)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, n * 0.9), 5.5))

    for i, model in enumerate(MODELS):
        col = f"{model}_{metric}"
        offset = (i - 1) * width
        bars = ax.bar(x + offset, df[col], width,
                        label=model, color=COLORS[model],
                        alpha=0.85, hatch=HATCHES[model],
                        edgecolor="white", linewidth=0.5)

    # ZeroR line per project
    ax.plot(x, df["RF_zeror"], color=COLORS["ZeroR"], marker="D",
            linewidth=1.5, markersize=5, linestyle="--", label="ZeroR", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["proj_short"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Per-Project {title_suffix}: RF vs. SRC vs. WSRC\n"
                 f"(S3 evaluation, merge-level grouping, 5-fold CV)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=10)

    # Annotate adaptive config project
    aosp_idx = df[df["project"].str.contains("aosp")].index
    if len(aosp_idx) > 0:
        pos = df.index.get_loc(aosp_idx[0])
        ax.annotate("*adaptive\nconfig", xy=(pos, 0.05),
                    ha="center", fontsize=7.5, color="gray", style="italic")

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 3: Normalized Improvement per project
def plot_normalized_improvement(df):
    """Bar chart of NI per model per project. NI=0 means equal to ZeroR."""
    n = len(df)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, n * 0.9), 5.5))

    for i, model in enumerate(MODELS):
        col = f"{model}_NI"
        offset = (i - 1) * width
        vals = df[col]
        colors = [COLORS[model] if v >= 0 else "#dddddd" for v in vals]
        ax.bar(x + offset, vals, width, label=model, color=colors,
               alpha=0.85, hatch=HATCHES[model], edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="-", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["proj_short"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Normalized Improvement over ZeroR")
    ax.set_title("Per-Project Normalized Improvement (NI)\n"
                 "NI = (Accuracy − ZeroR) / (1 − ZeroR) | NI < 0 means below ZeroR",
                 fontsize=13, fontweight="bold")

    legend_patches = [mpatches.Patch(color=COLORS[m], label=m) for m in MODELS]
    ax.legend(handles=legend_patches, frameon=False, loc="upper right")

    path = os.path.join(OUTPUT_DIR, "3_normalized_improvement.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 4: Accuracy gap RF − model
def plot_accuracy_gap(df):
    """Shows how much RF outperforms SRC, WSRC, and KNN per project."""
    fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.9), 5))

    x = np.arange(len(df))
    width = 0.25
    competitors = ["SRC", "WSRC", "KNN"]

    for i, model in enumerate(competitors):
        offset = (i - 1) * width
        gap = df["RF_accuracy"] - df[f"{model}_accuracy"]
        ax.bar(x + offset, gap, width, label=f"RF − {model}",
               color=COLORS[model], alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["proj_short"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy gap (RF − competitor)")
    ax.set_title("RF Accuracy Advantage over SRC, WSRC, and KNN per Project",
                 fontsize=13, fontweight="bold")

    for model in competitors:
        mean_gap = (df["RF_accuracy"] - df[f"{model}_accuracy"]).mean()
        ax.axhline(mean_gap, color=COLORS[model], linestyle=":",
                   linewidth=1.5, label=f"mean RF−{model}={mean_gap:.3f}")
    ax.legend(frameon=False, fontsize=9, ncol=2)

    path = os.path.join(OUTPUT_DIR, "4_accuracy_gap_rf_vs_models.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 5: WSRC vs SRC and WSRC vs KNN scatter
def plot_wsrc_vs_src_scatter(df):
    """Two scatter plots: WSRC vs SRC and WSRC vs KNN (key validation comparison)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    comparisons = [
        ("SRC",  "WSRC vs. SRC — does weighting help?"),
        ("KNN",  "WSRC vs. KNN — key validation (distance-based methods)"),
    ]

    for ax, (opponent, title) in zip(axes, comparisons):
        for _, row in df.iterrows():
            x_val = row[f"{opponent}_accuracy"]
            y_val = row["WSRC_accuracy"]
            color = COLORS["WSRC"] if y_val > x_val else COLORS[opponent]
            ax.scatter(x_val, y_val, color=color, s=80, zorder=5,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(row["proj_short"], (x_val, y_val),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7.5, color="gray")

        all_vals = pd.concat([df[f"{opponent}_accuracy"], df["WSRC_accuracy"]])
        lims = [all_vals.min() - 0.02, all_vals.max() + 0.02]
        ax.plot(lims, lims, "k--", linewidth=1.2, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)

        wsrc_wins = (df["WSRC_accuracy"] > df[f"{opponent}_accuracy"]).sum()
        delta = (df["WSRC_accuracy"] - df[f"{opponent}_accuracy"]).mean()
        ax.set_xlabel(f"{opponent} Accuracy")
        ax.set_ylabel("WSRC Accuracy")
        ax.set_title(f"{title}\n"
                     f"WSRC wins {wsrc_wins}/{len(df)} | mean Δ = {delta:+.3f}",
                     fontsize=10, fontweight="bold")

        legend_patches = [
            mpatches.Patch(color=COLORS["WSRC"],    label=f"WSRC > {opponent}"),
            mpatches.Patch(color=COLORS[opponent],  label=f"{opponent} ≥ WSRC"),
            plt.Line2D([0],[0], color="black", linestyle="--", label="equal"),
        ]
        ax.legend(handles=legend_patches, frameon=False, fontsize=9)

    fig.suptitle("WSRC Comparative Scatter Plots", fontsize=13, fontweight="bold")
    path = os.path.join(OUTPUT_DIR, "5_wsrc_scatter_vs_src_and_knn.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 6: Accuracy vs dataset size
def plot_accuracy_vs_size(df):
    """Scatter: x=chunks (log), y=accuracy, colored by model."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model in MODELS:
        ax.scatter(df["chunks"], df[f"{model}_accuracy"],
                   color=COLORS[model], s=70, alpha=0.8, label=model,
                   edgecolors="white", linewidths=0.8, zorder=5)
        # Trend line (log fit)
        log_chunks = np.log10(df["chunks"])
        z = np.polyfit(log_chunks, df[f"{model}_accuracy"], 1)
        p = np.poly1d(z)
        x_smooth = np.linspace(log_chunks.min(), log_chunks.max(), 100)
        ax.plot(10**x_smooth, p(x_smooth), color=COLORS[model],
                linewidth=1.5, linestyle="--", alpha=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("Project size (number of conflicting chunks, log scale)")
    ax.set_ylabel("Accuracy (5-fold CV, S3)")
    ax.set_title("Accuracy vs. Project Size\n"
                 "(dashed lines = log-linear trend per model)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False)

    path = os.path.join(OUTPUT_DIR, "6_accuracy_vs_dataset_size.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 7: Model accuracy vs ZeroR baseline
def plot_vs_zeror(df):
    """For each project, show how much each model beats ZeroR."""
    fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.9), 5.5))
    x = np.arange(len(df))

    # ZeroR baseline
    ax.bar(x, df["RF_zeror"], color=COLORS["ZeroR"], alpha=0.4,
           label="ZeroR", edgecolor="white")

    # Stack improvement on top of ZeroR
    for model in MODELS:
        improvement = (df[f"{model}_accuracy"] - df["RF_zeror"]).clip(lower=0)
        deficit = (df[f"{model}_accuracy"] - df["RF_zeror"]).clip(upper=0)
        ax.bar(x, improvement, bottom=df["RF_zeror"],
               color=COLORS[model], alpha=0.6, label=f"{model} gain",
               edgecolor="white", linewidth=0.5)
        if deficit.abs().sum() > 0:
            ax.bar(x, deficit, bottom=df["RF_zeror"],
                   color=COLORS[model], alpha=0.3,
                   edgecolor="white", linewidth=0.5, hatch="xx")

    ax.set_xticks(x)
    ax.set_xticklabels(df["proj_short"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Decomposition: ZeroR Baseline + Improvement\n"
                 "(hatched = below ZeroR)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=4, fontsize=9)

    path = os.path.join(OUTPUT_DIR, "7_accuracy_vs_zeror_decomposition.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 8: Summary bar — mean metrics
def plot_summary_means(df):
    """Clean summary bar chart of mean accuracy, F1, NI across all projects."""
    metrics = {
        "Mean Accuracy": {m: df[f"{m}_accuracy"].mean() for m in MODELS},
        "Mean Weighted F1": {m: df[f"{m}_f1"].mean() for m in MODELS},
        "Mean NI": {m: df[f"{m}_NI"].mean() for m in MODELS},
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Summary: Mean Performance Across All 16 Projects\n"
                 "(S3, merge-level grouping, 5-fold CV)",
                 fontsize=13, fontweight="bold")

    for ax, (metric_name, vals) in zip(axes, metrics.items()):
        models = list(vals.keys())
        values = list(vals.values())
        bars = ax.bar(models, values,
                      color=[COLORS[m] for m in models],
                      alpha=0.85, edgecolor="white", linewidth=0.5, width=0.5)

        # Value labels
        for bar, val in zip(bars, values):
            ypos = bar.get_height() + (0.005 if val >= 0 else -0.02)
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

        if metric_name == "Mean NI":
            ax.axhline(0, color="black", linewidth=1.2)
        else:
            zeror_mean = df["RF_zeror"].mean()
            ax.axhline(zeror_mean, color=COLORS["ZeroR"], linestyle="--",
                       linewidth=1.5, label=f"ZeroR ({zeror_mean:.3f})")
            ax.legend(frameon=False, fontsize=9)

        ax.set_title(metric_name, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric_name)

    path = os.path.join(OUTPUT_DIR, "8_summary_mean_metrics.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 9: Win/loss heatmap
def plot_win_matrix(df):
    """
    Heatmap: rows=projects, columns=models.
    Color encodes accuracy, with a marker for the winning model per project.
    """
    data = df[[f"{m}_accuracy" for m in MODELS]].values
    proj_labels = df["proj_short"].tolist()

    fig, ax = plt.subplots(figsize=(5, max(6, len(df) * 0.45)))

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=1.0)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(proj_labels)))
    ax.set_yticklabels(proj_labels, fontsize=9)
    ax.set_title("Accuracy Heatmap: RF vs. SRC vs. WSRC\nper Project",
                 fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(len(df)):
        winner_idx = np.argmax(data[i])
        for j in range(len(MODELS)):
            val = data[i, j]
            star = "★" if j == winner_idx else ""
            color = "white" if val < 0.55 else "black"
            ax.text(j, i, f"{val:.3f}{star}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if star else "normal")

    plt.colorbar(im, ax=ax, shrink=0.6, label="Accuracy")

    path = os.path.join(OUTPUT_DIR, "9_accuracy_heatmap_per_project.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 10: RF vs WSRC and RF vs KNN scatter (paper Fig. 2a style)
def plot_rf_vs_wsrc_scatter(df):
    """
    Two scatter plots in the style of paper Fig. 2a:
    Left:  x=WSRC accuracy, y=RF accuracy
    Right: x=KNN accuracy,  y=RF accuracy
    Points above diagonal = RF wins.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("RF vs. Distance-Based Methods (paper Fig. 2a style)",
                 fontsize=13, fontweight="bold")

    for ax, opponent in zip(axes, ["WSRC", "KNN"]):
        for _, row in df.iterrows():
            ax.scatter(row[f"{opponent}_accuracy"], row["RF_accuracy"],
                       color=COLORS["RF"], s=80, zorder=5,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(row["proj_short"],
                        (row[f"{opponent}_accuracy"], row["RF_accuracy"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7.5, color="gray")

        all_vals = pd.concat([df[f"{opponent}_accuracy"], df["RF_accuracy"]])
        lims = [all_vals.min() - 0.02, all_vals.max() + 0.02]
        ax.plot(lims, lims, "k--", linewidth=1.2, alpha=0.5, label="RF = " + opponent)
        ax.set_xlim(lims); ax.set_ylim(lims)

        rf_wins = (df["RF_accuracy"] > df[f"{opponent}_accuracy"]).sum()
        delta = (df["RF_accuracy"] - df[f"{opponent}_accuracy"]).mean()
        ax.set_xlabel(f"{opponent} Accuracy")
        ax.set_ylabel("RF Accuracy")
        ax.set_title(f"RF vs. {opponent}\n"
                     f"RF wins {rf_wins}/{len(df)} | mean Δ = {delta:+.3f}",
                     fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=9)

    path = os.path.join(OUTPUT_DIR, "10_rf_vs_wsrc_and_knn_scatter.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Main
def main():
    print(f"\nGenerating results analysis plots → {OUTPUT_DIR}\n")
    df = load_data()

    print("\n[1/10] Per-project accuracy grouped bar chart...")
    plot_grouped_bars(df, "accuracy", "Accuracy", "Accuracy", "1_per_project_accuracy.pdf")

    print("\n[2/10] Per-project F1 grouped bar chart...")
    plot_grouped_bars(df, "f1", "Weighted F1", "Weighted F1-Score", "2_per_project_f1.pdf")

    print("\n[3/10] Normalized Improvement per project...")
    plot_normalized_improvement(df)

    print("\n[4/10] Accuracy gap RF − model per project...")
    plot_accuracy_gap(df)

    print("\n[5/10] WSRC vs SRC scatter...")
    plot_wsrc_vs_src_scatter(df)

    print("\n[6/10] Accuracy vs dataset size...")
    plot_accuracy_vs_size(df)

    print("\n[7/10] Accuracy vs ZeroR decomposition...")
    plot_vs_zeror(df)

    print("\n[8/10] Summary mean metrics bar chart...")
    plot_summary_means(df)

    print("\n[9/10] Accuracy heatmap per project...")
    plot_win_matrix(df)

    print("\n[10/10] RF vs WSRC scatter (paper-style)...")
    plot_rf_vs_wsrc_scatter(df)

    print(f"\nAll plots saved to {OUTPUT_DIR}")
    print("Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()