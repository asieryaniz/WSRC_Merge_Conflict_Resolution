# plot_hyperparam_search.py
"""
Generates plots analyzing the hyperparameter search results from
hyperparam_search_wsrc.py. Saves all figures to plots/hyperparam/.

Plots produced:
  1. Accuracy vs dict_size per weight_method (one subplot per alpha)
  2. Accuracy vs alpha per weight_method (one subplot per dict_size)
  3. Heatmap: dict_size x alpha for each weight_method (mean across projects)
  4. Best accuracy per config across projects (top-N configs ranked)
  5. Accuracy gain: WSRC improvement over ZeroR per config
  6. Time vs dict_size tradeoff (accuracy vs time scatter)

Usage:
    python src/analysis/plot_hyperparam_search.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

INPUT_PATH  = os.path.join(BASE_DIR, "results", "wsrc_hyperparam_search.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "plots", "hyperparam")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aesthetics
PALETTE = {
    "similarity": "#2E86AB",
    "class": "#E84855",
    "uniform": "#F4A261",
}
MARKERS = {"similarity": "o", "class": "s", "uniform": "^"}

PROJECT_COLORS = {
    "getrailo/railo": "#2E86AB",
    "apache/accumulo": "#E84855",
    "zkoss/zk": "#3BB273",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right":False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 150,
})


def load_data():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")
    print(f"  Projects: {sorted(df['project'].unique())}")
    print(f"  Alphas:   {sorted(df['alpha'].unique())}")
    print(f"  Dict sizes: {sorted(df['max_per_class'].unique())}")
    print(f"  Weights:  {sorted(df['weight_method'].unique())}")
    return df


# Plot 1: Accuracy vs dict_size, faceted by alpha
def plot_accuracy_vs_dictsize(df):
    """Line plot: x=dict_size, y=accuracy, color=weight_method.
    One subplot per alpha value. One figure per project."""

    alphas = sorted(df["alpha"].unique())
    projects = sorted(df["project"].unique())

    for proj in projects:
        sub = df[df["project"] == proj]
        zeror = sub["zeror"].mean()

        # Extra top margin so two-line suptitle never clips
        fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5.2),
                                 sharey=True)
        proj_short = proj.split("/")[-1]
        fig.suptitle(f"WSRC: Accuracy vs Dictionary Size - {proj_short}",
                     fontsize=13, fontweight="bold")
        fig.subplots_adjust(top=0.88, bottom=0.22)

        for ax, alpha in zip(axes, alphas):
            sub_a = sub[sub["alpha"] == alpha]
            for wm in ["similarity", "class", "uniform"]:
                sub_w = sub_a[sub_a["weight_method"] == wm].sort_values("max_per_class")
                ax.plot(sub_w["max_per_class"], sub_w["accuracy"],
                        color=PALETTE[wm], marker=MARKERS[wm],
                        linewidth=2, markersize=6, label=wm)

            ax.axhline(zeror, color="gray", linestyle=":", linewidth=1.5,
                       label=f"ZeroR ({zeror:.3f})")
            ax.set_title(f"alpha = {alpha}", fontsize=11)
            ax.set_xlabel("Max samples per class (dict size)")
            if ax == axes[0]:
                ax.set_ylabel("Accuracy (5-fold CV)")

            # Rotate x-tick labels to avoid overlap when values are close
            dict_vals = sorted(sub["max_per_class"].unique())
            ax.set_xticks(dict_vals)
            ax.set_xticklabels([str(d) for d in dict_vals], rotation=45, ha="right")

        # Single shared legend below the subplots - weight_method + ZeroR
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=10,
                   title="Weight method", title_fontsize=10)

        proj_slug = proj.replace("/", "_")
        path = os.path.join(OUTPUT_DIR, f"1_acc_vs_dictsize_{proj_slug}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {os.path.basename(path)}")


# Plot 2: Accuracy vs alpha, faceted by dict_size
def plot_accuracy_vs_alpha(df):
    """Line plot: x=alpha (log scale), y=accuracy, color=weight_method.
    One subplot per dict_size. One figure per project."""

    dict_sizes = sorted(df["max_per_class"].unique())
    projects = sorted(df["project"].unique())

    for proj in projects:
        sub = df[df["project"] == proj]
        zeror = sub["zeror"].mean()

        fig, axes = plt.subplots(1, len(dict_sizes),
                                 figsize=(4.5 * len(dict_sizes), 5.2), sharey=True)
        proj_short = proj.split("/")[-1]
        fig.suptitle(f"WSRC: Accuracy vs Alpha (Lasso regularization) - {proj_short}",
                     fontsize=13, fontweight="bold")
        fig.subplots_adjust(top=0.88, bottom=0.22)

        for ax, ds in zip(axes, dict_sizes):
            sub_d = sub[sub["max_per_class"] == ds]
            for wm in ["similarity", "class", "uniform"]:
                sub_w = sub_d[sub_d["weight_method"] == wm].sort_values("alpha")
                ax.plot(sub_w["alpha"], sub_w["accuracy"],
                        color=PALETTE[wm], marker=MARKERS[wm],
                        linewidth=2, markersize=6, label=wm)

            ax.axhline(zeror, color="gray", linestyle=":", linewidth=1.5,
                       label=f"ZeroR ({zeror:.3f})")
            ax.set_xscale("log")
            ax.set_title(f"dict/cls = {ds}", fontsize=11)
            ax.set_xlabel("Alpha (log scale)")
            if ax == axes[0]:
                ax.set_ylabel("Accuracy (5-fold CV)")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=10,
                   title="Weight method", title_fontsize=10)

        proj_slug = proj.replace("/", "_")
        path = os.path.join(OUTPUT_DIR, f"2_acc_vs_alpha_{proj_slug}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {os.path.basename(path)}")


# Plot 3: Heatmap dict_size × alpha per weight_method
def plot_heatmaps(df):
    """Heatmap of mean accuracy (across projects) for each weight_method."""

    weight_methods = sorted(df["weight_method"].unique())
    fig, axes = plt.subplots(1, len(weight_methods),
                             figsize=(5.5 * len(weight_methods), 5.5))
    # Single-line title + explicit top margin so it never overlaps subplot titles
    fig.suptitle("WSRC Mean Accuracy Heatmap: dict_size x alpha (mean across search projects)",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.4)

    # Mean accuracy across projects for each config
    mean_df = df.groupby(["alpha", "max_per_class", "weight_method"])["accuracy"].mean().reset_index()

    alphas = sorted(df["alpha"].unique())
    dict_sizes = sorted(df["max_per_class"].unique())

    for ax, wm in zip(axes, weight_methods):
        sub = mean_df[mean_df["weight_method"] == wm]
        grid = sub.pivot(index="max_per_class", columns="alpha", values="accuracy")
        grid = grid.reindex(index=dict_sizes, columns=alphas)

        im = ax.imshow(grid.values, aspect="auto", cmap="RdYlGn",
                       vmin=mean_df["accuracy"].min(),
                       vmax=mean_df["accuracy"].max())

        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([str(a) for a in alphas], rotation=45, ha="right")
        ax.set_yticks(range(len(dict_sizes)))
        ax.set_yticklabels([str(d) for d in dict_sizes])
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Max samples per class" if ax == axes[0] else "")
        ax.set_title(f"weight = {wm}", fontweight="bold")

        # Annotate cells
        for i, ds in enumerate(dict_sizes):
            for j, a in enumerate(alphas):
                val = grid.loc[ds, a] if (ds in grid.index and a in grid.columns) else np.nan
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if val < (mean_df["accuracy"].min() + mean_df["accuracy"].max()) / 2
                            else "black")

        plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")

    path = os.path.join(OUTPUT_DIR, "3_heatmap_dictsize_alpha.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 4: Top-N configurations ranked by mean accuracy
def plot_top_configs(df, top_n=15):
    """Horizontal bar chart of all configs ranked by mean accuracy across projects."""

    mean_df = (df.groupby(["alpha", "max_per_class", "weight_method"])
                 .agg(mean_acc=("accuracy", "mean"),
                      std_acc=("accuracy", "std"),
                      mean_ni=("NI", "mean"))
                 .reset_index()
                 .sort_values("mean_acc", ascending=False))

    if top_n is not None:
        mean_df = mean_df.head(top_n)

    n_configs = len(mean_df)

    mean_df["label"] = mean_df.apply(
        lambda r: f"α={r['alpha']} | dict/cls={int(r['max_per_class'])} | weight={r['weight_method']}", axis=1
    )
    mean_df["color"] = mean_df["weight_method"].map(PALETTE)

    # Wide enough so labels on the left are never clipped
    fig, ax = plt.subplots(figsize=(13, n_configs * 0.45 + 2.0))
    ax.barh(range(n_configs), mean_df["mean_acc"],
            xerr=mean_df["std_acc"], capsize=3,
            color=mean_df["color"], alpha=0.85, height=0.65,
            error_kw={"elinewidth": 1.2, "ecolor": "gray"})

    # ZeroR reference line
    zeror_mean = df["zeror"].mean()
    ax.axvline(zeror_mean, color="gray", linestyle=":", linewidth=1.5)

    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(mean_df["label"], fontsize=9)
    ax.set_xlabel("Mean Accuracy (across search projects)")
    ax.set_title("All WSRC Configurations Ranked by Mean Accuracy",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # Value labels to the right of the error bar
    for i, (acc, std) in enumerate(zip(mean_df["mean_acc"], mean_df["std_acc"])):
        ax.text(acc + std + 0.003, i, f"{acc:.4f}", va="center", fontsize=8)

    # Legend placed in lower-right corner, outside the bars area
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=PALETTE[wm], label=f"weight = {wm}") for wm in PALETTE]
    legend_patches.append(
        plt.Line2D([0], [0], color="gray", linestyle=":", linewidth=1.5,
                   label=f"ZeroR mean ({zeror_mean:.3f})")
    )
    ax.legend(handles=legend_patches, frameon=True, loc="lower right",
              fontsize=9, framealpha=0.9)
    
    fig.subplots_adjust(left=0.38)

    path = os.path.join(OUTPUT_DIR, "4_top_configs_ranked.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 5: Accuracy vs dict_size - all projects on same axes
def plot_dictsize_curve_all_projects(df):
    """
    Best config per dict_size per project (maximized over alpha and weight_method).
    Shows the learning curve: how accuracy scales with dictionary size.
    """
    projects = sorted(df["project"].unique())

    # For each project × dict_size, take best accuracy across alpha+weight
    best = (df.groupby(["project", "max_per_class"])["accuracy"]
              .max()
              .reset_index())

    fig, ax = plt.subplots(figsize=(8, 5))

    for proj in projects:
        sub = best[best["project"] == proj].sort_values("max_per_class")
        color = PROJECT_COLORS.get(proj, "#888888")
        zeror = df[df["project"] == proj]["zeror"].mean()
        ax.plot(sub["max_per_class"], sub["accuracy"],
                color=color, marker="o", linewidth=2, markersize=6,
                label=proj.split("/")[-1])
        ax.axhline(zeror, color=color, linestyle=":", linewidth=1,
                   alpha=0.5)

    ax.set_xlabel("Max samples per class (dictionary size)")
    ax.set_ylabel("Best accuracy (max over alpha, weight_method)")
    ax.set_title("WSRC: Accuracy vs Dictionary Size\n(best config per size, ZeroR shown as dotted lines)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    path = os.path.join(OUTPUT_DIR, "5_dictsize_curve_all_projects.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 6: Accuracy vs Time tradeoff scatter
def plot_accuracy_vs_time(df):
    """Scatter: x=time_s, y=accuracy, color=weight_method, size=dict_size."""

    fig, axes = plt.subplots(1, len(df["project"].unique()),
                             figsize=(5.5 * df["project"].nunique(), 4.5),
                             sharey=False)

    if df["project"].nunique() == 1:
        axes = [axes]

    for ax, proj in zip(axes, sorted(df["project"].unique())):
        sub = df[df["project"] == proj]

        for wm in ["similarity", "class", "uniform"]:
            sub_w = sub[sub["weight_method"] == wm]
            ax.scatter(sub_w["time_s"], sub_w["accuracy"],
                       c=PALETTE[wm], marker=MARKERS[wm],
                       s=sub_w["max_per_class"] / 5 + 20,
                       alpha=0.75, label=wm, edgecolors="white", linewidths=0.5)

        zeror = sub["zeror"].mean()
        ax.axhline(zeror, color="gray", linestyle=":", linewidth=1.5,
                   label=f"ZeroR ({zeror:.3f})")

        ax.set_xlabel("Execution time (s, 5-fold CV)")
        ax.set_ylabel("Accuracy")
        ax.set_title(proj.split("/")[-1], fontsize=11, fontweight="bold")

        # Per-subplot legend showing weight_method colors + ZeroR
        ax.legend(frameon=True, fontsize=8, loc="lower right",
                  title="Weight method", title_fontsize=8, framealpha=0.9)

    # Shared size legend below the figure
    size_handles = [
        plt.scatter([], [], c="gray", s=ds / 5 + 20,
                    label=f"dict/cls = {ds}", edgecolors="white")
        for ds in sorted(df["max_per_class"].unique())
    ]
    fig.legend(handles=size_handles, loc="lower center",
               ncol=len(size_handles), bbox_to_anchor=(0.5, -0.08),
               frameon=False, fontsize=9, title="Dictionary size (point size)",
               title_fontsize=9)
    fig.suptitle("WSRC: Accuracy vs Computation Time Tradeoff",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(bottom=0.15)

    path = os.path.join(OUTPUT_DIR, "6_accuracy_vs_time_tradeoff.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Plot 7: Weight method comparison (boxplot across configs)
def plot_weight_method_comparison(df):
    """Boxplot comparing accuracy distributions per weight_method, per project."""

    projects = sorted(df["project"].unique())
    fig, axes = plt.subplots(1, len(projects),
                             figsize=(5 * len(projects), 5.5), sharey=False)
    if len(projects) == 1:
        axes = [axes]

    fig.suptitle("WSRC: Accuracy Distribution per Weight Method (across all alpha × dict_size combinations)",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.85, wspace=0.35)

    for ax, proj in zip(axes, projects):
        sub = df[df["project"] == proj]
        data = [sub[sub["weight_method"] == wm]["accuracy"].values
                  for wm in ["similarity", "class", "uniform"]]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})

        for patch, wm in zip(bp["boxes"], ["similarity", "class", "uniform"]):
            patch.set_facecolor(PALETTE[wm])
            patch.set_alpha(0.75)

        zeror = sub["zeror"].mean()
        ax.axhline(zeror, color="gray", linestyle=":", linewidth=1.5,
                   label=f"ZeroR ({zeror:.3f})")

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["similarity", "class", "uniform"], rotation=15)
        ax.set_ylabel("Accuracy")
        ax.set_title(proj.split("/")[-1], fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)

    path = os.path.join(OUTPUT_DIR, "7_weight_method_boxplot.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# Main
def main():
    print(f"\nGenerating hyperparameter analysis plots → {OUTPUT_DIR}\n")
    df = load_data()

    print("\n[1/7] Accuracy vs dict_size (per project, faceted by alpha)...")
    plot_accuracy_vs_dictsize(df)

    print("\n[2/7] Accuracy vs alpha (per project, faceted by dict_size)...")
    plot_accuracy_vs_alpha(df)

    print("\n[3/7] Heatmap dict_size x alpha per weight_method...")
    plot_heatmaps(df)

    print("\n[4/7] Top-N configurations ranked...")
    plot_top_configs(df)

    print("\n[5/7] Dict_size learning curve (all projects)...")
    plot_dictsize_curve_all_projects(df)

    print("\n[6/7] Accuracy vs time tradeoff scatter...")
    plot_accuracy_vs_time(df)

    print("\n[7/7] Weight method comparison (boxplot)...")
    plot_weight_method_comparison(df)

    print(f"\nAll plots saved to {OUTPUT_DIR}")
    print("Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()