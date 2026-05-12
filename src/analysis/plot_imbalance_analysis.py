# plot_imbalance_analysis.py
"""
Generates plots from imbalance_std_results.csv and imbalance_bal_results.csv.

Key findings from the data:
  - Balancing the training set hurts both RF (-0.26) and WSRC (-0.27) equally
  - RF_balanced barely changes vs RF (+0.003 mean, hurts in 6/16 projects)
  - WSRC_class > WSRC_uniform in 11/16 projects (class weights help marginally)
  - Correlation ZeroR vs RF-WSRC gap is moderate (r=0.317)
  → The WSRC disadvantage is structural, not primarily driven by class imbalance

Input:  results/imbalance/imbalance_std_results.csv
        results/imbalance/imbalance_bal_results.csv
Output: plots/imbalance/

Usage:
    python src/analysis/plot_imbalance_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STD_PATH = os.path.join(BASE_DIR, "results", "imbalance", "imbalance_std_results.csv")
BAL_PATH = os.path.join(BASE_DIR, "results", "imbalance", "imbalance_bal_results.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "imbalance")
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    "RF": "#2E86AB",
    "RF_balanced": "#7BC8E2",
    "WSRC_class": "#E84855",
    "WSRC_uniform": "#F4A261",
    "KNN": "#3BB273",
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

MODELS_STD = ["RF", "RF_balanced", "WSRC_class", "WSRC_uniform", "KNN"]


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# Plot 1: ZeroR vs RF−WSRC accuracy gap (correlation scatter)
def plot_imbalance_vs_gap(std):
    """
    x = ZeroR (class imbalance proxy), y = RF - WSRC_class accuracy gap.
    Tests whether higher imbalance = bigger WSRC disadvantage.
    """
    x = std["zeror"]
    y = std["RF"] - std["WSRC_class"]
    corr = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x, y, color=COLORS["WSRC_class"], s=80,
               edgecolors="white", linewidths=0.8, zorder=5)

    for _, row in std.iterrows():
        ax.annotate(row["proj_short"],
                    (row["zeror"], row["RF"] - row["WSRC_class"]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7.5, color="gray")

    # Trend line
    z = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), color=COLORS["WSRC_class"],
            linestyle="--", linewidth=1.5, alpha=0.7)

    ax.axhline(0, color="black", linewidth=1)
    ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("ZeroR baseline (class imbalance proxy — higher = more imbalanced)")
    ax.set_ylabel("Accuracy gap (RF − WSRC_class)")
    ax.set_title("Does Class Imbalance Explain the RF−WSRC Gap?\n"
                 f"(r={corr:.3f} — moderate correlation, not the full story)",
                 fontsize=12, fontweight="bold")

    save(fig, "1_imbalance_vs_accuracy_gap.pdf")


# Plot 2: Standard training — all models per project (grouped bar)
def plot_standard_per_project(std):
    """
    Grouped bar chart: x=project, groups=models, y=accuracy.
    Projects sorted by ZeroR to reveal imbalance trend.
    """
    df = std.sort_values("zeror")
    n = len(df)
    x = np.arange(n)
    w = 0.15

    fig, ax = plt.subplots(figsize=(15, 5.5))

    for i, model in enumerate(MODELS_STD):
        if model not in df.columns:
            continue
        offset = (i - 2) * w
        ax.bar(x + offset, df[model], w,
               label=model.replace("_", " "),
               color=COLORS[model], alpha=0.85, edgecolor="white")

    # ZeroR line
    ax.plot(x, df["zeror"], color="gray", marker="D", linewidth=1.5,
            markersize=5, linestyle="--", label="ZeroR", zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(df["proj_short"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (5-fold CV, S3)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Project Accuracy — Standard Training\n"
                 "(projects ordered left→right by increasing class imbalance / ZeroR)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, ncol=6, fontsize=9)

    save(fig, "2_per_project_accuracy_standard.pdf")


# Plot 3: Standard vs Balanced — effect of undersampling per model
def plot_balancing_effect(std, bal):
    """
    Side-by-side: for each model shows std vs balanced accuracy per project.
    Key finding: balancing hurts everyone, not just WSRC.
    """
    models_to_show = ["RF", "WSRC_class"]
    n = len(std)
    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    fig.suptitle("Effect of Training Set Balancing (undersampling to equal class sizes)\n"
                 "Balancing hurts both RF and WSRC — the distribution matters",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.85)

    for ax, model in zip(axes, models_to_show):
        std_vals = std[model].values
        bal_vals = bal[model].values
        delta = bal_vals - std_vals

        ax.bar(x - w/2, std_vals, w, label="Standard",
               color=COLORS[model], alpha=0.85, edgecolor="white")
        ax.bar(x + w/2, bal_vals, w, label="Balanced",
               color=COLORS[model], alpha=0.4, edgecolor="white",
               hatch="///")

        # Delta annotation
        for i, (s, b, d) in enumerate(zip(std_vals, bal_vals, delta)):
            ypos = max(s, b) + 0.015
            ax.text(i, ypos, f"{d:+.2f}", ha="center", fontsize=6.5,
                    color="green" if d > 0 else "red")

        # ZeroR
        ax.plot(x, std["zeror"].values, color="gray", marker="D",
                linewidth=1.2, markersize=4, linestyle=":",
                label="ZeroR", zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels(std["proj_short"], rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model.replace('_', ' ')}",
                     fontsize=11, fontweight="bold", color=COLORS[model])
        ax.legend(frameon=False, fontsize=9)

        mean_delta = delta.mean()
        ax.text(0.02, 0.97, f"Mean Δ = {mean_delta:+.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    save(fig, "3_balancing_effect_std_vs_bal.pdf")


# Plot 4: WSRC_class vs WSRC_uniform per project
def plot_wsrc_weights_comparison(std):
    """
    Scatter: x=WSRC_uniform, y=WSRC_class. Points above diagonal = class weights help.
    Reveals when frequency-based weighting adds value.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in std.iterrows():
        x_val = row["WSRC_uniform"]
        y_val = row["WSRC_class"]
        color = COLORS["WSRC_class"] if y_val > x_val else COLORS["WSRC_uniform"]
        ax.scatter(x_val, y_val, color=color, s=80,
                   edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(row["proj_short"], (x_val, y_val),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color="gray")

    all_vals = pd.concat([std["WSRC_uniform"], std["WSRC_class"]])
    lims = [all_vals.min() - 0.02, all_vals.max() + 0.02]
    ax.plot(lims, lims, "k--", linewidth=1.2, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)

    class_wins = (std["WSRC_class"] > std["WSRC_uniform"]).sum()
    delta_mean  = (std["WSRC_class"] - std["WSRC_uniform"]).mean()

    ax.set_xlabel("WSRC uniform weights accuracy")
    ax.set_ylabel("WSRC class-frequency weights accuracy")
    ax.set_title(f"WSRC: Class Weights vs. Uniform Weights\n"
                 f"Class weights win in {class_wins}/16 projects "
                 f"(mean Δ = {delta_mean:+.3f})",
                 fontsize=12, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=COLORS["WSRC_class"],
                       label=f"class > uniform ({class_wins} projects)"),
        mpatches.Patch(color=COLORS["WSRC_uniform"],
                       label=f"uniform ≥ class ({len(std)-class_wins} projects)"),
        plt.Line2D([0],[0], color="black", linestyle="--", label="equal"),
    ]
    ax.legend(handles=legend_patches, frameon=False, fontsize=9)

    save(fig, "4_wsrc_class_vs_uniform_scatter.pdf")


# Plot 5: Accuracy by imbalance tier — boxplot
def plot_by_imbalance_tier(std):
    """
    Projects split into low/medium/high imbalance tiers.
    Shows whether WSRC is competitive at low imbalance.
    """
    df = std.copy()
    df["tier"] = pd.cut(df["zeror"],
                        bins=[0, 0.55, 0.70, 1.0],
                        labels=["Low\n(ZeroR < 0.55)",
                                "Medium\n(0.55 – 0.70)",
                                "High\n(ZeroR > 0.70)"])
    tiers = ["Low\n(ZeroR < 0.55)", "Medium\n(0.55 – 0.70)", "High\n(ZeroR > 0.70)"]

    plot_models  = ["RF", "WSRC_class", "WSRC_uniform", "KNN"]
    plot_colors  = [COLORS[m] for m in plot_models]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    fig.suptitle("Model Accuracy by Class Imbalance Level\n"
                 "(is WSRC competitive when classes are balanced?)",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.85)

    for ax, tier in zip(axes, tiers):
        sub = df[df["tier"] == tier]
        if sub.empty:
            ax.set_title(f"{tier}\n(no projects)")
            continue

        data = [sub[m].dropna().values for m in plot_models]
        bp   = ax.boxplot(data, patch_artist=True,
                          medianprops={"color": "black", "linewidth": 2})

        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Individual points with jitter
        rng = np.random.default_rng(42)
        for i, m in enumerate(plot_models):
            vals   = sub[m].dropna().values
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.ones(len(vals)) * (i+1) + jitter, vals,
                       color="black", s=22, alpha=0.55, zorder=5)

        ax.set_xticks(range(1, len(plot_models)+1))
        ax.set_xticklabels([m.replace("_", "\n") for m in plot_models],
                           fontsize=8)
        ax.set_title(f"{tier}\n({len(sub)} project{'s' if len(sub)!=1 else ''})",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Accuracy" if ax == axes[0] else "")

        zeror_mean = sub["zeror"].mean()
        ax.axhline(zeror_mean, color="gray", linestyle=":",
                   linewidth=1.5, label=f"ZeroR ({zeror_mean:.2f})")
        ax.legend(frameon=False, fontsize=8)

    save(fig, "5_accuracy_by_imbalance_tier.pdf")


# Print key findings
def print_findings(std, bal):
    print(f"\n{'='*60}")
    print("KEY FINDINGS — IMBALANCE ANALYSIS")
    print(f"{'='*60}")

    print("\n  1. Does balancing help WSRC more than RF?")
    d_rf   = (bal["RF"]         - std["RF"]).mean()
    d_wsrc = (bal["WSRC_class"] - std["WSRC_class"]).mean()
    print(f"     RF   mean Δ: {d_rf:+.4f}  (helps in {(bal['RF']>std['RF']).sum()}/16)")
    print(f"     WSRC mean Δ: {d_wsrc:+.4f}  "
          f"(helps in {(bal['WSRC_class']>std['WSRC_class']).sum()}/16)")
    print(f"     → Balancing HURTS both equally. Imbalance is not the root cause.")

    print("\n  2. Class weights in WSRC (class vs uniform):")
    d = (std["WSRC_class"] - std["WSRC_uniform"]).mean()
    wins = (std["WSRC_class"] > std["WSRC_uniform"]).sum()
    print(f"     Mean Δ: {d:+.4f}  |  class wins: {wins}/16")
    print(f"     → Class weights marginally help but don't close the RF gap.")

    print("\n  3. RF_balanced vs RF:")
    d = (std["RF_balanced"] - std["RF"]).mean()
    wins = (std["RF_balanced"] > std["RF"]).sum()
    print(f"     Mean Δ: {d:+.4f}  |  balanced wins: {wins}/16")
    print(f"     → RF is robust to class_weight correction.")

    print("\n  4. Correlation ZeroR vs RF-WSRC gap:")
    gap = std["RF"] - std["WSRC_class"]
    corr = np.corrcoef(std["zeror"], gap)[0, 1]
    print(f"     r = {corr:.3f}  → moderate correlation, not deterministic")
    print(f"     Largest gap projects: "
          f"{std.nlargest(3,'zeror')['proj_short'].tolist()}")

    print("\n  5. WSRC in low-imbalance projects (ZeroR < 0.55):")
    lo = std[std["zeror"] < 0.55]
    print(f"     Mean RF:   {lo['RF'].mean():.4f}")
    print(f"     Mean WSRC: {lo['WSRC_class'].mean():.4f}")
    print(f"     WSRC>RF in: "
          f"{(lo['WSRC_class']>lo['RF']).sum()}/{len(lo)} low-imbalance projects")
    print(f"     → Even with balanced classes, WSRC doesn't surpass RF.")
    print(f"       The limitation is structural (no subspace assumption), not imbalance.")


# Main
def main():
    for path in [STD_PATH, BAL_PATH]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run analyze_imbalance.py first.")
            return

    std = pd.read_csv(STD_PATH)
    bal = pd.read_csv(BAL_PATH)

    print(f"\nImbalance analysis plots — {len(std)} projects\n")

    print("[1/5] ZeroR vs RF−WSRC gap scatter...")
    plot_imbalance_vs_gap(std)

    print("[2/5] Per-project accuracy standard training...")
    plot_standard_per_project(std)

    print("[3/5] Effect of balancing (std vs balanced)...")
    plot_balancing_effect(std, bal)

    print("[4/5] WSRC class vs uniform weights scatter...")
    plot_wsrc_weights_comparison(std)

    print("[5/5] Accuracy by imbalance tier (boxplot)...")
    plot_by_imbalance_tier(std)

    print_findings(std, bal)

    print(f"\n  All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()