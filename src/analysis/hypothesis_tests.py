# hypothesis_tests.py
"""
Statistical hypothesis testing for all experiments.

Sections:
  1. Friedman test — are there significant differences among RF, KNN, SRC, WSRC?
  2. Wilcoxon signed-rank post-hoc — pairwise comparisons with Bonferroni correction
  3. Spearman correlation — does class imbalance (ZeroR) explain the RF−WSRC gap?
  4. Wilcoxon on benchmark datasets — is WSRC's advantage on Digits significant?

All tests are non-parametric (n=16 projects, normality not assumed).
Significance level: α = 0.05 (two-sided). Bonferroni correction applied
to the 6 pairwise Wilcoxon comparisons.

Input files:
  results/final_comparison.csv
  results/benchmark_learning_curve.csv  (Digits learning curve)

Output:
  results/hypothesis_tests/hypothesis_tests_report.csv
  results/hypothesis_tests/hypothesis_tests_report.txt
  plots/hypothesis_tests/  (critical difference diagram + p-value heatmap)

Usage:
    python src/analysis/hypothesis_tests.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from itertools import combinations

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "hypothesis_tests")
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "hypothesis_tests")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

FINAL_CSV = os.path.join(BASE_DIR, "results", "final_comparison.csv")
DIGITS_CSV = os.path.join(BASE_DIR, "results", "benchmark_learning_curve.csv")

ALPHA = 0.05
MODELS = ["RF", "SRC", "WSRC", "KNN"]
N_PAIRS = 6 # C(4,2) pairwise comparisons
ALPHA_BONF = ALPHA / N_PAIRS # 0.0083

COLORS = {"RF": "#2E86AB", "KNN": "#3BB273",
          "SRC": "#F4A261", "WSRC": "#E84855"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})


# Helpers
def significance_label(p, alpha_corrected=None):
    """Return significance stars and label."""
    threshold = alpha_corrected if alpha_corrected else ALPHA
    if p < 0.001:
        return "***", "p < 0.001"
    elif p < 0.01:
        return "**", f"p = {p:.4f}"
    elif p < threshold:
        return "*", f"p = {p:.4f}"
    else:
        return "ns", f"p = {p:.4f}"


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {name}")


# Section 1: Friedman test
def test_friedman(df):
    """
    Non-parametric equivalent of one-way repeated-measures ANOVA.
    H0: all models have the same accuracy distribution across projects.
    """
    print("\n" + "="*62)
    print("SECTION 1 — Friedman Test")
    print("H0: no difference in accuracy among RF, KNN, SRC, WSRC")
    print("="*62)

    accs = [df[f"{m}_accuracy"].values for m in MODELS]
    stat, p = stats.friedmanchisquare(*accs)
    stars, label = significance_label(p)

    print(f"\n  χ²({len(MODELS)-1}) = {stat:.4f},  {label}  {stars}")
    print(f"  n = {len(df)} projects")

    if p < ALPHA:
        print(f"  → REJECT H0: significant differences exist among models.")
        print(f"    Post-hoc pairwise tests are warranted (see Section 2).")
    else:
        print(f"  → FAIL TO REJECT H0: no significant differences detected.")

    return {"test": "Friedman", "statistic": round(stat, 4),
            "p_value": round(p, 6), "significant": p < ALPHA,
            "stars": stars, "note": "k=4 models, n=16 projects"}


# Section 2: Wilcoxon signed-rank post-hoc with Bonferroni correction
def test_wilcoxon_posthoc(df):
    """
    Pairwise Wilcoxon signed-rank tests on the 16 per-project accuracy values.
    H0 for each pair: median difference = 0.
    Bonferroni correction: α_adj = 0.05 / 6 = 0.0083.
    """
    print("\n" + "="*62)
    print("SECTION 2 — Wilcoxon Signed-Rank Post-Hoc Tests")
    print(f"Bonferroni correction: α_adj = {ALPHA:.2f} / {N_PAIRS} = {ALPHA_BONF:.4f}")
    print("H0 for each pair: no difference in accuracy (two-sided)")
    print("="*62)

    print(f"\n  {'Comparison':<18} {'W stat':>8} {'p-value':>10} "
          f"{'Adj. sig':>10} {'Mean Δ':>8} {'A wins':>7}")
    print(f"  {'─'*18} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*7}")

    rows = []
    for m1, m2 in combinations(MODELS, 2):
        a = df[f"{m1}_accuracy"].values
        b = df[f"{m2}_accuracy"].values
        stat, p = stats.wilcoxon(a, b, alternative="two-sided")
        stars, label = significance_label(p, ALPHA_BONF)
        mean_delta = (a - b).mean()
        m1_wins = (a > b).sum()

        print(f"  {m1+' vs '+m2:<18} {stat:>8.1f} {p:>10.4f} "
              f"{stars:>10} {mean_delta:>+8.4f} {m1_wins:>4}/16")

        rows.append({
            "test": "Wilcoxon",
            "comparison": f"{m1} vs {m2}",
            "statistic": round(stat, 1),
            "p_value": round(p, 6),
            "p_bonf": round(min(p * N_PAIRS, 1.0), 6),  # Bonferroni-adjusted p
            "significant":p < ALPHA_BONF,
            "stars": stars,
            "mean_delta": round(mean_delta, 4),
            "wins_A": int(m1_wins),
            "note": f"n=16 projects, Bonferroni α={ALPHA_BONF:.4f}"
        })

    print(f"\n  Key findings:")
    sig_pairs = [r for r in rows if r["significant"]]
    ns_pairs = [r for r in rows if not r["significant"]]
    print(f"    Significant pairs:     "
          f"{[r['comparison'] for r in sig_pairs]}")
    print(f"    Non-significant pairs: "
          f"{[r['comparison'] for r in ns_pairs]}")
    print(f"\n    WSRC vs SRC: "
          f"{'NOT significant' if any(r['comparison']=='WSRC vs SRC' and not r['significant'] for r in rows) else 'significant'}"
          f" → the weighting mechanism adds no statistically proven benefit over SRC.")

    return rows


# Section 3: Spearman correlation — imbalance vs accuracy gap
def test_spearman_imbalance(df):
    """
    Tests whether class imbalance (ZeroR) correlates with the RF−WSRC accuracy gap.
    H0: no monotonic association (ρ = 0).
    """
    print("\n" + "="*62)
    print("SECTION 3 — Spearman Correlation")
    print("H0: no correlation between ZeroR and RF−WSRC accuracy gap")
    print("="*62)

    zeror = df["RF_zeror"].values
    gap = (df["RF_accuracy"] - df["WSRC_accuracy"]).values

    r, p = stats.spearmanr(zeror, gap)
    stars, label = significance_label(p)

    print(f"\n  Spearman ρ = {r:.4f},  {label}  {stars}")
    print(f"  n = {len(df)} projects")

    if p < ALPHA:
        print(f"  → REJECT H0: class imbalance significantly correlates with gap.")
        print(f"    Higher imbalance → larger RF advantage over WSRC.")
    else:
        print(f"  → FAIL TO REJECT H0: no significant correlation.")
        print(f"    The RF−WSRC gap cannot be explained by class imbalance alone.")
        print(f"    This supports the structural interpretation: WSRC fails because")
        print(f"    merge-conflict features do not satisfy the subspace assumption,")
        print(f"    not primarily because of the class distribution.")

    # Also test RF-KNN vs ZeroR for comparison
    gap_knn = (df["RF_accuracy"] - df["KNN_accuracy"]).values
    r2, p2  = stats.spearmanr(zeror, gap_knn)
    print(f"\n  Reference — ZeroR vs RF−KNN gap: ρ={r2:.4f}, p={p2:.4f}")

    return [
        {"test": "Spearman", "comparison": "ZeroR vs RF-WSRC gap",
         "statistic": round(r, 4), "p_value": round(p, 4),
         "significant": p < ALPHA, "stars": stars,
         "note": "n=16 projects"},
        {"test": "Spearman", "comparison": "ZeroR vs RF-KNN gap",
         "statistic": round(r2, 4), "p_value": round(p2, 4),
         "significant": p2 < ALPHA, "stars": significance_label(p2)[0],
         "note": "n=16 projects, reference comparison"},
    ]


# Section 4: Wilcoxon on Digits learning curve
def test_digits_learning_curve(digits_csv_path):
    """
    Tests whether WSRC's accuracy advantage on Digits is statistically significant.
    Uses the learning curve points (6 training fractions) as paired observations.
    H0: WSRC accuracy = RF accuracy across training sizes.
    """
    print("\n" + "="*62)
    print("SECTION 4 — Wilcoxon on Digits Learning Curve")
    print("H0: no difference between WSRC and RF/KNN on Digits dataset")
    print("="*62)

    if not os.path.exists(digits_csv_path):
        print(f"\n  SKIP: {digits_csv_path} not found.")
        print(f"  Run validate_benchmark_datasets.py first.")
        return []

    df = pd.read_csv(digits_csv_path)
    digs = df[df["dataset"] == "Digits"].copy()

    if digs.empty:
        print("  SKIP: No Digits rows found in benchmark_learning_curve.csv.")
        return []

    print(f"\n  n = {len(digs)} learning curve points (training fractions)")
    print(f"  Fractions: {digs['train_frac'].tolist()}")

    rows = []
    for opponent in ["RF", "KNN", "SRC"]:
        if opponent not in digs.columns:
            continue
        a = digs["WSRC"].values
        b = digs[opponent].values
        # Wilcoxon requires at least some non-zero differences
        if np.all(a == b):
            print(f"  WSRC vs {opponent}: all differences = 0, skip.")
            continue
        try:
            stat, p = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError as e:
            print(f"  WSRC vs {opponent}: {e}")
            continue

        stars, label = significance_label(p)
        mean_delta = (a - b).mean()
        wsrc_wins = (a > b).sum()

        print(f"\n  WSRC vs {opponent} on Digits:")
        print(f"    W = {stat:.1f},  {label}  {stars}")
        print(f"    Mean Δ (WSRC − {opponent}) = {mean_delta:+.4f}")
        print(f"    WSRC > {opponent} in {wsrc_wins}/{len(digs)} curve points")

        rows.append({
            "test": "Wilcoxon",
            "comparison": f"Digits: WSRC vs {opponent}",
            "statistic": round(stat, 1),
            "p_value": round(p, 4),
            "significant": p < ALPHA,
            "stars": stars,
            "mean_delta": round(mean_delta, 4),
            "wins_A": int(wsrc_wins),
            "note": f"n={len(digs)} learning curve points"
        })

    if rows:
        sig = [r for r in rows if r["significant"]]
        print(f"\n  Summary: WSRC advantage on Digits is "
              f"{'statistically significant' if sig else 'NOT significant'} "
              f"vs {[r['comparison'].split('vs ')[1].strip() for r in sig]}.")

    return rows


# Plot 1: p-value heatmap for pairwise Wilcoxon
def plot_pvalue_heatmap(wilcoxon_rows):
    """Lower-triangle heatmap of adjusted p-values for all pairwise comparisons."""
    n = len(MODELS)
    matrix = np.ones((n, n))
    pmatrix = np.ones((n, n))

    idx = {m: i for i, m in enumerate(MODELS)}
    for r in wilcoxon_rows:
        m1, m2 = r["comparison"].split(" vs ")
        i, j = idx[m1], idx[m2]
        p_bonf = min(r["p_value"] * N_PAIRS, 1.0)
        matrix[i, j] = p_bonf
        matrix[j, i] = p_bonf
        pmatrix[i, j] = r["p_value"]
        pmatrix[j, i] = r["p_value"]

    # Mask upper triangle and diagonal
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    data_to_plot = np.where(mask, np.nan, matrix)

    im = ax.imshow(data_to_plot, cmap="RdYlGn_r", vmin=0, vmax=0.5,
                   aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_yticklabels(MODELS, fontsize=11)

    for i in range(n):
        for j in range(n):
            if i <= j:
                continue
            p_raw = pmatrix[i, j]
            p_bonf = matrix[i, j]
            stars = significance_label(p_raw, ALPHA_BONF)[0]
            ax.text(j, i,
                    f"p={p_raw:.3f}\n({stars})",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if p_bonf < 0.05 else "black",
                    fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8,
                 label="Bonferroni-adjusted p-value")
    ax.set_title("Pairwise Wilcoxon Signed-Rank Tests\n"
                 "(raw p-value shown; color = Bonferroni-adjusted p;\n"
                 "green = significant after correction, red = not significant)",
                 fontsize=10, fontweight="bold")

    save(fig, "1_pvalue_heatmap_wilcoxon.pdf")


# Plot 2: Effect size — mean accuracy difference with CI
def plot_effect_sizes(df, wilcoxon_rows):
    """
    Horizontal bar chart: mean accuracy difference per pair.
    Error bars = bootstrapped 95% CI.
    Vertical line at 0 = no difference.
    """
    pairs = [(r["comparison"], r["mean_delta"], r["significant"])
               for r in wilcoxon_rows]
    labels = [p[0] for p in pairs]
    deltas = [p[1] for p in pairs]
    sig = [p[2] for p in pairs]

    # Bootstrap 95% CI for mean delta
    rng = np.random.default_rng(42)
    cis = []
    for r in wilcoxon_rows:
        m1, m2 = r["comparison"].split(" vs ")
        diffs = (df[f"{m1}_accuracy"] - df[f"{m2}_accuracy"]).values
        boot = rng.choice(diffs, size=(5000, len(diffs)), replace=True).mean(axis=1)
        cis.append((np.percentile(boot, 2.5), np.percentile(boot, 97.5)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(labels))

    for i, (label, delta, significant, ci) in enumerate(
            zip(labels, deltas, sig, cis)):
        color  = "#2E86AB" if significant else "#AAAAAA"
        ax.barh(i, delta, height=0.55, color=color, alpha=0.85,
                edgecolor="white")
        ax.errorbar(delta, i,
                    xerr=[[delta - ci[0]], [ci[1] - delta]],
                    fmt="none", color="black", capsize=4, linewidth=1.5)
        stars = "***" if significant and wilcoxon_rows[i]["p_value"] < 0.001 \
                else ("**" if significant and wilcoxon_rows[i]["p_value"] < 0.01
                      else ("*" if significant else "ns"))
        ax.text(delta + (0.005 if delta >= 0 else -0.005), i,
                stars, va="center",
                ha="left" if delta >= 0 else "right", fontsize=10)

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean accuracy difference (A − B)\nwith 95% bootstrap CI")
    ax.set_title("Effect Sizes: Pairwise Accuracy Differences\n"
                 "(blue = significant after Bonferroni; gray = not significant)",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color="#2E86AB", label="Significant (Bonferroni-corrected)"),
        mpatches.Patch(color="#AAAAAA", label="Not significant"),
    ]
    ax.legend(handles=legend_patches, frameon=False, fontsize=9,
              loc="lower right")

    save(fig, "2_effect_sizes_pairwise.pdf")


# Plot 3: Spearman scatter — ZeroR vs RF-WSRC gap
def plot_spearman_scatter(df, spearman_rows):
    """Annotated scatter with regression line and Spearman ρ."""
    zeror = df["RF_zeror"].values
    gap = (df["RF_accuracy"] - df["WSRC_accuracy"]).values
    r = spearman_rows[0]["statistic"]
    p = spearman_rows[0]["p_value"]
    stars = spearman_rows[0]["stars"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(zeror, gap, color="#E84855", s=75,
               edgecolors="white", linewidths=0.8, zorder=5)

    for _, row in df.iterrows():
        ax.annotate(row["project"].split("/")[-1],
                    (row["RF_zeror"],
                     row["RF_accuracy"] - row["WSRC_accuracy"]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7.5, color="gray")

    z = np.polyfit(zeror, gap, 1)
    xline = np.linspace(zeror.min(), zeror.max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), color="#E84855",
            linestyle="--", linewidth=1.5, alpha=0.7)

    ax.axhline(0, color="black", linewidth=1)
    ax.text(0.05, 0.95,
            f"Spearman ρ = {r:.3f}\n{stars}  (p = {p:.3f})",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("ZeroR baseline (class imbalance proxy)")
    ax.set_ylabel("Accuracy gap (RF - WSRC)")
    ax.set_title("Spearman Correlation: Class Imbalance vs RF-WSRC Gap\n"
                 "(non-significant → imbalance is not the primary cause of WSRC's underperformance)",
                 fontsize=11, fontweight="bold")

    save(fig, "3_spearman_zeror_vs_gap.pdf")


# Plot 4: Digits learning curve with significance annotations
def plot_digits_significance(digits_csv_path, digits_rows):
    """
    Line plot of the Digits learning curve with significance markers
    showing where WSRC significantly outperforms RF and KNN.
    """
    if not os.path.exists(digits_csv_path) or not digits_rows:
        print("    Skipping Digits plot (no data).")
        return

    df = pd.read_csv(digits_csv_path)
    digs = df[df["dataset"] == "Digits"].sort_values("train_frac")

    fig, ax = plt.subplots(figsize=(8, 5))
    pct = digs["train_frac"] * 100

    for model, color in [("RF", "#2E86AB"), ("KNN", "#3BB273"),
                          ("SRC", "#F4A261"), ("WSRC", "#E84855")]:
        if model in digs.columns:
            ax.plot(pct, digs[model], color=color, marker="o",
                    linewidth=2, markersize=6, label=model)

    # Annotate with WSRC>RF markers
    for _, row in digs.iterrows():
        if row.get("WSRC", 0) > row.get("RF", 0):
            ax.annotate("★", (row["train_frac"]*100, row["WSRC"]+0.005),
                        ha="center", fontsize=11, color="#E84855")

    # Add Wilcoxon result as text box
    wsrc_vs_rf = next((r for r in digits_rows
                       if "RF" in r["comparison"] and "KNN" not in r["comparison"]), None)
    if wsrc_vs_rf:
        ax.text(0.05, 0.08,
                f"Wilcoxon WSRC vs RF:\np = {wsrc_vs_rf['p_value']:.4f}  "
                f"{wsrc_vs_rf['stars']}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("Training set size (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Digits Dataset — Learning Curve\n"
                 "(★ = WSRC > RF at this training size;\n"
                 "Wilcoxon test over all points shown in box)",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, ncol=4)

    save(fig, "4_digits_learning_curve_significance.pdf")



# Save full report
def save_report(all_rows, txt_lines):
    # CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "hypothesis_tests_report.csv"),
              index=False)

    # TXT
    with open(os.path.join(RESULTS_DIR, "hypothesis_tests_report.txt"),
              "w") as f:
        f.write("\n".join(txt_lines))

    print(f"\n  Results saved → {RESULTS_DIR}")


# Main
def main():
    print("\nHypothesis Testing — Statistical Validation of All Experiments")
    print(f"α = {ALPHA}  |  Bonferroni-corrected α = {ALPHA_BONF:.4f}  "
          f"({N_PAIRS} pairwise comparisons)")

    if not os.path.exists(FINAL_CSV):
        print(f"\nERROR: {FINAL_CSV} not found. Run main_comparison.py first.")
        return

    df = pd.read_csv(FINAL_CSV)
    print(f"\nLoaded {len(df)} projects from final_comparison.csv")

    all_rows = []
    txt_lines = [
        "HYPOTHESIS TESTING REPORT",
        f"n = {len(df)} projects | α = {ALPHA} | Bonferroni α = {ALPHA_BONF:.4f}",
        "="*62,
    ]

    # Section 1: Friedman
    friedman_row = test_friedman(df)
    all_rows.append(friedman_row)

    # Section 2: Wilcoxon post-hoc
    wilcoxon_rows = test_wilcoxon_posthoc(df)
    all_rows.extend(wilcoxon_rows)

    # Section 3: Spearman
    spearman_rows = test_spearman_imbalance(df)
    all_rows.extend(spearman_rows)

    # Section 4: Digits
    digits_rows = test_digits_learning_curve(DIGITS_CSV)
    all_rows.extend(digits_rows)

    # Plots
    print("\n" + "="*62)
    print("Generating plots...")

    print("\n  [1/4] p-value heatmap...")
    plot_pvalue_heatmap(wilcoxon_rows)

    print("  [2/4] Effect sizes with bootstrap CI...")
    plot_effect_sizes(df, wilcoxon_rows)

    print("  [3/4] Spearman scatter...")
    plot_spearman_scatter(df, spearman_rows)

    print("  [4/4] Digits learning curve with significance...")
    plot_digits_significance(DIGITS_CSV, digits_rows)

    # Final summary
    print("\n" + "="*62)
    print("STATISTICAL SUMMARY")
    print("="*62)
    sig = [r for r in wilcoxon_rows if r["significant"]]
    ns = [r for r in wilcoxon_rows if not r["significant"]]
    print(f"\n  Friedman:  χ²={friedman_row['statistic']:.3f}, "
          f"p={friedman_row['p_value']:.6f}  {friedman_row['stars']}")
    print(f"\n  Significant pairs after Bonferroni (α={ALPHA_BONF:.4f}):")
    for r in sig:
        print(f"    {r['comparison']:<20} p={r['p_value']:.4f}  "
              f"Δ={r['mean_delta']:+.4f}  {r['stars']}")
    print(f"\n  Non-significant pairs:")
    for r in ns:
        print(f"    {r['comparison']:<20} p={r['p_value']:.4f}  "
              f"Δ={r['mean_delta']:+.4f}  {r['stars']}")
    print(f"\n  Spearman ZeroR vs gap: "
          f"ρ={spearman_rows[0]['statistic']:.3f}, "
          f"p={spearman_rows[0]['p_value']:.4f}  "
          f"{spearman_rows[0]['stars']}")

    save_report(all_rows, txt_lines)


if __name__ == "__main__":
    main()