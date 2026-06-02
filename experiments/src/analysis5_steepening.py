"""Analysis 5: Steepening case characterization — steepening vs flattening comparison."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config, utils, plotting, tables, data_loader


def run(df):
    """Compare steepening (meridian > 90°) vs flattening cases."""
    print("=" * 60)
    print("ANALYSIS 5: Steepening Case Characterization")
    print("=" * 60)

    # ── 5.1 Split into groups ──────────────────────────────────────────
    df = df.copy()
    df["steepening"] = (df["CSIA_meridian"] > 90).astype(int)
    steep = df[df["steepening"] == 1]
    flat = df[df["steepening"] == 0]
    print(f"\n  Steepening (meridian > 90°): n={len(steep)} ({100*len(steep)/len(df):.1f}%)")
    print(f"  Flattening (meridian <= 90°): n={len(flat)} ({100*len(flat)/len(df):.1f}%)")

    # ── 5.2 Compare all preoperative parameters ────────────────────────
    print("\n--- 5.2 Parameter comparison ---")
    comparison_vars = config.ALL_PREOP_PARAMS + ["age"]
    var_labels = {
        "PCT135": "PCT 135° (μm)", "AL": "AL (mm)", "WTW": "WTW (mm)",
        "AL1": "AL1 (mm)", "AV1": "AV1 (m/s)", "AL2": "AL2 (mm)", "AV2": "AV2 (m/s)",
        "PD": "PD (mm)", "HCR": "HCR (mm)", "HCDA": "HCDA (mm)",
        "IOPnc": "IOPnc (mmHg)", "bIOP": "bIOP (mmHg)", "CCT": "CCT (μm)",
        "SPA1": "SP-A1", "ARTh": "ARTh (μm)",
        "DA_Ratio": "DA Ratio", "IR": "IR (mm⁻¹)", "CBI": "CBI", "SSI": "SSI", "CBiF": "CBiF",
        "age": "Age (years)",
    }

    comp_rows = []
    n_tests = len(comparison_vars)

    print(f"  {'Variable':<20} {'Flattening':>15} {'Steepening':>15} {'p':>8} {'Bonf_p':>8} {'Effect':>8}")
    print("  " + "-" * 80)

    for var in comparison_vars:
        flat_vals = flat[var].dropna().values
        steep_vals = steep[var].dropna().values

        # normality
        try:
            _, p_norm_f = stats.shapiro(flat_vals[:5000])
            _, p_norm_s = stats.shapiro(steep_vals[:5000])
        except:
            p_norm_f, p_norm_s = 0, 0

        if p_norm_f > 0.05 and p_norm_s > 0.05:
            stat, p_val = stats.ttest_ind(flat_vals, steep_vals)
            test_name = "t-test"
        else:
            stat, p_val = stats.mannwhitneyu(flat_vals, steep_vals, alternative="two-sided")
            test_name = "Mann-Whitney"

        bonf_p = min(p_val * n_tests, 1.0)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(flat_vals) - 1) * np.std(flat_vals, ddof=1)**2 +
             (len(steep_vals) - 1) * np.std(steep_vals, ddof=1)**2) /
            (len(flat_vals) + len(steep_vals) - 2)
        )
        d = (np.mean(flat_vals) - np.mean(steep_vals)) / pooled_std if pooled_std > 0 else 0

        label = var_labels.get(var, var)
        sig = "*" if p_val < 0.05 else ""
        bsig = "**" if bonf_p < 0.05 else ""
        print(f"  {label:<20} {tables.fmt_mean_sd(flat_vals):>15} {tables.fmt_mean_sd(steep_vals):>15} "
              f"{p_val:>7.4f}{sig} {bonf_p:>7.4f}{bsig} {d:>7.3f}")

        comp_rows.append({
            "Variable": label,
            "Flattening (n={})".format(len(flat)): tables.fmt_mean_sd(flat_vals),
            "Steepening (n={})".format(len(steep)): tables.fmt_mean_sd(steep_vals),
            "Test": test_name,
            "p": p_val,
            "Bonferroni p": bonf_p,
            "Cohen's d": d,
        })

    # ── 5.3 HCDA specifically (Prof. Bu's finding) ────────────────────
    print("\n--- 5.3 HCDA comparison (Prof. Bu reported p=0.034) ---")
    t_hcda, p_hcda = stats.ttest_ind(flat["HCDA"], steep["HCDA"])
    u_hcda, p_hcda_mw = stats.mannwhitneyu(flat["HCDA"], steep["HCDA"], alternative="two-sided")
    print(f"  Flattening HCDA: {flat['HCDA'].mean():.4f} ± {flat['HCDA'].std():.4f}")
    print(f"  Steepening HCDA: {steep['HCDA'].mean():.4f} ± {steep['HCDA'].std():.4f}")
    print(f"  t-test: t={t_hcda:.3f}, p={p_hcda:.4f}")
    print(f"  Mann-Whitney: U={u_hcda:.0f}, p={p_hcda_mw:.4f}")

    # ── 5.4 Eye distribution in steepening vs flattening ───────────────
    print("\n--- 5.4 Eye distribution ---")
    ct = pd.crosstab(df["eye"], df["steepening"])
    chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
    print(f"  Steepening by eye:")
    print(f"    OD: {(steep['eye']=='OD').sum()}/{len(steep)} ({100*(steep['eye']=='OD').sum()/len(steep):.1f}%)")
    print(f"    OS: {(steep['eye']=='OS').sum()}/{len(steep)} ({100*(steep['eye']=='OS').sum()/len(steep):.1f}%)")
    print(f"  Chi-squared: χ²={chi2:.3f}, p={p_chi:.4f}")

    # ── 5.5 Logistic regression ────────────────────────────────────────
    print("\n--- 5.5 Logistic regression (predicting steepening) ---")
    X_log = df[config.ALL_PREOP_PARAMS + ["age", "eye_binary"]].values
    y_log = df["steepening"].values
    feature_names = config.ALL_PREOP_PARAMS + ["age", "eye_binary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    lr = LogisticRegression(max_iter=1000, penalty="l2", C=1.0, random_state=config.RANDOM_SEED)
    cv_scores = cross_val_score(lr, X_scaled, y_log, cv=10, scoring="roc_auc")
    print(f"  10-fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on full data for coefficients
    lr.fit(X_scaled, y_log)
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": lr.coef_[0],
        "abs_coefficient": np.abs(lr.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)
    print(f"\n  Top 5 logistic regression coefficients:")
    for _, row in coef_df.head(5).iterrows():
        print(f"    {row['feature']}: {row['coefficient']:.4f}")

    # ── 5.6 Figures ────────────────────────────────────────────────────
    print("\n--- 5.6 Figures ---")

    # Fig 15: Box plots for top discriminating variables with p-values
    top_vars_raw = coef_df.head(6)["feature"].tolist()
    top_vars = [v for v in top_vars_raw if v in df.columns]

    # Get p-values for each variable
    p_lookup = {row["Variable"]: row["p"] for row in comp_rows}

    n_plots = len(top_vars)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.8 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, var in enumerate(top_vars):
        ax = axes[i]
        label = var_labels.get(var, var)
        plot_df = df[[var, "steepening"]].copy()
        plot_df["Group"] = plot_df["steepening"].map({0: f"Flattening\n(n={len(flat)})", 1: f"Steepening\n(n={len(steep)})"})
        sns.boxplot(data=plot_df, x="Group", y=var, ax=ax,
                    palette=[config.COLOR_OD, config.COLOR_OS], width=0.5, linewidth=0.8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlabel("")

        # Add p-value annotation at top of panel
        p_val = p_lookup.get(label, None)
        if p_val is not None:
            p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
            ax.set_title(p_text, fontsize=8, fontstyle="italic", pad=4)

        sns.despine(ax=ax)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Steepening vs Flattening: Preoperative Parameters",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plotting.save_figure(fig, "fig15_steepening_comparison")
    print("  Saved fig15_steepening_comparison")

    # ── 5.7 Tables ─────────────────────────────────────────────────────
    print("\n--- 5.7 Tables ---")
    comp_df = pd.DataFrame(comp_rows).set_index("Variable")
    tables.save_table(comp_df, "table11_steepening_comparison")
    print("  Saved table11_steepening_comparison")

    # Save logistic regression results
    tables.save_table(coef_df.set_index("feature"), "table11b_logistic_regression")
    print("  Saved table11b_logistic_regression")

    print("\nAnalysis 5 complete.")
    return {
        "comparison": comp_rows,
        "hcda_p": p_hcda,
        "eye_chi2_p": p_chi,
        "logistic_auc": cv_scores.mean(),
    }
