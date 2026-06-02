"""Analysis 2: Association analysis — correlations, partial correlations, mixed-effects, MANOVA, VIF."""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.multivariate.manova import MANOVA

from . import config, utils, plotting, tables, data_loader


def run(df):
    """Run all association analyses and save outputs."""
    print("=" * 60)
    print("ANALYSIS 2: Association Analysis")
    print("=" * 60)

    od, os_ = data_loader.get_subsets(df)
    targets = ["CSIA_mag", "J0", "J45"]
    predictors = config.ALL_PREOP_PARAMS + ["age"]
    results = {}

    # ── 2.1 Bivariate Pearson correlations ─────────────────────────────
    print("\n--- 2.1 Bivariate Pearson correlations ---")
    corr_r = {t: {} for t in targets}
    corr_p = {t: {} for t in targets}

    for pred in predictors:
        mask = df[pred].notna()
        for target in targets:
            r, p = stats.pearsonr(df.loc[mask, pred], df.loc[mask, target])
            corr_r[target][pred] = r
            corr_p[target][pred] = p

    # FDR correction per target
    fdr_p = {t: {} for t in targets}
    for target in targets:
        pvals = [corr_p[target][pred] for pred in predictors]
        adj_pvals = utils.fdr_correction(pvals)
        for i, pred in enumerate(predictors):
            fdr_p[target][pred] = adj_pvals[i]

    # Print significant
    print(f"  {'Variable':<12} {'Mag_r':>8} {'Mag_p':>8} {'J0_r':>8} {'J0_p':>8} {'J45_r':>8} {'J45_p':>8}")
    for pred in predictors:
        line = f"  {pred:<12}"
        for t in targets:
            r = corr_r[t][pred]
            p = corr_p[t][pred]
            sig = "*" if p < 0.05 else ""
            line += f" {r:>7.3f}{sig} {p:>8.4f}"
        print(line)

    results["bivariate_correlations"] = {"r": corr_r, "p": corr_p, "fdr_p": fdr_p}

    # ── 2.2 Partial correlations (Prof. Bu's approach) ─────────────────
    print("\n--- 2.2 Partial correlations (adjusting for Age, CCT, bIOP) ---")
    cov_cols_bu = ["age", "CCT", "bIOP"]
    pcorr_bu = {t: {} for t in targets}
    pcorr_bu_p = {t: {} for t in targets}
    pcorr_bu_ci = {t: {} for t in targets}

    corvis_preds = config.CORVIS_PARAMS + config.BIOMETRIC_PARAMS
    for pred in corvis_preds:
        covs = df[cov_cols_bu].values
        for target in targets:
            r, p, ci_lo, ci_hi = utils.partial_correlation(df[pred].values, df[target].values, covs)
            pcorr_bu[target][pred] = r
            pcorr_bu_p[target][pred] = p
            pcorr_bu_ci[target][pred] = (ci_lo, ci_hi)

    print(f"  {'Variable':<12} {'Mag_r':>8} {'Mag_p':>8} {'J0_r':>8} {'J0_p':>8} {'J45_r':>8} {'J45_p':>8}")
    for pred in corvis_preds:
        line = f"  {pred:<12}"
        for t in targets:
            r = pcorr_bu[t][pred]
            p = pcorr_bu_p[t][pred]
            sig = "*" if p < 0.05 else ""
            line += f" {r:>7.3f}{sig} {p:>8.4f}"
        print(line)

    results["partial_corr_bu"] = {"r": pcorr_bu, "p": pcorr_bu_p, "ci": pcorr_bu_ci}

    # ── 2.3 Partial correlations (+ Eye) ───────────────────────────────
    print("\n--- 2.3 Partial correlations (adjusting for Age, CCT, bIOP, Eye) ---")
    cov_cols_eye = ["age", "CCT", "bIOP", "eye_binary"]
    pcorr_eye = {t: {} for t in targets}
    pcorr_eye_p = {t: {} for t in targets}
    pcorr_eye_ci = {t: {} for t in targets}

    for pred in corvis_preds:
        covs = df[cov_cols_eye].values
        for target in targets:
            r, p, ci_lo, ci_hi = utils.partial_correlation(df[pred].values, df[target].values, covs)
            pcorr_eye[target][pred] = r
            pcorr_eye_p[target][pred] = p
            pcorr_eye_ci[target][pred] = (ci_lo, ci_hi)

    # Print only those that are significant in either
    print("  Significant (p<0.05) after controlling for Eye:")
    for pred in corvis_preds:
        for t in targets:
            if pcorr_eye_p[t][pred] < 0.05:
                print(f"    {pred} -> {t}: r={pcorr_eye[t][pred]:.3f}, p={pcorr_eye_p[t][pred]:.4f}")

    results["partial_corr_eye"] = {"r": pcorr_eye, "p": pcorr_eye_p, "ci": pcorr_eye_ci}

    # ── 2.4 Eye as predictor ───────────────────────────────────────────
    print("\n--- 2.4 Eye laterality as predictor (partial corr adjusting Age, CCT, bIOP) ---")
    for target in targets:
        r, p, ci_lo, ci_hi = utils.partial_correlation(
            df["eye_binary"].values, df[target].values, df[cov_cols_bu].values
        )
        print(f"  Eye -> {target}: r={r:.3f}, p={p:.4f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    # ── 2.5 Linear mixed-effects models ────────────────────────────────
    print("\n--- 2.5 Linear mixed-effects models ---")
    lmm_results = []

    for target in targets:
        print(f"\n  Target: {target}")
        for pred in corvis_preds + ["age", "eye_binary"]:
            try:
                formula_df = df[["patient_id", target, pred]].dropna()
                model = MixedLM(
                    formula_df[target].values,
                    sm.add_constant(formula_df[pred].values),
                    groups=formula_df["patient_id"].values,
                )
                fit = model.fit(reml=True)
                coef = fit.params[1]
                se = fit.bse[1]
                pval = fit.pvalues[1]
                sig = "*" if pval < 0.05 else ""
                lmm_results.append({
                    "target": target, "predictor": pred,
                    "coefficient": coef, "SE": se, "p": pval,
                })
                if pval < 0.05:
                    print(f"    {pred}: coef={coef:.4f}, SE={se:.4f}, p={pval:.4f}{sig}")
            except Exception as e:
                lmm_results.append({
                    "target": target, "predictor": pred,
                    "coefficient": np.nan, "SE": np.nan, "p": np.nan,
                })

    results["mixed_effects"] = lmm_results

    # ── 2.6 Stepwise multivariate regression ───────────────────────────
    print("\n--- 2.6 Stepwise multivariate regression (AIC-based) ---")
    step_results = {}

    all_step_preds = corvis_preds + ["age", "eye_binary"]
    X_step = df[all_step_preds].values
    feature_names = all_step_preds

    for target in targets:
        y = df[target].values
        selected, aic, model_fit = utils.aic_stepwise(X_step, y, feature_names, verbose=False)
        print(f"\n  {target}: selected={selected}")
        print(f"    AIC={aic:.2f}, R²={model_fit.rsquared:.4f}, Adj-R²={model_fit.rsquared_adj:.4f}")
        if selected:
            print(f"    Coefficients:")
            for i, name in enumerate(["const"] + selected):
                print(f"      {name}: {model_fit.params[i]:.4f} (p={model_fit.pvalues[i]:.4f})")
        step_results[target] = {
            "selected_features": selected,
            "aic": aic,
            "r_squared": model_fit.rsquared,
            "adj_r_squared": model_fit.rsquared_adj,
            "model_summary": model_fit.summary().as_text(),
            "coefficients": dict(zip(["const"] + selected, model_fit.params)),
            "p_values": dict(zip(["const"] + selected, model_fit.pvalues)),
        }

    results["stepwise"] = step_results

    # ── 2.7 MANOVA ─────────────────────────────────────────────────────
    print("\n--- 2.7 MANOVA on CSIA vector (J0, J45) ---")
    try:
        manova_df = df[["J0", "J45"] + corvis_preds + ["age", "eye_binary"]].dropna()
        dep_vars = "J0 + J45"
        indep_vars = " + ".join(corvis_preds + ["age", "eye_binary"])
        formula = f"{dep_vars} ~ {indep_vars}"
        manova = MANOVA.from_formula(formula, data=manova_df)
        mv_test = manova.mv_test()
        print(mv_test.summary())
        results["manova"] = str(mv_test.summary())

        # Save full MANOVA output
        with open(config.OUTPUT_RESULTS / "manova_full_output.txt", "w") as f:
            f.write(str(mv_test.summary()))
    except Exception as e:
        print(f"  MANOVA failed: {e}")
        results["manova"] = f"Failed: {e}"

    # ── 2.8 VIF analysis ──────────────────────────────────────────────
    print("\n--- 2.8 VIF analysis ---")
    X_vif = df[config.CORVIS_PARAMS].dropna().values
    vif_df = utils.compute_vif(X_vif, config.CORVIS_PARAMS)
    vif_df = vif_df.sort_values("VIF", ascending=False)
    print(vif_df.to_string(index=False))
    results["vif"] = vif_df

    # ── 2.9 Figures ────────────────────────────────────────────────────
    print("\n--- 2.9 Generating figures ---")

    # Fig 05: Correlation heatmap
    corr_matrix = pd.DataFrame(corr_r).loc[predictors]
    p_matrix = pd.DataFrame(corr_p).loc[predictors]
    corr_matrix.columns = ["Magnitude", "J0", "J45"]
    p_matrix.columns = ["Magnitude", "J0", "J45"]
    plotting.plot_correlation_heatmap(corr_matrix, p_matrix, filename="fig05_correlation_heatmap",
                                      figsize=(5, 10))
    print("  Saved fig05_correlation_heatmap")

    # Fig 06: Forest plot of partial correlations
    forest_data = []
    for pred in corvis_preds:
        for target, color in [("CSIA_mag", config.COLOR_OD), ("J0", config.COLOR_OS), ("J45", config.COLOR_ACCENT)]:
            ci = pcorr_bu_ci[target][pred]
            forest_data.append({
                "label": f"{pred} → {target}",
                "r": pcorr_bu[target][pred],
                "ci_lo": ci[0], "ci_hi": ci[1],
                "color": color,
            })
    # only show significant or near-significant
    forest_sig = [d for d in forest_data if abs(d["r"]) > 0.10]
    if forest_sig:
        plotting.plot_forest(forest_sig, filename="fig06_partial_correlation_forest",
                             title="Partial correlations (adj. Age, CCT, bIOP)", figsize=(5, max(4, len(forest_sig) * 0.35)))
    print("  Saved fig06_partial_correlation_forest")

    # Fig 07: VIF bar plot
    plotting.plot_vif(vif_df, filename="fig07_vif_barplot")
    print("  Saved fig07_vif_barplot")

    # ── 2.10 Tables ───────────────────────────────────────────────────
    print("\n--- 2.10 Generating tables ---")

    # Table 02: Bivariate correlations
    corr_table = tables.build_correlation_table(predictors, corr_r, corr_p, fdr_p)
    tables.save_table(corr_table, "table02_bivariate_correlations")
    print("  Saved table02_bivariate_correlations")

    # Table 03: Partial correlations
    pcorr_table_rows = []
    for pred in corvis_preds:
        row = {"Variable": pred}
        for target in targets:
            for label, pcorr, pp in [("Bu", pcorr_bu, pcorr_bu_p), ("Eye", pcorr_eye, pcorr_eye_p)]:
                r = pcorr[target][pred]
                p = pp[target][pred]
                stars = "*" if p < 0.05 else ""
                row[f"{target}_{label}_r"] = f"{r:.3f}{stars}"
                row[f"{target}_{label}_p"] = tables.fmt_p(p)
        pcorr_table_rows.append(row)
    pcorr_table = pd.DataFrame(pcorr_table_rows).set_index("Variable")
    tables.save_table(pcorr_table, "table03_partial_correlations")
    print("  Saved table03_partial_correlations")

    # Table 04: Mixed-effects results
    lmm_df = pd.DataFrame(lmm_results)
    tables.save_table(lmm_df.set_index(["target", "predictor"]), "table04_mixed_effects")
    print("  Saved table04_mixed_effects")

    # Table 05: Stepwise regression
    step_rows = []
    for target in targets:
        sr = step_results[target]
        step_rows.append({
            "Target": target,
            "Selected Features": ", ".join(sr["selected_features"]) if sr["selected_features"] else "None",
            "R²": f"{sr['r_squared']:.4f}",
            "Adj-R²": f"{sr['adj_r_squared']:.4f}",
            "AIC": f"{sr['aic']:.2f}",
        })
    step_df = pd.DataFrame(step_rows).set_index("Target")
    tables.save_table(step_df, "table05_stepwise_regression")
    # Save full model summaries
    for target in targets:
        with open(config.OUTPUT_RESULTS / f"stepwise_{target}_summary.txt", "w") as f:
            f.write(step_results[target]["model_summary"])
    print("  Saved table05_stepwise_regression")

    # Table 06: VIF
    tables.save_table(vif_df.set_index("feature"), "table06_vif")
    print("  Saved table06_vif")

    print("\nAnalysis 2 complete.")
    return results
