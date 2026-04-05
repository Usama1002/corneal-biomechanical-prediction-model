"""Analysis 6: Comprehensive ablation studies."""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error

from . import config, vector_math, utils, plotting, tables, data_loader


def run(df):
    """Run all ablation experiments."""
    print("=" * 60)
    print("ANALYSIS 6: Comprehensive Ablation Studies")
    print("=" * 60)

    results = {}
    X_demo, names_demo = data_loader.get_feature_matrix(df, "biomech_demo")
    y_mag = df["CSIA_mag"].values
    y_j0 = df["J0"].values
    y_j45 = df["J45"].values
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=config.RANDOM_SEED)

    def quick_cv(X, y, model=None):
        if model is None:
            model = Pipeline([("s", StandardScaler()),
                              ("m", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5,
                                                  max_iter=10000, random_state=config.RANDOM_SEED))])
        r2 = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        mae = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        return {"R2_mean": r2.mean(), "R2_std": r2.std(),
                "MAE_mean": -mae.mean(), "MAE_std": mae.std()}

    # ── 6.1 Feature category ablation ──────────────────────────────────
    print("\n--- 6.1 Feature category ablation ---")
    categories = {
        "Applanation": ["AL1", "AV1", "AL2", "AV2"],
        "Concavity": ["PD", "HCR", "HCDA"],
        "Pressure": ["IOPnc", "bIOP"],
        "Thickness": ["CCT", "PCT135"],
        "Stiffness indices": ["SPA1", "DA_Ratio", "IR", "CBI", "SSI", "CBiF"],
        "Structural": ["ARTh"],
        "Biometric": ["AL", "WTW"],
        "Demographics": ["age", "sex_binary", "eye_binary"],
        "All biomech": config.BIOMECH_FEATURES,
        "All biomech + demo": config.BIOMECH_DEMO_FEATURES,
    }

    cat_results = []
    for cat_name, cols in categories.items():
        cols_available = [c for c in cols if c in df.columns]
        if not cols_available:
            continue
        X_cat = df[cols_available].values.astype(float)
        for target_name, y in [("Magnitude", y_mag), ("J0", y_j0), ("J45", y_j45)]:
            res = quick_cv(X_cat, y)
            res.update({"category": cat_name, "n_features": len(cols_available), "target": target_name})
            cat_results.append(res)
            if target_name == "Magnitude":
                print(f"  {cat_name:<25} ({len(cols_available)} feats) -> Mag R²={res['R2_mean']:.3f}±{res['R2_std']:.3f}")

    results["category_ablation"] = pd.DataFrame(cat_results)

    # ── 6.2 Leave-one-feature-out ──────────────────────────────────────
    print("\n--- 6.2 Leave-one-feature-out (magnitude target) ---")
    baseline = quick_cv(X_demo, y_mag)
    print(f"  Baseline (all {len(names_demo)} features): R²={baseline['R2_mean']:.3f}")

    lofo_results = []
    for i, fname in enumerate(names_demo):
        X_drop = np.delete(X_demo, i, axis=1)
        res = quick_cv(X_drop, y_mag)
        delta_r2 = res["R2_mean"] - baseline["R2_mean"]
        lofo_results.append({
            "dropped_feature": fname,
            "R2_without": res["R2_mean"],
            "delta_R2": delta_r2,  # positive = removing helps, negative = removing hurts
        })

    lofo_df = pd.DataFrame(lofo_results).sort_values("delta_R2", ascending=True)
    print("  Most important (removal hurts most):")
    for _, row in lofo_df.head(5).iterrows():
        print(f"    {row['dropped_feature']}: delta R²={row['delta_R2']:.4f}")
    print("  Least important (removal helps most):")
    for _, row in lofo_df.tail(5).iterrows():
        print(f"    {row['dropped_feature']}: delta R²={row['delta_R2']:.4f}")
    results["lofo"] = lofo_df

    # ── 6.3 Cumulative feature addition ────────────────────────────────
    print("\n--- 6.3 Cumulative feature addition (by SHAP importance) ---")
    # Order by SHAP importance for magnitude (from analysis 3)
    shap_order_mag = ["age", "HCDA", "AL2", "PCT135", "AL1", "ARTh", "eye_binary",
                      "CCT", "IR", "SPA1", "AV2", "CBiF", "WTW", "bIOP", "SSI",
                      "AL", "DA_Ratio", "AV1", "IOPnc", "HCR", "sex_binary", "CBI", "PD"]
    # Filter to available
    shap_order_mag = [f for f in shap_order_mag if f in names_demo]

    cum_results = []
    for n in range(1, len(shap_order_mag) + 1):
        feats = shap_order_mag[:n]
        feat_idx = [names_demo.index(f) for f in feats]
        X_cum = X_demo[:, feat_idx]
        res = quick_cv(X_cum, y_mag)
        cum_results.append({"n_features": n, "features": ", ".join(feats), **res})
        if n <= 5 or n == len(shap_order_mag):
            print(f"  Top {n}: R²={res['R2_mean']:.3f} ({feats[-1]})")

    results["cumulative"] = pd.DataFrame(cum_results)

    # ── 6.4 Interaction effects ────────────────────────────────────────
    print("\n--- 6.4 Interaction effects ---")
    interaction_results = []

    # Test key interactions
    interactions = [
        ("age", "eye_binary", "Age x Eye"),
        ("age", "IR", "Age x IR"),
        ("age", "DA_Ratio", "Age x DA Ratio"),
        ("eye_binary", "HCDA", "Eye x HCDA"),
        ("eye_binary", "IR", "Eye x IR"),
        ("PCT135", "CCT", "PCT135/CCT ratio"),
    ]

    for f1, f2, label in interactions:
        if f1 not in df.columns or f2 not in df.columns:
            continue
        # baseline: f1 + f2
        X_base = df[[f1, f2]].values.astype(float)
        # with interaction
        X_inter = np.column_stack([X_base, df[f1].values * df[f2].values])

        for target_name, y in [("Magnitude", y_mag), ("J0", y_j0), ("J45", y_j45)]:
            r_base = quick_cv(X_base, y)
            r_inter = quick_cv(X_inter, y)
            delta = r_inter["R2_mean"] - r_base["R2_mean"]
            interaction_results.append({
                "interaction": label, "target": target_name,
                "R2_base": r_base["R2_mean"], "R2_interaction": r_inter["R2_mean"],
                "delta_R2": delta,
            })
            if target_name == "Magnitude":
                print(f"  {label:<20} Mag: base R²={r_base['R2_mean']:.3f}, +inter={r_inter['R2_mean']:.3f}, delta={delta:.4f}")

    results["interactions"] = pd.DataFrame(interaction_results)

    # ── 6.5 Polynomial / nonlinear features ────────────────────────────
    print("\n--- 6.5 Polynomial features (degree 2) ---")
    top5 = ["age", "eye_binary", "IR", "HCDA", "DA_Ratio"]
    top5_idx = [names_demo.index(f) for f in top5 if f in names_demo]
    X_top5 = X_demo[:, top5_idx]

    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X_top5)
    poly_names = poly.get_feature_names_out([names_demo[i] for i in top5_idx])

    for target_name, y in [("Magnitude", y_mag), ("J0", y_j0), ("J45", y_j45)]:
        r_linear = quick_cv(X_top5, y)
        r_poly = quick_cv(X_poly, y)
        print(f"  {target_name}: linear R²={r_linear['R2_mean']:.3f}, poly R²={r_poly['R2_mean']:.3f}, "
              f"delta={r_poly['R2_mean'] - r_linear['R2_mean']:.4f}")

    results["polynomial"] = {"n_poly_features": X_poly.shape[1], "feature_names": list(poly_names)}

    # ── 6.6 Learning curve ─────────────────────────────────────────────
    print("\n--- 6.6 Learning curve (does more data help?) ---")
    pipe = Pipeline([("s", StandardScaler()),
                     ("m", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5,
                                         max_iter=10000, random_state=config.RANDOM_SEED))])
    train_sizes = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    train_sizes_abs, train_scores, test_scores = learning_curve(
        pipe, X_demo, y_mag, train_sizes=train_sizes, cv=10,
        scoring="r2", n_jobs=-1, random_state=config.RANDOM_SEED,
    )
    lc_results = []
    for i, ts in enumerate(train_sizes_abs):
        lc_results.append({
            "train_size": int(ts),
            "train_R2_mean": train_scores[i].mean(),
            "train_R2_std": train_scores[i].std(),
            "test_R2_mean": test_scores[i].mean(),
            "test_R2_std": test_scores[i].std(),
        })
        print(f"  n={int(ts):>4}: train R²={train_scores[i].mean():.3f}, test R²={test_scores[i].mean():.3f}")

    results["learning_curve"] = pd.DataFrame(lc_results)

    # ── 6.7 Sex-stratified analysis ────────────────────────────────────
    print("\n--- 6.7 Sex-stratified analysis ---")
    for sex_label, sex_code in [("Male", "M"), ("Female", "F")]:
        df_sex = df[df["sex"] == sex_code]
        X_sex, _ = data_loader.get_feature_matrix(df_sex, "biomech_demo")
        y_sex = df_sex["CSIA_mag"].values
        cv_sex = RepeatedKFold(n_splits=5, n_repeats=3, random_state=config.RANDOM_SEED)
        res = quick_cv(X_sex, y_sex)
        print(f"  {sex_label} (n={len(df_sex)}): Mag R²={res['R2_mean']:.3f}±{res['R2_std']:.3f}, MAE={res['MAE_mean']:.3f}")

        # centroid
        c = vector_math.compute_centroid(df_sex["CSIA_mag"], df_sex["CSIA_meridian"])
        print(f"    Centroid: {c['centroid_mag']:.3f} D @ {c['centroid_meridian']:.1f}°")

    # Sex comparison for CSIA
    male = df[df["sex"] == "M"]
    female = df[df["sex"] == "F"]
    for var in ["CSIA_mag", "CSIA_meridian", "J0", "J45"]:
        t, p = stats.ttest_ind(male[var], female[var])
        print(f"  {var}: M={male[var].mean():.3f}±{male[var].std():.3f}, "
              f"F={female[var].mean():.3f}±{female[var].std():.3f}, p={p:.4f}")

    # ── 6.8 Outlier sensitivity ────────────────────────────────────────
    print("\n--- 6.8 Outlier sensitivity (exclude CSIA > 2 SD) ---")
    mag_mean = df["CSIA_mag"].mean()
    mag_sd = df["CSIA_mag"].std()
    threshold = mag_mean + 2 * mag_sd
    df_no_outliers = df[df["CSIA_mag"] <= threshold]
    n_removed = len(df) - len(df_no_outliers)
    print(f"  Threshold: {threshold:.3f} D, removed {n_removed} eyes ({100*n_removed/len(df):.1f}%)")

    X_no, _ = data_loader.get_feature_matrix(df_no_outliers, "biomech_demo")
    y_no = df_no_outliers["CSIA_mag"].values
    res_no = quick_cv(X_no, y_no)
    print(f"  Without outliers: R²={res_no['R2_mean']:.3f}±{res_no['R2_std']:.3f}, MAE={res_no['MAE_mean']:.3f}")
    res_full = quick_cv(X_demo, y_mag)
    print(f"  With outliers:    R²={res_full['R2_mean']:.3f}±{res_full['R2_std']:.3f}, MAE={res_full['MAE_mean']:.3f}")

    c_no = vector_math.compute_centroid(df_no_outliers["CSIA_mag"], df_no_outliers["CSIA_meridian"])
    print(f"  Centroid without outliers: {c_no['centroid_mag']:.3f} D @ {c_no['centroid_meridian']:.1f}°")

    # ── 6.9 Steepening threshold sensitivity ───────────────────────────
    print("\n--- 6.9 Steepening threshold sensitivity ---")
    for thresh in [75, 80, 85, 90, 95, 100, 105]:
        n_steep = (df["CSIA_meridian"] > thresh).sum()
        pct = 100 * n_steep / len(df)
        print(f"  Meridian > {thresh}°: n={n_steep} ({pct:.1f}%)")

    # ── 6.10 Leave-one-out centroid (unbiased centroid prediction) ─────
    print("\n--- 6.10 Leave-one-out centroid prediction ---")
    loo_errors_overall = []
    loo_errors_eye = []
    loo_errors_eye_age = []

    actual_j0 = df["J0"].values
    actual_j45 = df["J45"].values

    for i in range(len(df)):
        # LOO overall centroid
        mask = np.ones(len(df), dtype=bool)
        mask[i] = False
        c = vector_math.compute_centroid(df.loc[mask, "CSIA_mag"], df.loc[mask, "CSIA_meridian"])
        pj0, pj45 = vector_math.decompose_to_j0_j45(c["centroid_mag"], c["centroid_meridian"])
        loo_errors_overall.append(vector_math.vector_error(pj0, pj45, actual_j0[i], actual_j45[i]))

        # LOO eye-specific centroid
        eye = df.iloc[i]["eye"]
        eye_mask = mask & (df["eye"] == eye).values
        if eye_mask.sum() > 2:
            c_eye = vector_math.compute_centroid(df.loc[eye_mask, "CSIA_mag"], df.loc[eye_mask, "CSIA_meridian"])
            pj0_e, pj45_e = vector_math.decompose_to_j0_j45(c_eye["centroid_mag"], c_eye["centroid_meridian"])
            loo_errors_eye.append(vector_math.vector_error(pj0_e, pj45_e, actual_j0[i], actual_j45[i]))
        else:
            loo_errors_eye.append(loo_errors_overall[-1])

        # LOO eye x age centroid
        age_t = df.iloc[i]["age_tertile"]
        ea_mask = mask & (df["eye"] == eye).values & (df["age_tertile"] == age_t).values
        if ea_mask.sum() > 5:
            c_ea = vector_math.compute_centroid(df.loc[ea_mask, "CSIA_mag"], df.loc[ea_mask, "CSIA_meridian"])
            pj0_ea, pj45_ea = vector_math.decompose_to_j0_j45(c_ea["centroid_mag"], c_ea["centroid_meridian"])
            loo_errors_eye_age.append(vector_math.vector_error(pj0_ea, pj45_ea, actual_j0[i], actual_j45[i]))
        else:
            loo_errors_eye_age.append(loo_errors_eye[-1])

    print(f"  LOO overall centroid:   mean VE = {np.mean(loo_errors_overall):.4f} ± {np.std(loo_errors_overall):.4f}")
    print(f"  LOO eye centroid:       mean VE = {np.mean(loo_errors_eye):.4f} ± {np.std(loo_errors_eye):.4f}")
    print(f"  LOO eye x age centroid: mean VE = {np.mean(loo_errors_eye_age):.4f} ± {np.std(loo_errors_eye_age):.4f}")

    results["loo_centroids"] = {
        "overall": {"mean": np.mean(loo_errors_overall), "std": np.std(loo_errors_overall)},
        "eye": {"mean": np.mean(loo_errors_eye), "std": np.std(loo_errors_eye)},
        "eye_age": {"mean": np.mean(loo_errors_eye_age), "std": np.std(loo_errors_eye_age)},
    }

    # ── 6.11 Multiple error metrics comparison ─────────────────────────
    print("\n--- 6.11 Error metric comparison for centroid strategies ---")
    # For eye-specific centroid, compute all three error types
    mag_errors = []
    ang_errors = []
    vec_errors = []

    for i in range(len(df)):
        eye = df.iloc[i]["eye"]
        eye_cent = vector_math.compute_centroid(
            df[df["eye"] == eye]["CSIA_mag"],
            df[df["eye"] == eye]["CSIA_meridian"],
        )
        # magnitude error
        mag_errors.append(abs(df.iloc[i]["CSIA_mag"] - eye_cent["centroid_mag"]))
        # angular error
        ang_errors.append(vector_math.angular_error(eye_cent["centroid_meridian"], df.iloc[i]["CSIA_meridian"]))
        # vector error
        pj0, pj45 = vector_math.decompose_to_j0_j45(eye_cent["centroid_mag"], eye_cent["centroid_meridian"])
        vec_errors.append(vector_math.vector_error(pj0, pj45, actual_j0[i], actual_j45[i]))

    print(f"  Eye-specific centroid:")
    print(f"    Magnitude error: {np.mean(mag_errors):.3f} ± {np.std(mag_errors):.3f} D")
    print(f"    Angular error:   {np.mean(ang_errors):.1f} ± {np.std(ang_errors):.1f}°")
    print(f"    Vector error:    {np.mean(vec_errors):.3f} ± {np.std(vec_errors):.3f} D")
    print(f"    % within 0.25D: {100*np.mean(np.array(mag_errors) <= 0.25):.1f}%")
    print(f"    % within 0.50D: {100*np.mean(np.array(mag_errors) <= 0.50):.1f}%")
    print(f"    % within 15°:   {100*np.mean(np.array(ang_errors) <= 15):.1f}%")
    print(f"    % within 30°:   {100*np.mean(np.array(ang_errors) <= 30):.1f}%")

    # ── 6.12 Bilateral patient sensitivity ─────────────────────────────
    print("\n--- 6.12 Bilateral patient sensitivity ---")
    # Compare results including vs excluding bilateral patients
    multi_patients = data_loader.get_patient_groups(df)
    multi_names = multi_patients["patient_id"].tolist() if len(multi_patients) > 0 else []
    df_single = df[~df["patient_id"].isin(multi_names)]
    print(f"  All eyes: n={len(df)}")
    print(f"  Single-entry only: n={len(df_single)} (removed {len(df)-len(df_single)} from {len(multi_names)} patients)")

    X_single, _ = data_loader.get_feature_matrix(df_single, "biomech_demo")
    y_single = df_single["CSIA_mag"].values
    res_single = quick_cv(X_single, y_single)
    print(f"  Single-entry R²={res_single['R2_mean']:.3f}±{res_single['R2_std']:.3f}")
    print(f"  All-eyes R²={res_full['R2_mean']:.3f}±{res_full['R2_std']:.3f}")

    c_single = vector_math.compute_centroid(df_single["CSIA_mag"], df_single["CSIA_meridian"])
    print(f"  Centroid (single only): {c_single['centroid_mag']:.3f} D @ {c_single['centroid_meridian']:.1f}°")

    # OD vs OS still significant in single-entry subset?
    od_s = df_single[df_single["eye"] == "OD"]
    os_s = df_single[df_single["eye"] == "OS"]
    t_j0, p_j0 = stats.ttest_ind(od_s["J0"], os_s["J0"])
    print(f"  OD vs OS J0 (single only): p={p_j0:.4f} (n_OD={len(od_s)}, n_OS={len(os_s)})")

    # ── Save all results ───────────────────────────────────────────────
    print("\n--- Saving ablation results ---")

    cat_df = results["category_ablation"]
    tables.save_table(cat_df.set_index(["category", "target"]), "table12_category_ablation")
    print("  Saved table12_category_ablation")

    tables.save_table(lofo_df.set_index("dropped_feature"), "table13_leave_one_feature_out")
    print("  Saved table13_leave_one_feature_out")

    cum_df = results["cumulative"]
    tables.save_table(cum_df.set_index("n_features"), "table14_cumulative_features")
    print("  Saved table14_cumulative_features")

    inter_df = results["interactions"]
    tables.save_table(inter_df.set_index(["interaction", "target"]), "table15_interactions")
    print("  Saved table15_interactions")

    lc_df = results["learning_curve"]
    tables.save_table(lc_df.set_index("train_size"), "table16_learning_curve")
    print("  Saved table16_learning_curve")

    # ── Figures ────────────────────────────────────────────────────────
    print("\n--- Generating ablation figures ---")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Fig 16: Learning curve
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.fill_between(lc_df["train_size"],
                     lc_df["train_R2_mean"] - lc_df["train_R2_std"],
                     lc_df["train_R2_mean"] + lc_df["train_R2_std"], alpha=0.2, color=config.COLOR_OD)
    ax.fill_between(lc_df["train_size"],
                     lc_df["test_R2_mean"] - lc_df["test_R2_std"],
                     lc_df["test_R2_mean"] + lc_df["test_R2_std"], alpha=0.2, color=config.COLOR_OS)
    ax.plot(lc_df["train_size"], lc_df["train_R2_mean"], "o-", color=config.COLOR_OD, label="Train", markersize=4)
    ax.plot(lc_df["train_size"], lc_df["test_R2_mean"], "s-", color=config.COLOR_OS, label="Test", markersize=4)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("$R^2$")
    ax.set_title("Learning Curve: Elastic Net (Magnitude)", fontweight="bold", fontsize=10)
    ax.legend(frameon=True, fancybox=True, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    plotting.save_figure(fig, "fig16_learning_curve")
    print("  Saved fig16_learning_curve")

    # Fig 17: Category ablation
    cat_mag = cat_df[cat_df["target"] == "Magnitude"].sort_values("R2_mean", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(cat_mag))
    bars = ax.barh(y_pos, cat_mag["R2_mean"].values,
                    xerr=cat_mag["R2_std"].values, capsize=3,
                    color=config.COLOR_OD, edgecolor="white", height=0.6)
    # Place value labels past the error bar cap
    for i, (bar, val, err) in enumerate(zip(bars, cat_mag["R2_mean"], cat_mag["R2_std"])):
        x_pos = max(val + err, 0) + 0.008
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7.5, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat_mag["category"].values, fontsize=9)
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Cross-validated $R^2$ (Magnitude)", fontsize=10)
    ax.set_title("Feature Category Ablation", fontweight="bold", fontsize=10)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.05)
    sns.despine(ax=ax)
    fig.tight_layout()
    plotting.save_figure(fig, "fig17_category_ablation")
    print("  Saved fig17_category_ablation")

    # Fig 18: Cumulative features
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(cum_df["n_features"], cum_df["R2_mean"], "o-", color=config.COLOR_OD, markersize=4)
    ax.fill_between(cum_df["n_features"],
                     cum_df["R2_mean"] - cum_df["R2_std"],
                     cum_df["R2_mean"] + cum_df["R2_std"], alpha=0.2, color=config.COLOR_OD)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Number of features (added by SHAP rank)")
    ax.set_ylabel("Cross-validated $R^2$")
    ax.set_title("Cumulative Feature Addition (Magnitude)", fontweight="bold", fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    plotting.save_figure(fig, "fig18_cumulative_features")
    print("  Saved fig18_cumulative_features")

    print("\nAnalysis 6 complete.")
    return results
