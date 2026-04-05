"""Analysis 4: Subgroup centroid analysis — eye-specific, age, combined centroids vs clinical baselines."""

import numpy as np
import pandas as pd
from scipy import stats

from . import config, vector_math, utils, plotting, tables, data_loader


def run(df, ml_predictions=None):
    """Run subgroup centroid analysis and strategy comparison."""
    print("=" * 60)
    print("ANALYSIS 4: Subgroup Centroid Analysis")
    print("=" * 60)

    od, os_ = data_loader.get_subsets(df)

    # ── 4.1 Compute all subgroup centroids ─────────────────────────────
    print("\n--- 4.1 Subgroup centroids ---")
    centroids = []

    # Overall
    c = vector_math.compute_centroid(df["CSIA_mag"], df["CSIA_meridian"])
    c["group"] = "Overall"
    centroids.append(c)

    # By eye
    for label, subset in [("OD", od), ("OS", os_)]:
        c = vector_math.compute_centroid(subset["CSIA_mag"], subset["CSIA_meridian"])
        c["group"] = label
        centroids.append(c)

    # By age tertile
    for age_group in df["age_tertile"].cat.categories:
        subset = df[df["age_tertile"] == age_group]
        c = vector_math.compute_centroid(subset["CSIA_mag"], subset["CSIA_meridian"])
        c["group"] = f"Age_{age_group}"
        centroids.append(c)

    # Combined: Eye x Age tertile
    for eye in ["OD", "OS"]:
        for age_group in df["age_tertile"].cat.categories:
            subset = df[(df["eye"] == eye) & (df["age_tertile"] == age_group)]
            if len(subset) < 5:
                continue
            c = vector_math.compute_centroid(subset["CSIA_mag"], subset["CSIA_meridian"])
            c["group"] = f"{eye}_{age_group}"
            centroids.append(c)

    for c in centroids:
        print(f"  {c['group']:<20}: {c['centroid_mag']:.3f} D @ {c['centroid_meridian']:.1f}° (n={c['n']})")

    # ── 4.2 Strategy comparison ────────────────────────────────────────
    print("\n--- 4.2 Strategy comparison (vector error) ---")

    actual_j0, actual_j45 = vector_math.decompose_to_j0_j45(df["CSIA_mag"], df["CSIA_meridian"])
    actual_j0 = actual_j0.values if hasattr(actual_j0, 'values') else actual_j0
    actual_j45 = actual_j45.values if hasattr(actual_j45, 'values') else actual_j45

    strategies = {}

    # Strategy 1: Clinical default (0.25D @ 45°)
    def_j0, def_j45 = vector_math.decompose_to_j0_j45(
        config.CLINICAL_DEFAULT_MAG, config.CLINICAL_DEFAULT_MERIDIAN
    )
    strategies["Clinical default\n(0.25D@45°)"] = vector_math.vector_error(
        def_j0, def_j45, actual_j0, actual_j45
    )

    # Strategy 2: Yin et al. (2025) formula (magnitude only, at population centroid meridian)
    bu_pred = config.BU_FORMULA(df["age"].values, df["IR"].values)
    overall_cent = [c for c in centroids if c["group"] == "Overall"][0]
    bu_j0, bu_j45 = vector_math.decompose_to_j0_j45(bu_pred, overall_cent["centroid_meridian"])
    strategies["Yin et al.\nformula"] = vector_math.vector_error(bu_j0, bu_j45, actual_j0, actual_j45)

    # Strategy 3: Population centroid
    pop_j0, pop_j45 = vector_math.decompose_to_j0_j45(
        overall_cent["centroid_mag"], overall_cent["centroid_meridian"]
    )
    strategies["Population\ncentroid"] = vector_math.vector_error(pop_j0, pop_j45, actual_j0, actual_j45)

    # Strategy 4: Eye-specific centroid
    eye_errors = np.zeros(len(df))
    for idx, row in df.iterrows():
        i = df.index.get_loc(idx)
        eye_cent = [c for c in centroids if c["group"] == row["eye"]][0]
        ej0, ej45 = vector_math.decompose_to_j0_j45(eye_cent["centroid_mag"], eye_cent["centroid_meridian"])
        eye_errors[i] = vector_math.vector_error(ej0, ej45, actual_j0[i], actual_j45[i])
    strategies["Eye-specific\ncentroid"] = eye_errors

    # Strategy 5: Age-tertile centroid
    age_errors = np.zeros(len(df))
    for idx, row in df.iterrows():
        i = df.index.get_loc(idx)
        age_cent = [c for c in centroids if c["group"] == f"Age_{row['age_tertile']}"][0]
        aj0, aj45 = vector_math.decompose_to_j0_j45(age_cent["centroid_mag"], age_cent["centroid_meridian"])
        age_errors[i] = vector_math.vector_error(aj0, aj45, actual_j0[i], actual_j45[i])
    strategies["Age-tertile\ncentroid"] = age_errors

    # Strategy 6: Eye x Age centroid
    combined_errors = np.zeros(len(df))
    for idx, row in df.iterrows():
        i = df.index.get_loc(idx)
        group_key = f"{row['eye']}_{row['age_tertile']}"
        matching = [c for c in centroids if c["group"] == group_key]
        if matching:
            cj0, cj45 = vector_math.decompose_to_j0_j45(matching[0]["centroid_mag"], matching[0]["centroid_meridian"])
        else:
            cj0, cj45 = pop_j0, pop_j45
        combined_errors[i] = vector_math.vector_error(cj0, cj45, actual_j0[i], actual_j45[i])
    strategies["Eye x Age\ncentroid"] = combined_errors

    # Strategy 7: Mean magnitude at eye-specific centroid direction
    mag_mean = df["CSIA_mag"].mean()
    eye_dir_errors = np.zeros(len(df))
    for idx, row in df.iterrows():
        i = df.index.get_loc(idx)
        eye_cent = [c for c in centroids if c["group"] == row["eye"]][0]
        ej0, ej45 = vector_math.decompose_to_j0_j45(mag_mean, eye_cent["centroid_meridian"])
        eye_dir_errors[i] = vector_math.vector_error(ej0, ej45, actual_j0[i], actual_j45[i])
    strategies["Mean mag @\neye direction"] = eye_dir_errors

    # Print comparison
    comp_rows = []
    print(f"\n  {'Strategy':<25} {'Mean VE':>8} {'SD':>8} {'Median':>8}")
    print("  " + "-" * 55)
    for name, errors in strategies.items():
        clean_name = name.replace("\n", " ")
        print(f"  {clean_name:<25} {np.mean(errors):>8.4f} {np.std(errors):>8.4f} {np.median(errors):>8.4f}")
        comp_rows.append({
            "Strategy": clean_name,
            "Mean Vector Error": np.mean(errors),
            "SD": np.std(errors),
            "Median": np.median(errors),
            "IQR": f"{np.percentile(errors, 25):.4f}-{np.percentile(errors, 75):.4f}",
        })

    # ── 4.3 Pairwise Wilcoxon tests ───────────────────────────────────
    print("\n--- 4.3 Pairwise Wilcoxon signed-rank tests ---")
    strategy_names = list(strategies.keys())
    pairwise_results = []
    ref = "Population\ncentroid"
    for name in strategy_names:
        if name == ref:
            continue
        stat, p = stats.wilcoxon(strategies[ref], strategies[name])
        clean = name.replace("\n", " ")
        print(f"  Population centroid vs {clean}: p={p:.4f}")
        pairwise_results.append({
            "Comparison": f"Pop centroid vs {clean}",
            "Wilcoxon_stat": stat, "p_value": p,
        })

    # ── 4.4 Bootstrap CIs for mean vector error ───────────────────────
    print("\n--- 4.4 Bootstrap 95% CIs for mean vector error ---")
    for name, errors in strategies.items():
        ci_lo, ci_hi = utils.bootstrap_ci(errors, np.mean, n_boot=5000)
        clean = name.replace("\n", " ")
        print(f"  {clean:<25}: {np.mean(errors):.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── 4.5 Figures ────────────────────────────────────────────────────
    print("\n--- 4.5 Figures ---")

    # Fig 13: Subgroup centroids on polar plot
    centroid_groups = []
    group_colors = {
        "Overall": config.COLOR_OVERALL, "OD": config.COLOR_OD, "OS": config.COLOR_OS,
    }
    group_markers = {"Overall": "s", "OD": "s", "OS": "s"}
    for c in centroids:
        if c["group"] in ["Overall", "OD", "OS"]:
            j0, j45 = vector_math.decompose_to_j0_j45(c["centroid_mag"], c["centroid_meridian"])
            centroid_groups.append({
                "j0": [j0], "j45": [j45],
                "color": group_colors.get(c["group"], config.COLOR_ACCENT),
                "size": 120, "alpha": 1.0,
                "marker": group_markers.get(c["group"], "s"),
                "label": f'{c["group"]} ({c["centroid_mag"]:.2f} D @ {c["centroid_meridian"]:.0f}\u00b0)',
            })

    # Yin et al. (2025) centroid
    bu_j0, bu_j45 = vector_math.decompose_to_j0_j45(config.BU_CENTROID_MAG, config.BU_CENTROID_MERIDIAN)
    centroid_groups.append({
        "j0": [bu_j0], "j45": [bu_j45],
        "color": "#999999", "size": 120, "alpha": 1.0, "marker": "D",
        "label": f"Yin et al. 2025 ({config.BU_CENTROID_MAG} D @ {config.BU_CENTROID_MERIDIAN}\u00b0)",
    })

    # Age tertile centroids
    age_colors = {"young": "#66c2a5", "middle": "#fc8d62", "old": "#8da0cb"}
    age_labels = {"young": "Young", "middle": "Middle", "old": "Old"}
    for c in centroids:
        if c["group"].startswith("Age_"):
            age_key = c["group"].replace("Age_", "")
            j0, j45 = vector_math.decompose_to_j0_j45(c["centroid_mag"], c["centroid_meridian"])
            centroid_groups.append({
                "j0": [j0], "j45": [j45],
                "color": age_colors.get(age_key, config.COLOR_ACCENT),
                "size": 80, "alpha": 0.9, "marker": "^",
                "label": f'{age_labels.get(age_key, age_key)} ({c["centroid_mag"]:.2f} D @ {c["centroid_meridian"]:.0f}\u00b0)',
            })

    plotting.plot_double_angle_polar(
        centroid_groups, filename="fig13_subgroup_centroids",
        title="Subgroup CSIA Centroids", max_ring=1.0, ring_step=0.25,
        annotate_centroids=False,
    )
    print("  Saved fig13_subgroup_centroids")

    # Fig 14: Strategy comparison box plot
    plotting.plot_strategy_boxplot(strategies, filename="fig14_strategy_comparison", figsize=(10, 5))
    print("  Saved fig14_strategy_comparison")

    # ── 4.6 Tables ─────────────────────────────────────────────────────
    print("\n--- 4.6 Tables ---")
    cent_df = tables.build_centroid_table(centroids)
    tables.save_table(cent_df.set_index("group"), "table09_subgroup_centroids")
    print("  Saved table09_subgroup_centroids")

    comp_df = pd.DataFrame(comp_rows).set_index("Strategy")
    tables.save_table(comp_df, "table10_strategy_comparison")
    print("  Saved table10_strategy_comparison")

    if pairwise_results:
        pw_df = pd.DataFrame(pairwise_results)
        tables.save_table(pw_df.set_index("Comparison"), "table10b_pairwise_wilcoxon")

    print("\nAnalysis 4 complete.")
    return {"centroids": centroids, "strategies": strategies, "comparison": comp_rows}
