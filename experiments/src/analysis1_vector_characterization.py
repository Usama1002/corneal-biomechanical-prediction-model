"""Analysis 1: CSIA vector characterization — centroids, double-angle plots, distributions."""

import numpy as np
import pandas as pd
from scipy import stats

from . import config, vector_math, plotting, tables, data_loader


def run(df):
    """Run all vector characterization analyses and save outputs."""
    print("=" * 60)
    print("ANALYSIS 1: Vector Characterization")
    print("=" * 60)

    od, os_ = data_loader.get_subsets(df)
    results = {}

    # ── 1.1 Compute centroids ──────────────────────────────────────────
    print("\n--- 1.1 Centroids ---")
    cent_all = vector_math.compute_centroid(df["CSIA_mag"], df["CSIA_meridian"])
    cent_od = vector_math.compute_centroid(od["CSIA_mag"], od["CSIA_meridian"])
    cent_os = vector_math.compute_centroid(os_["CSIA_mag"], os_["CSIA_meridian"])

    for label, c in [("Overall", cent_all), ("OD", cent_od), ("OS", cent_os)]:
        print(f"  {label}: {c['centroid_mag']:.3f} D @ {c['centroid_meridian']:.1f}° "
              f"(n={c['n']}, J0={c['mean_J0']:.4f}±{c['sd_J0']:.4f}, "
              f"J45={c['mean_J45']:.4f}±{c['sd_J45']:.4f})")

    print(f"  Yin et al. 2025: {config.BU_CENTROID_MAG} D @ {config.BU_CENTROID_MERIDIAN}°")
    results["centroids"] = {"overall": cent_all, "OD": cent_od, "OS": cent_os}

    # ── 1.2 Confidence ellipses ────────────────────────────────────────
    print("\n--- 1.2 Confidence ellipses ---")
    ellipses = {}
    for label, subset in [("overall", df), ("OD", od), ("OS", os_)]:
        j0, j45 = vector_math.decompose_to_j0_j45(subset["CSIA_mag"], subset["CSIA_meridian"])
        ell_cent = vector_math.compute_confidence_ellipse(j0, j45, confidence=0.95, is_centroid=True)
        ell_data = vector_math.compute_confidence_ellipse(j0, j45, confidence=0.95, is_centroid=False)
        ellipses[label] = {"centroid": ell_cent, "data": ell_data}
        print(f"  {label} centroid ellipse: w={ell_cent['width']:.3f}, h={ell_cent['height']:.3f}")
        print(f"  {label} data ellipse: w={ell_data['width']:.3f}, h={ell_data['height']:.3f}")

    results["ellipses"] = ellipses

    # ── 1.3 Descriptive statistics ─────────────────────────────────────
    print("\n--- 1.3 Descriptive statistics ---")
    desc_stats = {}
    for label, subset in [("Overall", df), ("OD", od), ("OS", os_)]:
        d = {
            "n": len(subset),
            "mag_mean": subset["CSIA_mag"].mean(),
            "mag_sd": subset["CSIA_mag"].std(),
            "mag_median": subset["CSIA_mag"].median(),
            "mag_iqr": f"{subset['CSIA_mag'].quantile(0.25):.3f}-{subset['CSIA_mag'].quantile(0.75):.3f}",
            "mag_range": f"{subset['CSIA_mag'].min():.3f}-{subset['CSIA_mag'].max():.3f}",
            "mer_mean": subset["CSIA_meridian"].mean(),
            "mer_sd": subset["CSIA_meridian"].std(),
            "mer_range": f"{subset['CSIA_meridian'].min():.1f}-{subset['CSIA_meridian'].max():.1f}",
        }
        desc_stats[label] = d

    # Normality tests
    sw_mag, p_mag = stats.shapiro(df["CSIA_mag"])
    sw_j0, p_j0 = stats.shapiro(df["J0"])
    sw_j45, p_j45 = stats.shapiro(df["J45"])
    print(f"  Shapiro-Wilk CSIA_mag: W={sw_mag:.4f}, p={p_mag:.6f}")
    print(f"  Shapiro-Wilk J0: W={sw_j0:.4f}, p={p_j0:.6f}")
    print(f"  Shapiro-Wilk J45: W={sw_j45:.4f}, p={p_j45:.6f}")
    results["normality"] = {
        "CSIA_mag": {"W": sw_mag, "p": p_mag},
        "J0": {"W": sw_j0, "p": p_j0},
        "J45": {"W": sw_j45, "p": p_j45},
    }

    # Meridian distribution
    mer = df["CSIA_meridian"].values
    near_45 = np.sum((mer >= 15) & (mer <= 75))
    steepening = np.sum(mer > 90)
    print(f"  Meridian 15-75° (near expected): {near_45}/{len(mer)} ({100*near_45/len(mer):.1f}%)")
    print(f"  Meridian >90° (steepening): {steepening}/{len(mer)} ({100*steepening/len(mer):.1f}%)")
    results["meridian_distribution"] = {
        "near_45_pct": near_45 / len(mer),
        "steepening_pct": steepening / len(mer),
        "steepening_n": steepening,
    }

    # OD vs OS comparison (full stats)
    print("\n--- 1.4 OD vs OS comparison ---")
    od_os_comp = {}
    for var, label in [("CSIA_mag", "Magnitude"), ("CSIA_meridian", "Meridian"),
                       ("J0", "J0"), ("J45", "J45")]:
        t_stat, p_val = stats.ttest_ind(od[var], os_[var])
        u_stat, p_mw = stats.mannwhitneyu(od[var], os_[var], alternative="two-sided")
        d_cohen = (od[var].mean() - os_[var].mean()) / np.sqrt(
            ((len(od) - 1) * od[var].std()**2 + (len(os_) - 1) * os_[var].std()**2) / (len(od) + len(os_) - 2)
        )
        print(f"  {label}: OD={od[var].mean():.3f}±{od[var].std():.3f}, "
              f"OS={os_[var].mean():.3f}±{os_[var].std():.3f}, "
              f"t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d_cohen:.3f}")
        od_os_comp[var] = {
            "od_mean": od[var].mean(), "od_sd": od[var].std(),
            "os_mean": os_[var].mean(), "os_sd": os_[var].std(),
            "t": t_stat, "p_ttest": p_val, "U": u_stat, "p_mannwhitney": p_mw,
            "cohens_d": d_cohen,
        }
    results["od_os_comparison"] = od_os_comp

    # ── 1.5 Figures ────────────────────────────────────────────────────
    print("\n--- 1.5 Generating figures ---")

    # Fig 01: Overall double-angle polar
    j0_all, j45_all = vector_math.decompose_to_j0_j45(df["CSIA_mag"], df["CSIA_meridian"])
    plotting.plot_double_angle_polar(
        groups=[{
            "j0": j0_all, "j45": j45_all,
            "color": config.COLOR_OVERALL, "alpha": 0.4, "size": 15,
            "label": f"All eyes (n={len(df)})",
            "centroid": cent_all,
            "centroid_color": config.COLOR_CENTROID,
            "ellipse_centroid": ellipses["overall"]["centroid"],
            "ellipse_data": ellipses["overall"]["data"],
            "ellipse_color": config.COLOR_CENTROID,
        }],
        filename="fig01_double_angle_polar_overall",
        title=f"CSIA Double-Angle Plot\nCentroid: {cent_all['centroid_mag']:.2f} D @ {cent_all['centroid_meridian']:.1f}°",
    )
    print("  Saved fig01_double_angle_polar_overall")

    # Fig 02: OD vs OS double-angle polar
    j0_od, j45_od = vector_math.decompose_to_j0_j45(od["CSIA_mag"], od["CSIA_meridian"])
    j0_os, j45_os = vector_math.decompose_to_j0_j45(os_["CSIA_mag"], os_["CSIA_meridian"])
    plotting.plot_double_angle_polar(
        groups=[
            {
                "j0": j0_od, "j45": j45_od,
                "color": config.COLOR_OD, "alpha": 0.4, "size": 15,
                "label": f"OD (n={len(od)})",
                "centroid": cent_od, "centroid_color": config.COLOR_OD,
                "ellipse_centroid": ellipses["OD"]["centroid"],
                "ellipse_color": config.COLOR_OD,
            },
            {
                "j0": j0_os, "j45": j45_os,
                "color": config.COLOR_OS, "alpha": 0.4, "size": 15,
                "label": f"OS (n={len(os_)})",
                "centroid": cent_os, "centroid_color": config.COLOR_OS,
                "ellipse_centroid": ellipses["OS"]["centroid"],
                "ellipse_color": config.COLOR_OS,
            },
        ],
        filename="fig02_double_angle_polar_by_eye",
        title="CSIA by Eye Laterality",
    )
    print("  Saved fig02_double_angle_polar_by_eye")

    # Fig 03: Rose diagram
    plotting.plot_rose_diagram(df["CSIA_meridian"].values, filename="fig03_rose_diagram", bins=12)
    print("  Saved fig03_rose_diagram")

    # Fig 04: Magnitude histogram
    plotting.plot_histogram(
        df["CSIA_mag"].values, xlabel="CSIA Magnitude (D)",
        filename="fig04_magnitude_histogram", bins=25,
        vlines=[
            (df["CSIA_mag"].mean(), f"Mean ({df['CSIA_mag'].mean():.2f} D)", config.COLOR_OD),
            (cent_all["centroid_mag"], f"Centroid ({cent_all['centroid_mag']:.2f} D)", config.COLOR_CENTROID),
        ],
    )
    print("  Saved fig04_magnitude_histogram")

    # ── 1.6 Tables ─────────────────────────────────────────────────────
    print("\n--- 1.6 Generating tables ---")

    # Table 01: Demographics
    demo_table = tables.build_demographics_table(df)
    tables.save_table(demo_table, "table01_descriptive_statistics")
    print("  Saved table01_descriptive_statistics")

    # Centroids summary table
    centroid_rows = []
    for label, c in [("Overall", cent_all), ("OD", cent_od), ("OS", cent_os),
                      ("Yin et al. 2025", {"centroid_mag": config.BU_CENTROID_MAG,
                                          "centroid_meridian": config.BU_CENTROID_MERIDIAN,
                                          "mean_J0": np.nan, "mean_J45": np.nan,
                                          "sd_J0": np.nan, "sd_J45": np.nan, "n": 149})]:
        centroid_rows.append({
            "Group": label, "n": c["n"],
            "Centroid Magnitude (D)": f"{c['centroid_mag']:.3f}",
            "Centroid Meridian (°)": f"{c['centroid_meridian']:.1f}",
            "Mean J0 ± SD": f"{c['mean_J0']:.4f} ± {c['sd_J0']:.4f}" if np.isfinite(c["mean_J0"]) else "N/A",
            "Mean J45 ± SD": f"{c['mean_J45']:.4f} ± {c['sd_J45']:.4f}" if np.isfinite(c["mean_J45"]) else "N/A",
        })
    cent_df = pd.DataFrame(centroid_rows).set_index("Group")
    tables.save_table(cent_df, "table01b_centroids")
    print("  Saved table01b_centroids")

    # Save results as CSV
    pd.DataFrame([od_os_comp]).to_csv(config.OUTPUT_RESULTS / "od_os_comparison.csv", index=False)
    pd.DataFrame([results["normality"]]).to_csv(config.OUTPUT_RESULTS / "normality_tests.csv", index=False)

    print("\nAnalysis 1 complete.")
    return results
