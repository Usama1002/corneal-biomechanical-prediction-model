"""Formatted table generators for all analyses."""

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def save_table(df, name, subdir=None):
    """Save DataFrame as CSV + XLSX to outputs/tables/."""
    base = config.OUTPUT_TABLES
    if subdir:
        base = base / subdir
        base.mkdir(parents=True, exist_ok=True)

    df.to_csv(base / f"{name}.csv", index=True)
    with pd.ExcelWriter(base / f"{name}.xlsx", engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Sheet1")
        worksheet = writer.sheets["Sheet1"]
        for i, col in enumerate(df.reset_index().columns):
            max_len = max(df.reset_index()[col].astype(str).str.len().max(), len(str(col))) + 2
            worksheet.set_column(i, i, min(max_len, 30))


def fmt_mean_sd(values, decimals=2):
    """Format as 'mean +/- SD'."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "N/A"
    return f"{np.mean(v):.{decimals}f} ± {np.std(v, ddof=1):.{decimals}f}"


def fmt_p(p):
    """Format p-value."""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def build_demographics_table(df):
    """Table 1: demographics and preop parameters, overall and by eye."""
    od = df[df["eye"] == "OD"]
    os_ = df[df["eye"] == "OS"]

    rows = []

    # Age
    t, p = stats.ttest_ind(od["age"], os_["age"])
    rows.append({"Variable": "Age (years)", "Overall": fmt_mean_sd(df["age"]),
                 "OD": fmt_mean_sd(od["age"]), "OS": fmt_mean_sd(os_["age"]), "p": fmt_p(p)})

    # Sex
    n_m = (df["sex"] == "M").sum()
    n_f = (df["sex"] == "F").sum()
    rows.append({"Variable": "Sex (M/F)", "Overall": f"{n_m}/{n_f}",
                 "OD": f"{(od['sex']=='M').sum()}/{(od['sex']=='F').sum()}",
                 "OS": f"{(os_['sex']=='M').sum()}/{(os_['sex']=='F').sum()}",
                 "p": fmt_p(stats.chi2_contingency(pd.crosstab(df["sex"], df["eye"]))[1])})

    # Numeric variables
    num_vars = config.ALL_PREOP_PARAMS + ["CSIA_mag", "CSIA_meridian", "J0", "J45"]
    var_labels = {
        "PCT135": "PCT 135° (μm)", "AL": "Axial Length (mm)", "WTW": "WTW (mm)",
        "AL1": "AL1 (mm)", "AV1": "AV1 (m/s)", "AL2": "AL2 (mm)", "AV2": "AV2 (m/s)",
        "PD": "PD (mm)", "HCR": "HCR (mm)", "HCDA": "HCDA (mm)",
        "IOPnc": "IOPnc (mmHg)", "bIOP": "bIOP (mmHg)", "CCT": "CCT (μm)",
        "SPA1": "SP-A1 (mmHg/mm)", "ARTh": "ARTh (μm)",
        "DA_Ratio": "DA Ratio", "IR": "IR (mm⁻¹)", "CBI": "CBI", "SSI": "SSI", "CBiF": "CBiF",
        "CSIA_mag": "CSIA Magnitude (D)", "CSIA_meridian": "CSIA Meridian (°)",
        "J0": "J0", "J45": "J45",
    }

    for var in num_vars:
        if var not in df.columns:
            continue
        label = var_labels.get(var, var)
        # normality test to decide t-test vs Mann-Whitney
        try:
            _, p_norm = stats.shapiro(df[var].dropna().values[:5000])
        except Exception:
            p_norm = 0

        if p_norm > 0.05:
            t_stat, p_val = stats.ttest_ind(od[var].dropna(), os_[var].dropna())
        else:
            t_stat, p_val = stats.mannwhitneyu(od[var].dropna(), os_[var].dropna(), alternative="two-sided")

        rows.append({
            "Variable": label,
            "Overall": fmt_mean_sd(df[var]),
            "OD": fmt_mean_sd(od[var]),
            "OS": fmt_mean_sd(os_[var]),
            "p": fmt_p(p_val),
        })

    result = pd.DataFrame(rows).set_index("Variable")
    return result


def build_correlation_table(predictors, targets_dict, p_dict, fdr_p_dict=None):
    """Build formatted correlation table.

    predictors: list of predictor names
    targets_dict: {target_name: {predictor: r_value}}
    p_dict: {target_name: {predictor: p_value}}
    fdr_p_dict: optional {target_name: {predictor: fdr_p_value}}
    """
    rows = []
    for pred in predictors:
        row = {"Variable": pred}
        for target in targets_dict:
            r = targets_dict[target].get(pred, np.nan)
            p = p_dict[target].get(pred, np.nan)
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row[f"{target}_r"] = f"{r:.3f}{stars}"
            row[f"{target}_p"] = fmt_p(p)
            if fdr_p_dict:
                fdr_p = fdr_p_dict[target].get(pred, np.nan)
                row[f"{target}_fdr_p"] = fmt_p(fdr_p)
        rows.append(row)

    return pd.DataFrame(rows).set_index("Variable")


def build_cv_results_table(results_list):
    """Build table from a list of CV result dicts.

    Each dict: {model, target, feature_set, R2_mean, R2_std, MAE_mean, MAE_std, ...}
    """
    df = pd.DataFrame(results_list)
    # format mean ± std columns
    for metric in ["R2", "MAE", "RMSE"]:
        if f"{metric}_mean" in df.columns:
            df[metric] = df.apply(
                lambda r: f"{r[f'{metric}_mean']:.3f} ± {r[f'{metric}_std']:.3f}", axis=1
            )
    return df


def build_centroid_table(centroids_list):
    """Build table from a list of centroid dicts.

    Each dict should have: group, n, centroid_mag, centroid_meridian, mean_J0, mean_J45, sd_J0, sd_J45
    """
    return pd.DataFrame(centroids_list)
