"""Publication-quality figure generation for all analyses."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.ticker as ticker
import seaborn as sns

from . import config


def setup_style():
    config.setup_matplotlib()


def save_figure(fig, name, subdir=None):
    """Save figure to outputs/figures/ as PNG + PDF."""
    base = config.OUTPUT_FIGURES
    if subdir:
        base = base / subdir
        base.mkdir(parents=True, exist_ok=True)

    for fmt in config.FIGURE_FORMATS:
        path = base / f"{name}.{fmt}"
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ── Double-angle polar plot ────────────────────────────────────────────────
def plot_double_angle_polar(
    groups,
    filename="double_angle_polar",
    title=None,
    figsize=(6.5, 6),
    ring_step=0.5,
    max_ring=None,
    annotate_centroids=True,
):
    """Double-angle polar plot for astigmatism vectors."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    if max_ring is None:
        all_mags = []
        for g in groups:
            all_mags.extend(np.sqrt(np.asarray(g["j0"])**2 + np.asarray(g["j45"])**2))
        max_ring = np.ceil(max(all_mags) / ring_step) * ring_step

    ax.set_rlim(0, max_ring)
    ticks = np.arange(ring_step, max_ring + 0.01, ring_step)
    ax.set_rticks(ticks)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f D"))
    ax.tick_params(axis="y", labelsize=7, pad=1)

    # Meridian tick labels
    meridian_labels = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    double_angles = [2 * m for m in meridian_labels]
    ax.set_thetagrids(
        double_angles,
        [f"{m:.0f}°" if m == int(m) else f"{m:.1f}°" for m in meridian_labels],
        fontsize=9,
    )

    ax.grid(True, linewidth=0.3, alpha=0.5)

    for g in groups:
        j0 = np.asarray(g["j0"])
        j45 = np.asarray(g["j45"])
        theta = np.arctan2(j45, j0)
        r = np.sqrt(j0**2 + j45**2)

        ax.scatter(
            theta, r,
            c=g.get("color", "#999999"),
            s=g.get("size", 20),
            alpha=g.get("alpha", 0.5),
            marker=g.get("marker", "o"),
            label=g.get("label", ""),
            edgecolors="none",
            zorder=2,
        )

        # centroid marker
        if "centroid" in g:
            c = g["centroid"]
            ct = np.arctan2(c["mean_J45"], c["mean_J0"])
            cr = np.sqrt(c["mean_J0"]**2 + c["mean_J45"]**2)
            centroid_color = g.get("centroid_color", config.COLOR_CENTROID)
            ax.scatter(
                ct, cr, marker="s", s=100,
                c=centroid_color,
                edgecolors="black", linewidths=1.0, zorder=6,
            )
            # annotate centroid value
            if annotate_centroids:
                ann_text = f'{c["centroid_mag"]:.2f} D @ {c["centroid_meridian"]:.0f}°'
                ax.annotate(
                    ann_text, xy=(ct, cr),
                    xytext=(15, 10), textcoords="offset points",
                    fontsize=7, fontweight="bold",
                    color=centroid_color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=centroid_color, alpha=0.85, lw=0.5),
                    zorder=7,
                )

        # confidence ellipses
        for ell_key, ls, lbl in [("ellipse_centroid", "-", "95% CI centroid"), ("ellipse_data", "--", "95% CI data")]:
            if ell_key in g:
                _draw_ellipse_on_polar(ax, g[ell_key], ls=ls,
                                        color=g.get("ellipse_color", g.get("color", "#999999")),
                                        linewidth=1.5 if ell_key == "ellipse_centroid" else 1.0)

    if title:
        ax.set_title(title, pad=20, fontsize=11, fontweight="bold")

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=True,
              fancybox=True, framealpha=0.9, fontsize=8)

    fig.subplots_adjust(right=0.75)
    save_figure(fig, filename)
    return fig


def _draw_ellipse_on_polar(ax, ell, ls="-", color="red", n_points=200, linewidth=1.2):
    """Draw an ellipse on a polar axis by converting parametric points."""
    cx, cy = ell["center_x"], ell["center_y"]
    w, h = ell["width"], ell["height"]
    angle_rad = np.radians(ell["angle_deg"])

    t = np.linspace(0, 2 * np.pi, n_points)
    x = (w / 2) * np.cos(t)
    y = (h / 2) * np.sin(t)

    xr = x * np.cos(angle_rad) - y * np.sin(angle_rad) + cx
    yr = x * np.sin(angle_rad) + y * np.cos(angle_rad) + cy

    theta = np.arctan2(yr, xr)
    r = np.sqrt(xr**2 + yr**2)
    ax.plot(theta, r, ls=ls, color=color, linewidth=linewidth, zorder=3)


# ── Rose diagram ───────────────────────────────────────────────────────────
def plot_rose_diagram(meridians, filename="rose_diagram", bins=12, figsize=(6, 5.5)):
    """Circular histogram (rose diagram) of CSIA meridians."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    meridians = np.asarray(meridians, dtype=float)
    bin_edges = np.linspace(0, 180, bins + 1)
    counts, _ = np.histogram(meridians, bins=bin_edges)

    widths = np.diff(bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    theta_centers = np.radians(2 * centers)
    theta_widths = np.radians(2 * widths)

    colors = [config.COLOR_CENTROID if c > 90 else config.COLOR_OD for c in centers]

    bars = ax.bar(
        theta_centers, counts, width=theta_widths,
        color=colors, edgecolor="white", linewidth=0.8, alpha=0.85,
    )

    # Add count labels on each bar
    for bar, count, theta in zip(bars, counts, theta_centers):
        if count > 0:
            ax.text(theta, count + 0.8, str(int(count)),
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    meridian_labels = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    ax.set_thetagrids([2 * m for m in meridian_labels],
                      [f"{m:.0f}°" if m == int(m) else f"{m:.1f}°" for m in meridian_labels],
                      fontsize=9)
    ax.set_title("CSIA Meridian Distribution", pad=20, fontsize=11, fontweight="bold")

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=config.COLOR_OD, edgecolor="white", label=f"Flattening (n={int((meridians <= 90).sum())})"),
        Patch(facecolor=config.COLOR_CENTROID, edgecolor="white", label=f"Steepening (n={int((meridians > 90).sum())})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.3, 1.05),
              frameon=True, fancybox=True, framealpha=0.9, fontsize=8)

    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.subplots_adjust(right=0.75)
    save_figure(fig, filename)
    return fig


# ── Histogram ──────────────────────────────────────────────────────────────
def plot_histogram(values, xlabel, filename, figsize=(5, 3.5), bins=25, vlines=None):
    """Histogram with KDE overlay and optional vertical lines."""
    fig, ax = plt.subplots(figsize=figsize)
    values = np.asarray(values, dtype=float)

    ax.hist(values, bins=bins, density=True, color="#CCCCCC", edgecolor="white", linewidth=0.5)
    sns.kdeplot(values, ax=ax, color=config.COLOR_OD, linewidth=1.8)

    if vlines:
        for v, label, color in vlines:
            ax.axvline(v, color=color, linestyle="--", linewidth=1.2, label=label)
        ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=8)

    # Clip x-axis to non-negative for magnitude
    if values.min() >= 0:
        ax.set_xlim(left=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, filename)
    return fig


# ── Correlation heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(corr_df, p_df, filename="correlation_heatmap", figsize=(5.5, 10)):
    """Annotated heatmap with significance stars."""
    fig, ax = plt.subplots(figsize=figsize)

    # Build annotation matrix
    annot = corr_df.copy().astype(str)
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            r_val = corr_df.iloc[i, j]
            p_val = p_df.iloc[i, j]
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            annot.iloc[i, j] = f"{r_val:.2f}{stars}"

    # Rename columns to proper math notation
    col_map = {"Magnitude": "Magnitude", "J0": "$J_0$", "J45": "$J_{45}$"}
    corr_plot = corr_df.rename(columns=col_map)
    annot_plot = annot.rename(columns=col_map)

    sns.heatmap(
        corr_plot, annot=annot_plot, fmt="", cmap="RdBu_r", center=0,
        vmin=-0.4, vmax=0.4, ax=ax, linewidths=0.5,
        cbar_kws={"label": "Pearson $r$", "shrink": 0.6},
        annot_kws={"fontsize": 8},
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_title("Bivariate Correlations with CSIA Components\n(* $p$ < 0.05, ** $p$ < 0.01, *** $p$ < 0.001)",
                 fontsize=10, fontweight="bold", pad=12)

    fig.tight_layout()
    save_figure(fig, filename)
    return fig


# ── Forest plot ────────────────────────────────────────────────────────────
def plot_forest(data, filename="forest_plot", figsize=(6, 8), title=""):
    """Forest plot for effect sizes with CIs."""
    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(len(data))

    # Clean up labels
    label_map = {"CSIA_mag": "Magnitude", "J0": "$J_0$", "J45": "$J_{45}$"}

    for i, d in enumerate(data):
        color = d.get("color", config.COLOR_OD)
        ax.plot([d["ci_lo"], d["ci_hi"]], [i, i], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter(d["r"], i, color=color, s=50, zorder=3, edgecolors="white", linewidths=0.8)

        # Value annotation
        ax.text(d["ci_hi"] + 0.01, i, f'{d["r"]:.2f}', va="center", fontsize=7, color=color)

    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_positions)

    # Clean labels
    clean_labels = []
    for d in data:
        lbl = d["label"]
        for old, new in label_map.items():
            lbl = lbl.replace(old, new)
        # Replace arrow with unicode
        lbl = lbl.replace("→", "\u2192")
        clean_labels.append(lbl)

    ax.set_yticklabels(clean_labels, fontsize=8)
    ax.set_xlabel("Partial correlation ($r$)")
    ax.invert_yaxis()
    if title:
        ax.set_title(title, fontweight="bold", fontsize=10)

    # Legend for target colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=config.COLOR_OD, lw=2, label="Magnitude"),
        Line2D([0], [0], color=config.COLOR_OS, lw=2, label="$J_0$"),
        Line2D([0], [0], color=config.COLOR_ACCENT, lw=2, label="$J_{45}$"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fancybox=True,
              framealpha=0.9, fontsize=8)

    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, filename)
    return fig


# ── Bland-Altman ───────────────────────────────────────────────────────────
def plot_bland_altman(predicted, actual, filename="bland_altman", figsize=(5.5, 4.5), ylabel="CSIA Magnitude (D)"):
    """Standard Bland-Altman plot."""
    from .utils import bland_altman_stats

    ba = bland_altman_stats(predicted, actual)
    mean_vals = (np.asarray(predicted) + np.asarray(actual)) / 2
    diff_vals = np.asarray(predicted) - np.asarray(actual)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(mean_vals, diff_vals, s=15, alpha=0.5, color=config.COLOR_OD, edgecolors="none")
    ax.axhline(ba["mean_diff"], color="black", linewidth=1, label=f'Mean: {ba["mean_diff"]:.3f}')
    ax.axhline(ba["upper_loa"], color=config.COLOR_CENTROID, linewidth=0.8, linestyle="--",
               label=f'+1.96 SD: {ba["upper_loa"]:.3f}')
    ax.axhline(ba["lower_loa"], color=config.COLOR_CENTROID, linewidth=0.8, linestyle="--",
               label=f'$-$1.96 SD: {ba["lower_loa"]:.3f}')

    ax.set_xlabel(f"Mean of predicted and actual\n{ylabel}")
    ax.set_ylabel(f"Predicted $-$ Actual\n{ylabel}")
    ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.9, fontsize=8)
    ax.set_title("Bland-Altman Analysis", fontweight="bold", fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()

    save_figure(fig, filename)
    return fig


# ── Model comparison bar chart ─────────────────────────────────────────────
def plot_model_comparison(results_df, metric, filename="model_comparison", figsize=(7, 4.5), title=None):
    """Grouped bar chart: models x feature sets for a given metric."""
    metric_labels = {
        "R2_mean": "$R^2$",
        "MAE_mean": "MAE (D)",
        "RMSE_mean": "RMSE (D)",
    }

    fig, ax = plt.subplots(figsize=figsize)

    pivot = results_df.pivot_table(values=metric, index="model", columns="feature_set", aggfunc="mean")
    bars = pivot.plot(kind="bar", ax=ax, width=0.7, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="-")
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    # Annotate bar values
    for container in ax.containers:
        for bar in container:
            val = bar.get_height()
            if abs(val) > 0.005:
                y_pos = val - 0.01 if val < 0 else val + 0.005
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        f'{val:.2f}', ha='center', va='top' if val < 0 else 'bottom',
                        fontsize=6, color='#333333')

    # Rename legend entries
    fs_labels = {"biomech": "Biomechanics", "biomech_demo": "Biomech + Demo", "reduced": "Reduced"}
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [fs_labels.get(l, l) for l in labels],
              title="Feature set", frameon=True, fancybox=True, framealpha=0.9,
              fontsize=8, title_fontsize=9, bbox_to_anchor=(1.02, 1))

    if title:
        ax.set_title(title, fontweight="bold", fontsize=10, pad=10)
    sns.despine(ax=ax)
    fig.tight_layout()

    save_figure(fig, filename)
    return fig


# ── Box plot comparison ────────────────────────────────────────────────────
def plot_strategy_boxplot(errors_dict, filename="strategy_comparison", figsize=(9, 5)):
    """Box plot comparing vector errors across prediction strategies."""
    import pandas as pd

    # Clean strategy names (remove newlines)
    clean_dict = {}
    for k, v in errors_dict.items():
        clean_key = k.replace("\n", " ")
        clean_dict[clean_key] = v

    records = []
    for strategy, errors in clean_dict.items():
        for e in errors:
            records.append({"Strategy": strategy, "Vector Error (D)": e})
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=figsize)
    order = df.groupby("Strategy")["Vector Error (D)"].median().sort_values().index.tolist()

    bp = sns.boxplot(data=df, x="Strategy", y="Vector Error (D)", order=order, ax=ax,
                     palette="Set2", width=0.55, linewidth=0.8, fliersize=2)

    # Annotate medians
    for i, strat in enumerate(order):
        med = df[df["Strategy"] == strat]["Vector Error (D)"].median()
        mean = df[df["Strategy"] == strat]["Vector Error (D)"].mean()
        ax.text(i, med - 0.06, f'{med:.3f}', ha='center', va='top', fontsize=7,
                fontweight='bold', color='#333333',
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Vector Error (D)", fontsize=10)
    ax.set_xlabel("")
    ax.set_title("CSIA Prediction Strategy Comparison\n(lower is better)", fontweight="bold", fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()

    save_figure(fig, filename)
    return fig


# ── VIF bar plot ───────────────────────────────────────────────────────────
def plot_vif(vif_df, filename="vif_barplot", figsize=(6, 6)):
    """Horizontal bar plot of VIF values."""
    fig, ax = plt.subplots(figsize=figsize)
    vif_sorted = vif_df.sort_values("VIF", ascending=True)

    bars = ax.barh(vif_sorted["feature"], vif_sorted["VIF"],
                   color=config.COLOR_OD, edgecolor="white", height=0.7)

    # Value labels at bar tips
    for bar, val in zip(bars, vif_sorted["VIF"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=7)

    ax.axvline(5, color="orange", linewidth=1.2, linestyle="--", label="VIF = 5")
    ax.axvline(10, color="red", linewidth=1.2, linestyle="--", label="VIF = 10")
    ax.set_xlabel("Variance Inflation Factor", fontsize=10)
    ax.set_title("Multicollinearity Assessment (VIF)", fontweight="bold", fontsize=10)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=8)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    sns.despine(ax=ax)
    fig.tight_layout()

    save_figure(fig, filename)
    return fig


# ── SHAP summary ───────────────────────────────────────────────────────────
def plot_shap_summary(shap_values, X_df, filename="shap_summary", figsize=(7, 7), title=None):
    """SHAP beeswarm plot."""
    import shap

    fig, ax = plt.subplots(figsize=figsize)
    shap.summary_plot(shap_values, X_df, show=False, plot_size=None)

    current_fig = plt.gcf()
    if title:
        current_fig.axes[0].set_title(title, fontweight="bold", fontsize=10, pad=10)

    save_figure(current_fig, filename)
    return fig


# ── Permutation importance ─────────────────────────────────────────────────
def plot_permutation_importance(importances, feature_names, filename="permutation_importance",
                                figsize=(6, 7), title=None):
    """Horizontal bar chart of permutation importance with error bars and value labels."""
    fig, ax = plt.subplots(figsize=figsize)
    mean_imp = importances.importances_mean
    std_imp = importances.importances_std
    idx = np.argsort(mean_imp)

    bars = ax.barh(range(len(idx)), mean_imp[idx], xerr=std_imp[idx],
                   color=config.COLOR_OD, edgecolor="white", capsize=2, height=0.7)

    # Value labels
    for i, (bar, ii) in enumerate(zip(bars, idx)):
        val = mean_imp[ii]
        if val > 0.001:
            ax.text(bar.get_width() + std_imp[ii] + 0.002, i,
                    f'{val:.3f}', va='center', fontsize=6.5)

    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    ax.set_xlabel("Mean decrease in $R^2$", fontsize=10)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=10)
    else:
        ax.set_title("Permutation Importance", fontweight="bold", fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()

    save_figure(fig, filename)
    return fig
