"""Statistical utilities: partial correlation, FDR, Bland-Altman, stepwise AIC, bootstrap."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


# ── Partial correlation ────────────────────────────────────────────────────
def partial_correlation(x, y, covariates):
    """Partial Pearson correlation between x and y controlling for covariates.

    Args:
        x: 1-D array
        y: 1-D array
        covariates: 2-D array (n, k)

    Returns:
        (r, p_value, ci_low, ci_high) — 95 % CI via Fisher z-transform
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    C = np.asarray(covariates, dtype=float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)

    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(C), axis=1)
    x, y, C = x[mask], y[mask], C[mask]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan

    lr = LinearRegression()
    res_x = x - lr.fit(C, x).predict(C)
    res_y = y - lr.fit(C, y).predict(C)

    r, p = stats.pearsonr(res_x, res_y)

    # Fisher z CI
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - C.shape[1] - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    ci_lo, ci_hi = np.tanh(z_lo), np.tanh(z_hi)

    return float(r), float(p), float(ci_lo), float(ci_hi)


# ── FDR correction ─────────────────────────────────────────────────────────
def fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = p.argsort()
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)

    adjusted = p * n / ranks
    # enforce monotonicity (step-up)
    adjusted = np.minimum.accumulate(adjusted[order[::-1]])[::-1]
    result = np.empty_like(p)
    result[order] = adjusted
    return np.minimum(result, 1.0)


# ── Bland-Altman ───────────────────────────────────────────────────────────
def bland_altman_stats(predicted, actual):
    """Bland-Altman analysis. Returns dict with mean_diff, sd_diff, limits."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    diff = predicted - actual
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    n = len(diff)
    se = sd_diff / np.sqrt(n)
    return {
        "mean_diff": float(mean_diff),
        "sd_diff": float(sd_diff),
        "upper_loa": float(mean_diff + 1.96 * sd_diff),
        "lower_loa": float(mean_diff - 1.96 * sd_diff),
        "mean_diff_ci": (float(mean_diff - 1.96 * se), float(mean_diff + 1.96 * se)),
        "n": n,
    }


# ── Stepwise AIC ───────────────────────────────────────────────────────────
def aic_stepwise(X, y, feature_names, direction="both", verbose=False):
    """Forward-backward AIC-based feature selection.

    Returns:
        list of selected feature names, final AIC, statsmodels OLS result
    """
    import statsmodels.api as sm

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    remaining = list(range(p))
    selected = []

    # null model AIC
    null_model = sm.OLS(y, np.ones((n, 1))).fit()
    best_aic = null_model.aic

    improved = True
    while improved:
        improved = False

        # Forward step
        best_candidate = None
        for idx in remaining:
            trial = selected + [idx]
            Xt = sm.add_constant(X[:, trial])
            try:
                model = sm.OLS(y, Xt).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_candidate = idx
            except Exception:
                continue

        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            improved = True
            if verbose:
                print(f"  + {feature_names[best_candidate]} (AIC={best_aic:.2f})")

        # Backward step (only if direction == "both")
        if direction == "both" and len(selected) > 1:
            worst_candidate = None
            for idx in selected:
                trial = [s for s in selected if s != idx]
                Xt = sm.add_constant(X[:, trial])
                try:
                    model = sm.OLS(y, Xt).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        worst_candidate = idx
                except Exception:
                    continue

            if worst_candidate is not None:
                selected.remove(worst_candidate)
                remaining.append(worst_candidate)
                improved = True
                if verbose:
                    print(f"  - {feature_names[worst_candidate]} (AIC={best_aic:.2f})")

    selected_names = [feature_names[i] for i in selected]

    # fit final model
    if selected:
        Xf = sm.add_constant(X[:, selected])
        final = sm.OLS(y, Xf).fit()
    else:
        final = null_model

    return selected_names, best_aic, final


# ── Bootstrap CI ───────────────────────────────────────────────────────────
def bootstrap_ci(data, stat_func=np.mean, n_boot=10000, ci=0.95, seed=42):
    """BCa bootstrap confidence interval.

    Args:
        data: 1-D array
        stat_func: callable that takes an array and returns a scalar
        n_boot: number of bootstrap resamples
        ci: confidence level

    Returns:
        (ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)
    boot_stats = np.array([stat_func(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)])

    alpha = 1.0 - ci
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ── VIF ────────────────────────────────────────────────────────────────────
def compute_vif(X, feature_names):
    """Variance Inflation Factor for each feature.

    Returns DataFrame with columns [feature, VIF].
    """
    from sklearn.linear_model import LinearRegression

    X = np.asarray(X, dtype=float)
    n, p = X.shape
    vifs = []
    for j in range(p):
        y_j = X[:, j]
        X_others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(X_others, y_j).score(X_others, y_j)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vifs.append(vif)

    return pd.DataFrame({"feature": feature_names, "VIF": vifs})
