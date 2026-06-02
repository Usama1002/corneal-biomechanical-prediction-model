"""Analysis 7: Non-negative CSIA formula development + external validation.

Addresses the collaborators' request to refine the published magnitude
formula (CSIA = 0.13 + 0.01*Age - 0.09*IR), which can return clinically
impossible negative values. We replace the linear form with a Gamma
generalized linear model (log link), which is strictly positive by
construction, refit on the corrected 202-eye training set, and validate on
the two independent external cohorts (Jiang + Bu, 54 eyes).

Evaluation is two-tier and leakage-free:
  * internal: repeated 10-fold CV on the 202 training eyes,
  * external: fit on all 202, predict once on the 54 held-out eyes
    (combined and per-cohort), against fair baselines (the published
    formula, the population mean, and the eye-specific centroid).

Direction is handled separately via the eye-specific vector centroid, since
no biomechanical model predicts the CSIA meridian (see analyses 2-4).
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold

from . import config, vector_math, tables, plotting, data_loader

warnings.filterwarnings("ignore")

# Predictor sets evaluated for the magnitude formula. Kept deliberately small:
# analyses 2-3 show that adding biomechanical features beyond age degrades
# out-of-sample performance, so a clinical formula should stay parsimonious.
PREDICTOR_SETS = {
    "Age+IR": ["age", "IR"],            # the published-formula predictors
    "Age+IR+Eye": ["age", "IR", "eye_binary"],
}


# ── Metrics ────────────────────────────────────────────────────────────────
def _metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return {
        "n": len(actual),
        "R2": 1 - ss_res / ss_tot,
        "MAE": np.mean(np.abs(actual - pred)),
        "RMSE": np.sqrt(np.mean((actual - pred) ** 2)),
        "pct_negative": 100.0 * np.mean(pred < 0),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
    }


# ── Model fitters: each returns a predict(df) -> ndarray closure ───────────
def _fit_gamma_log(train_df, predictors):
    """Gamma GLM with log link. Strictly positive predictions by construction."""
    X = sm.add_constant(train_df[predictors].astype(float), has_constant="add")
    y = train_df["CSIA_mag"].astype(float).values
    model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log()))
    res = model.fit()

    def predict(df):
        Xn = sm.add_constant(df[predictors].astype(float), has_constant="add")
        Xn = Xn[X.columns]  # enforce identical column order
        return np.asarray(res.predict(Xn), dtype=float)

    predict.result = res
    return predict


def _fit_logols(train_df, predictors):
    """OLS on log(CSIA) with Duan smearing so exp() targets the mean, not the
    median. Strictly positive. Robustness check against the Gamma GLM."""
    X = sm.add_constant(train_df[predictors].astype(float), has_constant="add")
    y = np.log(train_df["CSIA_mag"].astype(float).values)
    res = sm.OLS(y, X).fit()
    smear = np.mean(np.exp(res.resid))  # Duan's smearing estimator

    def predict(df):
        Xn = sm.add_constant(df[predictors].astype(float), has_constant="add")
        Xn = Xn[X.columns]
        return np.exp(np.asarray(res.predict(Xn), dtype=float)) * smear

    predict.result = res
    return predict


def _fit_bu(train_df, predictors=None):
    """The published linear formula (can be negative). Ignores training data."""
    def predict(df):
        return config.BU_FORMULA(df["age"].astype(float).values,
                                 df["IR"].astype(float).values)
    return predict


def _fit_mean(train_df, predictors=None):
    """Population-mean baseline: predict the training mean magnitude."""
    mu = float(train_df["CSIA_mag"].mean())

    def predict(df):
        return np.full(len(df), mu)
    return predict


def _fit_eye_centroid_mag(train_df, predictors=None):
    """Eye-specific centroid magnitude: predict the OD/OS centroid magnitude of
    the training set, selected by each eye's laterality."""
    cents = {}
    for eye in ["OD", "OS"]:
        sub = train_df[train_df["eye"] == eye]
        c = vector_math.compute_centroid(sub["CSIA_mag"].values, sub["CSIA_meridian"].values)
        cents[eye] = c["centroid_mag"]

    def predict(df):
        return df["eye"].map(cents).astype(float).values
    return predict


# ── Cross-validation harness (leakage-free) ────────────────────────────────
def _cv(df, fit_fn, predictors, n_splits=10, n_repeats=3):
    """Repeated K-fold CV. fit_fn(train_df, predictors) -> predict(df)."""
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                       random_state=config.RANDOM_SEED)
    rows = []
    X_dummy = np.zeros((len(df), 1))
    for tr, te in cv.split(X_dummy):
        train_df, test_df = df.iloc[tr], df.iloc[te]
        pred = fit_fn(train_df, predictors)(test_df)
        rows.append(_metrics(test_df["CSIA_mag"].values, pred))
    m = pd.DataFrame(rows)
    return {k: (m[k].mean(), m[k].std()) for k in ["R2", "MAE", "RMSE", "pct_negative"]}


# ── Main ────────────────────────────────────────────────────────────────────
def run(df):
    print("=" * 60)
    print("ANALYSIS 7: Non-negative CSIA formula + external validation")
    print("=" * 60)

    val = data_loader.get_validation_data()
    print(f"\nTraining: {len(df)} eyes | External validation: {len(val)} eyes "
          f"({(val.cohort=='Jiang').sum()} Jiang + {(val.cohort=='Bu').sum()} Bu)")

    model_fitters = {
        "Gamma-GLM(log)": _fit_gamma_log,
        "log-OLS(smear)": _fit_logols,
    }
    baselines = {
        "Published formula (Age,IR)": (_fit_bu, None),
        "Population mean": (_fit_mean, None),
        "Eye-specific centroid": (_fit_eye_centroid_mag, None),
    }

    results = []

    # ── 7.1 Internal CV on the 202 training eyes ───────────────────────────
    print("\n--- 7.1 Internal CV (202 eyes, repeated 10-fold x3) ---")
    print(f"{'Model':<26}{'Predictors':<14}{'R2':>16}{'MAE':>14}{'%neg':>8}")
    for mname, fitter in model_fitters.items():
        for pname, preds in PREDICTOR_SETS.items():
            cvm = _cv(df, fitter, preds)
            results.append({"stage": "internal_cv", "model": mname, "predictors": pname,
                            "R2_mean": cvm["R2"][0], "R2_sd": cvm["R2"][1],
                            "MAE_mean": cvm["MAE"][0], "MAE_sd": cvm["MAE"][1],
                            "RMSE_mean": cvm["RMSE"][0], "pct_negative": cvm["pct_negative"][0]})
            print(f"{mname:<26}{pname:<14}{cvm['R2'][0]:>8.3f}±{cvm['R2'][1]:.3f}"
                  f"{cvm['MAE'][0]:>8.3f}±{cvm['MAE'][1]:.3f}{cvm['pct_negative'][0]:>8.1f}")
    for bname, (fitter, _) in baselines.items():
        cvm = _cv(df, fitter, None)
        results.append({"stage": "internal_cv", "model": bname, "predictors": "-",
                        "R2_mean": cvm["R2"][0], "R2_sd": cvm["R2"][1],
                        "MAE_mean": cvm["MAE"][0], "MAE_sd": cvm["MAE"][1],
                        "RMSE_mean": cvm["RMSE"][0], "pct_negative": cvm["pct_negative"][0]})
        print(f"{bname:<26}{'-':<14}{cvm['R2'][0]:>8.3f}±{cvm['R2'][1]:.3f}"
              f"{cvm['MAE'][0]:>8.3f}±{cvm['MAE'][1]:.3f}{cvm['pct_negative'][0]:>8.1f}")

    # ── 7.2 External validation: fit on all 202, predict on the 54 ─────────
    print("\n--- 7.2 External validation (fit on 202, test on 54 held-out) ---")
    print(f"{'Model':<26}{'Predictors':<14}{'cohort':<10}{'R2':>9}{'MAE':>9}{'%neg':>7}")

    def eval_external(label, fitter, preds, mname, pname):
        predict = fitter(df, preds)
        for cohort, sub in [("combined", val), ("Jiang", val[val.cohort == "Jiang"]),
                            ("Bu", val[val.cohort == "Bu"])]:
            m = _metrics(sub["CSIA_mag"].values, predict(sub))
            results.append({"stage": "external", "model": mname, "predictors": pname,
                            "cohort": cohort, "R2_mean": m["R2"], "MAE_mean": m["MAE"],
                            "RMSE_mean": m["RMSE"], "pct_negative": m["pct_negative"],
                            "pred_min": m["pred_min"], "pred_max": m["pred_max"], "n": m["n"]})
            print(f"{mname:<26}{pname:<14}{cohort:<10}{m['R2']:>9.3f}{m['MAE']:>9.3f}{m['pct_negative']:>7.1f}")

    for mname, fitter in model_fitters.items():
        for pname, preds in PREDICTOR_SETS.items():
            eval_external(mname, fitter, preds, mname, pname)
    for bname, (fitter, _) in baselines.items():
        eval_external(bname, fitter, None, bname, "-")

    # ── 7.3 Final formula coefficients (Gamma-GLM, fit on all 202) ─────────
    print("\n--- 7.3 Final non-negative formula coefficients (Gamma GLM, log link) ---")
    coef_rows = []
    for pname, preds in PREDICTOR_SETS.items():
        predict = _fit_gamma_log(df, preds)
        res = predict.result
        ci = res.conf_int()
        terms = []
        for term in res.params.index:
            b = res.params[term]
            lo, hi = ci.loc[term]
            coef_rows.append({"predictor_set": pname, "term": term, "coef": b,
                              "ci_low": lo, "ci_high": hi, "p": res.pvalues[term]})
            terms.append(f"{b:+.4f}*{term}" if term != "const" else f"{b:+.4f}")
        formula = "CSIA = exp(" + " ".join(terms).replace("*const", "") + ")"
        print(f"  [{pname}] {formula}")
        print(f"           AIC={res.aic:.1f}  (predictions strictly > 0 by construction)")

    # ── 7.4 Direction: external centroid validation ────────────────────────
    print("\n--- 7.4 Direction (CSIA vector) on the 54 external eyes ---")
    # Predict each external eye's (J0,J45) from training centroids.
    pop = vector_math.compute_centroid(df["CSIA_mag"].values, df["CSIA_meridian"].values)
    eye_cent = {}
    for eye in ["OD", "OS"]:
        sub = df[df["eye"] == eye]
        eye_cent[eye] = vector_math.compute_centroid(sub["CSIA_mag"].values, sub["CSIA_meridian"].values)
    dir_rows = []
    for strat, getter in [
        ("Population centroid", lambda r: (pop["mean_J0"], pop["mean_J45"])),
        ("Eye-specific centroid", lambda r: (eye_cent[r.eye]["mean_J0"], eye_cent[r.eye]["mean_J45"])),
    ]:
        errs = []
        for _, r in val.iterrows():
            pj0, pj45 = getter(r)
            errs.append(vector_math.vector_error(pj0, pj45, r.J0, r.J45))
        errs = np.array(errs)
        dir_rows.append({"strategy": strat, "mean_VE": errs.mean(), "sd_VE": errs.std(),
                         "median_VE": np.median(errs)})
        print(f"  {strat:<24} mean vector error = {errs.mean():.4f} D  (median {np.median(errs):.4f})")

    # ── 7.5 Save tables + calibration figure ───────────────────────────────
    res_df = pd.DataFrame(results)
    tables.save_table(res_df, "table17_formula_validation")
    tables.save_table(pd.DataFrame(coef_rows), "table18_formula_coefficients")
    tables.save_table(pd.DataFrame(dir_rows), "table19_direction_external")
    print("\n  Saved table17_formula_validation, table18_formula_coefficients, table19_direction_external")

    # External calibration: refined formula vs published formula on the 54.
    try:
        gamma_pred = _fit_gamma_log(df, PREDICTOR_SETS["Age+IR"])(val)
        bu_pred = _fit_bu(df)(val)
        _plot_calibration(val["CSIA_mag"].values, gamma_pred, bu_pred)
        print("  Saved fig19_formula_calibration")
    except Exception as e:
        print(f"  calibration figure failed: {e}")

    print("\nAnalysis 7 complete.")
    return {"results": res_df, "coefficients": pd.DataFrame(coef_rows)}


def _plot_calibration(actual, refined, published):
    """Predicted vs actual on external eyes: refined (non-negative) vs published."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lim = [min(0, published.min(), actual.min()) - 0.1,
           max(actual.max(), refined.max(), published.max()) + 0.1]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6, label="ideal (y=x)")
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5)
    ax.scatter(actual, published, s=22, c=config.COLOR_OS, alpha=0.7,
               label="Published formula", edgecolor="none")
    ax.scatter(actual, refined, s=22, c=config.COLOR_OD, alpha=0.7,
               label="Refined Gamma-GLM (Age, IR)", edgecolor="none")
    ax.set_xlabel("Observed CSIA magnitude (D)")
    ax.set_ylabel("Predicted CSIA magnitude (D)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    plotting.save_figure(fig, "fig19_formula_calibration")
    plt.close(fig)
