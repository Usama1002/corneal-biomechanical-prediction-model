"""Analysis 3: Predictive modeling — nested CV, SHAP, ablation studies."""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

from . import config, vector_math, utils, plotting, tables, data_loader


def _get_model_configs():
    """Return model configurations for grid search."""
    return {
        "OLS": {
            "model": LinearRegression(),
            "param_grid": {},  # no tuning
        },
        "ElasticNet": {
            "model": ElasticNet(max_iter=10000, random_state=config.RANDOM_SEED),
            "param_grid": {
                "model__alpha": [0.01, 0.1, 1.0],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=config.RANDOM_SEED, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [200],
                "model__max_depth": [3, 5],
                "model__min_samples_leaf": [5, 10],
            },
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(random_state=config.RANDOM_SEED, verbosity=0, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8],
            },
        },
    }


def _run_nested_cv(X, y, model_name, model_config, cv_outer, is_multi=False):
    """Run nested cross-validation for a single model configuration.

    Returns list of per-fold results.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model_config["model"]),
    ])
    param_grid = model_config["param_grid"]
    fold_results = []
    all_preds = np.full_like(y, np.nan, dtype=float) if not is_multi else np.full((len(y), 2), np.nan)

    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if param_grid:
            inner_cv = config.CV_INNER_FOLDS
            gs = GridSearchCV(
                pipe, param_grid, cv=inner_cv, scoring="neg_mean_absolute_error",
                n_jobs=-1, refit=True,
            )
            gs.fit(X_train, y_train)
            best_pipe = gs.best_estimator_
        else:
            best_pipe = pipe
            best_pipe.fit(X_train, y_train)

        preds = best_pipe.predict(X_test)

        if is_multi:
            all_preds[test_idx] = preds
            # compute per-component metrics
            for comp_idx, comp_name in enumerate(["J0", "J45"]):
                fold_results.append({
                    "fold": fold_idx, "component": comp_name,
                    "R2": r2_score(y_test[:, comp_idx], preds[:, comp_idx]),
                    "MAE": mean_absolute_error(y_test[:, comp_idx], preds[:, comp_idx]),
                    "RMSE": np.sqrt(mean_squared_error(y_test[:, comp_idx], preds[:, comp_idx])),
                })
        else:
            all_preds[test_idx] = preds
            fold_results.append({
                "fold": fold_idx,
                "R2": r2_score(y_test, preds),
                "MAE": mean_absolute_error(y_test, preds),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            })

    return fold_results, all_preds


def _run_bu_formula(df, cv_outer, X_dummy):
    """Evaluate Yin et al. (2025) formula on the same CV folds."""
    age = df["age"].values
    ir = df["IR"].values
    actual = df["CSIA_mag"].values
    pred_bu = config.BU_FORMULA(age, ir)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X_dummy)):
        y_test = actual[test_idx]
        p_test = pred_bu[test_idx]
        fold_results.append({
            "fold": fold_idx,
            "R2": r2_score(y_test, p_test),
            "MAE": mean_absolute_error(y_test, p_test),
            "RMSE": np.sqrt(mean_squared_error(y_test, p_test)),
        })
    return fold_results, pred_bu


def _aggregate_folds(fold_results, model_name, target, feature_set):
    """Aggregate per-fold results into a summary row."""
    df_folds = pd.DataFrame(fold_results)
    row = {
        "model": model_name, "target": target, "feature_set": feature_set,
    }
    for metric in ["R2", "MAE", "RMSE"]:
        if metric in df_folds.columns:
            row[f"{metric}_mean"] = df_folds[metric].mean()
            row[f"{metric}_std"] = df_folds[metric].std()
    return row


def run(df):
    """Run all predictive modeling experiments and save outputs."""
    print("=" * 60)
    print("ANALYSIS 3: Predictive Modeling")
    print("=" * 60)

    model_configs = _get_model_configs()
    all_results = []
    all_predictions = {}

    # Feature sets
    X_biomech, names_biomech = data_loader.get_feature_matrix(df, "biomech")
    X_demo, names_demo = data_loader.get_feature_matrix(df, "biomech_demo")

    # Determine reduced feature set via ElasticNetCV on full data
    print("\n--- Determining reduced feature set via ElasticNetCV ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_demo)
    enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5,
                            max_iter=10000, random_state=config.RANDOM_SEED)
    enet_cv.fit(X_scaled, df["CSIA_mag"].values)
    selected_mask = np.abs(enet_cv.coef_) > 1e-6
    reduced_names = [n for n, s in zip(names_demo, selected_mask) if s]
    print(f"  Reduced features ({len(reduced_names)}): {reduced_names}")
    if not reduced_names:
        reduced_names = ["age", "eye_binary"]  # fallback
        print(f"  Fallback to: {reduced_names}")
    X_reduced = df[reduced_names].values.astype(float)

    feature_sets = {
        "biomech": (X_biomech, names_biomech),
        "biomech_demo": (X_demo, names_demo),
        "reduced": (X_reduced, reduced_names),
    }

    # Targets
    y_mag = df["CSIA_mag"].values
    y_j0 = df["J0"].values
    y_j45 = df["J45"].values
    y_vec = np.column_stack([y_j0, y_j45])

    cv_outer = RepeatedKFold(
        n_splits=config.CV_OUTER_FOLDS,
        n_repeats=3,  # 3 repeats x 10 folds = 30 evaluations
        random_state=config.RANDOM_SEED,
    )

    # ── 3.1 Main experiment loop ───────────────────────────────────────
    print("\n--- 3.1 Nested CV experiments ---")
    print(f"{'Model':<15} {'Target':<12} {'Features':<15} {'R2':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 72)

    for model_name, model_config in model_configs.items():
        for target_name, y_target in [("Magnitude", y_mag), ("J0", y_j0), ("J45", y_j45)]:
            for fs_name, (X_fs, fn) in feature_sets.items():
                fold_res, preds = _run_nested_cv(X_fs, y_target, model_name, model_config, cv_outer)
                summary = _aggregate_folds(fold_res, model_name, target_name, fs_name)
                all_results.append(summary)

                pred_key = f"{model_name}_{target_name}_{fs_name}"
                all_predictions[pred_key] = preds

                print(f"{model_name:<15} {target_name:<12} {fs_name:<15} "
                      f"{summary['R2_mean']:>7.3f}±{summary['R2_std']:.3f} "
                      f"{summary['MAE_mean']:>7.3f}±{summary['MAE_std']:.3f} "
                      f"{summary['RMSE_mean']:>7.3f}±{summary['RMSE_std']:.3f}",
                      flush=True)

    # ── 3.2 Multi-output (J0 + J45 joint) ─────────────────────────────
    print("\n--- 3.2 Multi-output regression (J0 + J45 joint) ---")
    for model_name in ["ElasticNet", "RandomForest"]:
        mc = _get_model_configs()[model_name]
        multi_model = MultiOutputRegressor(mc["model"])
        multi_pipe = Pipeline([("scaler", StandardScaler()), ("model", multi_model)])

        # simplified CV for multi-output (no grid search, too slow)
        cv_simple = RepeatedKFold(n_splits=10, n_repeats=3, random_state=config.RANDOM_SEED)
        fold_res_j0 = []
        fold_res_j45 = []

        for train_idx, test_idx in cv_simple.split(X_demo):
            multi_pipe.fit(X_demo[train_idx], y_vec[train_idx])
            preds = multi_pipe.predict(X_demo[test_idx])
            fold_res_j0.append({
                "R2": r2_score(y_vec[test_idx, 0], preds[:, 0]),
                "MAE": mean_absolute_error(y_vec[test_idx, 0], preds[:, 0]),
            })
            fold_res_j45.append({
                "R2": r2_score(y_vec[test_idx, 1], preds[:, 1]),
                "MAE": mean_absolute_error(y_vec[test_idx, 1], preds[:, 1]),
            })

        for comp, folds in [("J0_joint", fold_res_j0), ("J45_joint", fold_res_j45)]:
            df_f = pd.DataFrame(folds)
            all_results.append({
                "model": f"{model_name}_multi", "target": comp, "feature_set": "biomech_demo",
                "R2_mean": df_f["R2"].mean(), "R2_std": df_f["R2"].std(),
                "MAE_mean": df_f["MAE"].mean(), "MAE_std": df_f["MAE"].std(),
                "RMSE_mean": np.nan, "RMSE_std": np.nan,
            })
            print(f"  {model_name}_multi -> {comp}: R²={df_f['R2'].mean():.3f}±{df_f['R2'].std():.3f}, "
                  f"MAE={df_f['MAE'].mean():.3f}±{df_f['MAE'].std():.3f}")

    # ── 3.3 Yin et al. (2025) formula baseline ────────────────────────────────
    print("\n--- 3.3 Yin et al. (2025) formula baseline ---")
    bu_folds, bu_preds = _run_bu_formula(df, cv_outer, X_biomech)
    bu_summary = _aggregate_folds(bu_folds, "Bu_Formula", "Magnitude", "Age+IR")
    all_results.append(bu_summary)
    all_predictions["Bu_Formula"] = bu_preds
    print(f"  Bu Formula: R²={bu_summary['R2_mean']:.3f}±{bu_summary['R2_std']:.3f}, "
          f"MAE={bu_summary['MAE_mean']:.3f}±{bu_summary['MAE_std']:.3f}")

    # ── 3.4 Ablation: OD-only and OS-only models ──────────────────────
    print("\n--- 3.4 Ablation: Stratified by eye ---")
    for eye_label, eye_code in [("OD", "OD"), ("OS", "OS")]:
        df_eye = df[df["eye"] == eye_code]
        X_eye, _ = data_loader.get_feature_matrix(df_eye, "biomech_demo")
        y_eye_mag = df_eye["CSIA_mag"].values

        cv_eye = RepeatedKFold(n_splits=5, n_repeats=5, random_state=config.RANDOM_SEED)
        for model_name in ["ElasticNet", "RandomForest"]:
            mc = _get_model_configs()[model_name]
            fold_res, _ = _run_nested_cv(X_eye, y_eye_mag, model_name, mc, cv_eye)
            summary = _aggregate_folds(fold_res, model_name, f"Mag_{eye_label}", "biomech_demo")
            all_results.append(summary)
            print(f"  {model_name} ({eye_label}, n={len(df_eye)}): "
                  f"R²={summary['R2_mean']:.3f}±{summary['R2_std']:.3f}, "
                  f"MAE={summary['MAE_mean']:.3f}±{summary['MAE_std']:.3f}")

    # ── 3.5 Ablation: Age subgroups ───────────────────────────────────
    print("\n--- 3.5 Ablation: Age subgroups ---")
    for age_group in df["age_tertile"].cat.categories:
        df_age = df[df["age_tertile"] == age_group]
        if len(df_age) < 30:
            print(f"  {age_group}: skipped (n={len(df_age)})")
            continue
        X_age, _ = data_loader.get_feature_matrix(df_age, "biomech")
        y_age = df_age["CSIA_mag"].values
        cv_age = RepeatedKFold(n_splits=5, n_repeats=3, random_state=config.RANDOM_SEED)
        mc = _get_model_configs()["ElasticNet"]
        fold_res, _ = _run_nested_cv(X_age, y_age, "ElasticNet", mc, cv_age)
        summary = _aggregate_folds(fold_res, "ElasticNet", f"Mag_{age_group}", "biomech")
        all_results.append(summary)
        print(f"  ElasticNet ({age_group}, n={len(df_age)}): "
              f"R²={summary['R2_mean']:.3f}±{summary['R2_std']:.3f}")

    # ── 3.6 SHAP analysis ─────────────────────────────────────────────
    print("\n--- 3.6 SHAP analysis ---")
    shap_results = {}
    try:
        import shap

        for target_name, y_target in [("Magnitude", y_mag), ("J0", y_j0), ("J45", y_j45)]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(
                    n_estimators=300, max_depth=5, min_samples_leaf=10,
                    random_state=config.RANDOM_SEED, n_jobs=-1
                )),
            ])
            pipe.fit(X_demo, y_target)

            X_scaled = pipe.named_steps["scaler"].transform(X_demo)
            explainer = shap.TreeExplainer(pipe.named_steps["model"])
            sv = explainer.shap_values(X_scaled)

            # Mean absolute SHAP
            mean_abs_shap = np.abs(sv).mean(axis=0)
            shap_df = pd.DataFrame({
                "feature": names_demo,
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=False)
            shap_results[target_name] = shap_df

            print(f"\n  Top 5 SHAP features for {target_name}:")
            for _, row in shap_df.head(5).iterrows():
                print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")

            # Save SHAP plot
            X_demo_df = pd.DataFrame(X_scaled, columns=names_demo)
            try:
                plotting.plot_shap_summary(sv, X_demo_df,
                                            filename=f"fig09_shap_{target_name.lower()}")
                print(f"  Saved fig09_shap_{target_name.lower()}")
            except Exception as e:
                print(f"  SHAP plot for {target_name} failed: {e}")
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")

    # ── 3.7 Permutation importance ─────────────────────────────────────
    print("\n--- 3.7 Permutation importance ---")
    perm_results = {}
    for target_name, y_target in [("Magnitude", y_mag), ("J0", y_j0), ("J45", y_j45)]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=5, min_samples_leaf=10,
                random_state=config.RANDOM_SEED, n_jobs=-1
            )),
        ])
        pipe.fit(X_demo, y_target)

        perm = permutation_importance(
            pipe, X_demo, y_target, n_repeats=30,
            random_state=config.RANDOM_SEED, n_jobs=-1, scoring="r2",
        )
        perm_results[target_name] = perm

        try:
            plotting.plot_permutation_importance(
                perm, names_demo,
                filename=f"fig10_perm_importance_{target_name.lower()}")
            print(f"  Saved fig10_perm_importance_{target_name.lower()}")
        except Exception:
            pass

    # ── 3.8 Bland-Altman for best model ───────────────────────────────
    print("\n--- 3.8 Bland-Altman ---")
    # Use ElasticNet biomech_demo out-of-fold predictions for magnitude
    best_pred_key = "ElasticNet_Magnitude_biomech_demo"
    if best_pred_key in all_predictions:
        # get out-of-fold predictions from first repeat only
        cv_single = RepeatedKFold(n_splits=10, n_repeats=1, random_state=config.RANDOM_SEED)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=10000,
                                    random_state=config.RANDOM_SEED)),
        ])
        oof_preds = cross_val_predict(pipe, X_demo, y_mag, cv=cv_single)
        plotting.plot_bland_altman(oof_preds, y_mag, filename="fig12_bland_altman_magnitude")
        print("  Saved fig12_bland_altman_magnitude")

        ba_stats = utils.bland_altman_stats(oof_preds, y_mag)
        print(f"  Mean diff: {ba_stats['mean_diff']:.4f}, LoA: [{ba_stats['lower_loa']:.4f}, {ba_stats['upper_loa']:.4f}]")

    # ── 3.9 Generate summary figures ──────────────────────────────────
    print("\n--- 3.9 Summary figures ---")
    results_df = pd.DataFrame(all_results)

    # Model comparison for magnitude
    mag_results = results_df[results_df["target"] == "Magnitude"].copy()
    mag_results = mag_results[mag_results["feature_set"].isin(["biomech", "biomech_demo", "reduced"])]
    if not mag_results.empty:
        plotting.plot_model_comparison(mag_results, "R2_mean", filename="fig08_model_comparison_magnitude",
                                      title="Cross-validated $R^2$: CSIA Magnitude")
        print("  Saved fig08_model_comparison_magnitude")

    # Model comparison for J0
    j0_results = results_df[results_df["target"] == "J0"].copy()
    j0_results = j0_results[j0_results["feature_set"].isin(["biomech", "biomech_demo", "reduced"])]
    if not j0_results.empty:
        plotting.plot_model_comparison(j0_results, "R2_mean", filename="fig08_model_comparison_J0",
                                      title="Cross-validated $R^2$: CSIA $J_0$ Component")
        print("  Saved fig08_model_comparison_J0")

    # Model comparison for J45
    j45_results = results_df[results_df["target"] == "J45"].copy()
    j45_results = j45_results[j45_results["feature_set"].isin(["biomech", "biomech_demo", "reduced"])]
    if not j45_results.empty:
        plotting.plot_model_comparison(j45_results, "R2_mean", filename="fig08_model_comparison_J45",
                                      title="Cross-validated $R^2$: CSIA $J_{45}$ Component")
        print("  Saved fig08_model_comparison_J45")

    # MAE comparison for magnitude
    if not mag_results.empty:
        plotting.plot_model_comparison(mag_results, "MAE_mean", filename="fig08_mae_comparison_magnitude",
                                      title="Cross-validated MAE: CSIA Magnitude")
        print("  Saved fig08_mae_comparison_magnitude")

    # ── 3.10 Tables ───────────────────────────────────────────────────
    print("\n--- 3.10 Tables ---")
    tables.save_table(results_df, "table07_cv_results")
    print("  Saved table07_cv_results")

    if shap_results:
        shap_all = pd.concat(
            [v.assign(target=k) for k, v in shap_results.items()],
            ignore_index=True,
        )
        tables.save_table(shap_all.set_index(["target", "feature"]), "table08_shap_importance")
        print("  Saved table08_shap_importance")

    # Save raw predictions
    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv(config.OUTPUT_RESULTS / "cv_predictions.csv", index=False)

    # Save reduced feature set info
    with open(config.OUTPUT_RESULTS / "reduced_features.txt", "w") as f:
        f.write(f"Reduced features selected by ElasticNetCV:\n")
        f.write(f"{reduced_names}\n")

    print("\nAnalysis 3 complete.")
    return {"results_df": results_df, "predictions": all_predictions, "shap": shap_results}
