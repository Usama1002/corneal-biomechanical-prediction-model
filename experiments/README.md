# CSIA Vector Prediction Analysis

Analysis pipeline for investigating the relationship between preoperative corneal biomechanical parameters (Corvis ST) and surgically induced corneal astigmatism (CSIA) in cataract surgery. This extends Yin et al.'s 2025 paper by analyzing the full CSIA vector (magnitude and direction), not magnitude alone.

## Background

During cataract surgery, the corneal incision changes the eye's astigmatism. This change is called CSIA (Corneal Surgically Induced Astigmatism). Predicting CSIA before surgery would improve outcomes with toric intraocular lenses.

Yin et al.'s 2025 paper (Scientific Reports) found that CSIA magnitude could be predicted from two variables: `CSIA = 0.13 + 0.01 * Age - 0.09 * IR`. This project asks a harder question: can we predict both the magnitude and the direction (meridian) of CSIA from preoperative corneal biomechanics?

## Dataset

**Source**: `data/Corvis Data +CSIA.xlsx`

- 202 eyes (92 OD, 110 OS) from 182 patients
- 19 patients contribute multiple entries (11 bilateral, 7 same-eye repeats, 1 triple)
- 3 Excel sheets: `All(OD+OS)`, `OD`, `OS`
- Multi-row headers (rows 0-1 are headers, data starts at row 2)

**Predictor variables** (20 Corvis ST + biometric parameters, highlighted in red in the original Excel):

| Category | Variables |
|---|---|
| Applanation | AL1, AV1, AL2, AV2 |
| Highest concavity | PD, HCR, HCDA |
| Pressure | IOPnc, bIOP |
| Thickness | CCT, PCT135 |
| Stiffness/Biomechanics | SPA1, ARTh, DA Ratio, IR, CBI, SSI, CBiF |
| Biometric | AL (axial length), WTW (white-to-white) |

**Target variables**:
- `CSIA_mag`: CSIA magnitude in diopters
- `CSIA_meridian`: CSIA meridian in degrees (0-180)
- `J0`, `J45`: Cartesian components from double-angle vector decomposition

## Method: double-angle vector decomposition

Astigmatism is a vector quantity. The meridian is circular (0 and 180 degrees are identical), so standard regression on the raw meridian is inappropriate. We decompose each CSIA vector into Cartesian components using the double-angle method:

```
J0  = magnitude * cos(2 * meridian)   (horizontal/vertical component)
J45 = magnitude * sin(2 * meridian)   (oblique component)
```

This maps the 0-180 degree range to a full 0-360 degree circle, making standard statistical methods valid. Predictions are converted back:

```
magnitude = sqrt(J0^2 + J45^2)
meridian  = 0.5 * atan2(J45, J0)     (mapped to 0-180)
```

## Project structure

```
experiments/
├── src/
│   ├── config.py            Central configuration (paths, column names, seeds, plot style)
│   ├── data_loader.py       Excel parsing, cleaning, derived features, patient grouping
│   ├── vector_math.py       Double-angle decomposition, centroids, confidence ellipses, error metrics
│   ├── utils.py             Partial correlation, FDR correction, Bland-Altman, AIC stepwise, VIF, bootstrap
│   ├── plotting.py          All publication-quality figure functions (polar plots, heatmaps, forest plots, etc.)
│   ├── tables.py            Formatted table generators (demographics, correlations, CV results, centroids)
│   ├── analysis1_vector_characterization.py
│   ├── analysis2_association.py
│   ├── analysis3_predictive_modeling.py
│   ├── analysis4_subgroup_centroids.py
│   └── analysis5_steepening.py
├── outputs/
│   ├── figures/             PNG (300 DPI) + PDF for each figure
│   ├── tables/              CSV + XLSX for each table
│   └── results/             Intermediate results (MANOVA output, stepwise summaries, CV predictions)
├── run_all.py               Orchestrator (runs everything end-to-end)
└── requirements.txt         Python dependencies
```

## Running the pipeline

```bash
# Run everything
python experiments/run_all.py

# Run specific analyses only
python experiments/run_all.py --analyses 1,2,4

# Skip the slow predictive modeling (analysis 3)
python experiments/run_all.py --skip-modeling
```

Runtime: approximately 15 minutes for the full pipeline (analysis 3 is the bottleneck due to nested cross-validation with grid search).

## Analyses

### Analysis 1: Vector characterization

Computes CSIA centroids (overall, OD, OS) with 95% confidence ellipses, tests OD vs OS differences, and characterizes the meridian distribution.

**Key outputs**:
- `fig01`: Double-angle polar plot (overall) with centroid and confidence ellipses
- `fig02`: Double-angle polar plot by eye laterality (OD vs OS overlaid)
- `fig03`: Rose diagram of CSIA meridian distribution
- `fig04`: CSIA magnitude histogram with KDE
- `table01`: Descriptive statistics (overall and by eye)
- `table01b`: Centroid comparison (including Yin et al.'s published centroid)

### Analysis 2: Association analysis

Evaluates statistical associations between preoperative parameters and CSIA using multiple methods, replicating and extending Yin et al.'s approach.

**Methods**:
- Bivariate Pearson correlations with Benjamini-Hochberg FDR correction
- Partial correlations adjusting for Age, CCT, bIOP (Yin et al.'s covariates)
- Partial correlations additionally adjusting for Eye laterality
- Linear mixed-effects models (random intercept for patient, handles the 19 multi-entry patients)
- AIC-based stepwise multivariate regression
- MANOVA on the CSIA vector (J0, J45) jointly
- Variance Inflation Factor analysis

**Key outputs**:
- `fig05`: Correlation heatmap (predictors vs magnitude, J0, J45)
- `fig06`: Forest plot of partial correlations
- `fig07`: VIF bar plot
- `table02`-`table06`: Correlation tables, mixed-effects results, stepwise models, VIF values

### Analysis 3: Predictive modeling with ablations

Systematic comparison of four model families under nested cross-validation.

**Models**: OLS, Elastic Net, Random Forest, XGBoost

**Targets**: CSIA magnitude, J0, J45 (separately and jointly via multi-output regression)

**Feature sets**: Biomechanics only (20 variables), Biomechanics + demographics (+ age, sex, eye), Reduced (Elastic Net selected)

**Validation**: 10-fold outer CV repeated 3 times, with 5-fold inner CV for hyperparameter tuning. Metrics: R-squared, MAE, RMSE.

**Ablations**:
- Stratified by eye (OD-only, OS-only models)
- Stratified by age tertile
- Yin et al.'s formula evaluated on the same CV folds
- Multi-output regression (J0 + J45 jointly)

**Interpretability**: SHAP values, permutation importance, partial dependence plots, Bland-Altman analysis.

**Key outputs**:
- `fig08`: Model comparison bar charts (R-squared and MAE for each target)
- `fig09`: SHAP beeswarm plots for magnitude, J0, J45
- `fig10`: Permutation importance bar charts
- `fig12`: Bland-Altman plot for best magnitude model
- `table07`: Full nested CV results (model x target x feature set x metric)
- `table08`: SHAP feature importance rankings

### Analysis 4: Subgroup centroid analysis

Tests whether subgroup-specific CSIA centroids outperform fixed estimates and individualized models.

**Subgroups**: By eye (OD, OS), by age tertile (young, middle, old), and their 6 combinations (OD-young, OD-middle, etc.)

**Strategy comparison** (via vector error in double-angle space):
- Clinical default (0.25 D at 45 degrees)
- Yin et al.'s formula
- Population centroid
- Eye-specific centroid
- Age-tertile centroid
- Eye x Age centroid

Statistical testing via Wilcoxon signed-rank tests and bootstrap 95% confidence intervals.

**Key outputs**:
- `fig13`: Subgroup centroids plotted on a double-angle polar plot
- `fig14`: Box plot comparing vector errors across prediction strategies
- `table09`: All subgroup centroids with sample sizes
- `table10`: Strategy comparison (mean vector error, pairwise Wilcoxon p-values)

### Analysis 5: Steepening case characterization

Compares the 25 eyes (12.4%) with CSIA meridian above 90 degrees (steepening) against the 177 eyes with flattening. Tests all preoperative parameters with Bonferroni correction, evaluates Yin et al.'s HCDA finding, and fits a logistic regression.

**Key outputs**:
- `fig15`: Box plots of top discriminating variables
- `table11`: Full comparison table (flattening vs steepening)
- `table11b`: Logistic regression coefficients

## Key findings

### Our centroid replicates Yin et al.'s result
- Our centroid: 0.44 D at 45.7 degrees (n=202)
- Yin et al. (2025): 0.48 D at 43 degrees (n=149)

### Eye laterality is the strongest predictor of CSIA direction
- OD centroid: 0.37 D at 60.2 degrees
- OS centroid: 0.56 D at 37.9 degrees
- J0 difference: p < 0.0001, Cohen's d = -0.62
- MANOVA: eye is the only significant predictor of the CSIA vector (Wilks' lambda = 0.87, F = 13.26, p < 0.0001)

### Individual-level prediction is limited
- All models yield negative R-squared in cross-validation (worse than predicting the mean)
- Yin et al.'s formula: R-squared = -2.10 on our data
- Best model (Elastic Net, biomech + demographics): R-squared = -0.11, MAE = 0.36 D for magnitude
- SHAP confirms: age (for magnitude) and eye (for direction) are the dominant features

### Subgroup centroids are the best practical approach
- Eye x Age centroid: lowest mean vector error (0.59 D), significantly better than population centroid (Wilcoxon p = 0.007)
- Eye-specific centroid alone: 0.60 D, also significantly better than population centroid (p = 0.005)
- Yin et al.'s formula gives the highest error (0.69 D)

### Steepening is unpredictable
- No preoperative variable survives Bonferroni correction
- HCDA is not significant in our data (p = 0.57), unlike Yin et al.'s finding (p = 0.034)
- Logistic regression AUC = 0.43 (below chance)

## Module reference

### `src/config.py`
All project-wide constants. Column name mappings, feature group definitions, Yin et al.'s reference values, cross-validation parameters, plot style settings. Change paths or random seeds here.

### `src/data_loader.py`
- `get_clean_data()`: Loads the Excel file, applies column mappings, computes J0/J45, adds binary encodings and age tertiles. Saves to `data/processed/all_eyes.csv`. Returns the full DataFrame.
- `get_subsets(df)`: Returns (OD, OS) subsets.
- `get_patient_groups(df)`: Identifies the 19 multi-entry patients and classifies them as bilateral or same-eye repeat.
- `get_feature_matrix(df, feature_set)`: Returns (X, feature_names) for `"biomech"`, `"biomech_demo"`, or a custom list.

### `src/vector_math.py`
- `decompose_to_j0_j45(magnitude, meridian_deg)`: Standard double-angle decomposition.
- `reconstruct_from_j0_j45(j0, j45)`: Inverse, with meridian mapped to 0-180.
- `compute_centroid(magnitudes, meridians_deg)`: Vector mean in double-angle space.
- `compute_confidence_ellipse(j0, j45, confidence, is_centroid)`: 95% confidence ellipse using eigendecomposition of the covariance matrix.
- `vector_error(pred_j0, pred_j45, actual_j0, actual_j45)`: Euclidean distance in J0/J45 space.
- `angular_error(pred_meridian, actual_meridian)`: Handles 0-180 degree wraparound.

### `src/utils.py`
- `partial_correlation(x, y, covariates)`: Residualize both variables on covariates, then Pearson correlate. Returns r, p, and Fisher z-transform 95% CI.
- `fdr_correction(p_values)`: Benjamini-Hochberg procedure.
- `bland_altman_stats(predicted, actual)`: Mean difference, SD, limits of agreement.
- `aic_stepwise(X, y, feature_names)`: Forward-backward AIC selection using statsmodels OLS.
- `compute_vif(X, feature_names)`: Variance Inflation Factors.
- `bootstrap_ci(data, stat_func, n_boot)`: Percentile bootstrap 95% CI.

### `src/plotting.py`
All figure functions. Uses matplotlib with Arial font, 300 DPI, colorblind-safe palette. Every figure is saved as both PNG and PDF via `save_figure(fig, name)`.

Key functions:
- `plot_double_angle_polar(groups, ...)`: Polar scatter plot with centroids and confidence ellipses. Axis labels show actual meridians (0-180 degrees), not double angles.
- `plot_rose_diagram(meridians, ...)`: Circular histogram with steepening zone highlighted.
- `plot_correlation_heatmap(corr_df, p_df, ...)`: Annotated heatmap with significance stars.
- `plot_forest(data, ...)`: Forest plot for partial correlations or effect sizes.
- `plot_bland_altman(predicted, actual, ...)`: Standard Bland-Altman with limits of agreement.
- `plot_model_comparison(results_df, metric, ...)`: Grouped bar chart for model comparison.
- `plot_strategy_boxplot(errors_dict, ...)`: Box plot comparing prediction strategies.
- `plot_shap_summary(shap_values, X_df, ...)`: SHAP beeswarm plot.
- `plot_permutation_importance(importances, feature_names, ...)`: Horizontal bar chart with error bars.

### `src/tables.py`
- `save_table(df, name)`: Saves as CSV + auto-width-adjusted XLSX.
- `build_demographics_table(df)`: Table 1 with overall and by-eye columns, p-values (t-test or Mann-Whitney depending on normality).
- `build_correlation_table(...)`: Formatted r values with significance stars and FDR-corrected p-values.
- `build_cv_results_table(...)`: Model x target x feature set performance summary.
- `build_centroid_table(...)`: Subgroup centroid summary.

## Dependencies

See `requirements.txt`. All are standard scientific Python packages:
numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn, xgboost, shap, openpyxl, xlsxwriter, tabulate.

## Reproducibility

All random operations use `RANDOM_SEED = 42` (set in `config.py`). Running `python experiments/run_all.py` regenerates every output from the raw Excel file. The processed CSV at `data/processed/all_eyes.csv` is an intermediate artifact and can be deleted safely.
