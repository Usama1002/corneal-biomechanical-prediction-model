# Vector Analysis of CSIA and Corneal Biomechanical Properties

This repository contains the de-identified dataset and complete analysis code for the paper:

> **Vector Analysis of Surgically Induced Corneal Astigmatism and Its Association with Preoperative Corneal Biomechanical Properties: Eye Laterality as the Dominant Directional Factor**
>
> [Authors and journal details to be added upon publication]

## Overview

This study investigates the relationship between preoperative corneal biomechanical parameters (measured by Corvis ST) and surgically induced corneal astigmatism (CSIA) in cataract surgery. Unlike prior work that analyzed only CSIA magnitude, we analyze the full CSIA vector (both magnitude and meridian) using double-angle vector decomposition into Cartesian components (J0, J45).

### Key findings

- **Eye laterality (OD vs OS) is the strongest predictor of CSIA direction** (J0 difference: p < 0.0001, Cohen's d = 0.62), dominating all 20 Corvis ST biomechanical parameters in MANOVA (Wilks' lambda = 0.87, F = 13.26, p < 0.0001).
- **Individual-level CSIA prediction from preoperative biomechanics is not achievable**: all four model families (OLS, Elastic Net, Random Forest, XGBoost) yield negative cross-validated R-squared values.
- **Eye-specific subgroup centroids** provide a practical, statistically validated improvement over fixed population estimates (Wilcoxon p = 0.005).

## Dataset

The file `data/Corvis Data +CSIA.xlsx` contains de-identified clinical data from **202 eyes** (92 OD, 110 OS) of **182 cataract patients** who underwent phacoemulsification with a 2.2 mm clear corneal incision at 135 degrees. Patient identifiers have been replaced with anonymous codes (P001, P002, ...) to protect privacy. All clinical measurements are unmodified.

### Variables

**Demographics (4):**

| Variable | Description |
|----------|-------------|
| patient_id | De-identified patient identifier (P001, P002, ...) |
| sex | M = male, F = female |
| age | Patient age at surgery (years) |
| eye | OD = right eye, OS = left eye |

**IOL information (2):**

| Variable | Description |
|----------|-------------|
| iol_type | Intraocular lens model implanted |
| iol_diopter | IOL power (D) |

**Corvis ST dynamic corneal response parameters (17):**

| Variable | Full name | Unit | Interpretation |
|----------|-----------|------|----------------|
| AL1 | Applanation length, 1st applanation | mm | Cord length of flattened cornea at first applanation |
| AV1 | Applanation velocity, 1st applanation | m/s | Corneal speed during first applanation |
| AL2 | Applanation length, 2nd applanation | mm | Cord length at second applanation |
| AV2 | Applanation velocity, 2nd applanation | m/s | Corneal speed during second applanation |
| PD | Peak distance | mm | Distance between corneal peaks at highest concavity |
| HCR | Highest concavity radius | mm | Central concave curvature at maximum deformation |
| HCDA | Highest concavity deformation amplitude | mm | Apex displacement from initial position to max deformation |
| IOPnc | Uncorrected intraocular pressure | mmHg | Raw IOP from Corvis ST |
| bIOP | Biomechanically corrected IOP | mmHg | IOP corrected for corneal thickness and age effects |
| CCT | Central corneal thickness | um | Corneal thickness at center |
| SPA1 | Stiffness parameter at 1st applanation | mmHg/mm | Load-displacement ratio at first applanation |
| ARTh | Ambrosio relational thickness horizontal | um | Thickness progression ratio (thinnest to peripheral) |
| DA_Ratio | Deformation amplitude ratio | -- | Central to peripheral deformation amplitude ratio |
| IR | Integrated radius | mm^-1 | Integrated inverse radius during concavity phase |
| CBI | Corneal biomechanical index | -- | Composite index for ectasia screening |
| SSI | Stress-strain index | -- | Material stiffness estimate from stress-strain curve |
| CBiF | Corneal biomechanical factor | -- | Overall biomechanical stability composite |

**Additional biometric parameters (3):**

| Variable | Full name | Unit | Measured by |
|----------|-----------|------|-------------|
| PCT135 | Peripheral corneal thickness at 135 degrees | um | CASIA2 |
| AL | Axial length | mm | IOL Master 700 |
| WTW | White-to-white corneal diameter | mm | IOL Master 700 |

**Target variables (2):**

| Variable | Description | Unit |
|----------|-------------|------|
| CSIA magnitude | Surgically induced corneal astigmatism magnitude | D (diopters) |
| CSIA meridian | Surgically induced corneal astigmatism axis | degrees (0-180) |

CSIA was computed from pre- and one-month postoperative total corneal astigmatism (CASIA2) using Alpins vector analysis via the [ASSORT online calculator](https://ascrs.org/tools/corneal-sia-tool).

### Data structure

The Excel file contains three sheets:
- `All(OD+OS)`: All 202 eyes (primary analysis sheet)
- `OD`: 92 right eyes only
- `OS`: 110 left eyes only

Each sheet has a two-row header (row 1 = parameter groups, row 2 = specific variable names). Data begins at row 3.

### Patient structure

19 of the 182 patients contribute multiple entries:
- 11 patients had bilateral surgery (one OD + one OS entry)
- 7 patients had same-eye repeat surgery at different times
- 1 patient has three entries

This non-independence is accounted for in the analysis via linear mixed-effects models with a random intercept for patient.

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/corneal-biomechanical-prediction-model.git
cd corneal-biomechanical-prediction-model

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r experiments/requirements.txt
```

### Requirements

- Python 3.8 or later
- Dependencies (see `experiments/requirements.txt`):
  - numpy, pandas, scipy
  - scikit-learn, statsmodels, xgboost
  - matplotlib, seaborn, shap
  - openpyxl, xlsxwriter, tabulate

## Reproducing the analysis

### Quick start

```bash
# Run the complete analysis pipeline (approximately 15 minutes)
python experiments/run_all.py

# Run specific analyses only
python experiments/run_all.py --analyses 1,2,4

# Skip the slow predictive modeling (Analysis 3)
python experiments/run_all.py --skip-modeling
```

All outputs are regenerated from the raw Excel file. The random seed is fixed at 42 for full reproducibility.

### Output structure

Running `run_all.py` produces three output directories:

```
experiments/outputs/
├── figures/     24 figures, each saved as PNG (300 DPI) + PDF
├── tables/      18 tables, each saved as CSV + formatted XLSX
└── results/     8 intermediate result files (MANOVA output, stepwise summaries, etc.)
```

### Analysis pipeline

The pipeline consists of six sequential analyses. Each is a self-contained module that reads the cleaned dataset and writes its outputs to `experiments/outputs/`.

#### Analysis 1: CSIA vector characterization (`analysis1_vector_characterization.py`)

Computes CSIA centroids (overall, OD, OS) with 95% confidence ellipses using eigendecomposition of the (J0, J45) covariance matrix. Tests for normality (Shapiro-Wilk), characterizes the meridian distribution, and compares OD vs OS groups (t-test, Mann-Whitney U, Cohen's d).

**Outputs:** `fig01`-`fig04`, `table01`, `table01b`

**Key result:** OD centroid 0.37 D at 60.2 degrees, OS centroid 0.56 D at 37.9 degrees (J0 difference p < 0.0001).

#### Analysis 2: Association analysis (`analysis2_association.py`)

Evaluates statistical associations using five methods:

1. Bivariate Pearson correlations with Benjamini-Hochberg FDR correction
2. Partial correlations controlling for (i) age, CCT, bIOP and (ii) age, CCT, bIOP, eye laterality
3. Linear mixed-effects models (random intercept for patient, fitted via `statsmodels.MixedLM`)
4. AIC-based forward-backward stepwise regression
5. MANOVA on the joint (J0, J45) vector with all predictors entered simultaneously (Type III)
6. Variance Inflation Factor analysis

**Outputs:** `fig05`-`fig07`, `table02`-`table06`, MANOVA and stepwise summaries in `results/`

**Key result:** MANOVA confirms eye laterality is the only significant predictor of the CSIA vector (Wilks' lambda = 0.87, p < 0.0001).

#### Analysis 3: Predictive modeling (`analysis3_predictive_modeling.py`)

Benchmarks four regression models under nested cross-validation:

- **Models:** OLS, Elastic Net, Random Forest, XGBoost
- **Targets:** CSIA magnitude, J0, J45 (separately and jointly)
- **Feature sets:** Biomechanics only (20 variables), Biomechanics + demographics (+age, sex, eye), Reduced (Elastic Net selected)
- **Validation:** 10-fold outer CV (3 repeats), 5-fold inner CV for hyperparameter tuning
- **Hyperparameter grids:**
  - Elastic Net: alpha in {0.01, 0.1, 1.0}, l1_ratio in {0.1, 0.5, 0.9}
  - Random Forest: n_estimators = 200, max_depth in {3, 5}, min_samples_leaf in {5, 10}
  - XGBoost: n_estimators = 200, max_depth in {3, 5}, learning_rate in {0.05, 0.1}, subsample = 0.8

Also evaluates the formula from Yin et al. (2025): `CSIA = 0.13 + 0.01 * Age - 0.09 * IR`.

Includes OD-only, OS-only, and age-tertile stratified ablations.

Interpretability: SHAP values (TreeExplainer on Random Forest) and permutation importance (30 repeats).

**Outputs:** `fig08`-`fig12`, `table07`-`table08`, `cv_predictions.csv`

**Key result:** All models yield negative cross-validated R-squared for magnitude. Elastic Net selects only age as a non-zero feature.

#### Analysis 4: Subgroup centroid analysis (`analysis4_subgroup_centroids.py`)

Computes CSIA centroids for 12 subgroups (overall, OD, OS, 3 age tertiles, 6 eye-by-age combinations). Compares seven prediction strategies by vector error (Euclidean distance in J0/J45 space):

1. Clinical default (0.25 D at 45 degrees)
2. Yin et al. (2025) formula
3. Population centroid
4. Eye-specific centroid
5. Age-tertile centroid
6. Eye-by-age centroid
7. Mean magnitude at eye-specific centroid direction

Statistical testing via Wilcoxon signed-rank tests with bootstrap 95% confidence intervals.

**Outputs:** `fig13`-`fig14`, `table09`-`table10b`

**Key result:** Eye-by-age centroid achieves the lowest vector error (0.591 D, Wilcoxon p = 0.007 vs population centroid).

#### Analysis 5: Steepening characterization (`analysis5_steepening.py`)

Compares 25 eyes with CSIA meridian above 90 degrees (steepening) against 177 flattening eyes on all preoperative parameters. Uses t-tests or Mann-Whitney U tests with Bonferroni correction. Fits logistic regression under 10-fold CV.

**Outputs:** `fig15`, `table11`-`table11b`

**Key result:** No parameter survives Bonferroni correction. HCDA is not significant (p = 0.571), unlike the finding of Yin et al. (2025) (p = 0.034). Logistic regression AUC = 0.43.

#### Analysis 6: Ablation studies (`analysis6_ablations.py`)

Twelve ablation experiments:

1. Feature category ablation (applanation, concavity, pressure, thickness, stiffness, structural, biometric, demographics)
2. Leave-one-feature-out analysis
3. Cumulative feature addition by SHAP importance rank
4. Feature interaction effects (Age x Eye, Age x IR, etc.)
5. Polynomial features (degree 2)
6. Learning curves (training sizes 36 to 181)
7. Sex-stratified analysis
8. Outlier sensitivity (exclude CSIA magnitude > 2 SD)
9. Steepening threshold sensitivity (75 to 105 degrees)
10. Leave-one-out centroid prediction (unbiased)
11. Multiple error metrics comparison (magnitude, angular, vector)
12. Bilateral patient sensitivity (exclude all multi-entry patients)

**Outputs:** `fig16`-`fig18`, `table12`-`table16`

**Key result:** Demographics alone (R-squared = -0.063) outperform all 20 biomechanical features (R-squared = -0.135). Learning curve is flat. Results robust to all sensitivity checks.

## Code structure

```
experiments/
├── src/
│   ├── config.py              All constants, paths, column mappings, reference values, plot style
│   ├── data_loader.py         Excel parsing, cleaning, derived features, patient grouping
│   ├── vector_math.py         Double-angle decomposition, centroids, confidence ellipses, error metrics
│   ├── utils.py               Partial correlation, FDR, Bland-Altman, AIC stepwise, VIF, bootstrap
│   ├── plotting.py            All publication-quality figure functions (24 figure types)
│   ├── tables.py              Formatted table generators (CSV + auto-width XLSX)
│   ├── analysis1_vector_characterization.py
│   ├── analysis2_association.py
│   ├── analysis3_predictive_modeling.py
│   ├── analysis4_subgroup_centroids.py
│   ├── analysis5_steepening.py
│   └── analysis6_ablations.py
├── outputs/                   Generated by run_all.py (48 figures, 36 tables, 8 result files)
├── run_all.py                 Main orchestrator script
└── requirements.txt           Python dependencies
```

### Configuration

All project-wide constants are in `experiments/src/config.py`:

- **Column mappings**: Maps Excel column indices to clean Python variable names
- **Feature groups**: `CORVIS_PARAMS` (17), `BIOMETRIC_PARAMS` (3), `BIOMECH_FEATURES` (20), `BIOMECH_DEMO_FEATURES` (23)
- **Reference values**: Yin et al. (2025) centroid (0.48 D at 43 degrees) and formula (CSIA = 0.13 + 0.01 * Age - 0.09 * IR)
- **Cross-validation**: `RANDOM_SEED = 42`, `CV_OUTER_FOLDS = 10`, `CV_INNER_FOLDS = 5`
- **Plotting**: 300 DPI, colorblind-safe palette, Arial font

### Vector decomposition

The core mathematical operation is double-angle vector decomposition (implemented in `vector_math.py`):

```
J0  = magnitude * cos(2 * meridian_radians)    # horizontal/vertical component
J45 = magnitude * sin(2 * meridian_radians)    # oblique component
```

This maps the 0-180 degree astigmatism axis to a full 0-360 degree circle, making standard statistical methods (Pearson correlation, linear regression, MANOVA) applicable to the directional data.

## Generated outputs

### Figures (24 total, each as PNG 300 DPI + PDF)

| Figure | Description | Analysis |
|--------|-------------|----------|
| fig01 | Double-angle polar plot (overall) with centroid and 95% CI ellipses | 1 |
| fig02 | Double-angle polar plot by eye laterality (OD vs OS) | 1 |
| fig03 | Rose diagram of CSIA meridian distribution | 1 |
| fig04 | CSIA magnitude histogram with KDE | 1 |
| fig05 | Bivariate correlation heatmap (21 predictors x 3 targets) | 2 |
| fig06 | Forest plot of partial correlations | 2 |
| fig07 | Variance Inflation Factor bar plot | 2 |
| fig08 | Cross-validated R-squared model comparison (Magnitude, J0, J45) | 3 |
| fig09 | SHAP beeswarm plots (Magnitude, J0, J45) | 3 |
| fig10 | Permutation importance (Magnitude, J0, J45) | 3 |
| fig12 | Bland-Altman plot for magnitude prediction | 3 |
| fig13 | Subgroup centroids on double-angle polar plot | 4 |
| fig14 | Strategy comparison box plot (vector error) | 4 |
| fig15 | Steepening vs flattening parameter comparison | 5 |
| fig16 | Learning curve | 6 |
| fig17 | Feature category ablation | 6 |
| fig18 | Cumulative feature addition | 6 |

### Tables (18 total, each as CSV + formatted XLSX)

| Table | Description | Analysis |
|-------|-------------|----------|
| table01 | Demographics and descriptive statistics (overall and by eye) | 1 |
| table01b | CSIA centroid comparison with Yin et al. (2025) | 1 |
| table02 | Bivariate Pearson correlations with FDR-corrected p-values | 2 |
| table03 | Partial correlations (two covariate sets) | 2 |
| table04 | Linear mixed-effects model results | 2 |
| table05 | Stepwise regression models (AIC-based) | 2 |
| table06 | Variance Inflation Factors | 2 |
| table07 | Nested cross-validation results (all models, targets, feature sets) | 3 |
| table08 | SHAP feature importance rankings | 3 |
| table09 | Subgroup centroids (12 subgroups) | 4 |
| table10 | Strategy comparison with Wilcoxon p-values | 4 |
| table10b | Pairwise Wilcoxon signed-rank test results | 4 |
| table11 | Steepening vs flattening comparison (21 parameters) | 5 |
| table11b | Logistic regression coefficients for steepening prediction | 5 |
| table12 | Feature category ablation results | 6 |
| table13 | Leave-one-feature-out analysis | 6 |
| table14 | Cumulative feature addition results | 6 |
| table15 | Feature interaction effects | 6 |
| table16 | Learning curve data | 6 |

## Ethics and data sharing

This study was approved by the ethics committee of Tianjin Medical University Eye Hospital. The requirement for informed consent was waived due to the retrospective design. Patient identifiers have been replaced with anonymous codes. The de-identified dataset is shared in accordance with institutional data sharing policies.

## Citation

If you use this dataset or code in your research, please cite:

```
[Citation to be added upon publication]
```

## License

- **Code**: MIT License
- **Dataset**: CC-BY 4.0

See `LICENSE` for details.
