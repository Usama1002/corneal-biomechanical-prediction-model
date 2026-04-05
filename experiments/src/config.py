"""Central configuration: paths, column names, constants, plotting style."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
DATA_RAW = PROJECT_ROOT / "data" / "Corvis Data +CSIA.xlsx"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

OUTPUT_FIGURES = EXPERIMENTS_ROOT / "outputs" / "figures"
OUTPUT_TABLES = EXPERIMENTS_ROOT / "outputs" / "tables"
OUTPUT_RESULTS = EXPERIMENTS_ROOT / "outputs" / "results"

for d in [DATA_PROCESSED, OUTPUT_FIGURES, OUTPUT_TABLES, OUTPUT_RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Column mapping (Excel col index → clean name) ─────────────────────────
COLUMN_MAP = {
    0: "patient_id", 1: "sex", 2: "age", 3: "eye",
    4: "iol_type", 5: "iol_diopter",
    6: "PCT135", 7: "AL", 8: "WTW",
    9: "AL1", 10: "AV1", 11: "AL2", 12: "AV2",
    13: "PD", 14: "HCR", 15: "HCDA",
    16: "IOPnc", 17: "bIOP",
    18: "CCT", 19: "SPA1", 20: "ARTh",
    21: "DA_Ratio", 22: "IR", 23: "CBI", 24: "SSI", 25: "CBiF",
    26: "CSIA_mag", 27: "CSIA_meridian",
}

# ── Feature groups ─────────────────────────────────────────────────────────
CORVIS_PARAMS = [
    "AL1", "AV1", "AL2", "AV2", "PD", "HCR", "HCDA",
    "IOPnc", "bIOP", "CCT", "SPA1", "ARTh",
    "DA_Ratio", "IR", "CBI", "SSI", "CBiF",
]

BIOMETRIC_PARAMS = ["PCT135", "AL", "WTW"]

ALL_PREOP_PARAMS = CORVIS_PARAMS + BIOMETRIC_PARAMS  # 20 variables

BIOMECH_FEATURES = ALL_PREOP_PARAMS  # feature set 1
BIOMECH_DEMO_FEATURES = ALL_PREOP_PARAMS + ["age", "sex_binary", "eye_binary"]  # feature set 2
# Feature set 3 (reduced) determined dynamically by Elastic Net

DEMOGRAPHIC_COLS = ["patient_id", "sex", "age", "eye", "iol_type", "iol_diopter"]
TARGET_COLS = ["CSIA_mag", "CSIA_meridian"]
VECTOR_COLS = ["J0", "J45"]

# ── Reference values from Yin et al. (2025) ────────────────────────────────────
YIN2025_CENTROID_MAG = 0.48
YIN2025_CENTROID_MERIDIAN = 43.0
YIN2025_FORMULA = lambda age, ir: 0.13 + 0.01 * age - 0.09 * ir
# Aliases for backward compatibility
BU_CENTROID_MAG = YIN2025_CENTROID_MAG
BU_CENTROID_MERIDIAN = YIN2025_CENTROID_MERIDIAN
BU_FORMULA = YIN2025_FORMULA

# ── Clinical defaults ──────────────────────────────────────────────────────
CLINICAL_DEFAULT_MAG = 0.25
CLINICAL_DEFAULT_MERIDIAN = 45.0

# ── Cross-validation ───────────────────────────────────────────────────────
RANDOM_SEED = 42
CV_OUTER_FOLDS = 10
CV_INNER_FOLDS = 5
CV_REPEATS = 5

# ── Plotting ───────────────────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMATS = ["png", "pdf"]

# Colorblind-safe palette
COLOR_OD = "#0072B2"       # blue
COLOR_OS = "#D55E00"       # orange
COLOR_OVERALL = "#333333"  # dark grey
COLOR_CENTROID = "#CC0000" # red
COLOR_ACCENT = "#009E73"   # green

PALETTE = {
    "OD": COLOR_OD,
    "OS": COLOR_OS,
    "overall": COLOR_OVERALL,
    "centroid": COLOR_CENTROID,
    "accent": COLOR_ACCENT,
}

def setup_matplotlib():
    """Apply publication-quality matplotlib defaults."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 100,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,       # editable text in PDF
        "ps.fonttype": 42,
    })
