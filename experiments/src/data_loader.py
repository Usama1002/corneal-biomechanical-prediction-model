"""Data ingestion: Excel parsing, cleaning, derived features."""

import numpy as np
import pandas as pd

from . import config
from . import vector_math


def load_raw_data(sheet_name="All(OD+OS)"):
    """Load and clean the raw Excel file into a flat DataFrame."""
    df = pd.read_excel(config.DATA_RAW, sheet_name=sheet_name, header=None, skiprows=2)

    # Apply column map
    df = df.rename(columns=config.COLUMN_MAP)
    # keep only mapped columns
    df = df[[config.COLUMN_MAP[i] for i in sorted(config.COLUMN_MAP.keys())]]

    # Cast numeric columns
    numeric_cols = (
        config.ALL_PREOP_PARAMS
        + ["age", "iol_diopter", "CSIA_mag", "CSIA_meridian"]
    )
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cast categoricals
    df["sex"] = df["sex"].astype(str).str.strip()
    df["eye"] = df["eye"].astype(str).str.strip()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    return df


def add_vector_components(df):
    """Add J0 and J45 columns from CSIA magnitude + meridian."""
    df = df.copy()
    j0, j45 = vector_math.decompose_to_j0_j45(df["CSIA_mag"], df["CSIA_meridian"])
    df["J0"] = j0
    df["J45"] = j45
    return df


def add_derived_features(df):
    """Add binary encodings and age tertiles."""
    df = df.copy()
    df["sex_binary"] = (df["sex"] == "F").astype(int)
    df["eye_binary"] = (df["eye"] == "OS").astype(int)

    # Age tertiles
    t1, t2 = np.quantile(df["age"].dropna(), [1 / 3, 2 / 3])
    df["age_tertile"] = pd.cut(
        df["age"], bins=[-np.inf, t1, t2, np.inf], labels=["young", "middle", "old"]
    )
    df["age_tertile_boundaries"] = f"{t1:.0f},{t2:.0f}"  # metadata

    return df


def get_clean_data():
    """Full pipeline: load → clean → add vectors → add derived → save.

    Returns the cleaned DataFrame.
    """
    df = load_raw_data()
    df = add_vector_components(df)
    df = add_derived_features(df)

    # Save processed
    out_path = config.DATA_PROCESSED / "all_eyes.csv"
    df.to_csv(out_path, index=False)

    return df


def get_subsets(df):
    """Return OD and OS subsets."""
    od = df[df["eye"] == "OD"].copy()
    os_ = df[df["eye"] == "OS"].copy()
    return od, os_


def get_patient_groups(df):
    """Return mapping from patient_id to list of row indices (for mixed-effects grouping).

    Also returns summary stats about multi-entry patients.
    """
    groups = df.groupby("patient_id")
    multi = {name: grp for name, grp in groups if len(grp) > 1}
    summary = []
    for name, grp in multi.items():
        summary.append({
            "patient_id": name,
            "n_entries": len(grp),
            "eyes": grp["eye"].tolist(),
            "ages": grp["age"].tolist(),
            "type": "bilateral" if grp["eye"].nunique() > 1 else "same_eye_repeat",
        })
    return pd.DataFrame(summary) if summary else pd.DataFrame()


def get_feature_matrix(df, feature_set="biomech_demo"):
    """Return (X, feature_names) for a given feature set.

    feature_set: "biomech", "biomech_demo", or a list of column names
    """
    if feature_set == "biomech":
        cols = config.BIOMECH_FEATURES
    elif feature_set == "biomech_demo":
        cols = config.BIOMECH_DEMO_FEATURES
    elif isinstance(feature_set, list):
        cols = feature_set
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    # only keep columns that exist
    cols = [c for c in cols if c in df.columns]
    X = df[cols].values.astype(float)
    return X, cols
