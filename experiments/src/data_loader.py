"""Data ingestion: Excel parsing, cleaning, derived features."""

import re

import numpy as np
import pandas as pd

from . import config
from . import vector_math

_SEX_MAP = {"男": "M", "女": "F", "M": "M", "F": "F",
            "MALE": "M", "FEMALE": "F"}


def _normalize_sex(series):
    """Map sex labels (incl. Chinese 男/女) to 'M'/'F'; 'U' if unrecognized."""
    def norm(v):
        s = str(v).strip()
        return _SEX_MAP.get(s, _SEX_MAP.get(s.upper(), "U"))
    return series.map(norm)


def _parse_pct135(series):
    """Extract the leading thickness number from PCT135 cells.

    Some cohorts store PCT135 as ``"660(4.37mm)"`` (thickness plus the
    measurement distance); the main file stores it as a plain number. Both
    parse to the leading numeric value (660.0).
    """
    def num(v):
        if pd.isna(v):
            return np.nan
        m = re.match(r"\s*(-?[0-9]+(?:\.[0-9]+)?)", str(v))
        return float(m.group(1)) if m else np.nan
    return series.map(num)


def load_raw_data(path=None, sheet_name="All(OD+OS)"):
    """Load and clean the raw Excel file into a flat, de-identified DataFrame.

    Patient identity is taken from the true name (Chinese name) to group
    bilateral eyes correctly, then replaced by an anonymous code (``P001`` ...)
    so that no personally identifying information is written downstream.
    """
    if path is None:
        path = config.DATA_RAW
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, skiprows=2)

    # Apply column map and keep only mapped columns
    df = df.rename(columns=config.COLUMN_MAP)
    df = df[[config.COLUMN_MAP[i] for i in sorted(config.COLUMN_MAP.keys())]]

    # Drop any trailing/blank rows defensively: a valid eye must be OD or OS
    df["eye"] = df["eye"].astype(str).str.strip()
    df = df[df["eye"].isin(["OD", "OS"])].reset_index(drop=True)

    # Normalize sex (handles Chinese labels in the validation cohorts) and
    # parse PCT135 (handles the "660(4.37mm)" string format) before casting.
    df["sex"] = _normalize_sex(df["sex"])
    df["PCT135"] = _parse_pct135(df["PCT135"])

    # Cast numeric columns
    numeric_cols = (
        config.ALL_PREOP_PARAMS
        + ["age", "iol_diopter", "CSIA_mag", "CSIA_meridian"]
    )
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive anonymous patient_id from true identity, then drop PII.
    # The full name is the correct identity key: romanized names and initials
    # can collide across distinct patients (which previously inflated the
    # bilateral count). Codes are assigned in first-appearance order for stability.
    identity = df["_chinese_name"].astype(str).str.strip()
    codes = pd.factorize(identity)[0]
    df["patient_id"] = [f"P{c + 1:03d}" for c in codes]
    df = df.drop(columns=config.PII_COLS)

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


def _load_from_processed(csv_path):
    """Load the de-identified processed CSV and strip derived columns so the
    standard add_vector_components / add_derived_features steps regenerate them
    identically. Used for public reproduction when the name-containing source
    spreadsheets are absent (they are excluded from the public release)."""
    df = pd.read_csv(csv_path)
    derived = ["J0", "J45", "sex_binary", "eye_binary",
               "age_tertile", "age_tertile_boundaries"]
    return df.drop(columns=[c for c in derived if c in df.columns])


def get_clean_data():
    """Full pipeline: load → clean → add vectors → add derived → save.

    Returns the cleaned DataFrame. When the name-containing source spreadsheet
    is not present (e.g. on a public clone, where it is withheld for privacy),
    the identical cohort is reconstructed from the committed de-identified CSV.
    """
    if config.DATA_RAW.exists():
        df = load_raw_data()
    else:
        df = _load_from_processed(config.DATA_PROCESSED / "all_eyes.csv")
    df = add_vector_components(df)
    df = add_derived_features(df)

    # Save processed
    out_path = config.DATA_PROCESSED / "all_eyes.csv"
    df.to_csv(out_path, index=False)

    return df


def get_validation_data():
    """Load the two external validation cohorts (Jiang, Bu) combined.

    These are independent eyes (no overlap with the main 202), operated by the
    same surgeon with the same technique (2.2 mm clear corneal incision at
    135 deg). Returns a de-identified DataFrame with a ``cohort`` column,
    J0/J45, and derived encodings. Patient codes are cohort-prefixed so the two
    cohorts never collide when combined.
    """
    if config.DATA_VALIDATION_JIANG.exists() and config.DATA_VALIDATION_BU.exists():
        frames = []
        for tag, path in [("Jiang", config.DATA_VALIDATION_JIANG),
                          ("Bu", config.DATA_VALIDATION_BU)]:
            d = load_raw_data(path=path)
            d["patient_id"] = tag[0] + "_" + d["patient_id"]
            d["cohort"] = tag
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = _load_from_processed(config.DATA_PROCESSED / "validation_eyes.csv")
    df = add_vector_components(df)
    df = add_derived_features(df)

    out_path = config.DATA_PROCESSED / "validation_eyes.csv"
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
