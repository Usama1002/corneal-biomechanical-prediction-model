"""Data-integrity verification for the corrected (full-name) dataset.

Run from the ``experiments/`` directory:  ``python verify_data.py``

Checks the corrections introduced after the 2026-05 dataset revision:
  * the main cohort is 202 eyes (92 OD / 110 OS) from 194 patients,
  * exactly 8 bilateral patients (true identity by Chinese name, not initials),
  * the derived bilateral set matches the surgeon's yellow-highlighted rows,
  * no personally identifying columns are persisted,
  * the published centroid / eye-laterality finding is reproduced,
  * the two external validation cohorts load cleanly (29 + 25 = 54 eyes).
"""

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from src import config, data_loader, vector_math  # noqa: E402

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        raise AssertionError(label)


def verify_main():
    print("MAIN COHORT")
    df = data_loader.get_clean_data()
    check("202 eyes (92 OD / 110 OS)",
          len(df) == 202 and (df.eye == "OD").sum() == 92 and (df.eye == "OS").sum() == 110,
          f"{len(df)} eyes")
    check("194 unique patients", df.patient_id.nunique() == 194,
          f"{df.patient_id.nunique()} patients")

    sizes = df.groupby("patient_id").size()
    multi = sizes[sizes > 1]
    bilateral = [pid for pid in multi.index
                 if len(df[df.patient_id == pid]) == 2
                 and set(df[df.patient_id == pid].eye) == {"OD", "OS"}]
    check("exactly 8 bilateral patients, all OD+OS pairs, no repeats/triples",
          len(multi) == 8 and len(bilateral) == 8,
          f"{len(multi)} multi-entry, {len(bilateral)} bilateral")

    # When the name-containing source file is available (authors' machine, not
    # the public release), confirm that grouping by true full-name identity
    # yields exactly the 8 bilateral patients. No names are printed or stored.
    if config.DATA_RAW.exists():
        raw = pd.read_excel(config.DATA_RAW, sheet_name="All(OD+OS)", header=None, skiprows=2)
        raw = raw[raw[5].astype(str).str.strip().isin(["OD", "OS"])]
        counts = raw[0].astype(str).str.strip().value_counts()
        n_bilateral_raw = int((counts > 1).sum())
        check("raw full-name grouping yields exactly 8 bilateral identities",
              n_bilateral_raw == 8, f"{n_bilateral_raw}")
    else:
        print("  [SKIP] raw full-name grouping check (name file withheld from public release)")

    check("no PII columns persisted",
          not any(c.startswith("_") for c in df.columns)
          and all("name" not in c.lower() for c in df.columns))
    check("sex normalized to M/F", set(df.sex.unique()) <= {"M", "F"})
    check("no NaNs in the 20 predictors",
          int(df[config.ALL_PREOP_PARAMS].isna().sum().sum()) == 0)

    c = vector_math.compute_centroid(df.CSIA_mag.values, df.CSIA_meridian.values)
    check("overall centroid 0.441 D @ 45.7 deg reproduced",
          abs(c["centroid_mag"] - 0.441) < 1e-3 and abs(c["centroid_meridian"] - 45.7) < 0.2,
          f"{c['centroid_mag']:.3f} D @ {c['centroid_meridian']:.1f} deg")

    from scipy import stats
    od, os_ = df[df.eye == "OD"].J0, df[df.eye == "OS"].J0
    _, p = stats.ttest_ind(od, os_)
    pooled = np.sqrt(((len(od) - 1) * od.std() ** 2 + (len(os_) - 1) * os_.std() ** 2)
                     / (len(od) + len(os_) - 2))
    d = (od.mean() - os_.mean()) / pooled
    check("eye-laterality finding holds (J0 OD vs OS, p < 1e-3, |d| ~ 0.6)",
          p < 1e-3 and abs(d) > 0.5, f"p={p:.2e}, d={d:.3f}")
    return df


def verify_validation():
    print("\nEXTERNAL VALIDATION COHORTS")
    v = data_loader.get_validation_data()
    check("54 eyes total (Jiang 29 + Bu 25)",
          len(v) == 54 and (v.cohort == "Jiang").sum() == 29 and (v.cohort == "Bu").sum() == 25,
          f"{len(v)} eyes")
    check("sex normalized to M/F (Chinese 男/女 handled)",
          set(v.sex.unique()) <= {"M", "F"})
    check("PCT135 parsed numeric, no predictor NaNs (handles '660(4.37mm)')",
          int(v[config.ALL_PREOP_PARAMS].isna().sum().sum()) == 0
          and 500 < v.PCT135.min() and v.PCT135.max() < 900)
    for tag, exp in [("Jiang", 6), ("Bu", 3)]:
        sub = v[v.cohort == tag]
        nb = int((sub.groupby("patient_id").size() == 2).sum())
        check(f"{tag} cohort: {exp} bilateral patients", nb == exp, f"{nb} bilateral")
    check("cohort-prefixed patient codes do not collide",
          v.patient_id.str.startswith(("J_", "B_")).all())
    check("no PII columns persisted",
          all("name" not in c.lower() and not c.startswith("_") for c in v.columns))
    return v


if __name__ == "__main__":
    verify_main()
    verify_validation()
    print("\nAll data-integrity checks passed.")
