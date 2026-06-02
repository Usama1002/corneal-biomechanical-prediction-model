#!/usr/bin/env python3
"""Master orchestration script for CSIA vector prediction analysis.

Usage:
    python run_all.py                    # run everything
    python run_all.py --analyses 1,2,4   # run specific analyses
    python run_all.py --skip-modeling    # skip slow analysis 3
"""

import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

# ensure src is importable
sys.path.insert(0, str(__file__).rsplit("/", 1)[0] if "/" in str(__file__) else ".")

from src import config
from src.plotting import setup_style
from src.data_loader import get_clean_data

from src.analysis1_vector_characterization import run as run_analysis1
from src.analysis2_association import run as run_analysis2
from src.analysis3_predictive_modeling import run as run_analysis3
from src.analysis4_subgroup_centroids import run as run_analysis4
from src.analysis5_steepening import run as run_analysis5
from src.analysis6_ablations import run as run_analysis6
from src.analysis7_formula import run as run_analysis7


def main():
    parser = argparse.ArgumentParser(description="CSIA Vector Prediction Analysis Pipeline")
    parser.add_argument("--analyses", type=str, default="1,2,3,4,5,6,7",
                        help="Comma-separated list of analyses to run (default: 1,2,3,4,5,6,7)")
    parser.add_argument("--skip-modeling", action="store_true",
                        help="Skip analysis 3 (predictive modeling)")
    args = parser.parse_args()

    analyses = [int(x.strip()) for x in args.analyses.split(",")]
    if args.skip_modeling and 3 in analyses:
        analyses.remove(3)

    t0 = time.time()
    print("=" * 60)
    print("CSIA Vector Prediction Analysis Pipeline")
    print(f"Analyses to run: {analyses}")
    print("=" * 60)

    # Setup
    setup_style()
    print("\nLoading and cleaning data...")
    df = get_clean_data()
    print(f"  {len(df)} eyes loaded ({df['eye'].value_counts().to_dict()})")

    all_results = {}

    if 1 in analyses:
        t1 = time.time()
        all_results["analysis1"] = run_analysis1(df)
        print(f"\n  Analysis 1 took {time.time()-t1:.1f}s")

    if 2 in analyses:
        t1 = time.time()
        all_results["analysis2"] = run_analysis2(df)
        print(f"\n  Analysis 2 took {time.time()-t1:.1f}s")

    if 3 in analyses:
        t1 = time.time()
        all_results["analysis3"] = run_analysis3(df)
        print(f"\n  Analysis 3 took {time.time()-t1:.1f}s")

    if 4 in analyses:
        t1 = time.time()
        ml_preds = all_results.get("analysis3", {}).get("predictions", None)
        all_results["analysis4"] = run_analysis4(df, ml_predictions=ml_preds)
        print(f"\n  Analysis 4 took {time.time()-t1:.1f}s")

    if 5 in analyses:
        t1 = time.time()
        all_results["analysis5"] = run_analysis5(df)
        print(f"\n  Analysis 5 took {time.time()-t1:.1f}s")

    if 6 in analyses:
        t1 = time.time()
        all_results["analysis6"] = run_analysis6(df)
        print(f"\n  Analysis 6 took {time.time()-t1:.1f}s")

    if 7 in analyses:
        t1 = time.time()
        all_results["analysis7"] = run_analysis7(df)
        print(f"\n  Analysis 7 took {time.time()-t1:.1f}s")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)

    # Count outputs
    import os
    n_figs = len([f for f in os.listdir(config.OUTPUT_FIGURES) if f.endswith((".png", ".pdf"))])
    n_tabs = len([f for f in os.listdir(config.OUTPUT_TABLES) if f.endswith((".csv", ".xlsx"))])
    n_res = len([f for f in os.listdir(config.OUTPUT_RESULTS)])
    print(f"  Figures: {n_figs} files in {config.OUTPUT_FIGURES}")
    print(f"  Tables:  {n_tabs} files in {config.OUTPUT_TABLES}")
    print(f"  Results: {n_res} files in {config.OUTPUT_RESULTS}")

    return all_results


if __name__ == "__main__":
    main()
