# Corneal Biomechanics and Surgically Induced Corneal Astigmatism (CSIA)

Analysis code and de-identified data for the study of preoperative corneal biomechanical parameters (Corvis ST) and surgically induced corneal astigmatism (CSIA) after 2.2 mm clear corneal incision cataract surgery. The pipeline reproduces every figure, table, and statistical result in the manuscript from the de-identified data with a fixed random seed.

## Data

- **De-identified dataset on Hugging Face:** https://huggingface.co/datasets/usama10/corneal-biomechanical-prediction-model
- The same de-identified CSV files are included here under `data/processed/` so the pipeline runs out of the box.

Two cohorts, both operated by the same surgeon using an identical 2.2 mm clear corneal incision at 135 degrees:

| File | Eyes | Patients | Purpose |
|---|---|---|---|
| `data/processed/all_eyes.csv` | 202 (92 OD, 110 OS) | 194 (8 bilateral) | Development and association analysis |
| `data/processed/validation_eyes.csv` | 54 (29 + 25) | 45 (9 bilateral) | Independent external validation |

Patient identities were replaced with anonymous codes; eyes from the same patient share a code. The released data contain no names, no dates, and no direct identifiers. See the Hugging Face dataset card for the full column dictionary.

## Reproduce

```bash
pip install -r experiments/requirements.txt

# Full pipeline (analyses 1-7): regenerates all figures, tables, and results
python experiments/run_all.py

# Data-integrity checks
python experiments/verify_data.py
```

A fixed random seed (42) is used throughout. The pipeline reads `data/processed/*.csv` automatically when the original name-containing source files are not present, as is the case in this public release.

## Structure

- `experiments/src/` analysis modules (config, data loader, vector math, utilities, plotting, tables, analyses 1 to 7)
- `experiments/run_all.py` orchestrator for the full pipeline
- `experiments/verify_data.py` reproducible data-integrity checks
- `experiments/outputs/` generated figures, tables, and intermediate results
- `data/processed/` de-identified development and external validation data

## What the study finds

Eye laterality is the dominant predictor of CSIA direction, and the only variable that significantly predicts the CSIA vector (MANOVA Wilks' lambda = 0.87, F = 13.26, p < 0.0001). Individual-level CSIA magnitude is not predictable from preoperative corneal biomechanics: every model yields negative cross-validated R squared. A Gamma generalized linear model refit of the published magnitude formula removes its physically impossible negative predictions and reduces its out-of-sample error on the independent cohort by 45 percent, while matching the population mean. Eye-specific subgroup centroids generalize to the external eyes.

## Ethics and data sharing

The study was approved by the institutional ethics committee, and the requirement for informed consent was waived due to the retrospective design. Public release of the de-identified data is permitted under the approval. Patient identifiers were replaced with anonymous codes; the released data contain no names, no dates, and no direct identifiers.

## License

Code is released under the MIT License (see `LICENSE`). The de-identified dataset is released under CC-BY-4.0 on Hugging Face.

## Citation

If you use this code or data, please cite the accompanying manuscript and link this repository and the Hugging Face dataset. A formal citation will be added on publication.
