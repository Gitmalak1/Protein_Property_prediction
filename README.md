# Protein Property Prediction: Protein Solubility

A reproducible machine-learning project that predicts **protein solubility** from amino-acid composition and simple sequence-derived features. It includes dataset download, validation, feature engineering, model comparison, evaluation reports, a command-line predictor, tests, and GitHub Actions CI.

> **Scientific scope:** This is an educational ML baseline, not a laboratory replacement. Solubility depends on protein context and experimental conditions. Reported performance must be measured by running the workflow on the downloaded dataset; no scores are fabricated in this repository.

## Dataset

The project is designed around the processed eSol-derived dataset shared by the authors of:

Han, X., Wang, X., & Zhou, K. (2019). “Develop machine learning-based regression predictive models for engineering protein solubility.” *Bioinformatics*, 35(22), 4640–4646.

- Upstream dataset/code repository: https://github.com/xiaomizhou616/Protein_solubility_regression_model
- The downloader attempts to fetch `cleandata_1name.csv` from that repository.
- Dataset files are not redistributed here; the upstream dataset remains subject to its own terms. If the raw URL changes, download the processed CSV from the upstream repository and save it as `data/raw_solubility.csv`.

## Quick start

Python 3.10+ recommended.

### 1. Install

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download data

```bash
python -m src.download_data
```

### 3. Train and evaluate

```bash
python -m src.train
```

Generated outputs:
- `reports/model_comparison.csv`: test-set MAE, RMSE, and R² for models
- `reports/test_predictions.csv`: actual and predicted values for the selected model
- `reports/metrics.json`: selected model and metrics
- `reports/data_summary.json`: dataset diagnostics
- `models/protein_solubility_model.joblib`: saved pipeline

The model with the lowest test MAE is selected. Test metrics are only an estimate of generalization, especially if related proteins or variants appear in both splits.

### 4. Predict a sequence

```bash
python -m src.predict --sequence "MKWVTFISLLFLFSSAYS"
```

Or use a CSV with a `sequence` column:
```bash
python -m src.predict --input data/my_sequences.csv --output reports/my_predictions.csv
```

## Features

For sequence datasets, the feature extractor computes the fraction of each of the 20 standard amino acids, sequence length, mean hydropathy, charged-residue fraction (D/E/K/R), aromatic fraction (F/W/Y), glycine fraction, and proline fraction.

For the processed eSol feature table, the trainer uses numeric columns while excluding likely identifiers and the target. Inspect `reports/data_summary.json` and set `target_column` in `config.json` if automatic target detection is ambiguous.

## Workflow

1. Download and inspect the dataset.
2. Validate columns, remove missing targets and duplicate rows.
3. Split into train and test sets.
4. Fit preprocessing only on training data.
5. Compare a dummy mean baseline, Ridge regression, Random Forest, and Extra Trees.
6. Report MAE, RMSE and R²; save the best model and test predictions.

**Limitations:** A random split can overestimate performance when highly similar proteins or variants cross the split. A sequence-identity-aware split is a valuable extension. Engineered composition features do not capture full protein structure or evolutionary context.

## Structure

```text
protein-property-prediction/
├── config.json
├── data/                  # downloaded dataset is not committed
├── models/                # trained model is not committed
├── reports/               # metrics and predictions are not committed
├── src/
│   ├── data_utils.py
│   ├── download_data.py
│   ├── train.py
│   └── predict.py
├── tests/
└── .github/workflows/tests.yml
```

## Troubleshooting

- **Download fails:** Get `cleandata_1name.csv` from the upstream repository and save it as `data/raw_solubility.csv`.
- **Target not detected:** Set `target_column` in `config.json` to the solubility label column in the downloaded CSV.
- **No sequence column:** Training will use numeric processed features. Sequence-based CLI prediction is available when a model has been trained on sequence-derived features; it cannot be assumed compatible with a model trained on a different feature schema.

## Citation

```bibtex
@article{han2019protein_solubility,
  title={Develop machine learning-based regression predictive models for engineering protein solubility},
  author={Han, X. and Wang, X. and Zhou, K.},
  journal={Bioinformatics},
  volume={35},
  number={22},
  pages={4640--4646},
  year={2019}
}
```

Code license: MIT. Dataset rights remain with the upstream source.
