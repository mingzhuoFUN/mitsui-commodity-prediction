# MITSUI&CO. Commodity Prediction Challenge

This repository improves the original public notebook into a causal,
reproducible training project for the Kaggle competition:

https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge

## Verified baseline

The current baseline trains one Ridge model per official target using only
causal features from the target's declared asset pair.

- Official labels: `train_labels.csv`
- Validation: final 252 rows, chronological, with a 4-row embargo
- Targets trained: 424
- Local validation IC Sharpe: **0.1466**
- Mean daily IC: **0.0242**

The score is a baseline, not a leaderboard claim. It was produced locally from
the supplied competition files and is recorded in ignored training artifacts.

## Corrections to the original notebook

- Uses official labels instead of rebuilding and shifting targets.
- Removes `shift(-1)`, `shift(-2)`, negative differences, and backward fill.
- Parses pairs with `" - "` so asset names containing hyphens remain intact.
- Uses chronological validation rather than random splitting.
- Fits the preprocessing pipeline on training rows only.
- Starts with a transparent Ridge baseline before leakage-safe model blending.

## Competition overview

- Task: predict future commodity-related return/spread targets from historical market data.
- Data families: LME commodities, JPX commodities, US stocks, and FX.
- Target design: 424 target columns (`target_0` to `target_423`) built from asset pairs and mixed 1-4 day prediction horizons.
- Evaluation: daily Spearman rank correlation, aggregated as an IC Sharpe ratio: `mean(daily_ic) / std(daily_ic)`.
- Submission style: Kaggle notebook / time-series API style competition, with internet disabled for final submissions.

Sources checked on 2026-06-01:

- Kaggle competition page: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge
- CompeteHub summary: https://www.competehub.dev/en/competitions/kagglemitsui-commodity-prediction-challenge
- Problem-design article: https://zenn.dev/gamella/articles/7e944bd18cdbe6?locale=en

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The Kaggle CLI is not currently installed in this environment. Once installed and authenticated, download data with:

```powershell
kaggle competitions download -c mitsui-commodity-prediction-challenge -p data/raw
```

Then unzip the downloaded archive into `data/raw/`.

## Project layout

```text
data/raw/        # official Kaggle files
data/interim/    # temporary feature tables
data/processed/  # model-ready artifacts
models/          # trained models
outputs/         # submissions, reports, diagnostics
src/mitsui/      # reusable code
scripts/         # command-line experiment entrypoints
notebooks/       # Colab-ready training notebook
tests/           # leakage and metric checks
```

## Commands

Inspect available data files:

```powershell
python scripts/inspect_data.py --data-dir data/raw
```

Run metric smoke tests:

```powershell
pytest
```

Run a ten-target smoke train:

```powershell
$env:PYTHONPATH="src"
python scripts/train_baseline.py --data-dir data/raw --output-dir outputs/smoke --max-targets 10
```

Run all 424 targets:

```powershell
$env:PYTHONPATH="src"
python scripts/train_baseline.py --data-dir data/raw --output-dir outputs/ridge_full --valid-size 252 --save-models
```

## Colab

Open `notebooks/mitsui_commodity_prediction_improved_colab.ipynb` in Colab.
It clones this repository, downloads data with your private Kaggle credential,
runs a smoke train, and then trains all 424 targets.

Never commit `kaggle.json`, competition CSV files, ZIP archives, or trained
model artifacts. They are excluded by `.gitignore`.
