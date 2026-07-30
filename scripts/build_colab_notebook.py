from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUTPUT = Path("notebooks/mitsui_commodity_prediction_improved_colab.ipynb")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3"}
    nb.metadata["colab"] = {"name": OUTPUT.name, "provenance": []}
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# MITSUI Commodity Prediction — improved causal baseline

This notebook is the corrected, reproducible successor to the uploaded
`mitsui-commodity-prediction-challenge.ipynb`.

Key corrections:

- trains on the official `train_labels.csv` targets;
- removes negative lags and backward filling;
- uses a chronological holdout with a four-day embargo;
- builds test features with continuous historical context;
- keeps data, secrets, and trained models outside GitHub.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Runtime setup

Use a CPU runtime. Ridge is the verified fast baseline; LightGBM can be added
after this notebook reproduces the baseline score."""
        ),
        nbf.v4.new_code_cell(
            """!git clone https://github.com/mingzhuoFUN/mitsui-commodity-prediction.git
%cd mitsui-commodity-prediction
!pip -q install -r requirements.txt"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. Download competition data

In Colab, add a secret named `KAGGLE_API_TOKEN` containing the contents of your
Kaggle API token, or upload `kaggle.json` when prompted. Never commit the token."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import os

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

if not (DATA_DIR / "train.csv").exists():
    from google.colab import files
    uploaded = files.upload()  # choose kaggle.json
    Path.home().joinpath(".kaggle").mkdir(exist_ok=True)
    token_name = next(iter(uploaded))
    Path.home().joinpath(".kaggle/kaggle.json").write_bytes(uploaded[token_name])
    os.chmod(Path.home() / ".kaggle/kaggle.json", 0o600)
    !kaggle competitions download -c mitsui-commodity-prediction-challenge -p data/raw
    !unzip -q -o data/raw/mitsui-commodity-prediction-challenge.zip -d data/raw

!python scripts/inspect_data.py --data-dir data/raw"""
        ),
        nbf.v4.new_markdown_cell("## 3. Verify official-label alignment"),
        nbf.v4.new_code_cell(
            """import pandas as pd

train = pd.read_csv(DATA_DIR / "train.csv")
labels = pd.read_csv(DATA_DIR / "train_labels.csv")
pairs = pd.read_csv(DATA_DIR / "target_pairs.csv")

assert train["date_id"].equals(labels["date_id"])
print("train:", train.shape)
print("labels:", labels.shape)
print("targets:", pairs.shape)
pairs.head()"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Smoke train

Train ten targets first. This catches data, dependency, and feature errors in a
few seconds before the full run."""
        ),
        nbf.v4.new_code_cell(
            """!PYTHONPATH=src python scripts/train_baseline.py \\
  --data-dir data/raw \\
  --output-dir outputs/smoke \\
  --valid-size 128 \\
  --max-targets 10"""
        ),
        nbf.v4.new_markdown_cell("## 5. Full 424-target chronological validation"),
        nbf.v4.new_code_cell(
            """!PYTHONPATH=src python scripts/train_baseline.py \\
  --data-dir data/raw \\
  --output-dir outputs/ridge_full \\
  --valid-size 252 \\
  --save-models"""
        ),
        nbf.v4.new_code_cell(
            """import json
from pathlib import Path

metrics = json.loads(Path("outputs/ridge_full/metrics.json").read_text())
metrics"""
        ),
        nbf.v4.new_markdown_cell(
            """## 6. Persist artifacts to Google Drive (optional)

Model files are intentionally ignored by Git. Copy them to Drive if you need
them after the Colab runtime ends."""
        ),
        nbf.v4.new_code_cell(
            """# from google.colab import drive
# drive.mount("/content/drive")
# !cp -r outputs/ridge_full "/content/drive/MyDrive/mitsui-ridge-full" """
        ),
        nbf.v4.new_markdown_cell(
            """## Next model iteration

Once the Ridge score is reproduced, add LightGBM behind the same feature and
validation interfaces. Compare it against this baseline before considering
time-series OOF blending. Do not use in-sample stacking predictions."""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)


if __name__ == "__main__":
    main()
