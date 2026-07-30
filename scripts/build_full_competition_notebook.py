from __future__ import annotations

from pathlib import Path
import nbformat as nbf


OUTPUT = Path("notebooks/mitsui_competition_colab.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["colab"] = {"name": OUTPUT.name, "provenance": []}
    nb.cells = [
        md("""# MITSUI Commodity Prediction — Colab training and submission

End-to-end training, validation and sequential inference for all 424 targets.
The notebook uses the uploaded competition notebook as its blueprint: target
pair features, LightGBM, Random Forest, XGBoost, stacking and the Kaggle
inference server. The implementation fixes label alignment, future leakage and
in-sample stacking while retaining that model architecture."""),
        md("""## Blueprint mapping

| Uploaded notebook section | Repository implementation |
|---|---|
| `get_data_for_day` and generated targets | official labels aligned by `date_id` |
| `prepare_features_for_col/df` | `src/mitsui/features.py` |
| LGBM + RF + XGB base learners | `src/mitsui/ensemble.py` |
| XGB meta-model trained in-sample | time-series OOF Ridge meta-model |
| `predict_on_test` | `src/mitsui/inference.py` |
| `MitsuiInferenceServer` | `scripts/run_local_gateway.py` and final cells |

The original negative shifts and backward fill are not retained because they
read future rows. All positive lag, rolling, difference/spread concepts remain
in causal form."""),
        md("## 1. Clone and install"),
        code("""!git clone https://github.com/mingzhuoFUN/mitsui-commodity-prediction.git
%cd mitsui-commodity-prediction
!pip -q install -r requirements.txt
%env PYTHONPATH=src"""),
        md("""## 2. Competition data

Upload `kaggle.json` only to the runtime. Never commit it."""),
        code("""from pathlib import Path
import os
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

if not (DATA_DIR / "train.csv").exists():
    from google.colab import files
    uploaded = files.upload()
    token_file = next(iter(uploaded))
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    (kaggle_dir / "kaggle.json").write_bytes(uploaded[token_file])
    os.chmod(kaggle_dir / "kaggle.json", 0o600)
    !kaggle competitions download -c mitsui-commodity-prediction-challenge -p data/raw
    !unzip -q -o data/raw/mitsui-commodity-prediction-challenge.zip -d data/raw

!python scripts/inspect_data.py --data-dir data/raw"""),
        md("## 3. Verify official X/Y alignment and target horizons"),
        code("""import pandas as pd
train = pd.read_csv(DATA_DIR / "train.csv")
labels = pd.read_csv(DATA_DIR / "train_labels.csv")
pairs = pd.read_csv(DATA_DIR / "target_pairs.csv")
assert train["date_id"].equals(labels["date_id"])
display(pairs.groupby("lag").size().rename("targets"))
print("market", train.shape, "labels", labels.shape)"""),
        md("""## 4. Leakage checks

Tests include a future-mutation test: changing future market rows must not
change features already computed for past rows."""),
        code("!pytest -q"),
        md("""## 5. Inspect the actual model code

The notebook calls repository modules so the same tested implementation is
used in Colab, local validation and Kaggle inference."""),
        code("""from mitsui.features import make_target_features
from mitsui.ensemble import fit_stacked_models, predict_stacked_models
from mitsui.inference import SequentialPredictor

print("Base models: LightGBM, Random Forest, XGBoost")
print("Meta model: Ridge trained from time-series OOF predictions")"""),
        md("## 6. Eight-target smoke run"),
        code("""!python scripts/train_ensemble.py \\
  --data-dir data/raw \\
  --output-dir outputs/ensemble_smoke \\
  --valid-size 128 \\
  --max-targets 8"""),
        md("""## 7. Full chronological validation

Each target uses LightGBM, Random Forest and XGBoost. Their time-series OOF
predictions train a Ridge meta-model. The final 252 rows are untouched until
validation."""),
        code("""!python scripts/train_ensemble.py \\
  --data-dir data/raw \\
  --output-dir outputs/ensemble_full \\
  --valid-size 252"""),
        code("""import json
json.loads(Path("outputs/ensemble_full/metrics.json").read_text())"""),
        md("## 8. Fit submission models on all official training rows"),
        code("""!python scripts/train_ensemble.py \\
  --data-dir data/raw \\
  --output-dir outputs/ensemble_submit \\
  --fit-full"""),
        md("## 9. Run the complete local Kaggle gateway"),
        code("""!python scripts/run_local_gateway.py \\
  --data-dir data/raw \\
  --model-path outputs/ensemble_submit/stacked_models.pkl"""),
        md("""## 10. Competition rerun entrypoint

For a Kaggle submission, keep the trained model artifact in a Kaggle Dataset
attached to the notebook, initialize `SequentialPredictor`, then serve it:"""),
        code("""import os, sys
import pandas as pd
from mitsui.ensemble import load_stacked_models
from mitsui.inference import SequentialPredictor

sys.path.append(str(DATA_DIR.resolve()))
from kaggle_evaluation.mitsui_inference_server import MitsuiInferenceServer

predictor = SequentialPredictor(
    load_stacked_models("outputs/ensemble_submit/stacked_models.pkl"),
    pd.read_csv(DATA_DIR / "train.csv"),
    pd.read_csv(DATA_DIR / "train_labels.csv"),
)

def predict(test, label_lags_1, label_lags_2, label_lags_3, label_lags_4):
    return predictor.predict(
        test, label_lags_1, label_lags_2, label_lags_3, label_lags_4
    )

inference_server = MitsuiInferenceServer(predict)
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    print("Use the previous cell for local gateway validation.")"""),
        md("""## Artifacts

`outputs/ensemble_submit/stacked_models.pkl` contains all 424 target bundles.
Store it in Google Drive or a private Kaggle Dataset; it is intentionally not
committed to GitHub."""),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)


if __name__ == "__main__":
    main()
