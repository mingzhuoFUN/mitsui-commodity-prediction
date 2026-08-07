from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("mitsui-model-development.ipynb")
OUTPUT = Path("notebooks/mitsui_model_experiments_colab.ipynb")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    setup = [
        markdown(
            """# MITSUI model experiments in Colab

This notebook contains an experimental modeling path with generated targets,
lag/rolling/difference features, LightGBM + Random Forest + XGBoost, an
XGBoost stacking meta-model, and the Mitsui inference server.

The Colab setup configures authentication, paths, dependencies and bounded
training concurrency.
"""
        ),
        code(
            """!pip -q install pandas numpy polars lightgbm xgboost scikit-learn tqdm pyarrow grpcio kaggle

from pathlib import Path
import os

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)
if not (DATA_DIR / "train.csv").exists():
    from google.colab import userdata
    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
    !kaggle competitions download -c mitsui-commodity-prediction-challenge -p data/raw
    !unzip -q -o data/raw/mitsui-commodity-prediction-challenge.zip -d data/raw
"""
        ),
    ]

    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace(
            "ROOT = '/kaggle/input/mitsui-commodity-prediction-challenge/'",
            "ROOT = 'data/raw/'",
        )
        source = source.replace(
            "ThreadPoolExecutor(max_workers=len(train_df))",
            "ThreadPoolExecutor(max_workers=min(8, len(rows)))",
        )
        source = source.replace(
            "sys.path.append('/kaggle/working')",
            "sys.path.append(str(Path('data/raw').resolve()))",
        )
        source = source.replace(
            "('/kaggle/input/mitsui-commodity-prediction-challenge',)",
            "(str(Path('data/raw').resolve()),)",
        )
        cell["source"] = source.splitlines(True)
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)

    notebook["cells"] = setup + notebook["cells"]
    notebook.setdefault("metadata", {})["colab"] = {
        "name": OUTPUT.name,
        "provenance": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
