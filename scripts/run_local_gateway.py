from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from mitsui.ensemble import load_stacked_models
from mitsui.inference import SequentialPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    sys.path.append(str(args.data_dir.resolve()))
    from kaggle_evaluation.mitsui_inference_server import MitsuiInferenceServer

    predictor = SequentialPredictor(
        load_stacked_models(args.model_path),
        pd.read_csv(args.data_dir / "train.csv"),
        pd.read_csv(args.data_dir / "train_labels.csv"),
    )

    def predict(test, lag1, lag2, lag3, lag4):
        return predictor.predict(test, lag1, lag2, lag3, lag4)

    server = MitsuiInferenceServer(predict)
    server.run_local_gateway((str(args.data_dir),))


if __name__ == "__main__":
    main()
