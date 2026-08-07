from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from mitsui.ensemble import load_stacked_models
from mitsui.inference import SequentialPredictor


def main() -> None:
    """使用已保存的集成模型运行官方本地推理网关。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    # 竞赛推理包随数据集提供，并非通过 PyPI 安装，因此只在当前进程中
    # 将数据目录加入模块搜索路径。
    sys.path.append(str(args.data_dir.resolve()))
    from kaggle_evaluation.mitsui_inference_server import MitsuiInferenceServer

    predictor = SequentialPredictor(
        load_stacked_models(args.model_path),
        pd.read_csv(args.data_dir / "train.csv"),
        pd.read_csv(args.data_dir / "train_labels.csv"),
    )

    def predict(test, lag1, lag2, lag3, lag4):
        # 函数签名必须与竞赛推理服务器的接口约定完全一致。
        return predictor.predict(test, lag1, lag2, lag3, lag4)

    server = MitsuiInferenceServer(predict)
    server.run_local_gateway((str(args.data_dir),))


if __name__ == "__main__":
    main()
