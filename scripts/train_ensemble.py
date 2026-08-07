from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mitsui.ensemble import (
    fit_stacked_models,
    predict_stacked_models,
    save_stacked_models,
)
from mitsui.metric import daily_ic, ic_sharpe
from mitsui.validation import holdout_indices


def main() -> None:
    """训练并验证 424 目标集成模型，也可使用全量数据重新拟合。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ensemble"))
    parser.add_argument("--valid-size", type=int, default=252)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--models", nargs="+", default=["lgbm", "rf", "xgb"])
    parser.add_argument("--fit-full", action="store_true")
    args = parser.parse_args()

    market = pd.read_csv(args.data_dir / "train.csv")
    labels = pd.read_csv(args.data_dir / "train_labels.csv")
    pairs = pd.read_csv(args.data_dir / "target_pairs.csv")
    # 后续按行位置对齐特征和标签，因此日期错位时必须尽早报错，
    # 避免模型在不知情的情况下学习错误日期的标签。
    if not market["date_id"].equals(labels["date_id"]):
        raise ValueError("Market rows and official labels are not aligned")

    if args.fit_full:
        # 提交模型使用全部有标签数据，因此本次调用不会产生留出集分数。
        train_idx = market.index.to_numpy()
        valid_idx = None
    else:
        train_idx, valid_idx = holdout_indices(
            len(market), valid_size=args.valid_size, embargo=4
        )
    models = fit_stacked_models(
        market,
        labels,
        pairs,
        train_idx,
        base_names=tuple(args.models),
        max_targets=args.max_targets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_stacked_models(models, args.output_dir / "stacked_models.pkl")

    report = {"n_models": len(models), "fit_full": args.fit_full}
    if valid_idx is not None:
        # 保持模型输出的目标列顺序，确保指标计算时严格对齐。
        pred = predict_stacked_models(models, market, labels, valid_idx)
        truth = labels.loc[valid_idx, pred.columns].copy()
        truth.index = pred.index
        dates = market.loc[valid_idx, "date_id"].to_numpy()
        ics = daily_ic(truth, pred, dates)
        report.update(
            {
                "ic_sharpe": ic_sharpe(truth, pred, dates),
                "mean_daily_ic": float(ics.mean()),
                "std_daily_ic": float(ics.std(ddof=0)),
            }
        )
        pred.insert(0, "date_id", dates)
        pred.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
