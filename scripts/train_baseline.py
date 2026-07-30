from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mitsui.metric import daily_ic, ic_sharpe
from mitsui.modeling import fit_target_models, predict_target_models, save_models
from mitsui.validation import holdout_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a causal official-label baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--valid-size", type=int, default=252)
    parser.add_argument("--embargo", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    market = pd.read_csv(args.data_dir / "train.csv")
    labels = pd.read_csv(args.data_dir / "train_labels.csv")
    pairs = pd.read_csv(args.data_dir / "target_pairs.csv")

    if not market["date_id"].equals(labels["date_id"]):
        raise ValueError("train.csv and train_labels.csv date_id columns are not aligned")

    train_idx, valid_idx = holdout_indices(
        len(market), valid_size=args.valid_size, embargo=args.embargo
    )
    models = fit_target_models(
        market,
        labels,
        pairs,
        train_idx,
        alpha=args.alpha,
        max_targets=args.max_targets,
    )
    predictions = predict_target_models(models, market, valid_idx)
    truth = labels.loc[valid_idx, list(predictions.columns)].copy()
    truth.index = predictions.index
    dates = market.loc[valid_idx, "date_id"].to_numpy()

    score = ic_sharpe(truth, predictions, dates)
    ics = daily_ic(truth, predictions, dates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.insert(0, "date_id", dates)
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    ics.rename_axis("date_id").to_csv(args.output_dir / "daily_ic.csv")
    report = {
        "n_train_rows": int(len(train_idx)),
        "n_valid_rows": int(len(valid_idx)),
        "n_targets": int(len(models)),
        "valid_ic_sharpe": score,
        "mean_daily_ic": float(ics.mean()),
        "std_daily_ic": float(ics.std(ddof=0)),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if args.save_models:
        save_models(models, args.output_dir / "models.pkl")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
