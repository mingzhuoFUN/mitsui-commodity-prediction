from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUTPUT = Path("notebooks/mitsui_complete_workflow.ipynb")


def md(source: str):
    return nbf.v4.new_markdown_cell(source)


def code(source: str):
    return nbf.v4.new_code_cell(source)


def main() -> None:
    """生成可在本地 Jupyter 中独立阅读和运行的完整方法 notebook。"""
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
    nb.cells = [
        md(
            """# MITSUI 商品预测：从数据到顺序推理的完整方法

这份 notebook 将项目的完整建模思路放在一个文件中，适合本地 Jupyter、
JupyterLab 或 VS Code Notebook 直接阅读和运行，不使用 Colab 专用 API。

核心问题不是简单地选择某一种模型，而是针对 424 个不同资产组合与预测周期，
在严格避免未来数据泄漏的前提下构造有效特征，并保证训练与线上顺序推理使用
同一套信息边界。

完整流程：

1. 读取市场数据、官方标签和目标资产映射；
2. 理解每个 target 的资产 pair 与预测 horizon；
3. 构造单资产动量、滚动统计和双资产价差特征；
4. 按标签实际揭示时间构造历史标签特征；
5. 使用时间留出、embargo 和时间序列 OOF 控制泄漏；
6. 训练 LightGBM、Random Forest、XGBoost；
7. 使用 Ridge 合并三个基础模型的 OOF 预测；
8. 保存模型，并用有状态预测器模拟竞赛顺序推理。

> 默认配置只训练 8 个目标，便于快速检查。确认流程无误后，将
> `RUN_FULL_TRAINING` 改为 `True` 即可训练全部 424 个目标。"""
        ),
        md(
            """## 1. 环境与目录约定

先在项目根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

将竞赛文件放在 `data/raw/`：

```text
data/raw/
  train.csv
  train_labels.csv
  target_pairs.csv
  test.csv
  lagged_test_labels/
  kaggle_evaluation/
```

notebook 可以从项目根目录或 `notebooks/` 目录启动，下面的代码会自动定位
项目根目录。"""
        ),
        code(
            """from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "data" / "raw").exists() and PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "standalone_workflow"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_FULL_TRAINING = False
MAX_TARGETS = None if RUN_FULL_TRAINING else 8
VALID_SIZE = 252 if RUN_FULL_TRAINING else 128
RANDOM_STATE = 42

print("项目目录:", PROJECT_ROOT)
print("数据目录:", DATA_DIR)
print("训练目标数:", "全部 424 个" if MAX_TARGETS is None else MAX_TARGETS)"""
        ),
        md(
            """## 2. 读取数据并确认标签对齐

- `train.csv` 是按 `date_id` 排列的多市场价格序列；
- `train_labels.csv` 包含 424 个官方目标；
- `target_pairs.csv` 说明每个目标对应的预测周期和资产 pair。

训练代码按行对齐特征与标签，因此首先严格检查两个训练文件的 `date_id`
是否完全一致。"""
        ),
        code(
            """required_files = ["train.csv", "train_labels.csv", "target_pairs.csv"]
missing_files = [name for name in required_files if not (DATA_DIR / name).exists()]
if missing_files:
    raise FileNotFoundError(f"data/raw 缺少文件: {missing_files}")

market = pd.read_csv(DATA_DIR / "train.csv")
labels = pd.read_csv(DATA_DIR / "train_labels.csv")
target_pairs = pd.read_csv(DATA_DIR / "target_pairs.csv")

if not market["date_id"].equals(labels["date_id"]):
    raise ValueError("train.csv 与 train_labels.csv 的 date_id 未严格对齐")

target_names = sorted(
    [column for column in labels.columns if column.startswith("target_")],
    key=lambda name: int(name.split("_")[1]),
)

print("市场数据:", market.shape)
print("标签数据:", labels.shape)
print("目标映射:", target_pairs.shape)
print("目标数量:", len(target_names))
display(target_pairs.head(10))
display(target_pairs.groupby("lag").size().rename("目标数"))"""
        ),
        md(
            """## 3. 如何理解一个目标

以双资产目标为例：

```text
target_9 | lag=1 | FX_AUDJPY - LME_PB_Close
```

它表示模型需要利用澳元兑日元和 LME 铅价格的历史状态，预测给定 horizon
上的相对收益。项目不会把所有市场列无差别地输入每个模型，而是为每个目标
选取其 pair 资产，分别构造：

- 当前对数价格；
- 1、2、3、5、10、20 日对数收益；
- 5、20、60 日收益均值与波动率；
- 双资产对数价差、价差变化和滚动 z-score；
- 按 horizon 延迟后才允许使用的历史标签。

这样既减少无关变量噪声，也让特征直接对应目标的经济含义。"""
        ),
        code(
            """@dataclass(frozen=True)
class FeatureConfig:
    return_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    rolling_windows: tuple[int, ...] = (5, 20, 60)


def parse_pair(pair: str) -> list[str]:
    # 只按带空格的分隔符拆分，保留资产列名内部可能存在的连字符。
    return [part.strip() for part in pair.split(" - ") if part.strip()]


def safe_log(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log(values.where(values > 0))


def make_target_features(
    market_data: pd.DataFrame,
    pair: str,
    config: FeatureConfig = FeatureConfig(),
    label: pd.Series | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    columns = parse_pair(pair)
    missing = [column for column in columns if column not in market_data.columns]
    if missing:
        raise KeyError(f"市场数据缺少 pair 列: {missing}")

    # 前向填充只使用当前及过去信息；不能使用会读取未来数据的后向填充。
    raw = market_data[columns].apply(pd.to_numeric, errors="coerce").ffill()
    features: dict[str, pd.Series] = {}
    log_prices: dict[str, pd.Series] = {}

    for column in columns:
        log_price = safe_log(raw[column])
        log_prices[column] = log_price
        features[f"{column}__log_level"] = log_price

        return_1 = log_price.diff()
        features[f"{column}__return_1"] = return_1

        for lag in config.return_lags:
            features[f"{column}__log_return_{lag}"] = log_price.diff(lag)

        for window in config.rolling_windows:
            min_periods = max(2, window // 3)
            rolling = return_1.rolling(window=window, min_periods=min_periods)
            features[f"{column}__return_mean_{window}"] = rolling.mean()
            features[f"{column}__return_std_{window}"] = rolling.std()

    if len(columns) == 2:
        left, right = columns
        spread = log_prices[left] - log_prices[right]
        features["pair__log_spread"] = spread
        features["pair__spread_change_1"] = spread.diff()

        for window in config.rolling_windows:
            min_periods = max(2, window // 3)
            mean = spread.rolling(window, min_periods=min_periods).mean()
            std = spread.rolling(window, min_periods=min_periods).std()
            features[f"pair__spread_zscore_{window}"] = (
                (spread - mean) / std.replace(0, np.nan)
            )

    if label is not None:
        if horizon is None:
            raise ValueError("启用历史标签特征时必须提供 horizon")
        aligned_label = pd.to_numeric(label, errors="coerce").reindex(market_data.index)
        reveal_delay = int(horizon) + 1
        for extra_lag in (0, 1, 2, 5):
            features[f"label__available_{extra_lag}"] = aligned_label.shift(
                reveal_delay + extra_lag
            )

    result = pd.DataFrame(features, index=market_data.index)
    return result.replace([np.inf, -np.inf], np.nan)"""
        ),
        md(
            """## 4. 检查特征结果与因果性

单资产目标当前产生 18 列特征，双资产目标产生 37 列特征。下面同时进行
“未来变更测试”：修改未来市场价格后，过去已经生成的特征必须完全不变。"""
        ),
        code(
            """single_row = target_pairs[~target_pairs["pair"].str.contains(" - ", regex=False)].iloc[0]
pair_row = target_pairs[target_pairs["pair"].str.contains(" - ", regex=False)].iloc[0]

for row in (single_row, pair_row):
    feature_frame = make_target_features(
        market,
        row["pair"],
        label=labels[row["target"]],
        horizon=int(row["lag"]),
    )
    print(row["target"], "|", row["pair"], "| 特征数:", feature_frame.shape[1])
    display(feature_frame.tail(3))

check_market = market.copy()
cutoff = len(check_market) - 100
original = make_target_features(check_market, pair_row["pair"])
for column in parse_pair(pair_row["pair"]):
    check_market.loc[cutoff:, column] = check_market.loc[cutoff:, column] * 1000
changed = make_target_features(check_market, pair_row["pair"])

pd.testing.assert_frame_equal(original.iloc[:cutoff], changed.iloc[:cutoff])
print("因果性检查通过：改变未来市场数据不会影响过去特征。")"""
        ),
        md(
            """## 5. 时间留出与评价指标

随机切分不适合时间序列，因为它会让训练集包含验证期之后的数据。本项目把
时间轴末端作为验证集，并在训练与验证之间设置 4 日 embargo。

每个日期的评价是 424 个目标上的横截面 Spearman 相关（Daily IC），再用
日度 IC 均值除以标准差得到 IC Sharpe。"""
        ),
        code(
            """def holdout_indices(
    n_rows: int,
    valid_size: int = 252,
    embargo: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    split = n_rows - valid_size
    train_end = split - embargo
    if valid_size <= 0 or embargo < 0 or train_end <= 0:
        raise ValueError("无法按当前 valid_size 与 embargo 完成时间切分")
    return np.arange(train_end), np.arange(split, n_rows)


def daily_ic(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    dates: np.ndarray,
) -> pd.Series:
    if y_true.shape != y_pred.shape:
        raise ValueError("真实值与预测值形状不同")
    if list(y_true.columns) != list(y_pred.columns):
        raise ValueError("真实值与预测值的目标列未对齐")

    result = {}
    date_index = pd.Index(dates)
    for date_value in date_index.unique():
        mask = date_index == date_value
        true_values = y_true.loc[mask].to_numpy(dtype=float).ravel()
        pred_values = y_pred.loc[mask].to_numpy(dtype=float).ravel()
        valid = np.isfinite(true_values) & np.isfinite(pred_values)
        if valid.sum() < 2:
            result[date_value] = np.nan
        else:
            true_rank = pd.Series(true_values[valid]).rank(method="average")
            pred_rank = pd.Series(pred_values[valid]).rank(method="average")
            result[date_value] = float(true_rank.corr(pred_rank))
    return pd.Series(result, name="daily_ic", dtype=float)


def ic_sharpe(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    dates: np.ndarray,
) -> float:
    values = daily_ic(y_true, y_pred, dates).dropna()
    std = values.std(ddof=0)
    return float(values.mean() / std) if len(values) and std > 0 else float("nan")


train_idx, valid_idx = holdout_indices(
    len(market),
    valid_size=VALID_SIZE,
    embargo=4,
)
print("训练行:", len(train_idx))
print("验证行:", len(valid_idx))
print("隔离区:", valid_idx[0] - train_idx[-1] - 1, "行")"""
        ),
        md(
            """## 6. 三种树模型与 OOF stacking

每个目标分别训练：

- LightGBM：快速学习非线性关系和特征交互；
- Random Forest：提供与 boosting 不同的方差结构；
- XGBoost：作为另一种正则化 boosting 模型。

元模型不能使用基础模型对自身训练样本的拟合预测。下面通过
`TimeSeriesSplit` 生成严格的 OOF 预测，再使用 Ridge 学习三个基础模型的
稳定线性组合。完成元模型训练后，三个基础模型会在全部可用训练行上重新拟合，
供最终验证或顺序推理使用。"""
        ),
        code(
            """@dataclass
class StackedTargetModel:
    target: str
    pair: str
    horizon: int
    base_names: list[str]
    base_models: list[RegressorMixin]
    meta_model: Ridge
    feature_columns: list[str]


def model_factory(name: str, random_state: int = 42):
    if name == "rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", RandomForestRegressor(
                n_estimators=40,
                max_depth=6,
                min_samples_leaf=8,
                max_features=0.8,
                n_jobs=1,
                random_state=random_state,
            )),
        ])

    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=80,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=5,
            min_child_samples=25,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            verbosity=-1,
            n_jobs=1,
            random_state=random_state,
        )

    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=80,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=random_state,
        )

    raise ValueError(f"未知模型: {name}")"""
        ),
        code(
            """def fit_stacked_models(
    market_data: pd.DataFrame,
    label_data: pd.DataFrame,
    pair_data: pd.DataFrame,
    training_indices: np.ndarray,
    base_names: tuple[str, ...] = ("lgbm", "rf", "xgb"),
    n_splits: int = 3,
    max_targets: int | None = None,
) -> dict[str, StackedTargetModel]:
    pair_lookup = pair_data.set_index("target")
    targets = target_names if max_targets is None else target_names[:max_targets]
    fitted: dict[str, StackedTargetModel] = {}

    for position, target in enumerate(targets, start=1):
        pair = str(pair_lookup.loc[target, "pair"])
        horizon = int(pair_lookup.loc[target, "lag"])
        y = pd.to_numeric(label_data[target], errors="coerce")
        x = make_target_features(
            market_data,
            pair,
            label=y,
            horizon=horizon,
        )

        usable = training_indices[y.iloc[training_indices].notna().to_numpy()]
        if len(usable) < 100:
            print(f"跳过 {target}: 可用训练行不足")
            continue

        x_train = x.iloc[usable]
        y_train = y.iloc[usable]
        oof = np.full((len(usable), len(base_names)), np.nan)
        splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)

        for fold_train, fold_valid in splitter.split(x_train):
            for model_index, name in enumerate(base_names):
                model = model_factory(name, RANDOM_STATE)
                model.fit(x_train.iloc[fold_train], y_train.iloc[fold_train])
                oof[fold_valid, model_index] = model.predict(
                    x_train.iloc[fold_valid]
                )

        meta_rows = np.isfinite(oof).all(axis=1)
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(oof[meta_rows], y_train.iloc[meta_rows])

        base_models: list[RegressorMixin] = []
        for name in base_names:
            model = model_factory(name, RANDOM_STATE)
            model.fit(x_train, y_train)
            base_models.append(model)

        fitted[target] = StackedTargetModel(
            target=target,
            pair=pair,
            horizon=horizon,
            base_names=list(base_names),
            base_models=base_models,
            meta_model=meta_model,
            feature_columns=list(x.columns),
        )
        print(f"[{position}/{len(targets)}] 完成 {target}")

    return fitted"""
        ),
        md(
            """## 7. 训练与时间留出验证

默认只训练前 8 个目标，所以这里的指标只用于确认数据、模型和评价代码能够
连通，不能与 424 目标的完整实验直接比较。完整实验需要将
`RUN_FULL_TRAINING=True` 后重新运行 notebook。"""
        ),
        code(
            """models = fit_stacked_models(
    market,
    labels,
    target_pairs,
    train_idx,
    max_targets=MAX_TARGETS,
)
print("已训练目标数:", len(models))"""
        ),
        code(
            """def predict_stacked_models(
    models: dict[str, StackedTargetModel],
    market_data: pd.DataFrame,
    label_data: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    predictions: dict[str, np.ndarray] = {}
    pair_cache: dict[str, pd.DataFrame] = {}

    for target, bundle in models.items():
        label = (
            label_data[target]
            if target in label_data
            else pd.Series(np.nan, index=market_data.index)
        )

        if bundle.pair not in pair_cache:
            pair_cache[bundle.pair] = make_target_features(
                market_data,
                bundle.pair,
            )

        x = pair_cache[bundle.pair].copy()
        aligned_label = pd.to_numeric(label, errors="coerce").reindex(market_data.index)
        reveal_delay = bundle.horizon + 1
        for extra_lag in (0, 1, 2, 5):
            x[f"label__available_{extra_lag}"] = aligned_label.shift(
                reveal_delay + extra_lag
            )

        x = x.reindex(columns=bundle.feature_columns)
        base_predictions = np.column_stack([
            model.predict(x.iloc[indices])
            for model in bundle.base_models
        ])
        predictions[target] = bundle.meta_model.predict(base_predictions)

    return pd.DataFrame(predictions, index=market_data.index[indices])


validation_predictions = predict_stacked_models(
    models,
    market,
    labels,
    valid_idx,
)
validation_truth = labels.loc[valid_idx, validation_predictions.columns].copy()
validation_truth.index = validation_predictions.index
validation_dates = market.loc[valid_idx, "date_id"].to_numpy()

validation_ics = daily_ic(
    validation_truth,
    validation_predictions,
    validation_dates,
)
validation_report = {
    "targets": len(models),
    "train_rows": len(train_idx),
    "valid_rows": len(valid_idx),
    "mean_daily_ic": float(validation_ics.mean()),
    "ic_sharpe": ic_sharpe(
        validation_truth,
        validation_predictions,
        validation_dates,
    ),
}
display(validation_report)"""
        ),
        md(
            """## 8. 保存模型与验证产物

模型对象同时保存资产 pair、horizon、特征列顺序、三个基础模型和 Ridge
元模型。推理阶段必须恢复完全相同的特征列顺序。"""
        ),
        code(
            """model_path = OUTPUT_DIR / "stacked_models.pkl"
prediction_path = OUTPUT_DIR / "validation_predictions.csv"
metrics_path = OUTPUT_DIR / "metrics.json"

with model_path.open("wb") as handle:
    pickle.dump(models, handle)

output_predictions = validation_predictions.copy()
output_predictions.insert(0, "date_id", validation_dates)
output_predictions.to_csv(prediction_path, index=False)
metrics_path.write_text(
    json.dumps(validation_report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

for path in (model_path, prediction_path, metrics_path):
    assert path.exists() and path.stat().st_size > 0
    print(path, "|", path.stat().st_size, "bytes")"""
        ),
        md(
            """## 9. 顺序推理为什么需要状态

竞赛不会一次性提供完整未来测试集，而是按批次调用 `predict`。预测器必须：

1. 保存最近的市场历史；
2. 按 `label_date_id` 把新揭示标签写回对应历史日期；
3. 追加当前测试行；
4. 使用与训练完全相同的特征逻辑；
5. 按 `target_0` 到 `target_423` 的数字顺序返回结果；
6. 裁剪历史窗口，避免推理期间内存持续增长。"""
        ),
        code(
            """def ordered_target_columns(columns) -> list[str]:
    names = [str(column) for column in columns if str(column).startswith("target_")]
    return sorted(names, key=lambda name: int(name.split("_")[1]))


def as_pandas(frame) -> pd.DataFrame:
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame.copy()


class SequentialPredictor:
    def __init__(
        self,
        fitted_models: dict[str, StackedTargetModel],
        market_history: pd.DataFrame,
        label_history: pd.DataFrame,
        history_window: int = 128,
    ) -> None:
        self.models = fitted_models
        self.market_history = market_history.copy()
        self.label_history = label_history.copy()
        self.history_window = history_window
        self.started = False

    def append_revealed_labels(self, batches: tuple) -> None:
        for raw_batch in batches:
            batch = as_pandas(raw_batch)
            if batch.empty or "label_date_id" not in batch:
                continue

            for _, row in batch.iterrows():
                label_date = row["label_date_id"]
                mask = self.label_history["date_id"].eq(label_date)
                if not mask.any():
                    self.label_history = pd.concat(
                        [
                            self.label_history,
                            pd.DataFrame([{"date_id": label_date}]),
                        ],
                        ignore_index=True,
                    )
                    mask = self.label_history["date_id"].eq(label_date)

                for column in ordered_target_columns(batch.columns):
                    if pd.notna(row[column]):
                        self.label_history.loc[mask, column] = row[column]

    def predict(self, test, *label_batches) -> pd.DataFrame:
        current = as_pandas(test)

        if not self.started:
            first_date = current["date_id"].min()
            self.market_history = self.market_history[
                self.market_history["date_id"] < first_date
            ].tail(self.history_window).reset_index(drop=True)
            keep_dates = set(self.market_history["date_id"])
            self.label_history = self.label_history[
                self.label_history["date_id"].isin(keep_dates)
            ].reset_index(drop=True)
            self.started = True

        self.append_revealed_labels(label_batches)
        market_columns = [c for c in current.columns if c != "is_scored"]
        start = len(self.market_history)
        self.market_history = pd.concat(
            [self.market_history, current[market_columns]],
            ignore_index=True,
        )

        aligned_labels = self.market_history[["date_id"]].merge(
            self.label_history,
            on="date_id",
            how="left",
        )
        indices = np.arange(start, len(self.market_history))
        predictions = predict_stacked_models(
            self.models,
            self.market_history,
            aligned_labels,
            indices,
        )
        result = predictions.reindex(
            columns=ordered_target_columns(self.models.keys())
        ).reset_index(drop=True)

        self.market_history = self.market_history.tail(
            self.history_window
        ).reset_index(drop=True)
        keep_dates = set(self.market_history["date_id"])
        self.label_history = self.label_history[
            self.label_history["date_id"].isin(keep_dates)
        ].reset_index(drop=True)
        return result"""
        ),
        md(
            """## 10. 可选：运行官方本地推理网关

只有在 `data/raw/kaggle_evaluation/`、`test.csv` 和延迟标签文件都存在时，
才运行下面的单元格。它不依赖 Colab，但需要完整的 Kaggle 竞赛数据包。

对于正式 424 目标推理，应先使用全部训练数据拟合模型，而不是使用上面的
时间留出模型。"""
        ),
        code(
            """RUN_LOCAL_GATEWAY = False

if RUN_LOCAL_GATEWAY:
    if MAX_TARGETS is not None:
        raise ValueError("官方网关要求 424 个目标，请先启用完整训练")

    sys.path.append(str(DATA_DIR.resolve()))
    from kaggle_evaluation.mitsui_inference_server import MitsuiInferenceServer

    predictor = SequentialPredictor(models, market, labels)

    def predict(test, lag1, lag2, lag3, lag4):
        return predictor.predict(test, lag1, lag2, lag3, lag4)

    server = MitsuiInferenceServer(predict)
    server.run_local_gateway((str(DATA_DIR),))
else:
    print("RUN_LOCAL_GATEWAY=False：跳过官方本地网关。")"""
        ),
        md(
            """## 11. 下一步实验方向

这份 notebook 固定了可靠的训练与推理主线。后续提升应优先围绕特征而非
无约束地增加模型复杂度：

- 跨资产滚动相关与 beta；
- 波动率期限结构和高低波动 regime；
- spread 的半衰期、偏离持续时间和均值回归速度；
- 市场横截面 rank、行业或资产类别聚合；
- 不同 horizon 使用不同窗口组合；
- 对 424 个目标按资产类别或 horizon 分组调参；
- 在相同时间切分下进行特征消融，记录每组特征的增量贡献。

任何新特征都必须先通过未来数据变更测试，再进入完整训练。"""
        ),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)


if __name__ == "__main__":
    main()
