from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import target_columns
from .features import FeatureConfig, make_target_features


@dataclass
class StackedTargetModel:
    """单个预测目标的可序列化训练结果。

    除拟合后的模型外，该对象还保存资产 pair、预测周期和特征顺序，
    以便推理阶段准确重建训练时的输入结构。
    """

    target: str
    pair: str
    horizon: int
    base_names: list[str]
    base_models: list[RegressorMixin]
    meta_model: Ridge
    feature_columns: list[str]


def _model_factory(name: str, random_state: int = 42) -> Callable[[], RegressorMixin]:
    """返回模型构造器，确保每个 OOF 折都使用全新的模型实例。"""
    if name == "ridge":
        return lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
    if name == "rf":
        return lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=40,
                        max_depth=6,
                        min_samples_leaf=8,
                        max_features=0.8,
                        # 在模型外部统一控制并行度，避免 Colab CPU 核心
                        # 因嵌套并行而过度占用。
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "lgbm":
        from lightgbm import LGBMRegressor

        return lambda: LGBMRegressor(
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

        return lambda: XGBRegressor(
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
    raise ValueError(f"Unknown model: {name}")


def fit_stacked_models(
    market: pd.DataFrame,
    labels: pd.DataFrame,
    target_pairs: pd.DataFrame,
    train_indices: np.ndarray,
    *,
    base_names: tuple[str, ...] = ("lgbm", "rf", "xgb"),
    n_splits: int = 3,
    max_targets: int | None = None,
    feature_config: FeatureConfig = FeatureConfig(),
) -> dict[str, StackedTargetModel]:
    """为每个目标训练基础模型及无泄漏的 OOF 元模型。

    元模型不会看到基础模型对自身训练样本生成的预测。
    这是 stacking 在时间序列数据上保持验证可信度的关键。
    """
    pairs = target_pairs.set_index("target")
    targets = target_columns(labels.columns)
    if max_targets is not None:
        targets = targets[:max_targets]

    fitted: dict[str, StackedTargetModel] = {}
    for target in targets:
        pair = str(pairs.loc[target, "pair"])
        horizon = int(pairs.loc[target, "lag"])
        y = pd.to_numeric(labels[target], errors="coerce")
        x = make_target_features(
            market, pair, feature_config, label=y, horizon=horizon
        )
        # 保持调用方提供的时间顺序，同时移除官方标签缺失的行。
        usable = train_indices[y.iloc[train_indices].notna().to_numpy()]
        if len(usable) < 100:
            continue

        x_train = x.iloc[usable]
        y_train = y.iloc[usable]
        # 未被任何 OOF 验证折覆盖的行保持 NaN，不参与元模型训练。
        oof = np.full((len(usable), len(base_names)), np.nan)
        # gap=horizon 将每折训练集末尾与验证集开头隔开，
        # 减少远期收益标签窗口发生重叠。
        splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
        for fold_train, fold_valid in splitter.split(x_train):
            for model_idx, name in enumerate(base_names):
                model = _model_factory(name)()
                model.fit(x_train.iloc[fold_train], y_train.iloc[fold_train])
                oof[fold_valid, model_idx] = model.predict(x_train.iloc[fold_valid])

        # 只有全部基础模型都给出预测时，才构成完整的 stacking 特征向量。
        meta_rows = np.isfinite(oof).all(axis=1)
        meta = Ridge(alpha=1.0)
        meta.fit(oof[meta_rows], y_train.iloc[meta_rows])

        # OOF 折模型只用于训练元模型；推理使用的基础模型需要在全部
        # 可用训练行上重新拟合。
        base_models: list[RegressorMixin] = []
        for name in base_names:
            model = _model_factory(name)()
            model.fit(x_train, y_train)
            base_models.append(model)

        fitted[target] = StackedTargetModel(
            target=target,
            pair=pair,
            horizon=horizon,
            base_names=list(base_names),
            base_models=base_models,
            meta_model=meta,
            feature_columns=list(x.columns),
        )
    return fitted


def predict_stacked_models(
    models: dict[str, StackedTargetModel],
    market: pd.DataFrame,
    labels: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    """使用已保存的基础模型和元模型预测指定行。"""
    predictions: dict[str, np.ndarray] = {}
    # 多个目标可能共享同一资产 pair，因此先缓存只依赖 pair 的市场特征，
    # 再为各目标添加专属的延迟标签特征。
    pair_feature_cache: dict[str, pd.DataFrame] = {}
    for target, bundle in models.items():
        label = labels[target] if target in labels else pd.Series(np.nan, index=market.index)
        if bundle.pair not in pair_feature_cache:
            pair_feature_cache[bundle.pair] = make_target_features(market, bundle.pair)
        x = pair_feature_cache[bundle.pair].copy()
        reveal_delay = bundle.horizon + 1
        aligned_label = pd.to_numeric(label, errors="coerce").reindex(market.index)
        for extra_lag in (0, 1, 2, 5):
            x[f"label__available_{extra_lag}"] = aligned_label.shift(
                reveal_delay + extra_lag
            )
        # 恢复拟合时使用的精确特征结构和列顺序。
        x = x.reindex(columns=bundle.feature_columns)
        base_predictions = np.column_stack(
            [model.predict(x.iloc[indices]) for model in bundle.base_models]
        )
        predictions[target] = bundle.meta_model.predict(base_predictions)
    return pd.DataFrame(predictions, index=market.index[indices])


def save_stacked_models(
    models: dict[str, StackedTargetModel], path: str | Path
) -> None:
    """将全部目标模型打包保存为一个提交阶段使用的文件。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(models, handle)


def load_stacked_models(path: str | Path) -> dict[str, StackedTargetModel]:
    """加载由 :func:`save_stacked_models` 创建的模型文件。"""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
