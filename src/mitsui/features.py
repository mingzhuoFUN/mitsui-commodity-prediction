from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """训练和在线推理共用的特征周期配置。"""

    return_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    rolling_windows: tuple[int, ...] = (5, 20, 60)


def parse_pair(pair: str) -> list[str]:
    """解析目标资产 pair，同时保留列名内部的连字符。"""
    return [part.strip() for part in pair.split(" - ") if part.strip()]


def _safe_log(series: pd.Series) -> pd.Series:
    """取对数前，将非数值和非正价格转换为缺失值。"""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log(values.where(values > 0))


def make_target_features(
    market: pd.DataFrame,
    pair: str,
    config: FeatureConfig = FeatureConfig(),
    label: pd.Series | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """只使用目标对应的资产列构造因果特征。

    第 t 行只使用不晚于 t 时刻可获得的观测。市场缺失值只进行前向填充，
    不使用后向填充或负数 shift。
    """
    columns = parse_pair(pair)
    missing = [column for column in columns if column not in market.columns]
    if missing:
        raise KeyError(f"Pair columns missing from market data: {missing}")

    # 前向填充满足因果性：第 t 行只使用 t 时刻及之前的观测。
    # 不使用后向填充，因为它会读取未来市场数据。
    raw = market[columns].apply(pd.to_numeric, errors="coerce").ffill()
    features: dict[str, pd.Series] = {}
    logs: dict[str, pd.Series] = {}

    for column in columns:
        log_price = _safe_log(raw[column])
        logs[column] = log_price
        features[f"{column}__log_level"] = log_price

        # 对数差分可以沿时间累加，也便于比较价格尺度差异很大的资产。
        one_day_return = log_price.diff()
        features[f"{column}__return_1"] = one_day_return
        for lag in config.return_lags:
            features[f"{column}__log_return_{lag}"] = log_price.diff(lag)

        for window in config.rolling_windows:
            rolling = one_day_return.rolling(window=window, min_periods=max(2, window // 3))
            features[f"{column}__return_mean_{window}"] = rolling.mean()
            features[f"{column}__return_std_{window}"] = rolling.std()

    if len(columns) == 2:
        # 对于由两个资产定义的目标，相对对数价差比两个独立价格序列
        # 更直接地表达二者的相对关系。
        left, right = columns
        spread = logs[left] - logs[right]
        features["pair__log_spread"] = spread
        features["pair__spread_change_1"] = spread.diff()
        for window in config.rolling_windows:
            mean = spread.rolling(window=window, min_periods=max(2, window // 3)).mean()
            std = spread.rolling(window=window, min_periods=max(2, window // 3)).std()
            features[f"pair__spread_zscore_{window}"] = (spread - mean) / std.replace(0, np.nan)

    if label is not None:
        if horizon is None:
            raise ValueError("horizon is required when label features are enabled")
        aligned_label = pd.to_numeric(label, errors="coerce").reindex(market.index)
        # 只有远期收益窗口结束后，历史标签才可使用。
        # 额外的一天用于匹配顺序推理网关的标签揭示时机。
        reveal_delay = int(horizon) + 1
        for extra_lag in (0, 1, 2, 5):
            features[f"label__available_{extra_lag}"] = aligned_label.shift(
                reveal_delay + extra_lag
            )

    # 树模型和 sklearn 填补器能够处理 NaN，但不能处理正负无穷。
    result = pd.DataFrame(features, index=market.index)
    return result.replace([np.inf, -np.inf], np.nan)
