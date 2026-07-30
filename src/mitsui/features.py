from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    return_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    rolling_windows: tuple[int, ...] = (5, 20, 60)


def parse_pair(pair: str) -> list[str]:
    """Parse a target pair without breaking column names containing hyphens."""
    return [part.strip() for part in pair.split(" - ") if part.strip()]


def _safe_log(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log(values.where(values > 0))


def make_target_features(
    market: pd.DataFrame,
    pair: str,
    config: FeatureConfig = FeatureConfig(),
    label: pd.Series | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Create causal features for one target using only its pair columns.

    Row t uses observations available no later than row t. Missing market values
    are forward-filled only; no backward fill or negative shift is used.
    """
    columns = parse_pair(pair)
    missing = [column for column in columns if column not in market.columns]
    if missing:
        raise KeyError(f"Pair columns missing from market data: {missing}")

    raw = market[columns].apply(pd.to_numeric, errors="coerce").ffill()
    features: dict[str, pd.Series] = {}
    logs: dict[str, pd.Series] = {}

    for column in columns:
        log_price = _safe_log(raw[column])
        logs[column] = log_price
        features[f"{column}__log_level"] = log_price

        one_day_return = log_price.diff()
        features[f"{column}__return_1"] = one_day_return
        for lag in config.return_lags:
            features[f"{column}__log_return_{lag}"] = log_price.diff(lag)

        for window in config.rolling_windows:
            rolling = one_day_return.rolling(window=window, min_periods=max(2, window // 3))
            features[f"{column}__return_mean_{window}"] = rolling.mean()
            features[f"{column}__return_std_{window}"] = rolling.std()

    if len(columns) == 2:
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
        reveal_delay = int(horizon) + 1
        for extra_lag in (0, 1, 2, 5):
            features[f"label__available_{extra_lag}"] = aligned_label.shift(
                reveal_delay + extra_lag
            )

    result = pd.DataFrame(features, index=market.index)
    return result.replace([np.inf, -np.inf], np.nan)
