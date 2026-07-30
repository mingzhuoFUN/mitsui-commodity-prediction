from __future__ import annotations

import numpy as np
import pandas as pd


def _spearman_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average")
    right_rank = pd.Series(right).rank(method="average")
    return float(left_rank.corr(right_rank))


def daily_ic(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    dates: pd.Series | pd.Index | np.ndarray,
) -> pd.Series:
    """Compute per-date Spearman rank correlation across target columns."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}"
        )
    if list(y_true.columns) != list(y_pred.columns):
        raise ValueError("y_true and y_pred columns must be aligned in the same order")

    dates_index = pd.Index(dates, name="date")
    if len(dates_index) != len(y_true):
        raise ValueError("dates must have the same length as y_true/y_pred")

    rows: list[tuple[object, float]] = []
    for date_value in dates_index.unique():
        mask = dates_index == date_value
        true_values = y_true.loc[mask].to_numpy(dtype=float).ravel()
        pred_values = y_pred.loc[mask].to_numpy(dtype=float).ravel()
        valid = np.isfinite(true_values) & np.isfinite(pred_values)

        if valid.sum() < 2:
            ic = np.nan
        else:
            ic = _spearman_corr(pred_values[valid], true_values[valid])
        rows.append((date_value, ic))

    return pd.Series(dict(rows), name="daily_ic", dtype=float)


def ic_sharpe(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    dates: pd.Series | pd.Index | np.ndarray,
    ddof: int = 0,
) -> float:
    """Compute mean daily IC divided by standard deviation of daily IC."""
    ics = daily_ic(y_true=y_true, y_pred=y_pred, dates=dates).dropna()
    if ics.empty:
        return float("nan")

    std = ics.std(ddof=ddof)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(ics.mean() / std)
