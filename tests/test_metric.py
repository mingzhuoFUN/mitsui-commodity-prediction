from __future__ import annotations

import numpy as np
import pandas as pd

from mitsui.metric import daily_ic, ic_sharpe


def test_daily_ic_is_one_for_perfect_rank_order() -> None:
    y_true = pd.DataFrame(
        {
            "target_0": [1.0, 3.0],
            "target_1": [2.0, 2.0],
            "target_2": [3.0, 1.0],
        }
    )
    y_pred = y_true.copy()
    dates = pd.Series(["2025-01-01", "2025-01-02"])

    result = daily_ic(y_true, y_pred, dates)

    assert np.allclose(result.to_numpy(), [1.0, 1.0])


def test_ic_sharpe_returns_nan_when_daily_ic_has_no_variance() -> None:
    y_true = pd.DataFrame({"target_0": [1.0, 1.0], "target_1": [2.0, 2.0]})
    y_pred = y_true.copy()
    dates = pd.Series(["2025-01-01", "2025-01-02"])

    assert np.isnan(ic_sharpe(y_true, y_pred, dates))
