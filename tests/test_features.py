from __future__ import annotations

import numpy as np
import pandas as pd

from mitsui.features import make_target_features, parse_pair


def test_parse_pair_preserves_internal_hyphens() -> None:
    pair = "JPX_Gold_Rolling-Spot_Futures_Close - FX_AUDJPY"
    assert parse_pair(pair) == [
        "JPX_Gold_Rolling-Spot_Futures_Close",
        "FX_AUDJPY",
    ]


def test_features_do_not_change_when_future_rows_change() -> None:
    market = pd.DataFrame(
        {
            "A": np.arange(1.0, 101.0),
            "B": np.arange(101.0, 201.0),
        }
    )
    original = make_target_features(market, "A - B")
    changed = market.copy()
    changed.loc[80:, ["A", "B"]] *= 1000
    modified = make_target_features(changed, "A - B")

    pd.testing.assert_frame_equal(original.iloc[:80], modified.iloc[:80])
