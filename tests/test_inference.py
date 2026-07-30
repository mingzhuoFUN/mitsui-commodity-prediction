from __future__ import annotations

import pandas as pd

from mitsui.inference import SequentialPredictor


def test_revealed_labels_are_written_by_label_date() -> None:
    predictor = SequentialPredictor(
        models={},
        market_history=pd.DataFrame({"date_id": [1, 2], "A": [10.0, 11.0]}),
        label_history=pd.DataFrame({"date_id": [1, 2], "target_0": [0.1, None]}),
    )
    batch = pd.DataFrame(
        {"date_id": [4], "label_date_id": [2], "target_0": [0.25]}
    )
    predictor._append_revealed_labels((batch,))
    value = predictor.label_history.loc[
        predictor.label_history["date_id"].eq(2), "target_0"
    ].iloc[0]
    assert value == 0.25
