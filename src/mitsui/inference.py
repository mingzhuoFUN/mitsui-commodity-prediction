from __future__ import annotations

import numpy as np
import pandas as pd

from .data import target_columns
from .ensemble import StackedTargetModel, predict_stacked_models


def _as_pandas(frame) -> pd.DataFrame:
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame.copy()


class SequentialPredictor:
    """Stateful predictor matching the competition's sequential API."""

    def __init__(
        self,
        models: dict[str, StackedTargetModel],
        market_history: pd.DataFrame,
        label_history: pd.DataFrame,
        history_window: int = 128,
    ) -> None:
        self.models = models
        self.market_history = market_history.copy()
        self.label_history = label_history.copy()
        self.history_window = history_window
        self._started = False

    def _append_revealed_labels(self, batches: tuple) -> None:
        for raw_batch in batches:
            batch = _as_pandas(raw_batch)
            if batch.empty or "label_date_id" not in batch:
                continue
            for _, row in batch.iterrows():
                label_date = row["label_date_id"]
                mask = self.label_history["date_id"].eq(label_date)
                if not mask.any():
                    new_row = {"date_id": label_date}
                    self.label_history = pd.concat(
                        [self.label_history, pd.DataFrame([new_row])],
                        ignore_index=True,
                    )
                    mask = self.label_history["date_id"].eq(label_date)
                for column in target_columns(batch.columns):
                    if pd.notna(row[column]):
                        self.label_history.loc[mask, column] = row[column]

    def predict(self, test, *label_batches) -> pd.DataFrame:
        current = _as_pandas(test)
        if not self._started:
            first_date = current["date_id"].min()
            self.market_history = self.market_history[
                self.market_history["date_id"] < first_date
            ].copy()
            self.label_history = self.label_history[
                self.label_history["date_id"] < first_date
            ].copy()
            self.market_history = self.market_history.tail(self.history_window).reset_index(
                drop=True
            )
            keep_dates = set(self.market_history["date_id"])
            self.label_history = self.label_history[
                self.label_history["date_id"].isin(keep_dates)
            ].reset_index(drop=True)
            self._started = True

        self._append_revealed_labels(label_batches)
        market_columns = [c for c in current.columns if c != "is_scored"]
        start = len(self.market_history)
        self.market_history = pd.concat(
            [self.market_history, current[market_columns]], ignore_index=True
        )

        labels = self.market_history[["date_id"]].merge(
            self.label_history, on="date_id", how="left"
        )
        indices = np.arange(start, len(self.market_history))
        predictions = predict_stacked_models(
            self.models, self.market_history, labels, indices
        )
        ordered = target_columns(self.models.keys())
        result = predictions.reindex(columns=ordered).reset_index(drop=True)
        self.market_history = self.market_history.tail(self.history_window).reset_index(
            drop=True
        )
        keep_dates = set(self.market_history["date_id"])
        self.label_history = self.label_history[
            self.label_history["date_id"].isin(keep_dates)
        ].reset_index(drop=True)
        return result
