from __future__ import annotations

import numpy as np
import pandas as pd

from .data import target_columns
from .ensemble import StackedTargetModel, predict_stacked_models


def _as_pandas(frame) -> pd.DataFrame:
    """Normalize pandas/Polars gateway inputs without mutating the caller."""
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame.copy()


class SequentialPredictor:
    """Stateful predictor matching the competition's sequential API.

    The gateway invokes ``predict`` repeatedly. The object therefore keeps the
    recent market rows and labels revealed between calls instead of treating
    each test batch as an independent table.
    """

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
        """Merge newly revealed targets into history by ``label_date_id``."""
        for raw_batch in batches:
            batch = _as_pandas(raw_batch)
            if batch.empty or "label_date_id" not in batch:
                continue
            for _, row in batch.iterrows():
                label_date = row["label_date_id"]
                # label_date_id identifies the historical market row to which
                # the revealed target belongs; it is not the current test date.
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
        """Update sequential state and predict every row in the current batch."""
        current = _as_pandas(test)
        if not self._started:
            # Local gateway data can include the full training history. On the
            # first call, discard anything at or after the first test date to
            # prevent duplicated rows and retain only the required lookback.
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

        # Revealed labels become available before features for this batch are
        # assembled, matching the competition call order.
        self._append_revealed_labels(label_batches)
        market_columns = [c for c in current.columns if c != "is_scored"]
        start = len(self.market_history)
        self.market_history = pd.concat(
            [self.market_history, current[market_columns]], ignore_index=True
        )

        # Left-join onto market dates so unavailable labels remain NaN and are
        # handled exactly as they were during training.
        labels = self.market_history[["date_id"]].merge(
            self.label_history, on="date_id", how="left"
        )
        indices = np.arange(start, len(self.market_history))
        predictions = predict_stacked_models(
            self.models, self.market_history, labels, indices
        )
        # Kaggle validates both the number and ordering of output columns.
        ordered = target_columns(self.models.keys())
        result = predictions.reindex(columns=ordered).reset_index(drop=True)
        # Bound memory usage while keeping enough rows for the longest rolling
        # window and delayed-label feature.
        self.market_history = self.market_history.tail(self.history_window).reset_index(
            drop=True
        )
        keep_dates = set(self.market_history["date_id"])
        self.label_history = self.label_history[
            self.label_history["date_id"].isin(keep_dates)
        ].reset_index(drop=True)
        return result
