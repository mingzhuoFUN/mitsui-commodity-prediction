from __future__ import annotations

import numpy as np
import pandas as pd

from .data import target_columns
from .ensemble import StackedTargetModel, predict_stacked_models


def _as_pandas(frame) -> pd.DataFrame:
    """将 pandas/Polars 输入统一为 pandas，且不修改调用方对象。"""
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame.copy()


class SequentialPredictor:
    """与竞赛顺序推理 API 匹配的有状态预测器。

    网关会重复调用 ``predict``，因此对象必须在调用之间保存近期市场数据
    和新揭示的标签，不能把每个测试批次当作互相独立的表格。
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
        """按照 ``label_date_id`` 将新揭示的目标值写入历史记录。"""
        for raw_batch in batches:
            batch = _as_pandas(raw_batch)
            if batch.empty or "label_date_id" not in batch:
                continue
            for _, row in batch.iterrows():
                label_date = row["label_date_id"]
                # label_date_id 指向该标签所属的历史市场行，
                # 它并不是当前测试日期。
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
        """更新顺序推理状态，并预测当前批次中的全部行。"""
        current = _as_pandas(test)
        if not self._started:
            # 本地网关数据可能包含完整训练历史。首次调用时丢弃首个测试日
            # 及其之后的历史内容，避免重复行，并只保留必要回看窗口。
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

        # 新揭示标签在构造当前批次特征前可用，这与竞赛调用顺序一致。
        self._append_revealed_labels(label_batches)
        market_columns = [c for c in current.columns if c != "is_scored"]
        start = len(self.market_history)
        self.market_history = pd.concat(
            [self.market_history, current[market_columns]], ignore_index=True
        )

        # 按市场日期左连接，使尚未公开的标签保持 NaN，
        # 并沿用训练阶段相同的缺失值处理方式。
        labels = self.market_history[["date_id"]].merge(
            self.label_history, on="date_id", how="left"
        )
        indices = np.arange(start, len(self.market_history))
        predictions = predict_stacked_models(
            self.models, self.market_history, labels, indices
        )
        # Kaggle 会同时校验输出列数量和列顺序。
        ordered = target_columns(self.models.keys())
        result = predictions.reindex(columns=ordered).reset_index(drop=True)
        # 在保留最长滚动窗口及延迟标签所需历史的同时限制内存占用。
        self.market_history = self.market_history.tail(self.history_window).reset_index(
            drop=True
        )
        keep_dates = set(self.market_history["date_id"])
        self.label_history = self.label_history[
            self.label_history["date_id"].isin(keep_dates)
        ].reset_index(drop=True)
        return result
