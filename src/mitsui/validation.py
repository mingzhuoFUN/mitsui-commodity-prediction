from __future__ import annotations

import numpy as np


def holdout_indices(
    n_rows: int,
    valid_size: int = 252,
    embargo: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """返回带 embargo 的时间顺序训练集和验证集索引。

    验证集始终位于时间轴末端。``embargo`` 会移除紧邻验证集之前的行，
    防止边界附近的远期收益标签窗口与验证期重叠。
    """
    if valid_size <= 0 or embargo < 0:
        raise ValueError("valid_size must be positive and embargo non-negative")
    split = n_rows - valid_size
    # 训练集在验证边界之前提前结束，中间隔离区不属于任何返回索引。
    train_end = split - embargo
    if train_end <= 0:
        raise ValueError("Not enough rows for the requested validation split")
    return np.arange(train_end), np.arange(split, n_rows)
