from __future__ import annotations

import numpy as np


def holdout_indices(
    n_rows: int,
    valid_size: int = 252,
    embargo: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return chronological train/validation indices with a small embargo.

    The validation block is always the newest data. ``embargo`` removes rows
    immediately before it so forward-return labels near the boundary cannot
    overlap the validation period.
    """
    if valid_size <= 0 or embargo < 0:
        raise ValueError("valid_size must be positive and embargo non-negative")
    split = n_rows - valid_size
    # Training stops before the validation boundary; the gap is intentionally
    # absent from both returned index arrays.
    train_end = split - embargo
    if train_end <= 0:
        raise ValueError("Not enough rows for the requested validation split")
    return np.arange(train_end), np.arange(split, n_rows)
