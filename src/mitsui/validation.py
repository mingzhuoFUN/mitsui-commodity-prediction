from __future__ import annotations

import numpy as np


def holdout_indices(
    n_rows: int,
    valid_size: int = 252,
    embargo: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return chronological train/validation indices with a small embargo."""
    if valid_size <= 0 or embargo < 0:
        raise ValueError("valid_size must be positive and embargo non-negative")
    split = n_rows - valid_size
    train_end = split - embargo
    if train_end <= 0:
        raise ValueError("Not enough rows for the requested validation split")
    return np.arange(train_end), np.arange(split, n_rows)
