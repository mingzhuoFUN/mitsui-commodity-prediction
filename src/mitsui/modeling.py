from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import target_columns
from .features import FeatureConfig, make_target_features


@dataclass
class TargetModel:
    """Everything required to reproduce one target's feature/model contract."""

    target: str
    pair: str
    model: Pipeline
    feature_columns: list[str]


def make_ridge(alpha: float = 10.0) -> Pipeline:
    """Build the baseline pipeline with imputation and scale normalization."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def fit_target_models(
    market: pd.DataFrame,
    labels: pd.DataFrame,
    target_pairs: pd.DataFrame,
    train_indices: np.ndarray,
    *,
    alpha: float = 10.0,
    max_targets: int | None = None,
    feature_config: FeatureConfig = FeatureConfig(),
) -> dict[str, TargetModel]:
    """Fit one lightweight causal baseline model per official target."""
    # target_pairs is the authoritative mapping from target name to the market
    # columns that are allowed to describe it.
    pairs = target_pairs.set_index("target")
    targets = target_columns(labels.columns)
    if max_targets is not None:
        targets = targets[:max_targets]

    fitted: dict[str, TargetModel] = {}
    for target in targets:
        pair = str(pairs.loc[target, "pair"])
        x_all = make_target_features(market, pair, feature_config)
        y_all = pd.to_numeric(labels[target], errors="coerce")

        # Combine the caller's chronological split with label availability.
        # Feature NaNs are handled inside the sklearn pipeline.
        valid = np.zeros(len(market), dtype=bool)
        valid[train_indices] = True
        valid &= y_all.notna().to_numpy()
        if valid.sum() < 30:
            continue

        model = make_ridge(alpha=alpha)
        model.fit(x_all.loc[valid], y_all.loc[valid])
        fitted[target] = TargetModel(
            target=target,
            pair=pair,
            model=model,
            feature_columns=list(x_all.columns),
        )
    return fitted


def predict_target_models(
    models: dict[str, TargetModel],
    market: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    predictions: dict[str, np.ndarray] = {}
    for target, bundle in models.items():
        features = make_target_features(market, bundle.pair)
        # Persisted column order is part of the model contract. Reindexing also
        # protects inference from accidental feature-order changes.
        features = features.reindex(columns=bundle.feature_columns)
        predictions[target] = bundle.model.predict(features.iloc[indices])
    return pd.DataFrame(predictions, index=market.index[indices])


def save_models(models: dict[str, TargetModel], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(models, handle)
