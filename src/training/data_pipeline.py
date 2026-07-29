"""Time splits and train-only feature standardization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PREPROCESSING_VERSION = "qqq_spy_preprocessing_v1"


class TrainingDataError(ValueError):
    """Raised when training preparation would violate the time contract."""


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Contiguous target-date boundaries for the initial frozen experiment."""

    train_end: date
    validation_end: date
    test_end: date

    def __post_init__(self) -> None:
        if not self.train_end < self.validation_end < self.test_end:
            raise ValueError(
                "split endpoints must satisfy train < validation < test"
            )


DEFAULT_SPLIT_CONFIG = DatasetSplitConfig(
    train_end=date(2018, 12, 31),
    validation_end=date(2022, 12, 31),
    test_end=date(2026, 6, 30),
)


@dataclass(frozen=True)
class FeatureStandardizer:
    """Column-ordered z-score parameters fit on the training split only."""

    columns: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    epsilon: float = 1e-8

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        epsilon: float = 1e-8,
    ) -> "FeatureStandardizer":
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        columns = tuple(feature_columns)
        if not columns:
            raise TrainingDataError("feature_columns cannot be empty")
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise TrainingDataError(
                f"training frame is missing features: {missing}"
            )

        values = frame.loc[:, columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise TrainingDataError(
                "training features must contain only finite values"
            )
        means = values.mean(axis=0)
        scales = values.std(axis=0, ddof=0)
        scales = np.where(scales < epsilon, 1.0, scales)
        return cls(
            columns=columns,
            means=tuple(float(value) for value in means),
            scales=tuple(float(value) for value in scales),
            epsilon=epsilon,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with only declared feature columns standardized."""

        missing = [
            column for column in self.columns if column not in frame.columns
        ]
        if missing:
            raise TrainingDataError(
                f"frame is missing scaler features: {missing}"
            )
        result = frame.copy()
        values = result.loc[:, self.columns].to_numpy(dtype=np.float64)
        standardized = (
            values - np.asarray(self.means)
        ) / np.asarray(self.scales)
        if not np.isfinite(standardized).all():
            raise TrainingDataError(
                "standardized features contain non-finite values"
            )
        result.loc[:, self.columns] = standardized
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PREPROCESSING_VERSION,
            "columns": list(self.columns),
            "means": dict(zip(self.columns, self.means)),
            "scales": dict(zip(self.columns, self.scales)),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureStandardizer":
        columns = tuple(payload["columns"])
        means_by_name = payload["means"]
        scales_by_name = payload["scales"]
        return cls(
            columns=columns,
            means=tuple(float(means_by_name[name]) for name in columns),
            scales=tuple(float(scales_by_name[name]) for name in columns),
            epsilon=float(payload.get("epsilon", 1e-8)),
        )


def _validate_dataset(
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    required = {
        "target_date",
        "episode_id",
        "episode_complete",
        *feature_columns,
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise TrainingDataError(f"dataset is missing columns: {missing}")

    result = dataset.copy()
    result["target_date"] = pd.to_datetime(result["target_date"])
    if result["target_date"].isna().any():
        raise TrainingDataError("target_date cannot be missing")
    if not result["target_date"].is_monotonic_increasing:
        raise TrainingDataError("target_date must be increasing")
    if result["target_date"].duplicated().any():
        raise TrainingDataError("target_date cannot be duplicated")
    if not result["episode_complete"].astype(bool).all():
        raise TrainingDataError(
            "formal training dataset cannot include incomplete episodes"
        )
    return result


def prepare_training_splits(
    dataset: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    split_config: DatasetSplitConfig = DEFAULT_SPLIT_CONFIG,
) -> tuple[dict[str, pd.DataFrame], FeatureStandardizer]:
    """Split by outcome date and standardize using training rows only."""

    data = _validate_dataset(dataset, feature_columns)
    target_date = data["target_date"]
    train = data.loc[target_date <= pd.Timestamp(split_config.train_end)].copy()
    validation = data.loc[
        (target_date > pd.Timestamp(split_config.train_end))
        & (target_date <= pd.Timestamp(split_config.validation_end))
    ].copy()
    test = data.loc[
        (target_date > pd.Timestamp(split_config.validation_end))
        & (target_date <= pd.Timestamp(split_config.test_end))
    ].copy()
    after_test = data.loc[target_date > pd.Timestamp(split_config.test_end)]

    if any(frame.empty for frame in (train, validation, test)):
        raise TrainingDataError("train, validation, and test must be non-empty")
    if not after_test.empty:
        raise TrainingDataError(
            "dataset contains rows after the frozen test boundary"
        )

    episode_to_split: dict[str, set[str]] = {}
    for split_name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        for episode_id in frame["episode_id"].unique():
            episode_to_split.setdefault(episode_id, set()).add(split_name)
    crossing = {
        episode_id: sorted(names)
        for episode_id, names in episode_to_split.items()
        if len(names) != 1
    }
    if crossing:
        raise TrainingDataError(
            f"episodes cross split boundaries: {crossing}"
        )

    standardizer = FeatureStandardizer.fit(train, feature_columns)
    splits = {
        "train": standardizer.transform(train),
        "validation": standardizer.transform(validation),
        "test": standardizer.transform(test),
    }
    return splits, standardizer


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_preprocessing_artifact(
    *,
    output_path: str | Path,
    dataset_path: str | Path,
    splits: Mapping[str, pd.DataFrame],
    standardizer: FeatureStandardizer,
    split_config: DatasetSplitConfig = DEFAULT_SPLIT_CONFIG,
) -> Path:
    """Write split evidence and train-only scaler parameters."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    split_summary = {}
    for name, frame in splits.items():
        split_summary[name] = {
            "rows": len(frame),
            "episodes": int(frame["episode_id"].nunique()),
            "first_target_date": (
                frame["target_date"].min().date().isoformat()
            ),
            "last_target_date": (
                frame["target_date"].max().date().isoformat()
            ),
        }

    payload = {
        "version": PREPROCESSING_VERSION,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "split_field": "target_date",
        "split_config": {
            key: value.isoformat()
            for key, value in asdict(split_config).items()
        },
        "splits": split_summary,
        "standardizer": standardizer.to_dict(),
        "controls": {
            "standardizer_fit_split": "train",
            "validation_used_for_fit": False,
            "test_used_for_fit": False,
            "episode_crossing_allowed": False,
        },
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output
