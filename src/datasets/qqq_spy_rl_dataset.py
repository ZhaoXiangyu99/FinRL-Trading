"""Build a leakage-safe QQQ/SPY reinforcement-learning dataset.

Timing contract for each row
----------------------------
1. Observe market features after the regular-session close on ``feature_date``.
2. Existing holdings remain exposed from that close to the next session open.
3. Execute the new target weights at ``target_date`` regular-session open.
4. New holdings remain exposed from that open to the same session close.
5. Value the portfolio and settle the step at ``target_date`` close.

The environment therefore needs both overnight returns (old weights) and
intraday returns (new target weights).  This avoids same-close execution and
does not discard overnight risk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.qqq_spy_features import FEATURE_SET_VERSION


DATASET_VERSION = "qqq_spy_rl_daily_v1"
ASSETS = ("qqq", "spy")


class RLDatasetError(ValueError):
    """Raised when market, feature, or episode alignment is unsafe."""


@dataclass(frozen=True)
class RLDatasetConfig:
    """Frozen execution and episode rules for dataset v1."""

    include_incomplete_episodes: bool = False
    expected_feature_set_version: str = FEATURE_SET_VERSION
    cash_overnight_return: float = 0.0
    cash_intraday_return: float = 0.0

    def __post_init__(self) -> None:
        cash_returns = (
            self.cash_overnight_return,
            self.cash_intraday_return,
        )
        if not all(np.isfinite(value) for value in cash_returns):
            raise ValueError("cash returns must be finite")
        if any(value <= -1.0 for value in cash_returns):
            raise ValueError("cash returns must be greater than -100%")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_inputs(
    market: pd.DataFrame,
    features: pd.DataFrame,
    config: RLDatasetConfig,
) -> None:
    for label, frame in (("market", market), ("features", features)):
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise RLDatasetError(f"{label} index must be a DatetimeIndex")
        if frame.empty:
            raise RLDatasetError(f"{label} frame cannot be empty")
        if not frame.index.is_monotonic_increasing:
            raise RLDatasetError(f"{label} dates must be increasing")
        if frame.index.has_duplicates:
            raise RLDatasetError(f"{label} dates cannot contain duplicates")

    required_market_columns = [
        f"{asset}_{field}"
        for asset in ASSETS
        for field in ("open", "close")
    ]
    missing_market = [
        column
        for column in required_market_columns
        if column not in market.columns
    ]
    if missing_market:
        raise RLDatasetError(
            f"market frame is missing columns: {missing_market}"
        )

    market_prices = market.loc[:, required_market_columns]
    if market_prices.isna().any().any():
        raise RLDatasetError("market open/close prices cannot be missing")
    if not np.isfinite(market_prices.to_numpy(dtype=np.float64)).all():
        raise RLDatasetError("market open/close prices must be finite")
    if (market_prices <= 0.0).any().any():
        raise RLDatasetError("market open/close prices must be positive")

    missing_feature_dates = features.index.difference(market.index)
    if not missing_feature_dates.empty:
        raise RLDatasetError(
            "feature dates are absent from the market calendar: "
            f"{list(missing_feature_dates[:5].date)}"
        )
    if features.isna().any().any():
        raise RLDatasetError("features cannot contain missing values")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise RLDatasetError("features must contain only finite values")

    feature_version = features.attrs.get("feature_set_version")
    if (
        feature_version is not None
        and feature_version != config.expected_feature_set_version
    ):
        raise RLDatasetError(
            f"feature version {feature_version!r} does not match "
            f"{config.expected_feature_set_version!r}"
        )


def _half_year_id(value: pd.Timestamp) -> str:
    half = 1 if value.month <= 6 else 2
    return f"{value.year}-H{half}"


def _half_year_bounds(episode_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    year_text, half_text = episode_id.split("-H")
    year = int(year_text)
    if half_text == "1":
        return pd.Timestamp(year, 1, 1), pd.Timestamp(year, 6, 30)
    if half_text == "2":
        return pd.Timestamp(year, 7, 1), pd.Timestamp(year, 12, 31)
    raise RLDatasetError(f"invalid episode id: {episode_id}")


def _mark_episode_completeness(
    transitions: pd.DataFrame,
    market_index: pd.DatetimeIndex,
) -> pd.Series:
    completeness: dict[str, bool] = {}
    market_end = market_index.max()
    boundary_tolerance = pd.Timedelta(days=7)

    for episode_id, episode in transitions.groupby(
        "episode_id", sort=True
    ):
        half_start, half_end = _half_year_bounds(episode_id)
        expected_dates = market_index[
            (market_index >= half_start) & (market_index <= half_end)
        ]
        actual_dates = pd.DatetimeIndex(episode["target_date"])
        completeness[episode_id] = bool(
            not expected_dates.empty
            and actual_dates.equals(expected_dates)
            and actual_dates.min() <= half_start + boundary_tolerance
            and actual_dates.max() >= half_end - boundary_tolerance
            and market_end >= half_end - boundary_tolerance
        )

    return transitions["episode_id"].map(completeness).astype(bool)


def _add_episode_positions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["step_in_episode"] = result.groupby(
        "episode_id", sort=False
    ).cumcount()
    result["steps_in_episode"] = result.groupby(
        "episode_id", sort=False
    )["episode_id"].transform("size")
    result["is_episode_terminal"] = (
        result["step_in_episode"] == result["steps_in_episode"] - 1
    )
    result["eligible_for_terminal_reward"] = (
        result["episode_complete"] & result["is_episode_terminal"]
    )
    return result


def build_rl_dataset(
    market: pd.DataFrame,
    features: pd.DataFrame,
    config: RLDatasetConfig | None = None,
) -> pd.DataFrame:
    """Align point-in-time states with next-open execution transitions."""

    config = config or RLDatasetConfig()
    _validate_inputs(market, features, config)

    market_positions = pd.Series(
        np.arange(len(market), dtype=np.int64),
        index=market.index,
    )
    feature_positions = market_positions.reindex(features.index).to_numpy()
    has_next_session = feature_positions < len(market) - 1

    usable_features = features.iloc[np.flatnonzero(has_next_session)].copy()
    current_positions = feature_positions[has_next_session]
    next_positions = current_positions + 1

    feature_dates = market.index[current_positions]
    target_dates = market.index[next_positions]
    transitions = pd.DataFrame(
        {
            "feature_date": feature_dates,
            "target_date": target_dates,
        }
    )
    transitions["episode_id"] = [
        _half_year_id(value) for value in target_dates
    ]

    for asset in ASSETS:
        close_t = market[f"{asset}_close"].to_numpy(dtype=np.float64)[
            current_positions
        ]
        open_next = market[f"{asset}_open"].to_numpy(dtype=np.float64)[
            next_positions
        ]
        close_next = market[f"{asset}_close"].to_numpy(dtype=np.float64)[
            next_positions
        ]
        overnight_return = open_next / close_t - 1.0
        intraday_return = close_next / open_next - 1.0
        close_to_close_return = close_next / close_t - 1.0

        transitions[f"{asset}_overnight_return"] = overnight_return
        transitions[f"{asset}_intraday_return"] = intraday_return
        transitions[f"{asset}_close_to_close_return"] = (
            close_to_close_return
        )

        recomposed = (
            (1.0 + overnight_return) * (1.0 + intraday_return) - 1.0
        )
        if not np.allclose(
            recomposed, close_to_close_return, rtol=1e-12, atol=1e-12
        ):
            raise RLDatasetError(
                f"{asset} overnight/intraday returns do not recompose"
            )

    transitions["cash_overnight_return"] = config.cash_overnight_return
    transitions["cash_intraday_return"] = config.cash_intraday_return

    feature_values = usable_features.reset_index(drop=True)
    dataset = pd.concat((transitions, feature_values), axis=1)
    dataset["episode_complete"] = _mark_episode_completeness(
        dataset, market.index
    )

    if not config.include_incomplete_episodes:
        dataset = dataset.loc[dataset["episode_complete"]].copy()
    if dataset.empty:
        raise RLDatasetError("no eligible episode rows remain")

    dataset = _add_episode_positions(dataset)
    dataset = dataset.reset_index(drop=True)

    metadata_columns = [
        "feature_date",
        "target_date",
        "episode_id",
        "episode_complete",
        "step_in_episode",
        "steps_in_episode",
        "is_episode_terminal",
        "eligible_for_terminal_reward",
    ]
    transition_columns = [
        f"{asset}_{session}_return"
        for asset in ASSETS
        for session in ("overnight", "intraday", "close_to_close")
    ] + ["cash_overnight_return", "cash_intraday_return"]
    feature_columns = list(features.columns)
    dataset = dataset.loc[
        :, metadata_columns + feature_columns + transition_columns
    ]

    episode_summary = (
        dataset.groupby("episode_id", sort=True)
        .agg(
            first_target_date=("target_date", "min"),
            last_target_date=("target_date", "max"),
            steps=("episode_id", "size"),
            complete=("episode_complete", "all"),
        )
        .reset_index()
    )
    dataset.attrs.update(
        {
            "dataset_version": DATASET_VERSION,
            "feature_set_version": config.expected_feature_set_version,
            "feature_columns": feature_columns,
            "row_count": len(dataset),
            "feature_count": len(feature_columns),
            "episode_count": dataset["episode_id"].nunique(),
            "complete_episode_count": int(
                episode_summary["complete"].sum()
            ),
            "first_feature_date": (
                dataset["feature_date"].min().date().isoformat()
            ),
            "first_target_date": (
                dataset["target_date"].min().date().isoformat()
            ),
            "last_target_date": (
                dataset["target_date"].max().date().isoformat()
            ),
            "include_incomplete_episodes": (
                config.include_incomplete_episodes
            ),
            "episode_summary": [
                {
                    "episode_id": row.episode_id,
                    "first_target_date": (
                        row.first_target_date.date().isoformat()
                    ),
                    "last_target_date": (
                        row.last_target_date.date().isoformat()
                    ),
                    "steps": int(row.steps),
                    "complete": bool(row.complete),
                }
                for row in episode_summary.itertuples(index=False)
            ],
        }
    )
    return dataset


def build_rl_dataset_from_csv(
    *,
    market_path: str | Path,
    feature_path: str | Path,
    config: RLDatasetConfig | None = None,
) -> pd.DataFrame:
    """Load market/features CSV files and build the aligned RL dataset."""

    market = pd.read_csv(
        market_path, parse_dates=["date"], index_col="date"
    )
    features = pd.read_csv(
        feature_path, parse_dates=["date"], index_col="date"
    )

    metadata_path = Path(feature_path).with_suffix(
        Path(feature_path).suffix + ".metadata.json"
    )
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        features.attrs["feature_set_version"] = metadata.get(
            "feature_set_version"
        )

    return build_rl_dataset(market, features, config)


def write_rl_dataset(
    dataset: pd.DataFrame,
    *,
    output_path: str | Path,
    market_path: str | Path,
    feature_path: str | Path,
    config: RLDatasetConfig | None = None,
) -> tuple[Path, Path]:
    """Write an RL dataset and its reproducibility metadata."""

    config = config or RLDatasetConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False, date_format="%Y-%m-%d")

    metadata: dict[str, Any] = {
        **dataset.attrs,
        "dataset_version": DATASET_VERSION,
        "config": asdict(config),
        "market_path": str(market_path),
        "market_sha256": _sha256_file(market_path),
        "feature_path": str(feature_path),
        "feature_sha256": _sha256_file(feature_path),
        "dataset_sha256": _sha256_file(output),
        "timing_contract": {
            "state_observation": "after feature_date regular-session close",
            "old_weights_exposure": (
                "feature_date close to target_date open"
            ),
            "action_execution": "target_date regular-session open",
            "new_weights_exposure": (
                "target_date open to target_date close"
            ),
            "step_settlement": "target_date regular-session close",
            "same_close_execution_allowed": False,
        },
        "split_contract": {
            "split_date_field": "target_date",
            "reason": (
                "the outcome and execution belong to target_date; splitting "
                "on feature_date can cross a boundary"
            ),
        },
        "benchmark_contract": {
            "start": "first target_date open in each complete half-year",
            "end": "last target_date close in the same half-year",
            "symbols": ["QQQ", "SPY"],
        },
        "cash_return_contract": {
            "overnight": config.cash_overnight_return,
            "intraday": config.cash_intraday_return,
            "status": "provisional; zero-yield cash in dataset v1",
        },
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output, metadata_path
