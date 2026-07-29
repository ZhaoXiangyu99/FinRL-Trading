"""Point-in-time market features for the QQQ/SPY/cash agent.

Every feature at date ``t`` uses data no later than the regular-session close
on date ``t``.  The output is intentionally not normalized: scalers must be
fit on each training split only, never on the complete history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_SET_VERSION = "qqq_spy_market_v1"
ASSETS = ("qqq", "spy")


class FeatureDataError(ValueError):
    """Raised when a feature dataset cannot be built safely."""


@dataclass(frozen=True)
class MarketFeatureConfig:
    """Frozen window definitions for market feature set v1."""

    return_windows: tuple[int, ...] = (5, 20, 60, 120)
    volatility_windows: tuple[int, ...] = (20, 60)
    moving_average_windows: tuple[int, ...] = (50, 120, 200)
    relative_return_windows: tuple[int, ...] = (20, 60, 120)
    correlation_windows: tuple[int, ...] = (20, 60)
    drawdown_window: int = 252
    volume_window: int = 20
    annualization_factor: int = 252

    def __post_init__(self) -> None:
        window_groups = (
            self.return_windows,
            self.volatility_windows,
            self.moving_average_windows,
            self.relative_return_windows,
            self.correlation_windows,
        )
        if any(not windows for windows in window_groups):
            raise ValueError("feature window groups cannot be empty")
        if any(
            window < 2
            for windows in window_groups
            for window in windows
        ):
            raise ValueError("all rolling windows must be at least 2")
        if self.drawdown_window < 2:
            raise ValueError("drawdown_window must be at least 2")
        if self.volume_window < 2:
            raise ValueError("volume_window must be at least 2")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")

    @property
    def warmup_window(self) -> int:
        return max(
            *self.return_windows,
            *self.volatility_windows,
            *self.moving_average_windows,
            *self.relative_return_windows,
            *self.correlation_windows,
            self.drawdown_window,
            self.volume_window,
        )


def _required_columns() -> tuple[str, ...]:
    return tuple(
        column
        for asset in ASSETS
        for column in (f"{asset}_close", f"{asset}_volume")
    )


def validate_market_frame(frame: pd.DataFrame) -> None:
    """Validate the source columns used by feature set v1."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise FeatureDataError("market frame index must be a DatetimeIndex")
    if frame.empty:
        raise FeatureDataError("market frame cannot be empty")
    if not frame.index.is_monotonic_increasing:
        raise FeatureDataError("market frame dates must be increasing")
    if frame.index.has_duplicates:
        raise FeatureDataError("market frame dates cannot contain duplicates")

    missing = [
        column for column in _required_columns() if column not in frame.columns
    ]
    if missing:
        raise FeatureDataError(f"market frame is missing columns: {missing}")

    required = frame.loc[:, _required_columns()]
    if required.isna().any().any():
        raise FeatureDataError(
            "close and volume inputs cannot contain missing values"
        )
    if not np.isfinite(required.to_numpy(dtype=np.float64)).all():
        raise FeatureDataError(
            "close and volume inputs must contain only finite values"
        )
    for asset in ASSETS:
        if (frame[f"{asset}_close"] <= 0.0).any():
            raise FeatureDataError(f"{asset} close prices must be positive")
        if (frame[f"{asset}_volume"] < 0.0).any():
            raise FeatureDataError(f"{asset} volume cannot be negative")


def build_market_features(
    market_frame: pd.DataFrame,
    config: MarketFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build raw, backward-looking features and drop only warm-up rows."""

    config = config or MarketFeatureConfig()
    validate_market_frame(market_frame)
    source = market_frame.copy()

    features: dict[str, pd.Series] = {}
    daily_returns: dict[str, pd.Series] = {}

    for asset in ASSETS:
        close = source[f"{asset}_close"].astype(np.float64)
        volume = source[f"{asset}_volume"].astype(np.float64)
        daily_return = close.pct_change(fill_method=None)
        daily_returns[asset] = daily_return

        for window in config.return_windows:
            features[f"{asset}_total_return_{window}d"] = (
                close / close.shift(window) - 1.0
            )

        for window in config.volatility_windows:
            features[f"{asset}_realized_vol_{window}d"] = (
                daily_return.rolling(window, min_periods=window).std(ddof=0)
                * np.sqrt(config.annualization_factor)
            )

        for window in config.moving_average_windows:
            moving_average = close.rolling(
                window, min_periods=window
            ).mean()
            features[f"{asset}_price_to_ma_{window}d"] = (
                close / moving_average - 1.0
            )

        rolling_high = close.rolling(
            config.drawdown_window,
            min_periods=config.drawdown_window,
        ).max()
        features[f"{asset}_drawdown_{config.drawdown_window}d"] = (
            close / rolling_high - 1.0
        )

        rolling_volume = volume.rolling(
            config.volume_window,
            min_periods=config.volume_window,
        ).mean()
        features[f"{asset}_volume_to_mean_{config.volume_window}d"] = (
            volume / rolling_volume - 1.0
        )

    for window in config.relative_return_windows:
        qqq_return = (
            source["qqq_close"] / source["qqq_close"].shift(window) - 1.0
        )
        spy_return = (
            source["spy_close"] / source["spy_close"].shift(window) - 1.0
        )
        features[f"qqq_minus_spy_return_{window}d"] = (
            qqq_return - spy_return
        )

    for window in config.correlation_windows:
        features[f"qqq_spy_return_corr_{window}d"] = (
            daily_returns["qqq"]
            .rolling(window, min_periods=window)
            .corr(daily_returns["spy"])
        )

    for window in config.volatility_windows:
        qqq_volatility = (
            daily_returns["qqq"]
            .rolling(window, min_periods=window)
            .std(ddof=0)
            * np.sqrt(config.annualization_factor)
        )
        spy_volatility = (
            daily_returns["spy"]
            .rolling(window, min_periods=window)
            .std(ddof=0)
            * np.sqrt(config.annualization_factor)
        )
        features[f"qqq_minus_spy_vol_{window}d"] = (
            qqq_volatility - spy_volatility
        )

    feature_frame = pd.DataFrame(features, index=source.index)
    feature_frame = feature_frame.replace((np.inf, -np.inf), np.nan)
    feature_frame = feature_frame.dropna(axis=0, how="any")

    if feature_frame.empty:
        raise FeatureDataError(
            f"no complete rows remain after {config.warmup_window}-day warm-up"
        )
    if not np.isfinite(feature_frame.to_numpy(dtype=np.float64)).all():
        raise FeatureDataError("feature frame contains non-finite values")

    feature_frame.index.name = "date"
    feature_frame.attrs.update(
        {
            "feature_set_version": FEATURE_SET_VERSION,
            "feature_count": feature_frame.shape[1],
            "input_rows": len(source),
            "output_rows": len(feature_frame),
            "warmup_rows_dropped": (
                source.index.get_loc(feature_frame.index[0])
            ),
            "first_feature_date": feature_frame.index[0].date().isoformat(),
            "last_feature_date": feature_frame.index[-1].date().isoformat(),
            "normalization": "none",
        }
    )
    return feature_frame


def build_market_features_from_csv(
    input_path: str | Path,
    config: MarketFeatureConfig | None = None,
) -> pd.DataFrame:
    """Load a Longbridge market CSV and build feature set v1."""

    path = Path(input_path)
    frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return build_market_features(frame, config)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a local artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_feature_dataset(
    feature_frame: pd.DataFrame,
    *,
    output_path: str | Path,
    source_path: str | Path,
    config: MarketFeatureConfig | None = None,
) -> tuple[Path, Path]:
    """Write features and an auditable sidecar metadata file."""

    config = config or MarketFeatureConfig()
    output = Path(output_path)
    source = Path(source_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(output, index=True, date_format="%Y-%m-%d")

    source_metadata_path = source.with_suffix(
        source.suffix + ".metadata.json"
    )
    source_metadata: dict[str, Any] | None = None
    if source_metadata_path.exists():
        source_metadata = json.loads(
            source_metadata_path.read_text(encoding="utf-8")
        )

    metadata = {
        **feature_frame.attrs,
        "feature_set_version": FEATURE_SET_VERSION,
        "feature_columns": list(feature_frame.columns),
        "config": asdict(config),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_metadata": source_metadata,
        "feature_csv_sha256": sha256_file(output),
        "timestamp_contract": {
            "observation_date": (
                "US regular-session date; values use data through that close"
            ),
            "earliest_execution": (
                "a later tradable timestamp; same-close execution is forbidden"
            ),
        },
        "leakage_controls": {
            "rolling_direction": "backward_only",
            "future_returns_included": False,
            "normalization": (
                "none; fit any scaler on the training split only"
            ),
            "raw_price_levels_included": False,
        },
        "formula_notes": {
            "total_return": "close_t / close_t_minus_window - 1",
            "realized_volatility": (
                "population_std(daily_returns, window) * sqrt(252)"
            ),
            "moving_average_distance": "close_t / rolling_mean - 1",
            "drawdown": "close_t / rolling_252d_high - 1",
            "volume_ratio": "volume_t / rolling_20d_mean - 1",
            "relative_return": "QQQ total return - SPY total return",
            "correlation": "rolling correlation of daily total returns",
            "volatility_spread": "QQQ realized volatility - SPY realized volatility",
        },
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output, metadata_path
