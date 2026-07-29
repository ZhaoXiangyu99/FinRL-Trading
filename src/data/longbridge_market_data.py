"""Longbridge daily-market-data adapter for the QQQ/SPY agent.

The adapter uses the official ``longbridge`` CLI so the same code can run on a
developer machine or cloud server without depending on a Codex conversation.
Authentication is managed by ``longbridge auth login`` and is never read or
stored by this module.

Only ingestion and validation live here.  Execution-time alignment (for
example, observe after close and execute at the next open) belongs to a
separate environment-dataset contract and must not be inferred silently.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


LONG_BRIDGE_SYMBOLS = {
    "QQQ": "QQQ.US",
    "SPY": "SPY.US",
}
REQUIRED_KLINE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class LongbridgeDataError(RuntimeError):
    """Raised when Longbridge data cannot be fetched or safely normalized."""


@dataclass(frozen=True)
class LongbridgeFetchConfig:
    """Configuration for a reproducible daily-bar request."""

    start: date
    end: date
    forward_adjust: bool = True
    trade_session: str = "intraday"
    allow_ohlc_anomalies: bool = False
    allow_leading_calendar_trim: bool = False

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("start must be on or before end")
        if self.trade_session not in {"intraday", "all"}:
            raise ValueError("trade_session must be 'intraday' or 'all'")


class LongbridgeCLIClient:
    """Small read-only client around ``longbridge kline history``."""

    def __init__(
        self,
        executable: str = "longbridge",
        *,
        timeout_seconds: int = 120,
        max_calendar_days_per_request: int = 700,
        cache_dir: str | Path | None = None,
        progress_callback: (
            Callable[[str, date, date, str], None] | None
        ) = None,
    ) -> None:
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        if max_calendar_days_per_request < 1:
            raise ValueError("max_calendar_days_per_request must be positive")
        self.max_calendar_days_per_request = max_calendar_days_per_request
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.progress_callback = progress_callback

    def ensure_available(self) -> str:
        """Return the resolved CLI path or raise an actionable error."""

        resolved = shutil.which(self.executable)
        if resolved is None:
            raise LongbridgeDataError(
                "Longbridge CLI was not found. Install it and run "
                "`longbridge auth login` before fetching data."
            )
        return resolved

    def fetch_daily_candlesticks(
        self,
        symbol: str,
        config: LongbridgeFetchConfig,
    ) -> pd.DataFrame:
        """Fetch daily bars in chunks so the API's 1000-bar cap cannot truncate."""

        chunks = []
        chunk_start = config.start
        while chunk_start <= config.end:
            chunk_end = min(
                config.end,
                chunk_start
                + timedelta(days=self.max_calendar_days_per_request - 1),
            )
            chunks.append(
                self._fetch_daily_chunk(
                    symbol,
                    config,
                    start=chunk_start,
                    end=chunk_end,
                )
            )
            chunk_start = chunk_end + timedelta(days=1)

        frame = pd.concat(chunks).sort_index()
        if frame.index.has_duplicates:
            duplicates = frame.index[frame.index.duplicated()].unique()
            raise LongbridgeDataError(
                f"{symbol} contains duplicate dates across request chunks: "
                f"{[value.date().isoformat() for value in duplicates]}"
            )
        return frame

    def _fetch_daily_chunk(
        self,
        symbol: str,
        config: LongbridgeFetchConfig,
        *,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch one bounded date range from Longbridge."""

        cache_path = self._cache_path(symbol, config, start, end)
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LongbridgeDataError(
                    f"Invalid Longbridge cache file: {cache_path}"
                ) from exc
            source = "cache"
        else:
            payload = self._run_daily_chunk_request(
                symbol, config, start=start, end=end
            )
            source = "network"
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = cache_path.with_suffix(
                    cache_path.suffix + ".tmp"
                )
                temporary_path.write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
                temporary_path.replace(cache_path)

        if self.progress_callback is not None:
            self.progress_callback(symbol, start, end, source)

        records = extract_kline_records(payload)
        return normalize_kline_records(
            records,
            symbol=symbol,
            allow_ohlc_anomalies=config.allow_ohlc_anomalies,
        )

    def _run_daily_chunk_request(
        self,
        symbol: str,
        config: LongbridgeFetchConfig,
        *,
        start: date,
        end: date,
    ) -> Any:
        executable = self.ensure_available()
        command = [
            executable,
            "kline",
            "history",
            symbol,
            "--period",
            "day",
            "--start",
            start.isoformat(),
            "--end",
            end.isoformat(),
            "--adjust",
            "forward" if config.forward_adjust else "none",
            "--session",
            config.trade_session,
            "--format",
            "json",
        ]

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise LongbridgeDataError(
                f"Longbridge request timed out for {symbol}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise LongbridgeDataError(
                f"Longbridge request failed for {symbol}: {detail}"
            ) from exc

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise LongbridgeDataError(
                f"Longbridge returned invalid JSON for {symbol}"
            ) from exc
        return payload

    def _cache_path(
        self,
        symbol: str,
        config: LongbridgeFetchConfig,
        start: date,
        end: date,
    ) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_symbol = symbol.replace(".", "_")
        adjustment = "forward" if config.forward_adjust else "raw"
        filename = (
            f"{safe_symbol}_day_{start.isoformat()}_{end.isoformat()}_"
            f"{adjustment}_{config.trade_session}.json"
        )
        return self.cache_dir / filename


def extract_kline_records(payload: Any) -> Sequence[Mapping[str, Any]]:
    """Extract records from supported Longbridge CLI response envelopes."""

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("data", "items", "candlesticks"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            raise LongbridgeDataError(
                "Longbridge JSON did not contain a candlestick list"
            )
    else:
        raise LongbridgeDataError(
            "Longbridge JSON must be a list or an object"
        )

    if not records:
        raise LongbridgeDataError("Longbridge returned no candlesticks")
    if not all(isinstance(record, dict) for record in records):
        raise LongbridgeDataError("Candlestick entries must be objects")
    return records


def normalize_kline_records(
    records: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    allow_ohlc_anomalies: bool = False,
) -> pd.DataFrame:
    """Normalize raw Longbridge records and enforce OHLCV invariants."""

    raw = pd.DataFrame.from_records(records)
    # The MCP connector currently uses ``timestamp`` while CLI 0.23.x uses
    # ``time``.  Normalize both explicitly and reject ambiguous payloads.
    time_columns = [
        column for column in ("timestamp", "time") if column in raw.columns
    ]
    if len(time_columns) != 1:
        raise LongbridgeDataError(
            f"{symbol} must contain exactly one of 'timestamp' or 'time'"
        )
    raw = raw.rename(columns={time_columns[0]: "timestamp"})

    missing_columns = [
        column for column in REQUIRED_KLINE_COLUMNS if column not in raw.columns
    ]
    if missing_columns:
        raise LongbridgeDataError(
            f"{symbol} is missing required columns: {missing_columns}"
        )

    frame = raw.loc[:, ("timestamp", *REQUIRED_KLINE_COLUMNS)].copy()
    timestamp = pd.to_datetime(frame.pop("timestamp"), utc=True, errors="coerce")
    if timestamp.isna().any():
        raise LongbridgeDataError(f"{symbol} contains invalid timestamps")

    # Daily timestamps represent the exchange-local session date.  Convert to
    # New York before dropping timezone information so EST/EDT offsets do not
    # move a bar to the wrong calendar date.
    frame.index = (
        timestamp.dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    frame.index.name = "date"
    # Stable ordering lets us keep the provider's last revision when a session
    # is repeated with identical OHLC but revised volume.
    frame = frame.sort_index(kind="stable")

    numeric_columns = ("open", "high", "low", "close", "volume")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.loc[:, numeric_columns].isna().any().any():
        raise LongbridgeDataError(f"{symbol} contains non-numeric OHLCV values")
    duplicate_dates = frame.index[frame.index.duplicated(keep=False)].unique()
    for duplicate_date in duplicate_dates:
        duplicate_rows = frame.loc[[duplicate_date]]
        if (
            duplicate_rows.loc[:, ("open", "high", "low", "close")]
            .drop_duplicates()
            .shape[0]
            != 1
        ):
            raise LongbridgeDataError(
                f"{symbol} contains conflicting OHLC revisions for "
                f"{duplicate_date.date().isoformat()}"
            )
    if len(duplicate_dates):
        frame = frame.loc[~frame.index.duplicated(keep="last")].copy()

    if (frame.loc[:, ("open", "high", "low", "close")] <= 0.0).any().any():
        raise LongbridgeDataError(f"{symbol} contains non-positive prices")
    if (frame["volume"] < 0).any():
        raise LongbridgeDataError(f"{symbol} contains negative volume")
    invalid_high = frame["high"] < frame[["open", "close", "low"]].max(axis=1)
    if invalid_high.any() and not allow_ohlc_anomalies:
        bad_date = invalid_high[invalid_high].index[0]
        values = frame.loc[
            bad_date, ["open", "high", "low", "close"]
        ].to_dict()
        raise LongbridgeDataError(
            f"{symbol} contains invalid high prices on "
            f"{bad_date.date().isoformat()}: {values}"
        )
    invalid_low = frame["low"] > frame[["open", "close", "high"]].min(axis=1)
    if invalid_low.any() and not allow_ohlc_anomalies:
        bad_date = invalid_low[invalid_low].index[0]
        values = frame.loc[
            bad_date, ["open", "high", "low", "close"]
        ].to_dict()
        raise LongbridgeDataError(
            f"{symbol} contains invalid low prices on "
            f"{bad_date.date().isoformat()}: {values}"
        )

    frame["ohlc_valid"] = ~(invalid_high | invalid_low)
    if allow_ohlc_anomalies:
        # Preserve trustworthy close/volume fields while making it impossible
        # for a downstream high/low feature to consume a known-bad value.
        frame.loc[invalid_high, "high"] = float("nan")
        frame.loc[invalid_low, "low"] = float("nan")

    frame["symbol"] = symbol
    frame.attrs["duplicate_dates_resolved"] = [
        value.date().isoformat() for value in duplicate_dates
    ]
    frame.attrs["ohlc_anomaly_dates"] = [
        value.date().isoformat()
        for value in frame.index[~frame["ohlc_valid"]]
    ]
    return frame.loc[
        :,
        ("symbol", "open", "high", "low", "close", "volume", "ohlc_valid"),
    ]


def fetch_qqq_spy_daily(
    client: LongbridgeCLIClient,
    config: LongbridgeFetchConfig,
) -> pd.DataFrame:
    """Fetch QQQ/SPY daily bars and require an identical trading calendar."""

    # Two workers reduce wall time while keeping request concurrency modest.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            short_symbol: executor.submit(
                client.fetch_daily_candlesticks,
                longbridge_symbol,
                config,
            )
            for short_symbol, longbridge_symbol in LONG_BRIDGE_SYMBOLS.items()
        }
        frames = {
            short_symbol: future.result()
            for short_symbol, future in futures.items()
        }

    qqq_dates = frames["QQQ"].index
    spy_dates = frames["SPY"].index
    calendar_trim: dict[str, Any] | None = None
    if not qqq_dates.equals(spy_dates):
        missing_qqq = spy_dates.difference(qqq_dates)
        missing_spy = qqq_dates.difference(spy_dates)
        mismatches = missing_qqq.union(missing_spy)
        if not config.allow_leading_calendar_trim:
            raise LongbridgeDataError(
                "QQQ/SPY calendars differ; refusing a silent inner join. "
                f"missing QQQ count={len(missing_qqq)}, "
                f"missing SPY count={len(missing_spy)}, "
                f"last mismatch={mismatches.max().date()}"
            )

        common_dates = qqq_dates.intersection(spy_dates)
        usable_dates = common_dates[common_dates > mismatches.max()]
        if usable_dates.empty:
            raise LongbridgeDataError(
                "QQQ/SPY calendars never become consistently aligned"
            )

        effective_start = usable_dates.min()
        original_counts = {
            symbol: len(frame) for symbol, frame in frames.items()
        }
        frames = {
            symbol: frame.loc[frame.index >= effective_start].copy()
            for symbol, frame in frames.items()
        }
        if not frames["QQQ"].index.equals(frames["SPY"].index):
            raise LongbridgeDataError(
                "QQQ/SPY calendars still differ after leading-date trim"
            )
        calendar_trim = {
            "requested_start": config.start.isoformat(),
            "effective_start": effective_start.date().isoformat(),
            "last_calendar_mismatch": mismatches.max().date().isoformat(),
            "missing_qqq_count": len(missing_qqq),
            "missing_spy_count": len(missing_spy),
            "dropped_leading_rows": {
                symbol: original_counts[symbol] - len(frame)
                for symbol, frame in frames.items()
            },
        }

    wide_parts = []
    for symbol, frame in frames.items():
        values = frame.drop(columns="symbol").add_prefix(
            f"{symbol.lower()}_"
        )
        wide_parts.append(values)

    combined = pd.concat(wide_parts, axis=1)
    combined.attrs.update(
        {
            "source": "Longbridge",
            "symbols": LONG_BRIDGE_SYMBOLS.copy(),
            "period": "day",
            "forward_adjust": config.forward_adjust,
            "trade_session": config.trade_session,
            "allow_ohlc_anomalies": config.allow_ohlc_anomalies,
            "allow_leading_calendar_trim": (
                config.allow_leading_calendar_trim
            ),
            "calendar_trim": calendar_trim,
            "ohlc_anomaly_counts": {
                symbol: int((~frame["ohlc_valid"]).sum())
                for symbol, frame in frames.items()
            },
            "start": config.start.isoformat(),
            "end": config.end.isoformat(),
        }
    )
    return combined


def write_market_dataset(
    frame: pd.DataFrame,
    output_path: str | Path,
) -> tuple[Path, Path]:
    """Write CSV plus a sidecar JSON containing the request metadata."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True, date_format="%Y-%m-%d")

    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(frame.attrs, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path, metadata_path
