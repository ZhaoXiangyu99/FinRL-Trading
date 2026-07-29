#!/usr/bin/env python3
"""Fetch validated QQQ/SPY daily bars from Longbridge."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.longbridge_market_data import (
    LongbridgeCLIClient,
    LongbridgeFetchConfig,
    fetch_qqq_spy_daily,
    write_market_dataset,
)


DEFAULT_OUTPUT = Path(
    "data/market/longbridge/qqq_spy_daily_forward_adjusted.csv"
)
DEFAULT_CACHE_DIR = Path(".cache/longbridge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch forward-adjusted, regular-session QQQ/SPY daily bars "
            "using the authenticated Longbridge CLI."
        )
    )
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=(
            "Cache successful date chunks here so interrupted downloads can "
            "resume without refetching them."
        ),
    )
    parser.add_argument(
        "--no-adjust",
        action="store_true",
        help="Fetch raw prices instead of forward-adjusted prices.",
    )
    parser.add_argument(
        "--allow-ohlc-anomalies",
        action="store_true",
        help=(
            "Keep rows with provider OHLC inconsistencies, mark them invalid, "
            "and replace the bad high/low field with an empty value."
        ),
    )
    parser.add_argument(
        "--allow-leading-calendar-trim",
        action="store_true",
        help=(
            "If one symbol has only leading historical gaps, trim both "
            "series to the first date after the final mismatch and record "
            "that effective start in metadata."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LongbridgeFetchConfig(
        start=args.start,
        end=args.end,
        forward_adjust=not args.no_adjust,
        trade_session="intraday",
        allow_ohlc_anomalies=args.allow_ohlc_anomalies,
        allow_leading_calendar_trim=args.allow_leading_calendar_trim,
    )
    def report_progress(
        symbol: str,
        chunk_start: date,
        chunk_end: date,
        source: str,
    ) -> None:
        print(
            f"[{symbol}] {chunk_start.isoformat()}.."
            f"{chunk_end.isoformat()} ({source})",
            flush=True,
        )

    client = LongbridgeCLIClient(
        cache_dir=args.cache_dir,
        progress_callback=report_progress,
    )
    frame = fetch_qqq_spy_daily(client, config)
    csv_path, metadata_path = write_market_dataset(frame, args.output)
    print(
        f"Wrote {len(frame)} aligned sessions to {csv_path} "
        f"and {metadata_path}"
    )


if __name__ == "__main__":
    main()
