"""Tests for the read-only Longbridge daily-bar adapter."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.data.longbridge_market_data import (
    LongbridgeCLIClient,
    LongbridgeDataError,
    LongbridgeFetchConfig,
    extract_kline_records,
    fetch_qqq_spy_daily,
    normalize_kline_records,
)


SAMPLE_RECORDS = [
    {
        "close": "725.170",
        "open": "729.190",
        "low": "724.600",
        "high": "731.920",
        "volume": 40_803_791,
        "turnover": "29734183453.000",
        "timestamp": "2026-07-01T04:00:00Z",
        "trade_session": "Intraday",
    },
    {
        "close": "712.600",
        "open": "725.580",
        "low": "707.560",
        "high": "730.830",
        "volume": 51_086_080,
        "turnover": "36625905934.000",
        "timestamp": "2026-07-02T04:00:00Z",
        "trade_session": "Intraday",
    },
]


class NormalizationTests(unittest.TestCase):
    def test_normalizes_connector_shaped_records(self) -> None:
        frame = normalize_kline_records(SAMPLE_RECORDS, symbol="QQQ.US")

        self.assertEqual(list(frame.index), list(pd.to_datetime(
            ("2026-07-01", "2026-07-02")
        )))
        self.assertEqual(frame.index.name, "date")
        self.assertEqual(frame.iloc[0]["symbol"], "QQQ.US")
        self.assertAlmostEqual(frame.iloc[0]["close"], 725.17)
        self.assertEqual(frame.iloc[0]["volume"], 40_803_791)

    def test_rejects_invalid_ohlc(self) -> None:
        invalid = [dict(SAMPLE_RECORDS[0], high="700.00")]
        with self.assertRaisesRegex(LongbridgeDataError, "invalid high"):
            normalize_kline_records(invalid, symbol="QQQ.US")

    def test_can_quarantine_invalid_high_without_changing_close(self) -> None:
        invalid = [dict(SAMPLE_RECORDS[0], high="700.00")]

        frame = normalize_kline_records(
            invalid,
            symbol="QQQ.US",
            allow_ohlc_anomalies=True,
        )

        self.assertTrue(pd.isna(frame.iloc[0]["high"]))
        self.assertAlmostEqual(frame.iloc[0]["close"], 725.17)
        self.assertFalse(bool(frame.iloc[0]["ohlc_valid"]))
        self.assertEqual(
            frame.attrs["ohlc_anomaly_dates"], ["2026-07-01"]
        )

    def test_extracts_supported_response_envelopes(self) -> None:
        self.assertEqual(extract_kline_records(SAMPLE_RECORDS), SAMPLE_RECORDS)
        self.assertEqual(
            extract_kline_records({"data": SAMPLE_RECORDS}), SAMPLE_RECORDS
        )
        with self.assertRaisesRegex(LongbridgeDataError, "no candlesticks"):
            extract_kline_records([])

    def test_accepts_cli_time_field(self) -> None:
        cli_records = [
            {
                key: value
                for key, value in SAMPLE_RECORDS[0].items()
                if key != "timestamp"
            }
        ]
        cli_records[0]["time"] = SAMPLE_RECORDS[0]["timestamp"]

        frame = normalize_kline_records(cli_records, symbol="QQQ.US")

        self.assertEqual(frame.index[0], pd.Timestamp("2026-07-01"))

    def test_keeps_last_volume_revision_when_duplicate_ohlc_matches(
        self,
    ) -> None:
        revised = dict(SAMPLE_RECORDS[0], volume=40_900_000)

        frame = normalize_kline_records(
            [SAMPLE_RECORDS[0], revised], symbol="QQQ.US"
        )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["volume"], 40_900_000)
        self.assertEqual(
            frame.attrs["duplicate_dates_resolved"], ["2026-07-01"]
        )

    def test_rejects_conflicting_duplicate_ohlc(self) -> None:
        conflicting = dict(SAMPLE_RECORDS[0], close="700.00")
        with self.assertRaisesRegex(
            LongbridgeDataError, "conflicting OHLC revisions"
        ):
            normalize_kline_records(
                [SAMPLE_RECORDS[0], conflicting], symbol="QQQ.US"
            )


class CLIClientTests(unittest.TestCase):
    @patch("src.data.longbridge_market_data.subprocess.run")
    @patch(
        "src.data.longbridge_market_data.shutil.which",
        return_value="/usr/local/bin/longbridge",
    )
    def test_uses_forward_adjusted_regular_session_daily_request(
        self,
        _which,
        run,
    ) -> None:
        run.return_value.stdout = json.dumps(SAMPLE_RECORDS)
        config = LongbridgeFetchConfig(
            start=date(2026, 7, 1),
            end=date(2026, 7, 2),
        )

        frame = LongbridgeCLIClient().fetch_daily_candlesticks(
            "QQQ.US", config
        )

        self.assertEqual(len(frame), 2)
        command = run.call_args.args[0]
        self.assertIn("--adjust", command)
        self.assertEqual(command[command.index("--adjust") + 1], "forward")
        self.assertEqual(
            command[command.index("--session") + 1], "intraday"
        )
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    @patch("src.data.longbridge_market_data.subprocess.run")
    @patch(
        "src.data.longbridge_market_data.shutil.which",
        return_value="/usr/local/bin/longbridge",
    )
    def test_long_ranges_are_chunked_to_avoid_silent_truncation(
        self,
        _which,
        run,
    ) -> None:
        second_chunk = [
            dict(
                SAMPLE_RECORDS[0],
                timestamp="2026-07-03T04:00:00Z",
            )
        ]
        run.side_effect = [
            type("Result", (), {"stdout": json.dumps(SAMPLE_RECORDS)})(),
            type("Result", (), {"stdout": json.dumps(second_chunk)})(),
        ]
        config = LongbridgeFetchConfig(
            start=date(2026, 7, 1),
            end=date(2026, 7, 3),
        )

        frame = LongbridgeCLIClient(
            max_calendar_days_per_request=2
        ).fetch_daily_candlesticks("QQQ.US", config)

        self.assertEqual(len(frame), 3)
        self.assertEqual(run.call_count, 2)
        first_command = run.call_args_list[0].args[0]
        second_command = run.call_args_list[1].args[0]
        self.assertEqual(
            first_command[first_command.index("--end") + 1], "2026-07-02"
        )
        self.assertEqual(
            second_command[second_command.index("--start") + 1],
            "2026-07-03",
        )

    @patch("src.data.longbridge_market_data.subprocess.run")
    @patch(
        "src.data.longbridge_market_data.shutil.which",
        return_value="/usr/local/bin/longbridge",
    )
    def test_completed_chunks_are_reused_from_cache(
        self,
        _which,
        run,
    ) -> None:
        run.return_value.stdout = json.dumps(SAMPLE_RECORDS)
        config = LongbridgeFetchConfig(
            start=date(2026, 7, 1),
            end=date(2026, 7, 2),
        )

        with TemporaryDirectory() as directory:
            client = LongbridgeCLIClient(cache_dir=Path(directory))
            first = client.fetch_daily_candlesticks("QQQ.US", config)
            second = client.fetch_daily_candlesticks("QQQ.US", config)

        self.assertEqual(run.call_count, 1)
        pd.testing.assert_frame_equal(first, second)


class CalendarAlignmentTests(unittest.TestCase):
    def test_requires_identical_qqq_spy_dates(self) -> None:
        config = LongbridgeFetchConfig(
            start=date(2026, 7, 1),
            end=date(2026, 7, 2),
        )
        qqq = normalize_kline_records(SAMPLE_RECORDS, symbol="QQQ.US")
        spy = normalize_kline_records(SAMPLE_RECORDS[:1], symbol="SPY.US")

        class FakeClient:
            def fetch_daily_candlesticks(self, symbol, _config):
                return qqq if symbol == "QQQ.US" else spy

        with self.assertRaisesRegex(LongbridgeDataError, "calendars differ"):
            fetch_qqq_spy_daily(FakeClient(), config)  # type: ignore[arg-type]

    def test_can_trim_a_leading_calendar_gap_with_metadata(self) -> None:
        qqq = normalize_kline_records(SAMPLE_RECORDS, symbol="QQQ.US")
        spy = normalize_kline_records(SAMPLE_RECORDS[1:], symbol="SPY.US")
        config = LongbridgeFetchConfig(
            start=date(2026, 7, 1),
            end=date(2026, 7, 2),
            allow_leading_calendar_trim=True,
        )

        class FakeClient:
            def fetch_daily_candlesticks(self, symbol, _config):
                return qqq if symbol == "QQQ.US" else spy

        frame = fetch_qqq_spy_daily(
            FakeClient(), config  # type: ignore[arg-type]
        )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.index[0], pd.Timestamp("2026-07-02"))
        self.assertEqual(
            frame.attrs["calendar_trim"]["effective_start"], "2026-07-02"
        )


if __name__ == "__main__":
    unittest.main()
