"""Leakage and formula tests for QQQ/SPY feature set v1."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.qqq_spy_features import (
    FEATURE_SET_VERSION,
    MarketFeatureConfig,
    build_market_features,
)


def synthetic_market_frame(rows: int = 340) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=rows)
    qqq_close = 100.0 * np.power(1.01, np.arange(rows))
    spy_close = 100.0 * np.power(1.005, np.arange(rows))
    return pd.DataFrame(
        {
            "qqq_close": qqq_close,
            "qqq_volume": np.linspace(1_000_000, 2_000_000, rows),
            "spy_close": spy_close,
            "spy_volume": np.linspace(2_000_000, 3_000_000, rows),
        },
        index=dates,
    )


class FeatureFormulaTests(unittest.TestCase):
    def test_builds_expected_feature_contract(self) -> None:
        source = synthetic_market_frame()

        features = build_market_features(source)

        self.assertEqual(features.shape[1], 29)
        self.assertEqual(
            features.attrs["feature_set_version"], FEATURE_SET_VERSION
        )
        self.assertEqual(features.index[0], source.index[251])
        self.assertFalse(features.isna().any().any())
        self.assertTrue(
            np.isfinite(features.to_numpy(dtype=np.float64)).all()
        )

    def test_total_and_relative_returns_match_known_growth(self) -> None:
        features = build_market_features(synthetic_market_frame())
        row = features.iloc[0]

        expected_qqq_5d = 1.01**5 - 1.0
        expected_spy_20d = 1.005**20 - 1.0
        expected_relative_20d = 1.01**20 - 1.005**20

        self.assertAlmostEqual(
            row["qqq_total_return_5d"], expected_qqq_5d, places=12
        )
        self.assertAlmostEqual(
            row["spy_total_return_20d"], expected_spy_20d, places=12
        )
        self.assertAlmostEqual(
            row["qqq_minus_spy_return_20d"],
            expected_relative_20d,
            places=12,
        )

    def test_drawdown_is_never_positive(self) -> None:
        source = synthetic_market_frame()
        source.loc[source.index[280]:, "qqq_close"] *= 0.8

        features = build_market_features(source)

        self.assertLessEqual(features["qqq_drawdown_252d"].max(), 0.0)
        self.assertLess(features["qqq_drawdown_252d"].min(), -0.1)


class LeakageTests(unittest.TestCase):
    def test_future_changes_do_not_modify_past_features(self) -> None:
        original = synthetic_market_frame()
        changed_future = original.copy()
        change_start = original.index[310]
        changed_future.loc[change_start:, "qqq_close"] *= 1.7
        changed_future.loc[change_start:, "qqq_volume"] *= 4.0

        baseline = build_market_features(original)
        perturbed = build_market_features(changed_future)
        past_dates = baseline.index[baseline.index < change_start]

        pd.testing.assert_frame_equal(
            baseline.loc[past_dates],
            perturbed.loc[past_dates],
            check_exact=True,
        )

    def test_output_is_not_globally_standardized(self) -> None:
        features = build_market_features(synthetic_market_frame())

        self.assertEqual(features.attrs["normalization"], "none")
        self.assertNotAlmostEqual(
            float(features["qqq_total_return_5d"].std()), 1.0
        )


class ConfigurationTests(unittest.TestCase):
    def test_short_history_is_rejected(self) -> None:
        config = MarketFeatureConfig()
        with self.assertRaisesRegex(ValueError, "no complete rows"):
            build_market_features(
                synthetic_market_frame(config.warmup_window - 1),
                config,
            )


if __name__ == "__main__":
    unittest.main()
