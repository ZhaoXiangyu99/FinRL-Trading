"""Timing, episode, and leakage tests for the RL dataset."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.datasets.qqq_spy_rl_dataset import (
    RLDatasetConfig,
    build_rl_dataset,
)


def yearly_market_and_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2021-12-31", "2022-12-30")
    position = np.arange(len(dates), dtype=np.float64)
    market = pd.DataFrame(
        {
            "qqq_open": 100.0 + position,
            "qqq_close": 100.5 + position,
            "spy_open": 200.0 + 0.5 * position,
            "spy_close": 200.25 + 0.5 * position,
        },
        index=dates,
    )
    features = pd.DataFrame(
        {
            "feature_a": position / 100.0,
            "feature_b": np.sin(position / 10.0),
        },
        index=dates,
    )
    features.attrs["feature_set_version"] = "qqq_spy_market_v1"
    return market, features


class TimingContractTests(unittest.TestCase):
    def test_feature_date_executes_at_next_session_open(self) -> None:
        market, features = yearly_market_and_features()

        dataset = build_rl_dataset(market, features)
        first = dataset.iloc[0]

        self.assertEqual(first["feature_date"], pd.Timestamp("2021-12-31"))
        self.assertEqual(first["target_date"], pd.Timestamp("2022-01-03"))
        expected_overnight = (
            market.loc["2022-01-03", "qqq_open"]
            / market.loc["2021-12-31", "qqq_close"]
            - 1.0
        )
        expected_intraday = (
            market.loc["2022-01-03", "qqq_close"]
            / market.loc["2022-01-03", "qqq_open"]
            - 1.0
        )
        self.assertAlmostEqual(
            first["qqq_overnight_return"], expected_overnight
        )
        self.assertAlmostEqual(
            first["qqq_intraday_return"], expected_intraday
        )

    def test_overnight_and_intraday_recompose_close_to_close(self) -> None:
        market, features = yearly_market_and_features()
        dataset = build_rl_dataset(market, features)

        recomposed = (
            (1.0 + dataset["spy_overnight_return"])
            * (1.0 + dataset["spy_intraday_return"])
            - 1.0
        )
        np.testing.assert_allclose(
            recomposed,
            dataset["spy_close_to_close_return"],
            rtol=1e-12,
            atol=1e-12,
        )


class EpisodeTests(unittest.TestCase):
    def test_complete_half_years_have_one_terminal_row_each(self) -> None:
        market, features = yearly_market_and_features()

        dataset = build_rl_dataset(market, features)

        self.assertEqual(set(dataset["episode_id"]), {"2022-H1", "2022-H2"})
        self.assertTrue(dataset["episode_complete"].all())
        terminals = dataset.loc[dataset["is_episode_terminal"]]
        self.assertEqual(len(terminals), 2)
        self.assertTrue(terminals["eligible_for_terminal_reward"].all())
        self.assertEqual(
            dataset.loc[
                dataset["episode_id"] == "2022-H1", "target_date"
            ].min(),
            pd.Timestamp("2022-01-03"),
        )
        self.assertEqual(
            dataset.loc[
                dataset["episode_id"] == "2022-H2", "target_date"
            ].max(),
            pd.Timestamp("2022-12-30"),
        )

    def test_partial_episode_is_excluded_by_default(self) -> None:
        market, features = yearly_market_and_features()
        partial_market = market.loc[:"2022-09-30"]
        partial_features = features.loc[:"2022-09-30"]

        dataset = build_rl_dataset(partial_market, partial_features)

        self.assertEqual(set(dataset["episode_id"]), {"2022-H1"})

    def test_partial_episode_can_be_kept_for_diagnostics(self) -> None:
        market, features = yearly_market_and_features()
        partial_market = market.loc[:"2022-09-30"]
        partial_features = features.loc[:"2022-09-30"]

        dataset = build_rl_dataset(
            partial_market,
            partial_features,
            RLDatasetConfig(include_incomplete_episodes=True),
        )

        h2 = dataset.loc[dataset["episode_id"] == "2022-H2"]
        self.assertFalse(h2["episode_complete"].any())
        self.assertFalse(h2["eligible_for_terminal_reward"].any())


class LeakageTests(unittest.TestCase):
    def test_future_market_changes_do_not_modify_earlier_rows(self) -> None:
        market, features = yearly_market_and_features()
        changed = market.copy()
        cutoff = pd.Timestamp("2022-10-03")
        changed.loc[cutoff:, ["qqq_open", "qqq_close"]] *= 2.0

        baseline = build_rl_dataset(market, features)
        perturbed = build_rl_dataset(changed, features)
        earlier_target_dates = baseline["target_date"] < cutoff

        pd.testing.assert_frame_equal(
            baseline.loc[earlier_target_dates].reset_index(drop=True),
            perturbed.loc[earlier_target_dates].reset_index(drop=True),
            check_exact=True,
        )


if __name__ == "__main__":
    unittest.main()
