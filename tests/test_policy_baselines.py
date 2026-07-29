"""Tests for deterministic environment baselines."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.policy_baselines import evaluate_constant_policies


class BaselineTests(unittest.TestCase):
    def test_qqq_target_matches_benchmark_without_costs(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=3)
        episode = pd.DataFrame(
            {
                "episode_id": "2022-H1",
                "feature_date": dates,
                "target_date": dates + pd.offsets.BDay(1),
                "feature_a": (0.0, 0.1, 0.2),
                "qqq_overnight_return": (0.01, 0.02, -0.01),
                "spy_overnight_return": (0.00, 0.01, 0.00),
                "qqq_intraday_return": (0.02, -0.01, 0.03),
                "spy_intraday_return": (0.01, 0.00, 0.01),
                "cash_overnight_return": 0.0,
                "cash_intraday_return": 0.0,
            }
        )

        results = evaluate_constant_policies(
            episode,
            feature_columns=("feature_a",),
            policies={"qqq": 14},
            transaction_cost_rate=0.0,
        )

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(
            results.iloc[0]["portfolio_return"],
            results.iloc[0]["qqq_buy_hold_return"],
        )
        self.assertAlmostEqual(
            results.iloc[0]["reward_sum"],
            results.iloc[0]["terminal_score"],
        )


if __name__ == "__main__":
    unittest.main()
