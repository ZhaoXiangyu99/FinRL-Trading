"""Contract tests for the two-stage QQQ/SPY/cash environment."""

from __future__ import annotations

import unittest

import numpy as np

from src.environments.qqq_spy_cash_env import (
    DISCRETE_ACTION_WEIGHTS,
    QQQSpyCashEnv,
    RewardConfig,
    relative_performance_score,
)


def make_env(
    *,
    overnight_returns: np.ndarray | None = None,
    intraday_returns: np.ndarray | None = None,
    transaction_cost_rate: float = 0.001,
    reward_config: RewardConfig | None = None,
) -> QQQSpyCashEnv:
    overnight = (
        np.asarray(((0.02, 0.01), (0.03, -0.01)), dtype=np.float64)
        if overnight_returns is None
        else np.asarray(overnight_returns, dtype=np.float64)
    )
    intraday = (
        np.asarray(((0.10, 0.00), (0.00, 0.00)), dtype=np.float64)
        if intraday_returns is None
        else np.asarray(intraday_returns, dtype=np.float64)
    )
    if overnight.shape != intraday.shape:
        raise ValueError("test returns must have matching shapes")
    steps = overnight.shape[0]
    features = np.zeros((steps, 2), dtype=np.float64)
    feature_dates = np.arange(
        np.datetime64("2026-01-02"),
        np.datetime64("2026-01-02") + steps,
    )
    target_dates = feature_dates + np.timedelta64(1, "D")
    return QQQSpyCashEnv(
        features=features,
        overnight_returns=overnight,
        intraday_returns=intraday,
        feature_dates=feature_dates,
        target_dates=target_dates,
        transaction_cost_rate=transaction_cost_rate,
        reward_config=reward_config,
    )


class ActionSpaceTests(unittest.TestCase):
    def test_action_grid_is_complete_valid_and_symmetric(self) -> None:
        self.assertEqual(DISCRETE_ACTION_WEIGHTS.shape, (15, 3))
        np.testing.assert_allclose(
            DISCRETE_ACTION_WEIGHTS.sum(axis=1), 1.0
        )
        self.assertTrue(np.all(DISCRETE_ACTION_WEIGHTS >= 0.0))
        actions = {tuple(row) for row in DISCRETE_ACTION_WEIGHTS}
        for qqq, spy, cash in actions:
            self.assertIn((spy, qqq, cash), actions)

    def test_invalid_action_is_rejected(self) -> None:
        env = make_env()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(15)
        with self.assertRaises(TypeError):
            env.step(1.0)  # type: ignore[arg-type]


class AccountingTests(unittest.TestCase):
    def test_initial_cash_ignores_first_overnight_then_trades_at_open(
        self,
    ) -> None:
        env = make_env(transaction_cost_rate=0.001)
        env.reset()

        # A14 is 100% QQQ. Initial cash earns zero overnight; the open trade
        # costs 0.1%, then QQQ earns 10% intraday.
        _, _, terminated, _, info = env.step(14)

        self.assertFalse(terminated)
        self.assertAlmostEqual(info["overnight_portfolio_return"], 0.0)
        self.assertAlmostEqual(info["turnover"], 1.0)
        self.assertAlmostEqual(info["transaction_cost_fraction"], 0.001)
        self.assertAlmostEqual(info["portfolio_value"], 0.999 * 1.10)
        np.testing.assert_allclose(info["weights"], (1.0, 0.0, 0.0))

    def test_old_weights_earn_overnight_before_new_action(self) -> None:
        env = make_env(transaction_cost_rate=0.0)
        env.reset()
        env.step(14)  # 100% QQQ after first open.

        _, _, terminated, _, info = env.step(0)  # Move to cash at second open.

        self.assertTrue(terminated)
        self.assertAlmostEqual(info["overnight_portfolio_return"], 0.03)
        self.assertAlmostEqual(info["portfolio_value"], 1.10 * 1.03)
        self.assertAlmostEqual(info["turnover"], 1.0)
        np.testing.assert_allclose(info["weights"], (0.0, 0.0, 1.0))

    def test_no_cost_qqq_policy_matches_qqq_benchmark(self) -> None:
        env = make_env(transaction_cost_rate=0.0)
        env.reset()
        final_info = None
        for _ in range(2):
            _, _, _, _, final_info = env.step(14)

        assert final_info is not None
        self.assertAlmostEqual(
            final_info["portfolio_return"], final_info["qqq_return"]
        )

    def test_dense_rewards_telescope_to_terminal_score(self) -> None:
        env = make_env(
            overnight_returns=np.asarray(
                ((0.01, 0.00), (-0.01, 0.02), (0.02, 0.00))
            ),
            intraday_returns=np.asarray(
                ((0.02, 0.01), (0.00, -0.01), (0.01, 0.00))
            ),
            transaction_cost_rate=0.0,
            reward_config=RewardConfig(
                partial_credit=0.25,
                drawdown_penalty=0.1,
                turnover_regularizer=0.001,
                score_clip=None,
            ),
        )
        env.reset()

        rewards = []
        final_info = None
        for action in (11, 11, 11):
            _, reward, terminated, truncated, final_info = env.step(action)
            rewards.append(reward)
            self.assertFalse(truncated)

        self.assertTrue(terminated)
        assert final_info is not None
        self.assertAlmostEqual(
            sum(rewards), final_info["terminal_score"], places=12
        )
        with self.assertRaises(RuntimeError):
            env.step(11)


class LeakageContractTests(unittest.TestCase):
    def test_reset_observation_does_not_contain_future_returns(self) -> None:
        calm = make_env(
            overnight_returns=np.zeros((2, 2)),
            intraday_returns=np.zeros((2, 2)),
        )
        shock = make_env(
            overnight_returns=np.asarray(((0.5, -0.4), (-0.3, 0.2))),
            intraday_returns=np.asarray(((-0.2, 0.3), (0.4, -0.2))),
        )

        calm_observation, _ = calm.reset()
        shock_observation, _ = shock.reset()
        np.testing.assert_array_equal(calm_observation, shock_observation)

    def test_target_dates_must_stay_in_one_half_year(self) -> None:
        with self.assertRaisesRegex(ValueError, "one calendar half-year"):
            QQQSpyCashEnv(
                features=np.zeros((2, 1)),
                overnight_returns=np.zeros((2, 2)),
                intraday_returns=np.zeros((2, 2)),
                feature_dates=("2026-06-29", "2026-06-30"),
                target_dates=("2026-06-30", "2026-07-01"),
            )


class RewardTests(unittest.TestCase):
    def test_reward_tiers_match_design_scenarios(self) -> None:
        bull_full = relative_performance_score(0.14, 0.08, 0.11)
        bull_partial = relative_performance_score(0.10, 0.08, 0.11)
        bull_fail = relative_performance_score(0.05, 0.08, 0.11)
        bear_full = relative_performance_score(-0.05, -0.08, -0.15)
        bear_partial = relative_performance_score(-0.10, -0.08, -0.15)

        self.assertEqual(bull_full[1], "beat_both")
        self.assertEqual(bull_partial[1], "beat_one")
        self.assertEqual(bull_fail[1], "beat_neither")
        self.assertEqual(bear_full[1], "beat_both")
        self.assertEqual(bear_partial[1], "beat_one")
        self.assertGreater(bull_full[0], bull_partial[0])
        self.assertGreater(bull_partial[0], bull_fail[0])
        self.assertGreater(bear_full[0], bear_partial[0])


if __name__ == "__main__":
    unittest.main()
