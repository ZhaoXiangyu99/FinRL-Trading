"""Gymnasium adapter tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env

from src.environments.gym_qqq_spy_cash import QQQSpyCashGymEnv


def sample_episode(episode_id: str = "2022-H1") -> pd.DataFrame:
    feature_dates = pd.bdate_range("2021-12-31", periods=4)
    target_dates = feature_dates + pd.offsets.BDay(1)
    return pd.DataFrame(
        {
            "episode_id": episode_id,
            "feature_date": feature_dates,
            "target_date": target_dates,
            "feature_a": (0.0, 0.1, -0.2, 0.3),
            "feature_b": (1.0, 0.5, -0.5, 0.0),
            "qqq_overnight_return": (0.0, 0.01, -0.01, 0.0),
            "spy_overnight_return": (0.0, 0.005, -0.005, 0.0),
            "qqq_intraday_return": (0.01, -0.01, 0.02, 0.0),
            "spy_intraday_return": (0.005, 0.0, 0.01, -0.005),
            "cash_overnight_return": 0.0,
            "cash_intraday_return": 0.0,
        }
    )


class GymAdapterTests(unittest.TestCase):
    def test_passes_stable_baselines_environment_checker(self) -> None:
        env = QQQSpyCashGymEnv(
            episodes=[sample_episode()],
            feature_columns=("feature_a", "feature_b"),
            episode_selection="sequential",
        )

        check_env(env, warn=True, skip_render_check=True)

    def test_reset_and_step_shapes_are_stable(self) -> None:
        env = QQQSpyCashGymEnv(
            episodes=[sample_episode()],
            feature_columns=("feature_a", "feature_b"),
        )

        observation, info = env.reset(seed=42)
        self.assertEqual(observation.shape, (13,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(info["episode_id"], "2022-H1")

        next_observation, reward, terminated, truncated, _ = env.step(0)
        self.assertEqual(next_observation.shape, (13,))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
