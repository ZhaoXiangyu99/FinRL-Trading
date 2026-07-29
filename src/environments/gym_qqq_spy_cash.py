"""Gymnasium adapter over the framework-agnostic accounting core."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .qqq_spy_cash_env import (
    DISCRETE_ACTION_WEIGHTS,
    QQQSpyCashEnv,
    RewardConfig,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised before optional install.
    gym = None
    spaces = None


_GymBase = gym.Env if gym is not None else object


class QQQSpyCashGymEnv(_GymBase):
    """Sample complete half-year episodes through a Gymnasium interface."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        episodes: Sequence[pd.DataFrame],
        feature_columns: Sequence[str],
        transaction_cost_rate: float = 0.001,
        reward_config: RewardConfig | None = None,
        episode_selection: Literal["random", "sequential"] = "random",
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError(
                "gymnasium is required for QQQSpyCashGymEnv; "
                "install the project's ml dependencies"
            )
        super().__init__()
        if not episodes:
            raise ValueError("episodes cannot be empty")
        if episode_selection not in {"random", "sequential"}:
            raise ValueError(
                "episode_selection must be 'random' or 'sequential'"
            )

        self.episodes = [episode.reset_index(drop=True) for episode in episodes]
        self.feature_columns = tuple(feature_columns)
        self.transaction_cost_rate = transaction_cost_rate
        self.reward_config = reward_config or RewardConfig()
        self.episode_selection = episode_selection
        self._sequential_index = 0
        self._core: QQQSpyCashEnv | None = None
        self._episode_id: str | None = None

        observation_size = len(self.feature_columns) + 11
        self.action_space = spaces.Discrete(len(DISCRETE_ACTION_WEIGHTS))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

    def _select_episode_index(
        self, options: dict[str, Any] | None
    ) -> int:
        if options and "episode_index" in options:
            index = int(options["episode_index"])
            if not 0 <= index < len(self.episodes):
                raise ValueError("episode_index is out of range")
            return index
        if self.episode_selection == "sequential":
            index = self._sequential_index
            self._sequential_index = (
                self._sequential_index + 1
            ) % len(self.episodes)
            return index
        return int(self.np_random.integers(0, len(self.episodes)))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        episode_index = self._select_episode_index(options)
        episode = self.episodes[episode_index]
        self._episode_id = str(episode["episode_id"].iloc[0])
        self._core = QQQSpyCashEnv.from_dataset(
            episode,
            feature_columns=self.feature_columns,
            transaction_cost_rate=self.transaction_cost_rate,
            reward_config=self.reward_config,
        )
        observation, info = self._core.reset()
        info.update(
            {
                "episode_id": self._episode_id,
                "episode_index": episode_index,
            }
        )
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._core is None:
            raise RuntimeError("reset() must be called before step()")
        observation, reward, terminated, truncated, info = self._core.step(
            action
        )
        info["episode_id"] = self._episode_id
        return observation, reward, terminated, truncated, info
