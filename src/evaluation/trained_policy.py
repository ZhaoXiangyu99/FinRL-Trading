"""Evaluation of a trained policy on explicitly selected episodes."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.environments.gym_qqq_spy_cash import QQQSpyCashGymEnv


def evaluate_model_on_episodes(
    model: Any,
    environment: QQQSpyCashGymEnv,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Evaluate deterministically without sampling or opening the test split."""

    results = []
    aggregate_actions: Counter[int] = Counter()
    for episode_index in range(len(environment.episodes)):
        observation, reset_info = environment.reset(
            options={"episode_index": episode_index}
        )
        terminated = False
        final_info = reset_info
        reward_sum = 0.0
        episode_actions: Counter[int] = Counter()
        while not terminated:
            action, _ = model.predict(observation, deterministic=True)
            action_value = int(np.asarray(action).item())
            episode_actions[action_value] += 1
            aggregate_actions[action_value] += 1
            (
                observation,
                reward,
                terminated,
                truncated,
                final_info,
            ) = environment.step(action_value)
            if truncated:
                raise RuntimeError("evaluation episode was truncated")
            reward_sum += float(reward)
        results.append(
            {
                "episode_id": final_info["episode_id"],
                "portfolio_return": final_info["portfolio_return"],
                "spy_return": final_info["spy_return"],
                "qqq_return": final_info["qqq_return"],
                "terminal_score": final_info["terminal_score"],
                "reward_sum": reward_sum,
                "reward_tier": final_info["reward_tier"],
                "max_drawdown": final_info["max_drawdown"],
                "cumulative_turnover": final_info["cumulative_turnover"],
                "action_counts": dict(sorted(episode_actions.items())),
            }
        )
    return results, dict(sorted(aggregate_actions.items()))
