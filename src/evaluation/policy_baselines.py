"""Deterministic policy baselines for environment sanity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.environments.qqq_spy_cash_env import QQQSpyCashEnv, RewardConfig


CONSTANT_POLICIES = {
    "cash_target": 0,
    "spy_target": 4,
    "qqq_spy_50_50_target": 11,
    "qqq_target": 14,
}


def evaluate_constant_policies(
    dataset: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    policies: Mapping[str, int] = CONSTANT_POLICIES,
    transaction_cost_rate: float = 0.001,
    reward_config: RewardConfig | None = None,
    split_name: str = "unknown",
) -> pd.DataFrame:
    """Evaluate constant target-weight actions one half-year at a time."""

    reward_config = reward_config or RewardConfig()
    rows = []
    for episode_id, episode in dataset.groupby("episode_id", sort=True):
        for policy_name, action in policies.items():
            env = QQQSpyCashEnv.from_dataset(
                episode,
                feature_columns=feature_columns,
                transaction_cost_rate=transaction_cost_rate,
                reward_config=reward_config,
            )
            env.reset()
            rewards = []
            final_info = None
            while final_info is None or not terminated:
                _, reward, terminated, _, final_info = env.step(action)
                rewards.append(reward)

            rows.append(
                {
                    "split": split_name,
                    "episode_id": episode_id,
                    "policy": policy_name,
                    "action": action,
                    "portfolio_return": final_info["portfolio_return"],
                    "spy_buy_hold_return": final_info["spy_return"],
                    "qqq_buy_hold_return": final_info["qqq_return"],
                    "alpha_strong": final_info["alpha_strong"],
                    "alpha_weak": final_info["alpha_weak"],
                    "reward_tier": final_info["reward_tier"],
                    "terminal_score": final_info["terminal_score"],
                    "reward_sum": float(sum(rewards)),
                    "max_drawdown": final_info["max_drawdown"],
                    "cumulative_turnover": final_info[
                        "cumulative_turnover"
                    ],
                    "transaction_cost_value": final_info[
                        "cumulative_transaction_cost"
                    ],
                }
            )
    return pd.DataFrame(rows)


def summarize_policy_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate half-year policy outcomes without hiding episode dispersion."""

    required = {
        "split",
        "policy",
        "portfolio_return",
        "terminal_score",
        "max_drawdown",
        "cumulative_turnover",
        "reward_tier",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing columns: {sorted(missing)}")

    summary = (
        results.groupby(["split", "policy"], sort=True)
        .agg(
            episodes=("episode_id", "size"),
            mean_half_year_return=("portfolio_return", "mean"),
            median_half_year_return=("portfolio_return", "median"),
            worst_half_year_return=("portfolio_return", "min"),
            mean_terminal_score=("terminal_score", "mean"),
            worst_max_drawdown=("max_drawdown", "max"),
            mean_turnover=("cumulative_turnover", "mean"),
            beat_both_rate=(
                "reward_tier",
                lambda values: float(np.mean(values == "beat_both")),
            ),
        )
        .reset_index()
    )
    return summary


def write_baseline_results(
    *,
    episode_results: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    episode_path = output / "constant_policy_episode_results.csv"
    summary_path = output / "constant_policy_summary.csv"
    episode_results.to_csv(episode_path, index=False)
    summary.to_csv(summary_path, index=False)
    return episode_path, summary_path
