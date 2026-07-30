import json

import numpy as np
import pandas as pd

from src.agentic.half_year_rewards import build_half_year_reward_rows
from src.agentic.trajectory_dataset import build_agentic_trajectories


FEATURES = (
    "qqq_total_return_120d",
    "spy_total_return_120d",
    "qqq_minus_spy_return_60d",
    "qqq_drawdown_252d",
    "spy_drawdown_252d",
)


def _episode(
    start: str,
    episode_id: str,
    qqq_returns: tuple[float, float],
    spy_returns: tuple[float, float],
) -> list[dict]:
    targets = pd.bdate_range(start, periods=2)
    rows = []
    for index, target in enumerate(targets):
        rows.append(
            {
                "feature_date": target - pd.offsets.BDay(1),
                "target_date": target,
                "episode_id": episode_id,
                "episode_complete": True,
                "qqq_total_return_120d": 0.1,
                "spy_total_return_120d": 0.08,
                "qqq_minus_spy_return_60d": 0.04,
                "qqq_drawdown_252d": -0.05,
                "spy_drawdown_252d": -0.04,
                "qqq_overnight_return": 0.0,
                "spy_overnight_return": 0.0,
                "qqq_intraday_return": qqq_returns[index],
                "spy_intraday_return": spy_returns[index],
                "qqq_close_to_close_return": qqq_returns[index],
                "spy_close_to_close_return": spy_returns[index],
                "cash_overnight_return": 0.0,
                "cash_intraday_return": 0.0,
            }
        )
    return rows


def _dataset() -> pd.DataFrame:
    rows = []
    rows += _episode(
        "2018-01-02", "2018-H1", (0.02, 0.02), (0.0, 0.0)
    )
    rows += _episode(
        "2022-01-03", "2022-H1", (0.0, 0.0), (0.02, 0.02)
    )
    rows += _episode(
        "2025-01-02", "2025-H1", (0.01, 0.01), (0.01, 0.01)
    )
    return pd.DataFrame(rows)


def test_reward_rows_use_train_validation_only_and_all_actions():
    dataset = _dataset()
    trajectories = build_agentic_trajectories(
        dataset,
        feature_columns=FEATURES,
    )
    rows = build_half_year_reward_rows(
        dataset,
        feature_columns=FEATURES,
        trajectory_records=trajectories,
    )
    assert set(rows) == {"train", "validation"}
    assert all(
        len(row["action_rewards"]) == 15
        for values in rows.values()
        for row in values
    )
    assert "2025-" not in json.dumps(rows)
    assert all(
        row["reward_contract"]["uses_future_returns_in_prompt"] is False
        for values in rows.values()
        for row in values
    )


def test_train_reward_prefers_qqq_when_only_qqq_rises():
    dataset = _dataset()
    trajectories = build_agentic_trajectories(
        dataset,
        feature_columns=FEATURES,
    )
    rows = build_half_year_reward_rows(
        dataset,
        feature_columns=FEATURES,
        trajectory_records=trajectories,
        transaction_cost_rate=0.0,
    )
    first = rows["train"][0]
    assert 14 in first["best_action_ids"]
    assert np.isfinite(first["action_rewards"]).all()
