import json

import pandas as pd
import pytest

from src.agentic.contracts import (
    AgenticContractError,
    parse_and_validate_action_json,
    validate_trajectory_record,
)
from src.agentic.trajectory_dataset import build_agentic_trajectories


FEATURES = (
    "qqq_total_return_120d",
    "spy_total_return_120d",
    "qqq_minus_spy_return_60d",
    "qqq_drawdown_252d",
    "spy_drawdown_252d",
)


def _episode_rows(
    start: str,
    end: str,
    episode_id: str,
) -> list[dict]:
    dates = pd.bdate_range(start, end)
    rows = []
    for index, target_date in enumerate(dates):
        rows.append(
            {
                "feature_date": target_date - pd.offsets.BDay(1),
                "target_date": target_date,
                "episode_id": episode_id,
                "episode_complete": True,
                "qqq_total_return_120d": 0.1 + index / 10000,
                "spy_total_return_120d": 0.08 + index / 10000,
                "qqq_minus_spy_return_60d": 0.04,
                "qqq_drawdown_252d": -0.05,
                "spy_drawdown_252d": -0.04,
                "qqq_intraday_return": 99.0,
            }
        )
    return rows


def _dataset() -> pd.DataFrame:
    rows = []
    rows.extend(_episode_rows("2018-01-01", "2018-06-29", "2018-H1"))
    rows.extend(_episode_rows("2022-01-03", "2022-06-30", "2022-H1"))
    rows.extend(_episode_rows("2025-01-02", "2025-06-30", "2025-H1"))
    return pd.DataFrame(rows)


def test_build_agentic_data_excludes_test_and_future_returns():
    records = build_agentic_trajectories(
        _dataset(),
        feature_columns=FEATURES,
    )
    assert set(records) == {"train", "validation"}
    assert records["train"]
    assert records["validation"]
    assert all(
        record["split"] != "test"
        for split_records in records.values()
        for record in split_records
    )
    serialized = json.dumps(records)
    assert "qqq_intraday_return" not in serialized
    assert "2025-" not in serialized


def test_records_are_deterministic_and_contract_valid():
    first = build_agentic_trajectories(
        _dataset(),
        feature_columns=FEATURES,
    )
    second = build_agentic_trajectories(
        _dataset(),
        feature_columns=FEATURES,
    )
    assert first == second
    for split_records in first.values():
        for record in split_records:
            validate_trajectory_record(record)


def test_action_weights_must_match_action_id():
    records = build_agentic_trajectories(
        _dataset(),
        feature_columns=FEATURES,
    )
    content = records["train"][0]["completion"][0]["content"]
    proposal = json.loads(content)
    proposal["target_weights"]["CASH"] = 0.99
    with pytest.raises(AgenticContractError):
        parse_and_validate_action_json(json.dumps(proposal))
