import json

from src.agentic.contracts import ACTION_SCHEMA_VERSION, action_weights
from src.agentic.grpo_rewards import (
    half_year_outcome_reward,
    strict_format_reward,
)


def _completion(action_id: int):
    content = json.dumps(
        {
            "schema_version": ACTION_SCHEMA_VERSION,
            "action_id": action_id,
            "target_weights": action_weights(action_id),
            "confidence": "low",
            "rationale_codes": ["outcome_optimized"],
            "requires_human_approval": True,
        }
    )
    return [{"role": "assistant", "content": content}]


def test_outcome_reward_indexes_generated_action():
    rewards = list(range(15))
    result = half_year_outcome_reward(
        [_completion(12), _completion(3)],
        [rewards, rewards],
    )
    assert result == [12.0, 3.0]


def test_invalid_completion_is_penalized():
    invalid = [{"role": "assistant", "content": "not json"}]
    assert half_year_outcome_reward(
        [invalid], [[0.0] * 15]
    ) == [-1.0]
    assert strict_format_reward([invalid, _completion(0)]) == [-1.0, 1.0]
