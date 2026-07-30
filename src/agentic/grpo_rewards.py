"""Reward functions used by TRL GRPO for the trading action contract."""

from __future__ import annotations

from typing import Any, Sequence

from src.agentic.contracts import parse_and_validate_action_json


def _completion_content(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if (
        isinstance(completion, list)
        and len(completion) == 1
        and isinstance(completion[0], dict)
    ):
        return str(completion[0].get("content", ""))
    return ""


def half_year_outcome_reward(
    completions: Sequence[Any],
    action_rewards: Sequence[Sequence[float]],
    **_: Any,
) -> list[float]:
    """Return the counterfactual terminal score for each generated action."""

    if len(completions) != len(action_rewards):
        raise ValueError("completions and action_rewards lengths differ")
    rewards = []
    for completion, scores in zip(
        completions, action_rewards, strict=True
    ):
        try:
            proposal = parse_and_validate_action_json(
                _completion_content(completion)
            )
            if len(scores) != 15:
                raise ValueError("action reward table must contain 15 values")
            rewards.append(float(scores[proposal["action_id"]]))
        except ValueError:
            rewards.append(-1.0)
    return rewards


def strict_format_reward(
    completions: Sequence[Any],
    **_: Any,
) -> list[float]:
    """Reward strict JSON/action compliance independently from outcome."""

    rewards = []
    for completion in completions:
        try:
            parse_and_validate_action_json(_completion_content(completion))
            rewards.append(1.0)
        except ValueError:
            rewards.append(-1.0)
    return rewards
