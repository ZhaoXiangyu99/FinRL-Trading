"""Score generated action reports against a sealed validation reward table."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def score_format_report(
    format_report: Mapping[str, Any],
    reward_rows: Sequence[Mapping[str, Any]],
    *,
    invalid_penalty: float = -1.0,
) -> dict[str, Any]:
    """Join generated actions to validation rewards by stable sample_id."""

    if format_report.get("test_samples_used") != 0:
        raise ValueError("format report must not use test samples")
    rewards_by_id = {
        str(row["sample_id"]): row
        for row in reward_rows
        if row.get("split") == "validation"
    }
    if len(rewards_by_id) != len(reward_rows):
        raise ValueError("reward rows must be unique validation samples")

    policy_rewards = []
    valid_policy_rewards = []
    teacher_rewards = []
    oracle_rewards = []
    action_counts: Counter[int] = Counter()
    for detail in format_report["details"]:
        row = rewards_by_id.get(str(detail["sample_id"]))
        if row is None:
            raise ValueError(
                f"reward missing for {detail['sample_id']}"
            )
        scores = row["action_rewards"]
        if len(scores) != 15:
            raise ValueError("each row must contain 15 action rewards")
        predicted = detail["predicted_action_id"]
        if predicted is None:
            policy_rewards.append(float(invalid_penalty))
        else:
            reward = float(scores[int(predicted)])
            policy_rewards.append(reward)
            valid_policy_rewards.append(reward)
            action_counts[int(predicted)] += 1
        teacher_action = int(detail["expected_teacher_action_id"])
        teacher_rewards.append(float(scores[teacher_action]))
        oracle_rewards.append(float(max(scores)))

    count = len(policy_rewards)
    if count == 0:
        raise ValueError("format report has no details")
    mean = lambda values: sum(values) / len(values)
    mean_policy = mean(policy_rewards)
    mean_teacher = mean(teacher_rewards)
    mean_oracle = mean(oracle_rewards)
    return {
        "evaluation": (
            "sampled_validation_counterfactual_half_year_reward"
        ),
        "samples": count,
        "schema_valid_count": len(valid_policy_rewards),
        "schema_valid_rate": len(valid_policy_rewards) / count,
        "mean_policy_reward_with_invalid_penalty": mean_policy,
        "mean_valid_policy_reward": (
            mean(valid_policy_rewards)
            if valid_policy_rewards
            else None
        ),
        "mean_sft_teacher_reward": mean_teacher,
        "mean_oracle_reward": mean_oracle,
        "policy_minus_sft_teacher": mean_policy - mean_teacher,
        "oracle_regret": mean_oracle - mean_policy,
        "action_counts": {
            str(action): value
            for action, value in sorted(action_counts.items())
        },
        "invalid_penalty": invalid_penalty,
        "test_samples_used": 0,
    }
