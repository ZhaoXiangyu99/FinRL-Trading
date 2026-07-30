from src.agentic.reward_evaluation import score_format_report


def test_score_format_report_penalizes_invalid_and_joins_ids():
    format_report = {
        "test_samples_used": 0,
        "details": [
            {
                "sample_id": "validation:a",
                "predicted_action_id": 2,
                "expected_teacher_action_id": 1,
            },
            {
                "sample_id": "validation:b",
                "predicted_action_id": None,
                "expected_teacher_action_id": 0,
            },
        ],
    }
    reward_rows = [
        {
            "sample_id": "validation:a",
            "split": "validation",
            "action_rewards": [0.0, 0.1, 0.2] + [0.0] * 12,
        },
        {
            "sample_id": "validation:b",
            "split": "validation",
            "action_rewards": [0.3] + [0.0] * 14,
        },
    ]
    result = score_format_report(format_report, reward_rows)
    assert result["schema_valid_rate"] == 0.5
    assert result["mean_policy_reward_with_invalid_penalty"] == -0.4
    assert result["mean_sft_teacher_reward"] == 0.2
    assert result["test_samples_used"] == 0
