#!/usr/bin/env python3
"""Apply the frozen validation gate to SFT and Agentic-RL reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    challenger = json.loads(
        args.challenger.read_text(encoding="utf-8")
    )
    for name, report in (
        ("baseline", baseline),
        ("challenger", challenger),
    ):
        if report["test_samples_used"] != 0:
            raise ValueError(f"{name} report used test samples")
        if report["samples"] != challenger["samples"]:
            raise ValueError("reports must use identical sample counts")

    baseline_reward = baseline[
        "mean_policy_reward_with_invalid_penalty"
    ]
    challenger_reward = challenger[
        "mean_policy_reward_with_invalid_penalty"
    ]
    reward_delta = challenger_reward - baseline_reward
    format_pass = (
        challenger["schema_valid_rate"] == 1.0
        and challenger["schema_valid_rate"]
        >= baseline["schema_valid_rate"]
    )
    reward_pass = reward_delta > 1e-6
    status = (
        "validation_challenger"
        if format_pass and reward_pass
        else "validation_rejected"
    )
    result = {
        "status": status,
        "baseline": "qwen3_8b_qqq_spy_format_sft_v1",
        "challenger": "qwen3_8b_qqq_spy_half_year_grpo_v1",
        "samples": challenger["samples"],
        "baseline_schema_valid_rate": baseline["schema_valid_rate"],
        "challenger_schema_valid_rate": challenger[
            "schema_valid_rate"
        ],
        "baseline_mean_reward": baseline_reward,
        "challenger_mean_reward": challenger_reward,
        "reward_delta": reward_delta,
        "minimum_strict_reward_improvement": 1e-6,
        "format_pass": format_pass,
        "reward_pass": reward_pass,
        "test_samples_used": 0,
        "promotion_effect": (
            "keep_sft_adapter"
            if status == "validation_rejected"
            else "eligible_for_further_validation_only"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
