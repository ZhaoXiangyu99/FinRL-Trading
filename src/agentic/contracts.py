"""Strict contracts for QQQ/SPY/cash agent decisions and SFT records."""

from __future__ import annotations

import json
from typing import Any, Mapping

from src.environments.qqq_spy_cash_env import DISCRETE_ACTION_WEIGHTS


ACTION_SCHEMA_VERSION = "qqq_spy_cash_action_v1"
TRAJECTORY_SCHEMA_VERSION = "qqq_spy_agentic_trajectory_v1"
ASSET_NAMES = ("QQQ", "SPY", "CASH")


class AgenticContractError(ValueError):
    """Raised when an agent record violates the frozen interface."""


def action_weights(action_id: int) -> dict[str, float]:
    """Return named target weights from the environment's action table."""

    if (
        not isinstance(action_id, int)
        or isinstance(action_id, bool)
        or action_id < 0
        or action_id >= len(DISCRETE_ACTION_WEIGHTS)
    ):
        raise AgenticContractError(f"unknown action_id: {action_id}")
    weights = DISCRETE_ACTION_WEIGHTS[action_id]
    return {
        asset: float(weight)
        for asset, weight in zip(ASSET_NAMES, weights, strict=True)
    }


def validate_action_proposal(payload: Mapping[str, Any]) -> None:
    """Validate the exact JSON object emitted by the policy model."""

    expected_keys = {
        "schema_version",
        "action_id",
        "target_weights",
        "confidence",
        "rationale_codes",
        "requires_human_approval",
    }
    if set(payload) != expected_keys:
        raise AgenticContractError(
            "action proposal keys differ from the frozen schema"
        )
    if payload["schema_version"] != ACTION_SCHEMA_VERSION:
        raise AgenticContractError("unexpected action schema version")

    action_id = payload["action_id"]
    if not isinstance(action_id, int) or isinstance(action_id, bool):
        raise AgenticContractError("action_id must be an integer")
    expected_weights = action_weights(action_id)
    supplied_weights = payload["target_weights"]
    if not isinstance(supplied_weights, Mapping):
        raise AgenticContractError("target_weights must be an object")
    if set(supplied_weights) != set(ASSET_NAMES):
        raise AgenticContractError("target_weights has unexpected assets")
    for asset, expected in expected_weights.items():
        actual = supplied_weights[asset]
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise AgenticContractError(f"{asset} weight must be numeric")
        if abs(float(actual) - expected) > 1e-9:
            raise AgenticContractError(
                f"{asset} weight does not match action_id {action_id}"
            )

    if payload["confidence"] not in {"low", "medium"}:
        raise AgenticContractError(
            "bootstrap policy confidence must be low or medium"
        )
    rationale_codes = payload["rationale_codes"]
    if (
        not isinstance(rationale_codes, list)
        or not rationale_codes
        or not all(isinstance(item, str) and item for item in rationale_codes)
    ):
        raise AgenticContractError(
            "rationale_codes must be a non-empty string list"
        )
    if payload["requires_human_approval"] is not True:
        raise AgenticContractError("human approval must remain mandatory")


def parse_and_validate_action_json(content: str) -> dict[str, Any]:
    """Parse model text as a single strict JSON action proposal."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AgenticContractError("assistant content is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AgenticContractError("assistant content must be a JSON object")
    validate_action_proposal(payload)
    return payload


def validate_trajectory_record(record: Mapping[str, Any]) -> None:
    """Validate one conversational prompt-completion training sample."""

    expected_keys = {
        "schema_version",
        "sample_id",
        "split",
        "feature_date",
        "target_date",
        "prompt",
        "completion",
        "provenance",
    }
    if set(record) != expected_keys:
        raise AgenticContractError(
            "trajectory record keys differ from the frozen schema"
        )
    if record["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise AgenticContractError("unexpected trajectory schema version")
    if record["split"] not in {"train", "validation"}:
        raise AgenticContractError("test records must never enter SFT data")

    prompt = record["prompt"]
    completion = record["completion"]
    if (
        not isinstance(prompt, list)
        or len(prompt) != 2
        or [item.get("role") for item in prompt] != ["system", "user"]
    ):
        raise AgenticContractError(
            "prompt must contain one system and one user message"
        )
    if (
        not isinstance(completion, list)
        or len(completion) != 1
        or completion[0].get("role") != "assistant"
    ):
        raise AgenticContractError(
            "completion must contain one assistant message"
        )
    user_payload = json.loads(prompt[1]["content"])
    if set(user_payload) != {
        "request",
        "feature_date",
        "execution_date",
        "tool_observations",
    }:
        raise AgenticContractError("user observation keys are not frozen")
    observations = user_payload["tool_observations"]
    if [item.get("tool") for item in observations] != [
        "observe_market_state",
        "observe_portfolio_state",
    ]:
        raise AgenticContractError("required tool observations are missing")

    serialized_prompt = json.dumps(prompt, ensure_ascii=False)
    forbidden = (
        "overnight_return",
        "intraday_return",
        "close_to_close_return",
        "terminal_reward",
        "benchmark_return",
    )
    if any(name in serialized_prompt for name in forbidden):
        raise AgenticContractError(
            "future return or reward field leaked into the prompt"
        )
    parse_and_validate_action_json(completion[0]["content"])

    provenance = record["provenance"]
    if provenance != {
        "teacher": "deterministic_format_bootstrap_v1",
        "purpose": "format_bootstrap_not_alpha",
        "uses_future_returns": False,
    }:
        raise AgenticContractError("unexpected trajectory provenance")
