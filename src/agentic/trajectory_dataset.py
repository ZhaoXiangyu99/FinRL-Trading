"""Build leakage-safe conversational data for the Qwen policy bootstrap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.agentic.contracts import (
    ACTION_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    action_weights,
    validate_trajectory_record,
)
from src.training.data_pipeline import (
    DEFAULT_SPLIT_CONFIG,
    prepare_training_splits,
)


SYSTEM_PROMPT = """You are a constrained QQQ/SPY/cash allocation policy.
The supplied observations are read-only tool results available after the
feature-date close. Propose exactly one target allocation for the next regular
session open. Return one JSON object only, with no markdown or extra text.
Allowed action_id values are the frozen environment actions 0 through 14.
target_weights must exactly match the selected action. Never submit an order.
Human approval and the independent risk engine remain mandatory. Do not claim
that macro context alone predicts returns. Confidence must be low or medium."""


def _rounded(value: Any) -> float:
    return round(float(value), 6)


def _bootstrap_teacher(row: Mapping[str, Any]) -> tuple[int, list[str]]:
    """Select a transparent current-state label, never an outcome label.

    This teacher exists only to teach the model the interface and coherent
    state-conditioned outputs. It is intentionally not presented as alpha.
    """

    qqq_120 = float(row["qqq_total_return_120d"])
    spy_120 = float(row["spy_total_return_120d"])
    relative_60 = float(row["qqq_minus_spy_return_60d"])
    qqq_drawdown = float(row["qqq_drawdown_252d"])
    spy_drawdown = float(row["spy_drawdown_252d"])

    if (
        qqq_120 < 0.0
        and spy_120 < 0.0
        and min(qqq_drawdown, spy_drawdown) < -0.10
    ):
        return 0, ["both_120d_trends_negative", "material_drawdown", "cash_guard"]
    if qqq_120 > 0.0 and spy_120 <= 0.0:
        return 12, ["qqq_120d_positive", "spy_120d_nonpositive", "cash_buffer"]
    if spy_120 > 0.0 and qqq_120 <= 0.0:
        return 3, ["spy_120d_positive", "qqq_120d_nonpositive", "cash_buffer"]
    if qqq_120 > 0.0 and spy_120 > 0.0 and relative_60 > 0.03:
        return 12, ["both_120d_trends_positive", "qqq_relative_strength", "cash_buffer"]
    if qqq_120 > 0.0 and spy_120 > 0.0 and relative_60 < -0.03:
        return 3, ["both_120d_trends_positive", "spy_relative_strength", "cash_buffer"]
    return 6, ["mixed_or_close_signals", "diversified_risk", "cash_buffer"]


def _make_record(
    *,
    row: Mapping[str, Any],
    standardized_row: Mapping[str, Any],
    feature_columns: Sequence[str],
    split: str,
    current_weights: Mapping[str, float],
) -> dict[str, Any]:
    action_id, rationale_codes = _bootstrap_teacher(row)
    feature_date = pd.Timestamp(row["feature_date"]).date().isoformat()
    target_date = pd.Timestamp(row["target_date"]).date().isoformat()
    user_payload = {
        "request": "propose_next_open_target_allocation",
        "feature_date": feature_date,
        "execution_date": target_date,
        "tool_observations": [
            {
                "tool": "observe_market_state",
                "status": "ok",
                "feature_space": "train_standardized_v1",
                "features": {
                    column: _rounded(standardized_row[column])
                    for column in feature_columns
                },
            },
            {
                "tool": "observe_portfolio_state",
                "status": "ok",
                "current_weights": {
                    asset: _rounded(weight)
                    for asset, weight in current_weights.items()
                },
            },
        ],
    }
    proposal = {
        "schema_version": ACTION_SCHEMA_VERSION,
        "action_id": action_id,
        "target_weights": action_weights(action_id),
        "confidence": "low",
        "rationale_codes": rationale_codes,
        "requires_human_approval": True,
    }
    record = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "sample_id": f"{split}:{feature_date}:{target_date}",
        "split": split,
        "feature_date": feature_date,
        "target_date": target_date,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": json.dumps(
                    proposal,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        ],
        "provenance": {
            "teacher": "deterministic_format_bootstrap_v1",
            "purpose": "format_bootstrap_not_alpha",
            "uses_future_returns": False,
        },
    }
    validate_trajectory_record(record)
    return record


def build_agentic_trajectories(
    dataset: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return train/validation records while keeping test fully sealed."""

    standardized_splits, _ = prepare_training_splits(
        dataset,
        feature_columns=feature_columns,
        split_config=DEFAULT_SPLIT_CONFIG,
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation"):
        standardized = standardized_splits[split]
        raw = dataset.loc[standardized.index]
        records: list[dict[str, Any]] = []
        current_episode: str | None = None
        current_weights = action_weights(0)
        for index in standardized.index:
            raw_row = raw.loc[index]
            standardized_row = standardized.loc[index]
            episode_id = str(raw_row["episode_id"])
            if episode_id != current_episode:
                current_episode = episode_id
                current_weights = action_weights(0)
            record = _make_record(
                row=raw_row,
                standardized_row=standardized_row,
                feature_columns=feature_columns,
                split=split,
                current_weights=current_weights,
            )
            records.append(record)
            proposal = json.loads(record["completion"][0]["content"])
            current_weights = proposal["target_weights"]
        result[split] = records
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> Path:
    """Write deterministic UTF-8 JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return path


def build_metadata(
    *,
    dataset_path: Path,
    feature_columns: Sequence[str],
    outputs: Mapping[str, Path],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Create audit metadata for generated train/validation files."""

    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": _sha256_file(dataset_path),
        "feature_columns": list(feature_columns),
        "split_field": "target_date",
        "split_boundaries": {
            "train_end": DEFAULT_SPLIT_CONFIG.train_end.isoformat(),
            "validation_end": DEFAULT_SPLIT_CONFIG.validation_end.isoformat(),
            "test_end": DEFAULT_SPLIT_CONFIG.test_end.isoformat(),
        },
        "outputs": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "rows": len(records[name]),
                "first_target_date": records[name][0]["target_date"],
                "last_target_date": records[name][-1]["target_date"],
            }
            for name, path in outputs.items()
        },
        "controls": {
            "teacher": "deterministic_format_bootstrap_v1",
            "purpose": "format_bootstrap_not_alpha",
            "standardizer_fit_split": "train",
            "future_returns_in_prompt": False,
            "test_records_written": False,
            "test_used_for_fit_or_selection": False,
            "human_approval_required": True,
            "automatic_order_submission": False,
        },
    }
