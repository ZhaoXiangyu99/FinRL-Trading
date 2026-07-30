"""Counterfactual half-year rewards for offline Agentic-RL.

For each decision state, the already-observed episode prefix is replayed with
the deterministic format-bootstrap policy.  Each candidate action is then
evaluated by holding that target allocation through the rest of the same
half-year.  Future returns are used only by this reward evaluator and are never
serialized into the model prompt.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.agentic.contracts import parse_and_validate_action_json
from src.environments.qqq_spy_cash_env import QQQSpyCashEnv, RewardConfig
from src.training.data_pipeline import prepare_training_splits


REWARD_TABLE_VERSION = "qqq_spy_half_year_counterfactual_v1"
DEFAULT_REWARD_CONFIG = RewardConfig(
    partial_credit=0.25,
    drawdown_penalty=0.1,
    turnover_regularizer=0.001,
    score_clip=1.0,
)


class HalfYearRewardError(ValueError):
    """Raised when a reward table would violate the split contract."""


def _terminal_info_for_constant_continuation(
    env: QQQSpyCashEnv,
    action_id: int,
) -> Mapping[str, Any]:
    candidate = deepcopy(env)
    terminated = False
    final_info: Mapping[str, Any] | None = None
    while not terminated:
        _, _, terminated, truncated, final_info = candidate.step(action_id)
        if truncated:
            raise RuntimeError("counterfactual episode was truncated")
    if final_info is None:
        raise RuntimeError("counterfactual episode produced no terminal info")
    return final_info


def _records_by_sample_id(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_split: str,
) -> dict[str, Mapping[str, Any]]:
    result = {}
    for record in records:
        if record["split"] != expected_split:
            raise HalfYearRewardError(
                f"unexpected split {record['split']!r} in {expected_split}"
            )
        sample_id = str(record["sample_id"])
        if sample_id in result:
            raise HalfYearRewardError(f"duplicate sample_id: {sample_id}")
        result[sample_id] = record
    return result


def build_half_year_reward_rows(
    dataset: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    trajectory_records: Mapping[str, Sequence[Mapping[str, Any]]],
    transaction_cost_rate: float = 0.001,
    reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> dict[str, list[dict[str, Any]]]:
    """Build train/validation GRPO rows and keep test sealed."""

    splits, _ = prepare_training_splits(
        dataset,
        feature_columns=feature_columns,
    )
    output: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "validation"):
        records = _records_by_sample_id(
            trajectory_records[split_name],
            expected_split=split_name,
        )
        split_frame = splits[split_name]
        rows: list[dict[str, Any]] = []
        for episode_id, episode in split_frame.groupby(
            "episode_id", sort=True
        ):
            env = QQQSpyCashEnv.from_dataset(
                episode,
                feature_columns=feature_columns,
                transaction_cost_rate=transaction_cost_rate,
                reward_config=reward_config,
            )
            env.reset()
            for _, data_row in episode.iterrows():
                feature_date = pd.Timestamp(
                    data_row["feature_date"]
                ).date().isoformat()
                target_date = pd.Timestamp(
                    data_row["target_date"]
                ).date().isoformat()
                sample_id = (
                    f"{split_name}:{feature_date}:{target_date}"
                )
                record = records.get(sample_id)
                if record is None:
                    raise HalfYearRewardError(
                        f"missing trajectory record: {sample_id}"
                    )

                terminal_infos = [
                    _terminal_info_for_constant_continuation(env, action_id)
                    for action_id in range(env.n_actions)
                ]
                rewards = [
                    float(info["terminal_score"])
                    for info in terminal_infos
                ]
                best_reward = max(rewards)
                best_actions = [
                    action_id
                    for action_id, reward in enumerate(rewards)
                    if np.isclose(reward, best_reward, atol=1e-12)
                ]
                rows.append(
                    {
                        "reward_table_version": REWARD_TABLE_VERSION,
                        "sample_id": sample_id,
                        "split": split_name,
                        "episode_id": str(episode_id),
                        "feature_date": feature_date,
                        "target_date": target_date,
                        "prompt": record["prompt"],
                        "action_rewards": rewards,
                        "best_action_ids": best_actions,
                        "best_terminal_score": best_reward,
                        "worst_terminal_score": min(rewards),
                        "reward_dispersion": float(np.std(rewards)),
                        "reward_contract": {
                            "settlement": "full_calendar_half_year",
                            "observed_prefix_policy": (
                                "deterministic_format_bootstrap_v1"
                            ),
                            "candidate_continuation": (
                                "constant_target_weight_to_episode_end"
                            ),
                            "uses_future_returns_in_prompt": False,
                            "uses_future_returns_in_reward": True,
                        },
                    }
                )

                teacher_proposal = parse_and_validate_action_json(
                    record["completion"][0]["content"]
                )
                _, _, terminated, truncated, _ = env.step(
                    teacher_proposal["action_id"]
                )
                if truncated:
                    raise RuntimeError("teacher prefix was truncated")
            if not terminated:
                raise RuntimeError(
                    f"teacher replay did not terminate: {episode_id}"
                )
        output[split_name] = rows
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_reward_jsonl(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return path


def build_reward_metadata(
    *,
    dataset_path: Path,
    outputs: Mapping[str, Path],
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    transaction_cost_rate: float,
    reward_config: RewardConfig,
) -> dict[str, Any]:
    split_metadata = {}
    for split_name, split_rows in rows.items():
        best_actions = Counter(
            action
            for row in split_rows
            for action in row["best_action_ids"]
        )
        split_metadata[split_name] = {
            "path": str(outputs[split_name]),
            "sha256": _sha256(outputs[split_name]),
            "rows": len(split_rows),
            "episodes": len(
                {row["episode_id"] for row in split_rows}
            ),
            "first_target_date": split_rows[0]["target_date"],
            "last_target_date": split_rows[-1]["target_date"],
            "mean_reward_dispersion": float(
                np.mean(
                    [row["reward_dispersion"] for row in split_rows]
                )
            ),
            "best_action_counts": {
                str(key): value
                for key, value in sorted(best_actions.items())
            },
        }
    return {
        "reward_table_version": REWARD_TABLE_VERSION,
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": _sha256(dataset_path),
        "transaction_cost_rate": transaction_cost_rate,
        "reward_config": {
            "partial_credit": reward_config.partial_credit,
            "drawdown_penalty": reward_config.drawdown_penalty,
            "turnover_regularizer": reward_config.turnover_regularizer,
            "score_clip": reward_config.score_clip,
        },
        "splits": split_metadata,
        "controls": {
            "settlement": "full_calendar_half_year",
            "standardizer_fit_split": "train",
            "future_returns_in_prompt": False,
            "future_returns_in_reward": True,
            "test_rows_written": False,
            "test_used_for_training_or_selection": False,
            "automatic_order_submission": False,
        },
    }
