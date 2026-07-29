#!/usr/bin/env python3
"""Evaluate deterministic target-weight policies on frozen splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environments.qqq_spy_cash_env import RewardConfig
from src.evaluation.policy_baselines import (
    evaluate_constant_policies,
    summarize_policy_results,
    write_baseline_results,
)
from src.training.data_pipeline import prepare_training_splits


DEFAULT_DATASET = Path(
    "data/market/longbridge/qqq_spy_rl_training_dataset_v1.csv"
)
DEFAULT_METADATA = DEFAULT_DATASET.with_suffix(
    DEFAULT_DATASET.suffix + ".metadata.json"
)
DEFAULT_OUTPUT = Path("artifacts/qqq_spy_agent/baselines_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed QQQ/SPY/cash target-weight policies."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-metadata", type=Path, default=DEFAULT_METADATA
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = pd.read_csv(
        args.dataset,
        parse_dates=["feature_date", "target_date"],
    )
    metadata = json.loads(
        args.dataset_metadata.read_text(encoding="utf-8")
    )
    feature_columns = metadata["feature_columns"]
    splits, _ = prepare_training_splits(
        dataset, feature_columns=feature_columns
    )
    reward_config = RewardConfig(
        partial_credit=0.25,
        drawdown_penalty=0.10,
        turnover_regularizer=0.001,
        score_clip=1.0,
    )
    results = pd.concat(
        [
            evaluate_constant_policies(
                frame,
                feature_columns=feature_columns,
                transaction_cost_rate=0.001,
                reward_config=reward_config,
                split_name=split_name,
            )
            for split_name, frame in splits.items()
        ],
        ignore_index=True,
    )
    summary = summarize_policy_results(results)
    episode_path, summary_path = write_baseline_results(
        episode_results=results,
        summary=summary,
        output_directory=args.output_directory,
    )
    print(summary.to_string(index=False))
    print(f"Wrote {episode_path} and {summary_path}")


if __name__ == "__main__":
    main()
