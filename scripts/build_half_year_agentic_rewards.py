#!/usr/bin/env python3
"""Build counterfactual half-year reward tables for Qwen GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic.half_year_rewards import (
    DEFAULT_REWARD_CONFIG,
    build_half_year_reward_rows,
    build_reward_metadata,
    write_reward_jsonl,
)


DEFAULT_DATASET = Path(
    "data/market/longbridge/qqq_spy_rl_training_dataset_v1.csv"
)
DEFAULT_DATASET_METADATA = DEFAULT_DATASET.with_suffix(
    DEFAULT_DATASET.suffix + ".metadata.json"
)
DEFAULT_TRAJECTORY_DIR = Path("data/agentic/qqq_spy_qwen_sft_v1")
DEFAULT_OUTPUT_DIR = Path(
    "data/agentic/qqq_spy_half_year_rewards_v1"
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=DEFAULT_DATASET_METADATA,
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=DEFAULT_TRAJECTORY_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    dataset_metadata = json.loads(
        args.dataset_metadata.read_text(encoding="utf-8")
    )
    dataset = pd.read_csv(
        args.dataset,
        parse_dates=["feature_date", "target_date"],
    )
    trajectory_records = {
        split: _read_jsonl(args.trajectory_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    rows = build_half_year_reward_rows(
        dataset,
        feature_columns=dataset_metadata["feature_columns"],
        trajectory_records=trajectory_records,
        transaction_cost_rate=0.001,
        reward_config=DEFAULT_REWARD_CONFIG,
    )
    outputs = {
        split: write_reward_jsonl(
            split_rows,
            args.output_dir / f"{split}.jsonl",
        )
        for split, split_rows in rows.items()
    }
    metadata = build_reward_metadata(
        dataset_path=args.dataset,
        outputs=outputs,
        rows=rows,
        transaction_cost_rate=0.001,
        reward_config=DEFAULT_REWARD_CONFIG,
    )
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        ", ".join(f"{name}={len(value)}" for name, value in rows.items())
        + f"; wrote {metadata_path}"
    )


if __name__ == "__main__":
    main()
