#!/usr/bin/env python3
"""Freeze target-date splits and train-only feature scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.data_pipeline import (
    DEFAULT_SPLIT_CONFIG,
    prepare_training_splits,
    write_preprocessing_artifact,
)


DEFAULT_DATASET = Path(
    "data/market/longbridge/qqq_spy_rl_training_dataset_v1.csv"
)
DEFAULT_METADATA = DEFAULT_DATASET.with_suffix(
    DEFAULT_DATASET.suffix + ".metadata.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/qqq_spy_agent/preprocessing_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare frozen QQQ/SPY train/validation/test splits."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-metadata", type=Path, default=DEFAULT_METADATA
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    splits, standardizer = prepare_training_splits(
        dataset,
        feature_columns=feature_columns,
        split_config=DEFAULT_SPLIT_CONFIG,
    )
    output = write_preprocessing_artifact(
        output_path=args.output,
        dataset_path=args.dataset,
        splits=splits,
        standardizer=standardizer,
        split_config=DEFAULT_SPLIT_CONFIG,
    )
    print(
        "Prepared splits: "
        + ", ".join(
            f"{name}={len(frame)} rows/{frame['episode_id'].nunique()} episodes"
            for name, frame in splits.items()
        )
        + f"; wrote {output}"
    )


if __name__ == "__main__":
    main()
