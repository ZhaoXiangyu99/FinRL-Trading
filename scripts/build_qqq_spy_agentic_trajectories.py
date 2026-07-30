#!/usr/bin/env python3
"""Build leakage-safe Qwen SFT train and validation trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic.trajectory_dataset import (
    build_agentic_trajectories,
    build_metadata,
    write_jsonl,
)


DEFAULT_DATASET = Path(
    "data/market/longbridge/qqq_spy_rl_training_dataset_v1.csv"
)
DEFAULT_METADATA = DEFAULT_DATASET.with_suffix(
    DEFAULT_DATASET.suffix + ".metadata.json"
)
DEFAULT_OUTPUT_DIR = Path("data/agentic/qqq_spy_qwen_sft_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-metadata", type=Path, default=DEFAULT_METADATA
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(
        args.dataset_metadata.read_text(encoding="utf-8")
    )
    dataset = pd.read_csv(
        args.dataset,
        parse_dates=["feature_date", "target_date"],
    )
    records = build_agentic_trajectories(
        dataset,
        feature_columns=metadata["feature_columns"],
    )
    outputs = {
        name: write_jsonl(rows, args.output_dir / f"{name}.jsonl")
        for name, rows in records.items()
    }
    audit = build_metadata(
        dataset_path=args.dataset,
        feature_columns=metadata["feature_columns"],
        outputs=outputs,
        records=records,
    )
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        ", ".join(f"{name}={len(rows)}" for name, rows in records.items())
        + f"; wrote {metadata_path}"
    )


if __name__ == "__main__":
    main()
