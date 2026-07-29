#!/usr/bin/env python3
"""Build the complete-half-year QQQ/SPY RL training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.qqq_spy_rl_dataset import (
    RLDatasetConfig,
    build_rl_dataset_from_csv,
    write_rl_dataset,
)


DEFAULT_MARKET = Path(
    "data/market/longbridge/qqq_spy_daily_forward_adjusted.csv"
)
DEFAULT_FEATURES = Path(
    "data/market/longbridge/qqq_spy_daily_features_v1.csv"
)
DEFAULT_OUTPUT = Path(
    "data/market/longbridge/qqq_spy_rl_training_dataset_v1.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align close-observed features with next-open execution and "
            "retain only complete calendar half-year episodes."
        )
    )
    parser.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-incomplete-episodes",
        action="store_true",
        help="Include partial first/current half-years for diagnostics only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RLDatasetConfig(
        include_incomplete_episodes=args.include_incomplete_episodes
    )
    dataset = build_rl_dataset_from_csv(
        market_path=args.market,
        feature_path=args.features,
        config=config,
    )
    output, metadata = write_rl_dataset(
        dataset,
        output_path=args.output,
        market_path=args.market,
        feature_path=args.features,
        config=config,
    )
    print(
        f"Wrote {len(dataset)} transitions across "
        f"{dataset['episode_id'].nunique()} half-year episodes "
        f"to {output} and {metadata}"
    )


if __name__ == "__main__":
    main()
