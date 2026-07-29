#!/usr/bin/env python3
"""Build point-in-time QQQ/SPY market features from Longbridge daily bars."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.qqq_spy_features import (
    MarketFeatureConfig,
    build_market_features_from_csv,
    write_feature_dataset,
)


DEFAULT_INPUT = Path(
    "data/market/longbridge/qqq_spy_daily_forward_adjusted.csv"
)
DEFAULT_OUTPUT = Path(
    "data/market/longbridge/qqq_spy_daily_features_v1.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build backward-looking QQQ/SPY market features."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MarketFeatureConfig()
    features = build_market_features_from_csv(args.input, config)
    output, metadata = write_feature_dataset(
        features,
        output_path=args.output,
        source_path=args.input,
        config=config,
    )
    print(
        f"Wrote {len(features)} rows x {features.shape[1]} features "
        f"to {output} and {metadata}"
    )


if __name__ == "__main__":
    main()
