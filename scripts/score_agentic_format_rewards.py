#!/usr/bin/env python3
"""Score a generated-format report with validation half-year rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic.reward_evaluation import score_format_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format-report", type=Path, required=True)
    parser.add_argument(
        "--reward-table",
        type=Path,
        default=Path(
            "data/agentic/qqq_spy_half_year_rewards_v1/validation.jsonl"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    format_report = json.loads(
        args.format_report.read_text(encoding="utf-8")
    )
    with args.reward_table.open(encoding="utf-8") as handle:
        reward_rows = [json.loads(line) for line in handle]
    report = score_format_report(format_report, reward_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
