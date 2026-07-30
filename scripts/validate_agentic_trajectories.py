#!/usr/bin/env python3
"""Validate JSONL contracts without loading Qwen or touching the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic.contracts import validate_trajectory_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    total = 0
    counts: dict[str, int] = {}
    for path in args.paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                    validate_trajectory_record(record)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise SystemExit(
                        f"{path}:{line_number}: invalid: {exc}"
                    ) from exc
                total += 1
                counts[record["split"]] = counts.get(record["split"], 0) + 1
    print(json.dumps({"valid_records": total, "splits": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
