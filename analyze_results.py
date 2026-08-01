#!/usr/bin/env python3
"""Regenerate analysis files from a completed response artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kcb.analysis import write_analysis
from kcb.io_utils import load_jsonl, read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()
    analysis = write_analysis(
        args.out_dir,
        read_json(args.packet),
        load_jsonl(args.responses),
        seed=args.seed,
    )
    print(json.dumps({
        "model_key": analysis["model_key"],
        "n_responses": analysis["validation"]["n_responses"],
        "manual_review_count": analysis["manual_review_count"],
        "output": str(args.out_dir / "analysis.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
