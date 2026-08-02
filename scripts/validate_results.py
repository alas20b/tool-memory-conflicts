#!/usr/bin/env python3
"""Recompute every stored response label and require a complete triplet set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kcb.analysis import validate_responses
from kcb.io_utils import load_jsonl, read_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="accept an incomplete run (used with --limit-tasks/--skip-conditions)",
    )
    args = parser.parse_args()
    check = validate_responses(
        read_json(args.packet),
        load_jsonl(args.responses),
        require_complete=not args.allow_partial,
    )
    print(json.dumps({**check, "responses_sha256": sha256_file(args.responses)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
