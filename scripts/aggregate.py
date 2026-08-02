from __future__ import annotations

import argparse
import json
import sys

from _common import ROOT, path_from_root
from kcb_open.aggregate import aggregate_full_runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate one complete five-model backend cohort")
    parser.add_argument("profiles", nargs=5, help="exactly five profile JSON paths")
    args = parser.parse_args()
    output = aggregate_full_runs(ROOT, [path_from_root(value) for value in args.profiles])
    print(json.dumps({"status": "complete", "aggregate_directory": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"AGGREGATION FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
