from __future__ import annotations

import argparse
import json
import sys

from _common import ROOT, path_from_root
from kcb_open.validation import validate_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a completed stage without model inference")
    parser.add_argument("stage", choices=("pilot", "calibrate", "smoke", "full"))
    parser.add_argument("profile")
    args = parser.parse_args()
    result = validate_stage(ROOT, path_from_root(args.profile), args.stage)
    print(json.dumps({"status": "valid", **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"VALIDATION FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
