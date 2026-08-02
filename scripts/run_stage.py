from __future__ import annotations

import argparse
import json
import os
import sys

from _common import ROOT, path_from_root
from kcb_open.runner import run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one resumable experiment stage")
    parser.add_argument("stage", choices=("pilot", "calibrate", "smoke", "full"))
    parser.add_argument("profile", help="profile JSON created by scripts/probe.py")
    parser.add_argument(
        "--cache-dir", default=os.environ.get("KCB_MODEL_CACHE", str(ROOT / ".model-cache")),
        help="Transformers cache (unused by Ollama/LM Studio)",
    )
    args = parser.parse_args()
    result = run_stage(
        ROOT, path_from_root(args.profile), args.stage, cache_dir=path_from_root(args.cache_dir)
    )
    print(json.dumps({"status": "complete", **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"RUN FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        print("The run is resumable. Correct the cause, then repeat the identical command.", file=sys.stderr)
        sys.exit(2)
