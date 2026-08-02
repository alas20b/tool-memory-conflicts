from __future__ import annotations

import argparse
import json
import sys

from _common import ROOT, path_from_root
from kcb_open.io_utils import read_json
from kcb_open.profiles import create_profile
from kcb_open.registry import load_registry, model_key_from_index


def selector_to_key(value: str) -> str:
    registry = load_registry(ROOT)
    try:
        return model_key_from_index(registry, int(value))
    except ValueError:
        if value in registry:
            return value
        raise ValueError(f"model must be index 0..4 or one of: {', '.join(registry)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the actual run backend and create an immutable profile")
    parser.add_argument("model", help="registry index 0..4 or model key")
    parser.add_argument("backend", choices=("ollama", "lmstudio", "transformers"))
    parser.add_argument("--server-url", default="")
    parser.add_argument("--server-model", default="")
    parser.add_argument("--artifact-path", help="optional local GGUF file for SHA256 verification")
    args = parser.parse_args()
    profile_path = create_profile(
        ROOT, selector_to_key(args.model), args.backend,
        server_url=args.server_url, server_model=args.server_model,
        artifact_path=path_from_root(args.artifact_path) if args.artifact_path else None,
    )
    profile = read_json(profile_path)
    print(json.dumps({
        "status": "ok", "profile_path": str(profile_path),
        "profile_id": profile["profile_id"], "model_key": profile["model_key"],
        "backend": profile["backend"], "weight_verification": profile["weight_verification"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"PROBE FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
