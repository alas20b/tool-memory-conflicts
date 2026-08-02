#!/usr/bin/env python3
"""Build one deterministic model-specific review or full packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kcb.io_utils import atomic_write_json, sha256_file
from kcb.packets import build_packet, validate_packet


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--mode", choices=("review", "full"), required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packet = build_packet(ROOT, args.model_key, args.mode, args.count, args.seed)
    check = validate_packet(packet, expected_model_key=args.model_key)
    atomic_write_json(args.output, packet)
    print(json.dumps({**check, "path": str(args.output), "file_sha256": sha256_file(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
