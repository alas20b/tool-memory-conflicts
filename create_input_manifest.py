#!/usr/bin/env python3
"""Regenerate hashes for immutable experiment inputs."""

from __future__ import annotations

from pathlib import Path

from kcb.io_utils import atomic_write_json, sha256_file, utc_now
from kcb.packets import load_registry


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    registry = load_registry(ROOT)
    paths = {
        Path("model_registry.json"),
        Path("experiment_config.json"),
        Path("data/episodes_v1_full_near_pv0_poolv3.json"),
        *(Path(model["memory_labels"]) for model in registry.values()),
    }
    files = {
        str(path).replace("\\", "/"): {
            "bytes": (ROOT / path).stat().st_size,
            "sha256": sha256_file(ROOT / path),
        }
        for path in sorted(paths, key=str)
    }
    atomic_write_json(
        ROOT / "input_manifest.json",
        {
            "schema_version": "kcb-input-manifest-v1",
            "generated_at": utc_now(),
            "files": files,
        },
    )
    print(f"wrote input_manifest.json for {len(files)} immutable inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
