from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from _common import ROOT
from kcb_open.io_utils import atomic_write_json, sha256_file, utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a return archive without model weights or caches")
    parser.add_argument("--name", default="kcb-results")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.name):
        raise ValueError("bundle name must be 1-80 safe filename characters")
    destination = ROOT / "return_bundles"
    destination.mkdir(exist_ok=True)
    staging = (destination / f".{args.name}-staging").resolve()
    if staging.parent != destination.resolve():
        raise RuntimeError("unsafe staging directory")
    archive_path = (destination / f"{args.name}.zip").resolve()
    if archive_path.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {archive_path}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for name in ("artifacts", "profiles", "packets", "model_registry.json", "experiment_config.json", "input_manifest.json"):
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, staging / name)
        elif source.exists():
            shutil.copy2(source, staging / name)
    manifest = {"created_at": utc_now(), "files": []}
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            manifest["files"].append({
                "path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    atomic_write_json(staging / "RETURN_MANIFEST.json", manifest)
    archive = shutil.make_archive(str(destination / args.name), "zip", root_dir=staging)
    shutil.rmtree(staging)
    print(json.dumps({"status": "complete", "archive": archive, "sha256": sha256_file(Path(archive))}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
