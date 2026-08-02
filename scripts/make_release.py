from __future__ import annotations

import argparse
import json
import stat
import sys
import zipfile
from pathlib import Path

from _common import ROOT
from kcb_open.io_utils import sha256_file


EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".model-cache", "return_bundles"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic source-only release ZIP")
    parser.add_argument("--output", default=str(ROOT.parent / "kcb_open_backends_study_ready.zip"))
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing release: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in (".pyc", ".pyo", ".zip"):
            continue
        files.append((path, relative))
    prefix = ROOT.name
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, relative in files:
            name = f"{prefix}/{relative.as_posix()}"
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if path.suffix == ".sh" else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, path.read_bytes())
    print(json.dumps({
        "status": "complete", "archive": str(output), "file_count": len(files),
        "size_bytes": output.stat().st_size, "sha256": sha256_file(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
