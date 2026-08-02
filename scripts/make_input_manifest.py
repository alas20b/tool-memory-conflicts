from __future__ import annotations

from _common import ROOT
from kcb_open.io_utils import atomic_write_json, sha256_file, utc_now


EXCLUDED_PARTS = {
    ".git", "__pycache__", ".pytest_cache", ".model-cache", "return_bundles",
}
EXCLUDED_TOP_LEVEL = {"artifacts", "packets", "profiles"}
EXCLUDED_FILES = {"input_manifest.json"}

files = []
for path in sorted(ROOT.rglob("*")):
    relative = path.relative_to(ROOT)
    if not path.is_file() or relative.name in EXCLUDED_FILES:
        continue
    if relative.parts[0] in EXCLUDED_TOP_LEVEL or any(part in EXCLUDED_PARTS for part in relative.parts):
        continue
    if path.suffix.lower() == ".zip" or path.name.endswith((".pyc", ".pyo")):
        continue
    files.append({
        "path": relative.as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
    })

atomic_write_json(ROOT / "input_manifest.json", {
    "schema_version": "kcb-open-input-manifest-v1",
    "created_at": utc_now(),
    "file_count": len(files),
    "note": "Dynamic profiles, packets, artifacts, caches, and this non-self-hashing manifest are excluded.",
    "files": files,
})
print(f"Wrote {len(files)} entries to {ROOT / 'input_manifest.json'}")
