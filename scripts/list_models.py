from __future__ import annotations

from _common import ROOT
from kcb_open.registry import load_registry, model_key_from_index


registry = load_registry(ROOT)
for index in range(5):
    key = model_key_from_index(registry, index)
    model = registry[key]
    gguf = model["gguf"]
    print(f"[{index}] {key}")
    print(f"    upstream: {model['upstream_hf_repo']} ({model['parameter_count']}, {model['license']})")
    print(f"    GGUF: {gguf['repo']}/{gguf['filename']} ({gguf['size_bytes']/1_000_000_000:.2f} GB)")
    print(f"    Ollama: ollama pull {gguf['ollama_pull']}")
    print(f"    LM Studio: lms get https://huggingface.co/{gguf['repo']}/blob/main/{gguf['filename']}")
