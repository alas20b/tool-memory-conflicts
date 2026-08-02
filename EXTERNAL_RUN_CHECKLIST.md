# External execution checklist

## Before compute is allocated

1. Upload or clone the entire repository, including `data/` and `memory_labels/`.
2. Use Linux with Python 3.10 or 3.11 and one CUDA-capable NVIDIA GPU.
3. Use the paper configuration (one A100-80GB) when available. A 32GB GPU may
   be sufficient for one 7-9B float16 model, but the external operator must
   verify this with the smoke run.
4. Provide at least 22 GiB of free model-cache storage. Ten GiB is insufficient
   for these unquantized checkpoints and quantization would change the experiment.
5. Install `requirements-runtime.txt` without changing versions.
6. Set `KCB_HF_HOME` to persistent storage.
7. Set `HF_TOKEN` and accept the licenses before indices 0, 1, or 3.

## Required command sequence

```sh
sh run_scripts.sh --dry_run
sh run_scripts.sh --list_models
sh run_scripts.sh --smoke_test 0
sh run_scripts.sh --full_test 0
```

After model 0 completes, repeat `--smoke_test N` and `--full_test N` for
indices 1, 2, 3, and 4. `sh run_scripts.sh --full_test all` is available only
when storage can retain all downloaded checkpoints; it still runs them
sequentially.

Do not start a full run if the dry run, runtime preflight, or real smoke run
returns a nonzero status.

## Files to return

Return the complete `artifacts/` directory. For every completed model it must
contain:

- `run_spec.json`;
- `run_meta.json`;
- `journal.jsonl`;
- `responses.jsonl`;
- `model_weights_manifest.json`;
- `analysis.json`;
- `group_metrics.csv`;
- `manual_review_queue.jsonl`;
- `completion.json`.

Also return the generated `packets/` directory so every response can be traced
to its exact prompt.
