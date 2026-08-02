# KCB prepared tool-error extension

This repository runs the follow-up experiment on failure semantics in
tool-augmented language models. It reuses the same post-QC KCB questions,
model-specific memory labels, five 7-9B models, prompt version, and deterministic
decoding settings. It changes only the prepared tool response:

- original `503 Service Unavailable`;
- `timeout`;
- `permission denied`.

The code never invokes a real tool. Every prompt already contains one complete,
fixed error payload.

## Model indices

| Index | Experiment key | Hugging Face repository | Size | Interface |
|---:|---|---|---:|---|
| 0 | `completion:llama-3.1-8b` | `meta-llama/Llama-3.1-8B` | 8B | completion |
| 1 | `llama-3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | 8B | chat |
| 2 | `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | 7.61B | chat |
| 3 | `gemma-2-9b-it` | `google/gemma-2-9b-it` | 9B | chat |
| 4 | `mistral-7b-instruct-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | chat |

Each command loads exactly one model. Model indices are a run selector, not
experimental conditions. A model can be given as its **index** (`0`..`4`) or as
its **registry key** (for example `llama-3.1-8b-instruct`). Both forms select
the same five fixed models.

## Backends

Every real run picks one inference backend with `--backend`:

| Backend | Model loading | Hardware | Notes |
|---|---|---|---|
| `transformers` | Hugging Face `transformers` | CUDA or CPU | **default**; always available |
| `vllm` | vLLM 0.8.5 | CUDA GPU | KCB-paper serving stack |
| `ollama` | remote Ollama server | any | no local weights needed |
| `lms` | remote LM Studio server | any | no local weights needed |
| `mock` | synthetic answers | none | offline tests only, never report |

Remote backends do not download weights and need no GPU. Local backends
(`transformers`, `vllm`) resolve the model commit, check the cache, and hash the
downloaded weights.

## 1. Offline dry run

Run this first on any machine with Python 3.10+:

```sh
sh run_scripts.sh --dry_run
```

The dry run needs no GPU, model download, network, vLLM, or Hugging Face token.
It performs all unit tests, verifies immutable input hashes, constructs review
and full packets for all five models, proves that the 280-question source set is
shared, runs 36 synthetic responses through the production scorer, tests
resume behavior, and regenerates the analysis.

Synthetic artifacts are stored under `artifacts/dry_run/` and must never be
reported as model results.

## 2. Install the real runtime

Use Linux, Python 3.10 or 3.11, and a CUDA-compatible NVIDIA GPU:

```sh
python3 -m pip install --no-cache-dir -r requirements-runtime.txt
export KCB_HF_HOME=/path/to/persistent/hf-cache
export HF_TOKEN=hf_your_read_token
```

The token is needed only for gated repositories, but the corresponding model
license must also be accepted in the same Hugging Face account. Never commit a
token to this repository.

The published KCB runs used one A100-80GB. This extension still loads one
unquantized float16 7-9B model at a time. Reserve at least 22 GiB of free cache
storage. A 10 GiB disk is insufficient; changing to quantized weights would
change the experiment.

Remote backends (`ollama`, `lms`) skip this section: they only need a running
server. `transformers` needs `transformers` and `torch` installed.

## 3. Real-model smoke test

Run 12 balanced questions and all three errors (36 generations) before the full
experiment:

```sh
sh run_scripts.sh --smoke_test 0
```

`--smoke_test` always builds a review packet (12 questions by default, override
with `--count N`). With the default `transformers` backend it downloads and
loads the full selected model, so it validates actual Hugging Face access,
CUDA, serialization, generation, checkpointing, scoring, and analysis. With a
remote backend it validates connectivity to the server instead.

Inspect `artifacts/smoke/<backend>/<model>/completion.json`. Continue only when
its status is `complete` and the command exits with code 0.

## 4. Full experiment

Run the first model:

```sh
sh run_scripts.sh --full_test 0
```

Then repeat with indices `1`, `2`, `3`, and `4`. Each full model run contains:

- 280 shared questions;
- 3 matched prepared-error prompts per question;
- 840 independent generations.

The complete five-model extension contains 4,200 generations.

After all five individual runs finish, validate and combine them with:

```sh
sh run_scripts.sh --aggregate
```

This aggregates the `transformers` backend and writes
`artifacts/full/transformers/study_summary.json` and
`artifacts/full/transformers/study_group_metrics.csv`. To aggregate a different
backend, pass it back:

```sh
sh run_scripts.sh --aggregate --backend lms
```

To combine **every** backend that has results in one shot:

```sh
sh run_scripts.sh --aggregate all
```

`--aggregate all` scans `artifacts/full/<backend>/` for each backend, aggregates
the ones that have data, and writes a combined report:
`artifacts/full/study_summary.json` and `artifacts/full/study_group_metrics.csv`
(each row carries a `backend` column). A per-backend summary is also written
under each backend's own folder.

Incomplete results are never fatal. If a model is missing or fewer than 4,200
responses are present (some models still running), the aggregator prints
`WARNING: ...` lines to stderr and continues, marking that backend `partial`
instead of `complete`.

To list the mapping at the command line:

```sh
sh run_scripts.sh --list_models
```

If enough persistent disk is available for all checkpoints, this convenience
command runs the models sequentially:

```sh
sh run_scripts.sh --full_test all
```

The compatibility wrapper `eun_scripts.sh` accepts the same arguments.

## 5. Running one model with any backend

The general command form is:

```sh
sh run_scripts.sh MODEL [OPTIONS]
```

`MODEL` may be an index or a registry key. `--run MODEL [OPTIONS]` is the
explicit spelling of the same action. Artifacts are written under
`artifacts/full/<backend>/<model>/` so different backends never overwrite each
other.

### Backend options

```sh
# Ollama server (default host http://localhost:11434)
sh run_scripts.sh 2 --backend ollama \
  --ollama-host http://localhost:11434 \
  --ollama-model qwen2.5:7b

# LM Studio server (default host http://localhost:1234)
sh run_scripts.sh llama-3.1-8b-instruct --backend lms \
  --lms-host http://10.16.98.67:1234 \
  --lms-model llama-3.1-8b-instruct

# Local transformers backend (the default)
sh run_scripts.sh 0 --backend transformers

# Local vLLM backend
sh run_scripts.sh 0 --backend vllm
```

If `--ollama-model` / `--lms-model` are omitted, the server-side model name is
derived from the last path segment of the Hugging Face repository.

### Limiting work (quick tests)

`--limit-tasks N` runs at most `N` prompts. `--skip-conditions` drops whole
error kinds (`service_503`, `timeout`, `permission_denied`), comma-separated.
These are ideal for verifying that a server, prompt, or backend works before
launching a full run:

```sh
# Quick connectivity + generation check against LM Studio:
sh run_scripts.sh llama-3.1-8b-lexi-uncensored-v2 --backend lms \
  --lms-host http://10.16.98.67:1234 \
  --limit-tasks 10

# Skip two of the three error conditions entirely:
sh run_scripts.sh 1 --backend transformers --skip-conditions timeout,service_503
```

`MODEL` must exist in `model_registry.json`: each model needs its own memory
labels, so a custom name such as `llama-3.1-8b-lexi-uncensored-v2` only works
after you register it there (add a key, its `hf_repo`/`memory_labels`, and a
unique `index`). `sh run_scripts.sh --list_models` shows the registered set.

A limited or partially-skipped run writes `"status": "partial"` in
`completion.json` and is validated without the completeness requirement. Partial
results must never be aggregated into the 4,200-response study.

### Forcing a rerun

Responses are appended and resume from the last checkpoint. To ignore existing
results and start over, add `--force`:

```sh
sh run_scripts.sh 1 --backend transformers --force
```

`--force` also lets you switch the backend or conditions on an existing output
directory; without it the runner refuses to touch an output whose immutable run
specification changed.

## Result interpretation

The automatic primary behavior labels are:

- `answers_from_memory`: matches the previously elicited parametric answer;
- `honest_abstention`: explicitly refuses or states that the answer is unknown;
- `other_answer`: neither of the above.

The last category is intentionally not called a hallucination automatically.
Every `other_answer` and every format failure is copied to
`manual_review_queue.jsonl`. The record also states whether the final answer is
correct, allowing unexpected correct answers to be distinguished from
unsupported or incorrect answers.

`analysis.json` reports error-specific rates, memory-correctness strata,
question-bootstrap 95% intervals, matched pairwise contrasts, and behavior
transitions. `group_metrics.csv` is the tabular version.

## Resume and integrity behavior

Responses are appended and fsynced one record at a time. Rerunning the exact
same command skips completed variant IDs. The runner refuses to resume if the
packet, model commit, decoding settings, backend, remote hosts, or conditions
changed. It also refuses duplicate IDs, unknown IDs, incomplete triplets,
prompt-hash mismatches, and score mismatches.

The Hugging Face branch is resolved to an immutable commit before loading.
That commit is passed to both the model and tokenizer. Cached safetensor files
are hashed into `model_weights_manifest.json` before generation.

See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for the methodology and
[EXTERNAL_RUN_CHECKLIST.md](EXTERNAL_RUN_CHECKLIST.md) for the execution
handoff.

## Direct Python commands

The shell interface is preferred, but each stage is independently callable:

```sh
PYTHONPATH=. python3 scripts/preflight.py \
  --runtime --model-key llama-3.1-8b-instruct --backend lms --lms-host http://localhost:1234

PYTHONPATH=. python3 scripts/build_packet.py \
  --model-key llama-3.1-8b-instruct --mode review --count 12 \
  --output packets/llama-review.json

PYTHONPATH=. python3 scripts/run_experiment.py \
  --model-key llama-3.1-8b-instruct \
  --packet packets/llama-review.json \
  --out-dir artifacts/manual-smoke \
  --backend transformers --limit-tasks 10

PYTHONPATH=. python3 scripts/validate_results.py \
  --packet packets/llama-review.json \
  --responses artifacts/manual-smoke/responses.jsonl \
  --allow-partial
```

All paths are resolved from the repository root by the shell runner.
