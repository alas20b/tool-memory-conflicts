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
experimental conditions.

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

## 3. Real-model smoke test

Run 12 balanced questions and all three errors (36 generations) before the full
experiment:

```sh
sh run_scripts.sh --smoke_test 0
```

This test downloads and loads the full selected model, so it validates actual
Hugging Face access, CUDA, vLLM, chat/completion serialization, generation,
checkpointing, scoring, and analysis. It is computation-light, not
weight-download-light.

Inspect `artifacts/smoke/<model>/completion.json`. Continue only when its status
is `complete` and the command exits with code 0.

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

This requires exactly 4,200 validated responses and writes
`artifacts/full/study_summary.json` and
`artifacts/full/study_group_metrics.csv`.

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
packet, model commit, or decoding settings changed. It also refuses duplicate
IDs, unknown IDs, incomplete triplets, prompt-hash mismatches, and score
mismatches.

The Hugging Face branch is resolved to an immutable commit before loading.
That commit is passed to both the model and tokenizer. Cached safetensor files
are hashed into `model_weights_manifest.json` before generation.

See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for the methodology and
[EXTERNAL_RUN_CHECKLIST.md](EXTERNAL_RUN_CHECKLIST.md) for the execution
handoff.

## Direct Python commands

The shell interface is preferred, but each stage is independently callable:

```sh
PYTHONPATH=. python3 scripts/preflight.py
PYTHONPATH=. python3 scripts/build_packet.py \
  --model-key llama-3.1-8b-instruct --mode review --count 12 \
  --output packets/llama-review.json
PYTHONPATH=. python3 scripts/run_experiment.py \
  --model-key llama-3.1-8b-instruct \
  --packet packets/llama-review.json \
  --out-dir artifacts/manual-smoke --backend vllm
```

All paths are resolved from the repository root by the shell runner.
