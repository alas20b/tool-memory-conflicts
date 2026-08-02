# Prepared tool-error experiment on open 7–9B models

This repository is ready for an institution that permits only Ollama, LM Studio, or Hugging Face Transformers. It contains no Meta Llama model, makes no real tool calls, and does not need a hosted inference API. Each model receives an already prepared tool response containing one of three errors: HTTP 503, timeout, or permission denied.

The repository deliberately recalibrates each replacement model's closed-book answer before the error experiment. The earlier “model knows the answer” labels cannot be reused because those labels belong to the earlier model checkpoint and execution setup.

## Registered model cohort

| Index | Model family | Parameters | Official/upstream checkpoint | Exact 10 GB-compatible GGUF |
|---:|---|---:|---|---|
| 0 | Qwen 2.5 | 7.61B | `Qwen/Qwen2.5-7B-Instruct` | bartowski Q6_K, 6.25 GB |
| 1 | Mistral 7B | 7B | `mistralai/Mistral-7B-Instruct-v0.3` | bartowski Q6_K, 5.95 GB |
| 2 | IBM Granite 3.3 | 8B | `ibm-granite/granite-3.3-8b-instruct` | IBM Q6_K, 6.71 GB |
| 3 | OLMo 2 | 7B | `allenai/OLMo-2-1124-7B-Instruct` | bartowski Q6_K, 5.99 GB |
| 4 | Yi 1.5 | 9B | `01-ai/Yi-1.5-9B-Chat` | LM Studio Community Q6_K, 7.25 GB |

Every registered upstream model and artifact is Apache-2.0. Exact repository names, filenames, byte counts, SHA256 hashes, and Ollama pull names are in `model_registry.json`. Bartowski and LM Studio Community are artifact converters/distributors; they are not substitute model families.

## First command: model-free dry run

From the repository root:

```sh
sh run_scripts.sh --dry_run
```

Python 3.10 or newer is required. The dry run uses only the Python standard library. It validates all 280 questions, the five-model registry, all prompt triplets, scoring, resume behavior, and fake Ollama/LM Studio API servers. It neither downloads a model nor contacts an inference server.

## Recommended route with a 10 GB storage limit

Use either Ollama or LM Studio, Q6_K, and one model at a time. Do not use the Transformers route under a strict 10 GB disk quota: standard Transformers checkpoints for these 7–9B models are larger than the corresponding GGUF and are dequantized/native PyTorch executions, so they form a different precision cohort.

List exact artifacts:

```sh
sh run_scripts.sh --list_models
```

### Ollama

Example for model 0:

```sh
ollama pull hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q6_K
ollama serve
ollama list
sh run_scripts.sh --probe 0 ollama \
  --server-url http://127.0.0.1:11434 \
  --server-model hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q6_K
```

Copy the server model name exactly from `ollama list`. The probe verifies that the server reports Q6_K and records the Ollama version, tag/digest, model details, parameters, and complete chat template.

### LM Studio

Download the exact file listed in `model_registry.json` with the LM Studio model browser or `lms get`, then load it with a stable identifier and start the server:

```sh
lms get https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/blob/main/Qwen2.5-7B-Instruct-Q6_K.gguf
lms ls
lms load <LM_STUDIO_MODEL_KEY> --identifier qwen25-q6 --context-length 4096 --gpu=auto
lms server start
sh run_scripts.sh --probe 0 lmstudio \
  --server-url http://127.0.0.1:1234 \
  --server-model qwen25-q6
```

The probe checks the `/api/v0/models` record and rejects a non-GGUF or non-Q6_K model. If the server requires authentication, set `LM_API_TOKEN` in the environment; never commit it.

If the original `.gguf` file is accessible, append `--artifact-path /absolute/path/model.gguf` to either probe. This performs an exact SHA256 check. Without it, the profile clearly records `server_metadata_only` rather than claiming file-level verification.

### Transformers (optional separate cohort)

Install the institution's CUDA-compatible PyTorch first, then:

```sh
python3 -m pip install -r requirements-transformers.txt
sh run_scripts.sh --probe 0 transformers
```

The probe resolves mutable Hugging Face `main` to an immutable commit, records weight-file metadata and library/GPU versions, and refuses CPU execution. Store Transformers outputs separately; the aggregator refuses to mix them with Ollama or LM Studio.

## Safe execution sequence for each model

The probe prints a path such as `profiles/ollama/qwen2.5-7b-instruct-….json`. Use that exact path below.

1. Run the small real-model pilot (12 closed-book prompts plus 36 error prompts):

   ```sh
   sh run_scripts.sh --pilot profiles/ollama/<PROFILE>.json
   ```

2. Inspect/bundle the pilot. Only after its format is accepted, run the 280-question closed-book calibration:

   ```sh
   sh run_scripts.sh --calibrate profiles/ollama/<PROFILE>.json
   ```

3. Run the balanced 12-question smoke experiment (36 prompts), then the full experiment (840 prompts):

   ```sh
   sh run_scripts.sh --smoke profiles/ollama/<PROFILE>.json
   sh run_scripts.sh --full profiles/ollama/<PROFILE>.json
   ```

   As a convenience, the last three commands can be run as:

   ```sh
   sh run_scripts.sh --full_test profiles/ollama/<PROFILE>.json
   ```

4. Bundle results for return:

   ```sh
   sh run_scripts.sh --bundle qwen-model0-results
   ```

5. Unload/remove that model only after the result bundle is safe, then download the next index. Repeat indices 0 through 4 one at a time.

6. Once all five full runs from the same backend are present, aggregate them:

   ```sh
   sh run_scripts.sh --aggregate <PROFILE0> <PROFILE1> <PROFILE2> <PROFILE3> <PROFILE4>
   ```

The compatibility wrapper `eun_scripts.sh` is included, but `run_scripts.sh` is the canonical filename.

## Where results appear

- `profiles/`: immutable execution profiles.
- `packets/`: exact generated prompt packets.
- `artifacts/<backend>/<model>/<profile-prefix>/`: raw answers, labels, run specifications, completion hashes, analyses, CSV metrics, manual-review queues, and journals.
- `return_bundles/`: ZIP archives generated for return; model weights and caches are excluded.

Every stage is append-only and resumable. Repeating the identical command skips completed task IDs. Changed prompts, settings, profiles, duplicate IDs, missing responses, or altered automatic scores cause validation to fail instead of silently continuing.

See `OPERATOR_INSTRUCTIONS.md` for the hand-off checklist and `PROTOCOL.md` for the full research design.
