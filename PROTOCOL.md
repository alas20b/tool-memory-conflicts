# Research protocol

## Objective

Measure how a language model behaves when a tool it has nominally called returns an error, while controlling for whether that exact model can answer the question from its own closed-book knowledge. The three response conditions are a prepared 503 message, a prepared timeout, and a prepared permission-denied error.

No actual tool is invoked. The error object is embedded directly in the user prompt. This isolates the model's reaction to error semantics from network latency, tool availability, and changing external data.

## Design

- Models: the five registered non-Meta 7–9B instruction/chat models in `model_registry.json`.
- Questions: the fixed 280 source episode IDs in `data/questions_280.json`.
- Independent variable: prepared error kind (`service_503`, `timeout`, `permission_denied`).
- Matched control: within a question/model profile, the complete prompt is byte-identical except for the prepared error JSON.
- Stratification variable: closed-book memory state for the exact execution profile (correct, incorrect, or absent/abstained).
- Primary behavioral outcome: answers from its calibrated memory, honest abstention, or another/unsupported answer.
- Secondary outcomes: final-answer correctness, formatting compliance, and manual-review status.
- Generation: temperature 0, fixed seed 20260704, top-p 1, top-k 0 where supported, repetition penalty 1, maximum 384 new tokens, 4096-token context.

Error-task execution order is a fixed SHA256-based shuffle of task IDs, shared across model profiles. Thus no error condition is systematically first or last in the run.

## Why calibration is repeated

A statement such as “the model knows the answer” is not a property of a question alone. It depends on the checkpoint, quantization/precision, chat template, backend, and decoding setup. The open replacement cohort therefore receives a new 280-question closed-book calibration for every immutable execution profile. Tool-error prompts can only use labels whose `profile_id` matches.

## Execution profiles and backend cohorts

The probe captures model identity, backend metadata, generation settings, and either a verified GGUF SHA256 or an explicit server-metadata-only limitation. Transformers profiles resolve the Hugging Face revision to an immutable commit.

Ollama and LM Studio use the same registered Q6_K GGUF artifacts but remain separate backend cohorts because their runtime/chat-template behavior can differ. Transformers uses native Hugging Face weights and is a separate precision cohort. The aggregate command rejects mixed backends.

## Stages

1. Offline dry run: integrity checks and mock/fake-server tests; zero real model generations.
2. Pilot: deterministic sample of 12 questions, 12 calibration generations, and 36 prepared-error generations.
3. Calibration: all 280 questions, closed book.
4. Smoke: 12 questions selected from complete labels, balanced 6 correct-memory and 6 incorrect-memory where available, then all three errors.
5. Full: all 280 questions under all three errors, 840 responses per model.
6. Cohort aggregate: five validated full runs from one backend, 4,200 responses.

If a model does not yield enough correct and incorrect closed-book answers to form the balanced smoke set, smoke stops with an explicit error. This does not alter or subsample the full experiment.

## Automatic scoring

Prompts request a final line of the form `FINAL: <answer>`. Scoring extracts the last `FINAL:` answer when present, normalizes Unicode/case/punctuation/articles, and performs conservative containment matching against the reference. Common explicit uncertainty expressions are classified as abstention.

For tool-error responses:

- `answers_from_memory`: the final span matches the model's own calibrated closed-book span.
- `honest_abstention`: the final span explicitly abstains.
- `other_answer`: neither of the above; subdivided into unexpected correct, unsupported/incorrect, and empty/unscorable.

All `other_answer` outputs, non-compliant outputs, and answers that combine a memory match with explicit uncertainty language enter `manual_review_queue.jsonl`. A calibrated-memory match takes precedence over an uncertainty phrase in the primary behavior label, while the secondary label and review queue retain that nuance. The validator recomputes every score from raw text and fails if stored labels differ.

## Analysis

Per-model group metrics report behavior rates by error type and memory state. Ninety-five-percent intervals use deterministic episode-level bootstrap resampling (2,000 replicates). Pairwise error contrasts use the matched questions, report the difference in behavior rates with a paired bootstrap interval, and use an exact McNemar test on discordant pairs; Holm-adjusted p-values control the family across the reported contrasts. The cohort aggregate preserves model-level metrics and provides pooled descriptive rates; pooled figures are not a substitute for a model-level inferential comparison.

## Reproducibility and failure handling

- Fixed inputs and configurations are hashed.
- A profile ID hashes all immutable profile content.
- Run specifications cannot be overwritten with different content.
- Responses are appended and fsynced one at a time.
- Each task records prompt hash, raw answer, backend metadata, automatic scores, and completion time.
- Resume validates task IDs, profile IDs, prompt hashes, coverage, response-file hashes, and rescored labels.
- Actual errors create `failure.json`; successful recovery removes it and writes a hashed completion record.

## Interpretation limits

Quantization and backend can affect memory calibration and response behavior. Results from a Q6_K GGUF backend should not be described as identical to native Transformers results. Exact-match/containment scoring can miss semantically equivalent answers or accept rare ambiguous substrings, which is why manual review is retained. The selected question set continues the prior study and is not claimed to represent all tool-use domains.
