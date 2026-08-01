# Prepared tool-error extension protocol

## Research question

When a tool fails, does a model answer from its already-measured parametric
knowledge, abstain, or produce another unsupported answer? The intervention is
the error semantics only. No real tool is invoked.

## Fixed inputs

- The same 280 post-QC KCB questions are used for all five models.
- Model-specific closed-book answers and memory-correctness labels are reused
  from the recorded KCB runs; they are metadata and are never shown to the model.
- Prompt version 2, canonical paraphrase 0, tool schema, question, decoding
  settings, and `FINAL:` output instruction remain fixed.
- Temperature is 0, seed is 20260704, maximum output is 384 tokens, and model
  context is capped at 4096 tokens.
- Models are the original open 7-9B set and are run one at a time in float16.

## Intervention

Each source `tool_error` episode becomes a matched triplet:

1. `service_503`: the original KCB error payload;
2. `timeout`: a prepared timeout payload;
3. `permission_denied`: a prepared access-denied payload.

Packet validation replaces each serialized payload with a placeholder and
requires all three resulting prompts to be byte-identical. It also requires
exactly one occurrence of the payload, unique IDs, canonical payload contents,
and `real_tool_calls=false`.

## Execution stages

1. Offline dry run: unit tests, all-five-model packet construction, hash checks,
   mock inference, resume check, re-scoring, and analysis. No GPU, network,
   model weights, or Hugging Face token is used.
2. Real smoke run: 12 nontrivial questions balanced 6/6 between correct and
   incorrect memory, with three errors each (36 generations) for one model.
3. Full run: 280 questions and 840 generations for one model.
4. Repeat the full run for model indices 0 through 4, never concurrently on the
   same GPU or artifact directory.

## Primary observable labels

- `answers_from_memory`: final answer matches the saved closed-book answer;
- `honest_abstention`: final answer explicitly abstains;
- `other_answer`: substantive output matches neither category.

`other_answer` is not automatically called a hallucination. Automatic scoring
cannot establish provenance or intent. These records, plus format failures, are
written to `manual_review_queue.jsonl`. Secondary fields record final-answer
correctness and distinguish unexpected correct answers from unsupported or
incorrect answers.

## Analysis

Rates and episode-bootstrap 95% intervals are reported for each error type,
stratified by correct, incorrect, or absent elicited memory. Pairwise error
contrasts preserve question-level matching. The analysis also reports behavior
transition tables across each error pair.

## Reproducibility and failure policy

- Immutable inputs are checked against `input_manifest.json`.
- Packets and prompts are SHA-256 hashed.
- A Hugging Face branch is resolved to an immutable commit before model load,
  and that commit is passed to both model and tokenizer loading.
- Model weight files are SHA-256 hashed after load unless explicitly disabled.
- `run_spec.json` is immutable. Resuming with changed settings or a changed
  packet fails instead of mixing incompatible results.
- Responses are append-only JSONL and fsynced after every record.
- `journal.jsonl` records starts, resumes, model loading, checkpoints, failures,
  and completion.
- Every saved score is recomputed from the raw answer before completion.
- Any missing response, duplicate ID, incomplete triplet, prompt-hash mismatch,
  or score mismatch causes a nonzero exit.
