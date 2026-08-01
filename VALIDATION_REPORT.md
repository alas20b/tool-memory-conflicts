# Validation report

Validation date: 2026-08-01

## Passed offline checks

- Six unit/integration tests pass.
- The exact handoff command `sh run_scripts.sh --dry_run` exits with status 0.
- All five full packets contain the same 280 source episode IDs.
- Every full packet contains 840 prompts: 280 questions times three errors.
- Every review packet contains 12 questions and 36 prompts.
- Review packets exclude absent memory and gold-in-question cases and are
  balanced between six correct-memory and six incorrect-memory questions.
- Packet validation proves that only the serialized error payload changes
  inside each matched triplet.
- Immutable source files pass size and SHA-256 verification.
- The synthetic runner writes 36 responses, exercises all three primary
  behavior labels, resumes without duplication, re-scores every answer, and
  writes a complete analysis artifact.
- Shell syntax and all Python modules compile successfully.
- No real tool call occurs in packet construction, dry inference, scoring, or
  analysis.

## Full-packet memory strata

| Model index | Correct | Incorrect | Absent | Total |
|---:|---:|---:|---:|---:|
| 0 | 147 | 130 | 3 | 280 |
| 1 | 143 | 137 | 0 | 280 |
| 2 | 110 | 160 | 10 | 280 |
| 3 | 137 | 112 | 31 | 280 |
| 4 | 129 | 143 | 8 | 280 |

Absent memory remains a separate analysis stratum and is never relabeled as
incorrect memory.

## Deliberately unexecuted locally

GPU inference was not run locally because the required Linux CUDA/vLLM
environment and model weights are external. The repository therefore requires
the real 36-generation smoke test before any full run. Runtime preflight checks
the exact NumPy and vLLM versions, CUDA visibility, Hugging Face access, and
available cache storage before model loading.
