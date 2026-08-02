# Instructions for the person running the experiment

Please run the repository from its top-level folder. Do not edit `model_registry.json`, `experiment_config.json`, `data/questions_280.json`, or an execution profile after it has been created.

## Before any model run

1. Record the repository commit: `git rev-parse HEAD`.
2. Run `sh run_scripts.sh --dry_run` and save the complete terminal output.
3. Confirm that there is at least 8 GB free for the largest registered Q6_K file plus normal runtime overhead. Under a 10 GB model-storage quota, keep only one model file at a time.
4. Choose one backend cohort for the main study: Ollama or LM Studio. Do not alternate backends between models.
5. Download exactly the Q6_K filename registered for the selected model index. Do not accept an application's default Q4 quantization.
6. Load the model with a 4096-token context and start the local server.
7. Run `--probe`; copy the exact profile path printed by the command.

## Required pilot to return first

```sh
sh run_scripts.sh --pilot <PROFILE_PATH>
sh run_scripts.sh --bundle pilot-model-<INDEX>
```

Return the ZIP from `return_bundles/`, the terminal output, and the Git commit. The pilot makes 48 generations in total: 12 closed-book calibration prompts and the same 12 questions under three prepared errors (36 prompts). It calls no tool.

## Full run after pilot approval

```sh
sh run_scripts.sh --calibrate <PROFILE_PATH>
sh run_scripts.sh --smoke <PROFILE_PATH>
sh run_scripts.sh --full <PROFILE_PATH>
sh run_scripts.sh --bundle full-model-<INDEX>
```

Expected completed counts for one model:

| Stage | Closed-book generations | Error generations |
|---|---:|---:|
| Pilot | 12 | 36 |
| Calibration | 280 | 0 |
| Smoke | 0 | 36 |
| Full | 0 | 840 |

The full error stage is 280 questions × 3 prepared error messages. A completed five-model cohort contains 4,200 full-stage error responses.

## If a command stops

Do not delete or edit `responses.jsonl`. Save the error message, correct only the external cause (for example, restart the same server and reload the same model identifier), then repeat exactly the same command. The runner validates every existing task and resumes at the first missing one.

If the model artifact, quantization, server identifier, server version, chat template, or generation configuration changes, run a new `--probe` and start a new profile. Never reuse calibration labels from another profile.

## Storage cleanup

First create and copy the return bundle to safe storage. Then unload and remove only the current model through Ollama or LM Studio before downloading the next model. Never delete `profiles/`, `packets/`, or `artifacts/` between model runs.

## Final return checklist

- Repository commit hash.
- Dry-run terminal log.
- Five profile JSON files from one backend.
- Five complete full-result bundles, or one final bundle containing them.
- Confirmation that each model used Q6_K.
- Exact server model identifiers.
- Any interruption/error logs.
- Aggregate output produced by `--aggregate`.
