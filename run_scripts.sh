#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-python3}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable '$PYTHON_BIN' was not found." >&2
  echo "Set PYTHON_BIN to a Python 3.10+ executable." >&2
  exit 2
fi

model_key_for_index() {
  case "$1" in
    0) printf '%s\n' 'completion:llama-3.1-8b' ;;
    1) printf '%s\n' 'llama-3.1-8b-instruct' ;;
    2) printf '%s\n' 'qwen2.5-7b-instruct' ;;
    3) printf '%s\n' 'gemma-2-9b-it' ;;
    4) printf '%s\n' 'mistral-7b-instruct-v0.3' ;;
    *) echo "ERROR: model index must be 0, 1, 2, 3, or 4." >&2; exit 2 ;;
  esac
}

slug_for_index() {
  case "$1" in
    0) printf '%s\n' 'base-llama-3.1-8b' ;;
    1) printf '%s\n' 'llama-3.1-8b-instruct' ;;
    2) printf '%s\n' 'qwen2.5-7b-instruct' ;;
    3) printf '%s\n' 'gemma-2-9b-it' ;;
    4) printf '%s\n' 'mistral-7b-instruct-v0.3' ;;
    *) echo "ERROR: model index must be 0, 1, 2, 3, or 4." >&2; exit 2 ;;
  esac
}

list_models() {
  echo '0  completion:llama-3.1-8b       meta-llama/Llama-3.1-8B'
  echo '1  llama-3.1-8b-instruct          meta-llama/Llama-3.1-8B-Instruct'
  echo '2  qwen2.5-7b-instruct            Qwen/Qwen2.5-7B-Instruct'
  echo '3  gemma-2-9b-it                  google/gemma-2-9b-it'
  echo '4  mistral-7b-instruct-v0.3       mistralai/Mistral-7B-Instruct-v0.3'
}

real_run() {
  mode=$1
  index=$2
  model_key=$(model_key_for_index "$index")
  slug=$(slug_for_index "$index")
  packet="packets/${slug}-${mode}.json"
  out_dir="artifacts/${mode}/${slug}"

  "$PYTHON_BIN" scripts/preflight.py --runtime --model-key "$model_key"
  if [ "$mode" = 'smoke' ]; then
    "$PYTHON_BIN" scripts/build_packet.py \
      --model-key "$model_key" --mode review --count 12 --output "$packet"
  else
    "$PYTHON_BIN" scripts/build_packet.py \
      --model-key "$model_key" --mode full --output "$packet"
  fi
  "$PYTHON_BIN" scripts/run_experiment.py \
    --model-key "$model_key" --packet "$packet" --out-dir "$out_dir" --backend vllm
  "$PYTHON_BIN" scripts/validate_results.py \
    --packet "$packet" --responses "$out_dir/responses.jsonl"
}

usage() {
  cat <<'EOF'
Usage:
  sh run_scripts.sh --dry_run
  sh run_scripts.sh --smoke_test MODEL_INDEX
  sh run_scripts.sh --full_test MODEL_INDEX
  sh run_scripts.sh --full_test all
  sh run_scripts.sh --preflight MODEL_INDEX
  sh run_scripts.sh --aggregate
  sh run_scripts.sh --list_models

MODEL_INDEX is 0..4. Each real invocation loads exactly one model.
EOF
}

case "${1:-}" in
  --dry_run)
    "$PYTHON_BIN" -m unittest discover -s tests -v
    "$PYTHON_BIN" scripts/preflight.py
    "$PYTHON_BIN" scripts/build_packet.py \
      --model-key 'llama-3.1-8b-instruct' --mode review --count 12 \
      --output 'packets/dry-run-review.json'
    "$PYTHON_BIN" scripts/run_experiment.py \
      --model-key 'llama-3.1-8b-instruct' \
      --packet 'packets/dry-run-review.json' \
      --out-dir 'artifacts/dry_run' --backend mock --skip-weight-hash
    "$PYTHON_BIN" scripts/validate_results.py \
      --packet 'packets/dry-run-review.json' \
      --responses 'artifacts/dry_run/responses.jsonl'
    ;;
  --smoke_test)
    [ "$#" -eq 2 ] || { usage >&2; exit 2; }
    real_run smoke "$2"
    ;;
  --full_test)
    [ "$#" -eq 2 ] || { usage >&2; exit 2; }
    if [ "$2" = 'all' ]; then
      for index in 0 1 2 3 4; do
        real_run full "$index"
      done
      "$PYTHON_BIN" scripts/aggregate_study.py
    else
      real_run full "$2"
    fi
    ;;
  --aggregate)
    [ "$#" -eq 1 ] || { usage >&2; exit 2; }
    "$PYTHON_BIN" scripts/aggregate_study.py
    ;;
  --preflight)
    [ "$#" -eq 2 ] || { usage >&2; exit 2; }
    model_key=$(model_key_for_index "$2")
    "$PYTHON_BIN" scripts/preflight.py --runtime --model-key "$model_key"
    ;;
  --list_models)
    list_models
    ;;
  --help|-h|'')
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
