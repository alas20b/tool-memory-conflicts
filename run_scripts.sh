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

# Accept either a registry key ("llama-3.1-8b-instruct") or an index (0..4).
# Sets MODEL_KEY and SLUG. Exits with the registry listing on unknown keys.
resolve_model() {
  selector=$1
  case "$selector" in
    ''|*[!0-9]*)
      MODEL_KEY=$selector
      SLUG=$("$PYTHON_BIN" -c 'import sys; from kcb.io_utils import model_slug; print(model_slug(sys.argv[1]))' "$selector")
      ;;
    *)
      MODEL_KEY=$(model_key_for_index "$selector")
      SLUG=$(slug_for_index "$selector")
      ;;
  esac
  if ! "$PYTHON_BIN" -c 'import sys; from pathlib import Path; from kcb.packets import load_registry; sys.exit(0 if sys.argv[1] in load_registry(Path(".")) else 1)' "$MODEL_KEY" >/dev/null 2>&1; then
    echo "ERROR: unknown model key '$MODEL_KEY'." >&2
    echo "Known models:" >&2
    list_models >&2
    exit 2
  fi
}

# Parse run options into shell globals. Defaults are stored in the globals
# before any call so that reuse across modes is safe.
backend=transformers
count=12
cache_dir=${KCB_HF_HOME:-}
ollama_host="http://localhost:11434"
ollama_model=""
lms_host="http://localhost:1234"
lms_model=""
limit_tasks=""
force=""
skip_conditions=""

parse_run_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --backend) backend=$2; shift 2 ;;
      --count) count=$2; shift 2 ;;
      --cache-dir) cache_dir=$2; shift 2 ;;
      --ollama-host) ollama_host=$2; shift 2 ;;
      --ollama-model) ollama_model=$2; shift 2 ;;
      --lms-host) lms_host=$2; shift 2 ;;
      --lms-model) lms_model=$2; shift 2 ;;
      --limit-tasks) limit_tasks=$2; shift 2 ;;
      --force) force=1; shift ;;
      --skip-conditions) skip_conditions=$2; shift 2 ;;
      *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    esac
  done
  case "$backend" in
    mock|vllm|transformers|ollama|lms) ;;
    *) echo "ERROR: --backend must be mock, vllm, transformers, ollama, or lms." >&2; exit 2 ;;
  esac
}

# mode is smoke or full. MODEL_KEY and SLUG must already be resolved.
real_run() {
  mode=$1
  if [ "$mode" = 'smoke' ]; then
    packet="packets/${SLUG}-review.json"
  else
    packet="packets/${SLUG}-full.json"
  fi
  out_dir="artifacts/${mode}/${backend}/${SLUG}"

  if [ "$backend" = 'mock' ]; then
    "$PYTHON_BIN" scripts/preflight.py
  else
    set -- --runtime --model-key "$MODEL_KEY" --backend "$backend"
    if [ "$backend" = 'ollama' ]; then
      set -- "$@" --ollama-host "$ollama_host"
    fi
    if [ "$backend" = 'lms' ]; then
      set -- "$@" --lms-host "$lms_host"
    fi
    "$PYTHON_BIN" scripts/preflight.py "$@"
  fi

  if [ "$mode" = 'smoke' ]; then
    "$PYTHON_BIN" scripts/build_packet.py \
      --model-key "$MODEL_KEY" --mode review --count "$count" --output "$packet"
  else
    "$PYTHON_BIN" scripts/build_packet.py \
      --model-key "$MODEL_KEY" --mode full --output "$packet"
  fi

  set -- --model-key "$MODEL_KEY" --packet "$packet" --out-dir "$out_dir" --backend "$backend"
  if [ -n "$cache_dir" ]; then
    set -- "$@" --cache-dir "$cache_dir"
  fi
  if [ -n "$ollama_host" ]; then
    set -- "$@" --ollama-host "$ollama_host"
  fi
  if [ -n "$ollama_model" ]; then
    set -- "$@" --ollama-model "$ollama_model"
  fi
  if [ -n "$lms_host" ]; then
    set -- "$@" --lms-host "$lms_host"
  fi
  if [ -n "$lms_model" ]; then
    set -- "$@" --lms-model "$lms_model"
  fi
  if [ -n "$limit_tasks" ]; then
    set -- "$@" --limit-tasks "$limit_tasks"
  fi
  if [ "$force" = 1 ]; then
    set -- "$@" --force
  fi
  if [ -n "$skip_conditions" ]; then
    set -- "$@" --skip-conditions "$skip_conditions"
  fi

  "$PYTHON_BIN" scripts/run_experiment.py "$@"

  set -- --packet "$packet" --responses "$out_dir/responses.jsonl"
  if [ -n "$limit_tasks" ] || [ -n "$skip_conditions" ]; then
    set -- "$@" --allow-partial
  fi
  "$PYTHON_BIN" scripts/validate_results.py "$@"
}

usage() {
  cat <<'EOF'
Usage:
  sh run_scripts.sh --dry_run
  sh run_scripts.sh --run MODEL [OPTIONS]
  sh run_scripts.sh --smoke_test MODEL [OPTIONS]
  sh run_scripts.sh --full_test MODEL|all [OPTIONS]
  sh run_scripts.sh --preflight MODEL [--backend BACKEND]
  sh run_scripts.sh --aggregate [--backend BACKEND]
  sh run_scripts.sh --list_models

MODEL is a registry model key (e.g. llama-3.1-8b-instruct) or an index 0..4.
A bare MODEL as the first argument is the same as --run MODEL.

OPTIONS:
  --backend mock|vllm|transformers|ollama|lms   inference backend
                                                (default: transformers)
  --count N                  review/smoke packet size (default: 12)
  --cache-dir PATH           Hugging Face cache dir (default: $KCB_HF_HOME)
  --ollama-host URL          Ollama server base URL (default: http://localhost:11434)
  --ollama-model NAME        Ollama model tag (default: last path segment of hf repo)
  --lms-host URL             LM Studio server base URL (default: http://localhost:1234)
  --lms-model NAME           LM Studio model name (default: last path segment of hf repo)
  --limit-tasks N            run at most N tasks (good for quick tests)
  --force                    ignore existing results and rerun everything
  --skip-conditions LIST     comma-separated error kinds to skip
                             (service_503,timeout,permission_denied)

Examples:
  sh run_scripts.sh llama-3.1-8b-instruct --backend transformers
  sh run_scripts.sh 2 --backend lms --lms-host http://10.16.98.67:1234 --limit-tasks 10
  sh run_scripts.sh --smoke_test 0 --backend ollama --limit-tasks 6
  sh run_scripts.sh --full_test all --backend transformers
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
    [ "$#" -ge 2 ] || { usage >&2; exit 2; }
    selector=$2
    shift 2
    resolve_model "$selector"
    parse_run_args "$@"
    real_run smoke
    ;;
  --full_test)
    [ "$#" -ge 2 ] || { usage >&2; exit 2; }
    selector=$2
    shift 2
    parse_run_args "$@"
    if [ "$selector" = 'all' ]; then
      for index in 0 1 2 3 4; do
        MODEL_KEY=$(model_key_for_index "$index")
        SLUG=$(slug_for_index "$index")
        real_run full
      done
      "$PYTHON_BIN" scripts/aggregate_study.py --backend "$backend"
    else
      resolve_model "$selector"
      real_run full
    fi
    ;;
  --run)
    [ "$#" -ge 2 ] || { usage >&2; exit 2; }
    selector=$2
    shift 2
    resolve_model "$selector"
    parse_run_args "$@"
    real_run full
    ;;
  --preflight)
    [ "$#" -ge 2 ] || { usage >&2; exit 2; }
    selector=$2
    shift 2
    resolve_model "$selector"
    parse_run_args "$@"
    set -- --runtime --model-key "$MODEL_KEY" --backend "$backend"
    if [ "$backend" = 'ollama' ]; then
      set -- "$@" --ollama-host "$ollama_host"
    fi
    if [ "$backend" = 'lms' ]; then
      set -- "$@" --lms-host "$lms_host"
    fi
    "$PYTHON_BIN" scripts/preflight.py "$@"
    ;;
  --aggregate)
    shift
    parse_run_args "$@"
    "$PYTHON_BIN" scripts/aggregate_study.py --backend "$backend"
    ;;
  --list_models)
    list_models
    ;;
  --help|-h|'')
    usage
    ;;
  *)
    case "${1:-}" in
      -*)
        usage >&2
        exit 2
        ;;
      *)
        selector=$1
        shift
        resolve_model "$selector"
        parse_run_args "$@"
        real_run full
        ;;
    esac
    ;;
esac
