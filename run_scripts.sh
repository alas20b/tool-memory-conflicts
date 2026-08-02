#!/usr/bin/env sh
set -eu

SCRIPT_DIR=${0%/*}
[ "$SCRIPT_DIR" = "$0" ] && SCRIPT_DIR=.
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  sh run_scripts.sh --dry_run
  sh run_scripts.sh --list_models
  sh run_scripts.sh --probe MODEL BACKEND [probe options]
  sh run_scripts.sh --pilot PROFILE.json
  sh run_scripts.sh --calibrate PROFILE.json
  sh run_scripts.sh --smoke PROFILE.json
  sh run_scripts.sh --full PROFILE.json
  sh run_scripts.sh --full_test PROFILE.json
  sh run_scripts.sh --validate STAGE PROFILE.json
  sh run_scripts.sh --aggregate PROFILE0.json ... PROFILE4.json
  sh run_scripts.sh --bundle [bundle-name]

MODEL is a registry index 0..4 or a model key. BACKEND is ollama,
lmstudio, or transformers. Run --probe first; experiment commands require
the exact immutable profile path printed by the probe.
EOF
}

[ "$#" -ge 1 ] || { usage; exit 2; }
COMMAND=$1
shift

case "$COMMAND" in
  --dry_run)
    [ "$#" -eq 0 ] || { usage; exit 2; }
    "$PYTHON_BIN" scripts/preflight.py
    "$PYTHON_BIN" -m unittest discover -s tests -v
    ;;
  --list_models)
    [ "$#" -eq 0 ] || { usage; exit 2; }
    "$PYTHON_BIN" scripts/list_models.py
    ;;
  --probe)
    [ "$#" -ge 2 ] || { usage; exit 2; }
    "$PYTHON_BIN" scripts/probe.py "$@"
    ;;
  --pilot|--calibrate|--smoke|--full)
    [ "$#" -eq 1 ] || { usage; exit 2; }
    STAGE=${COMMAND#--}
    "$PYTHON_BIN" scripts/run_stage.py "$STAGE" "$1"
    "$PYTHON_BIN" scripts/validate.py "$STAGE" "$1"
    ;;
  --full_test)
    [ "$#" -eq 1 ] || { usage; exit 2; }
    PROFILE=$1
    "$PYTHON_BIN" scripts/run_stage.py calibrate "$PROFILE"
    "$PYTHON_BIN" scripts/validate.py calibrate "$PROFILE"
    "$PYTHON_BIN" scripts/run_stage.py smoke "$PROFILE"
    "$PYTHON_BIN" scripts/validate.py smoke "$PROFILE"
    "$PYTHON_BIN" scripts/run_stage.py full "$PROFILE"
    "$PYTHON_BIN" scripts/validate.py full "$PROFILE"
    ;;
  --validate)
    [ "$#" -eq 2 ] || { usage; exit 2; }
    "$PYTHON_BIN" scripts/validate.py "$1" "$2"
    ;;
  --aggregate)
    [ "$#" -eq 5 ] || { usage; exit 2; }
    "$PYTHON_BIN" scripts/aggregate.py "$@"
    ;;
  --bundle)
    [ "$#" -le 1 ] || { usage; exit 2; }
    if [ "$#" -eq 1 ]; then
      "$PYTHON_BIN" scripts/bundle_results.py --name "$1"
    else
      "$PYTHON_BIN" scripts/bundle_results.py
    fi
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
