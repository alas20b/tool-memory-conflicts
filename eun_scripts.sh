#!/usr/bin/env sh
# Compatibility wrapper for the earlier misspelled filename.
SCRIPT_DIR=${0%/*}
[ "$SCRIPT_DIR" = "$0" ] && SCRIPT_DIR=.
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR" && pwd)
exec sh "$ROOT/run_scripts.sh" "$@"
