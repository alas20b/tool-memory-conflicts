#!/bin/sh
# Compatibility wrapper for the command spelling used in the handoff request.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$ROOT/run_scripts.sh" "$@"
