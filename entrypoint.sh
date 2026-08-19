#!/usr/bin/env bash
# Compatibility shim. The pipeline entrypoint is now entrypoint.py; this keeps
# existing `entrypoint.sh ...` invocations working. See entrypoint.py --help.
exec python3 "$(dirname "$(readlink -f "$0")")/entrypoint.py" "$@"
