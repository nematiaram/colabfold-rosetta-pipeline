#!/usr/bin/env bash
# Run steps 2–5 on an existing ColabFold output directory (no prediction step).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: UNIPROT=... ROSETTA_BIN=... $0 /path/to/workdir" >&2
  echo "  Expects: workdir/colabfold/*.pdb" >&2
  exit 1
fi

export SKIP_COLABFOLD=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_pipeline.sh" "$1"
