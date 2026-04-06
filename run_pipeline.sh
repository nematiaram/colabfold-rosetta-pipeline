#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run ColabFold -> clustering -> Rosetta NC -> decision pipeline.

Required environment variables:
  UNIPROT       UniProt ID (e.g., Q9X6R4)
  OUT_DIR       Output directory (will be created if missing)

Required if PRED_DIR is not set:
  FASTA         Input FASTA file for ColabFold

Optional environment variables:
  PRED_DIR              Directory with predicted PDBs (skip ColabFold if set)
  WORK_DIR              Working directory root (default: OUT_DIR)
  COLABFOLD_BIN          ColabFold executable (default: colabfold_batch)
  COLABFOLD_NUM_MODELS   Target number of models (default: 1500)
  COLABFOLD_EXTRA_ARGS   Extra args for colabfold_batch (quoted string)
  ROSETTA_BIN            per_residue_solvent_exposure.linuxgccrelease path
  LABELS_SOURCE          TSV mapping resname -> labels/label_non_specific
  LABELS_DIR             Directory with *_top10_all_reps.tsv files

Example:
  UNIPROT=Q9X6R4 \
  FASTA=/data/input.fasta \
  OUT_DIR=/data/output \
  ROSETTA_BIN=/rosetta/bin/per_residue_solvent_exposure.linuxgccrelease \
  ./run_pipeline.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

UNIPROT="${UNIPROT:-}"
OUT_DIR="${OUT_DIR:-}"
FASTA="${FASTA:-}"
PRED_DIR="${PRED_DIR:-}"
WORK_DIR="${WORK_DIR:-}"
COLABFOLD_BIN="${COLABFOLD_BIN:-colabfold_batch}"
COLABFOLD_NUM_MODELS="${COLABFOLD_NUM_MODELS:-1500}"
ROSETTA_BIN="${ROSETTA_BIN:-per_residue_solvent_exposure.linuxgccrelease}"
LABELS_SOURCE="${LABELS_SOURCE:-}"
LABELS_DIR="${LABELS_DIR:-}"

if [[ -z "$UNIPROT" || -z "$OUT_DIR" ]]; then
  echo "ERROR: UNIPROT and OUT_DIR are required." >&2
  usage
  exit 1
fi

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$OUT_DIR"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${PIPELINE_SCRIPTS_DIR:-$SCRIPT_DIR/pipeline_steps}"

COLABFOLD_ARGS=()
if [[ -n "${COLABFOLD_EXTRA_ARGS:-}" ]]; then
  read -r -a COLABFOLD_ARGS <<< "$COLABFOLD_EXTRA_ARGS"
fi

mkdir -p "$WORK_DIR"

if [[ -z "$PRED_DIR" ]]; then
  if [[ -z "$FASTA" ]]; then
    echo "ERROR: Set FASTA or PRED_DIR to skip ColabFold." >&2
    usage
    exit 1
  fi
  PRED_DIR="${WORK_DIR}/colabfold"
  mkdir -p "$PRED_DIR"
  if [[ ${#COLABFOLD_ARGS[@]} -gt 0 ]]; then
    python "$SCRIPTS_DIR/01_run_colabfold_batch.py" \
      --fasta "$FASTA" \
      --out-dir "$PRED_DIR" \
      --colabfold-bin "$COLABFOLD_BIN" \
      --num-models "$COLABFOLD_NUM_MODELS" \
      --extra-args "${COLABFOLD_ARGS[@]}"
  else
    python "$SCRIPTS_DIR/01_run_colabfold_batch.py" \
      --fasta "$FASTA" \
      --out-dir "$PRED_DIR" \
      --colabfold-bin "$COLABFOLD_BIN" \
      --num-models "$COLABFOLD_NUM_MODELS"
  fi
fi

ANALYSIS_DIR="${WORK_DIR}/analysis"
mkdir -p "$ANALYSIS_DIR"
python "$SCRIPTS_DIR/02_plddt_rmsd_kmeans.py" \
  --uniprot "$UNIPROT" \
  --pred-dir "$PRED_DIR" \
  --out-dir "$ANALYSIS_DIR"

ROSETTA_OUT="${WORK_DIR}/rosetta"
mkdir -p "$ROSETTA_OUT"
python "$SCRIPTS_DIR/03_run_rosetta_nc.py" \
  --uniprot "$UNIPROT" \
  --rep-info "${ANALYSIS_DIR}/${UNIPROT}_rep_info.tsv" \
  --out-dir "$ROSETTA_OUT" \
  --rosetta-bin "$ROSETTA_BIN"

DECISION_OUT="${WORK_DIR}/decision"
mkdir -p "$DECISION_OUT"

LABEL_ARGS=()
if [[ -n "$LABELS_SOURCE" ]]; then
  LABEL_ARGS+=(--labels-source "$LABELS_SOURCE")
fi
if [[ -n "$LABELS_DIR" ]]; then
  LABEL_ARGS+=(--labels-dir "$LABELS_DIR")
fi

python "$SCRIPTS_DIR/04_labeling_decision_pipeline_all_in_one.py" \
  --uniprot "$UNIPROT" \
  --nc-tsv "${ROSETTA_OUT}/${UNIPROT}_rosetta_nc.tsv" \
  --out-dir "$DECISION_OUT" \
  "${LABEL_ARGS[@]}"

echo "[DONE] Pipeline outputs in: ${WORK_DIR}"
