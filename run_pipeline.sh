#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run ColabFold -> clustering -> Rosetta NC -> decision pipeline.

Usage:
  UNIPROT=Q9X6R4 \
  FASTA=/data/input.fasta \
  ROSETTA_BIN=/rosetta/bin/per_residue_solvent_exposure.linuxgccrelease \
  ./run_pipeline.sh /data/output

Required:
  1st argument   Working directory / output root
  UNIPROT        UniProt ID
  FASTA          Input FASTA file
  ROSETTA_BIN    Path to per_residue_solvent_exposure.linuxgccrelease

Optional:
  COLABFOLD_BIN          ColabFold executable (default: colabfold_batch)
  COLABFOLD_NUM_MODELS   Target number of models (default: 1500)
  COLABFOLD_EXTRA_ARGS   Extra args for colabfold_batch (quoted string)
  LABELS_SOURCE          TSV mapping resname -> labels/label_non_specific
  LABELS_DIR             Directory with *_top10_all_reps.tsv files
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "ERROR: Please provide a working directory as the first argument." >&2
  usage
  exit 1
fi

WORK_DIR="$1"

UNIPROT="${UNIPROT:-}"
FASTA="${FASTA:-}"
COLABFOLD_BIN="${COLABFOLD_BIN:-colabfold_batch}"
COLABFOLD_NUM_MODELS="${COLABFOLD_NUM_MODELS:-1500}"
ROSETTA_BIN="${ROSETTA_BIN:-per_residue_solvent_exposure.linuxgccrelease}"
LABELS_SOURCE="${LABELS_SOURCE:-}"
LABELS_DIR="${LABELS_DIR:-}"

if [[ -z "$UNIPROT" || -z "$FASTA" ]]; then
  echo "ERROR: UNIPROT and FASTA are required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${PIPELINE_SCRIPTS_DIR:-$SCRIPT_DIR/pipeline_steps}"

COLABFOLD_ARGS=()
if [[ -n "${COLABFOLD_EXTRA_ARGS:-}" ]]; then
  read -r -a COLABFOLD_ARGS <<< "$COLABFOLD_EXTRA_ARGS"
fi

mkdir -p "$WORK_DIR"

PRED_DIR="${WORK_DIR}/colabfold"
ANALYSIS_DIR="${WORK_DIR}/analysis"
ROSETTA_OUT="${WORK_DIR}/rosetta"
DECISION_OUT="${WORK_DIR}/decision"

mkdir -p "$PRED_DIR" "$ANALYSIS_DIR" "$ROSETTA_OUT" "$DECISION_OUT"

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

python "$SCRIPTS_DIR/02_plddt_rmsd_kmeans.py" \
  --uniprot "$UNIPROT" \
  --pred-dir "$PRED_DIR" \
  --out-dir "$ANALYSIS_DIR"

python "$SCRIPTS_DIR/03_run_rosetta_nc.py" \
  --uniprot "$UNIPROT" \
  --rep-info "${ANALYSIS_DIR}/${UNIPROT}_rep_info.tsv" \
  --out-dir "$ROSETTA_OUT" \
  --rosetta-bin "$ROSETTA_BIN"

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
