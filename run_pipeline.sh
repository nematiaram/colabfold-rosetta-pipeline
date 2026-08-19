#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run ColabFold -> clustering -> Rosetta cone NC -> pairwise ΔNC + reagent mapping.

Usage:
  UNIPROT=Q9X6R4 \
  FASTA=/data/input.fasta \
  ROSETTA_BIN=/rosetta/bin/per_residue_solvent_exposure.linuxgccrelease \
  ./run_pipeline.sh /data/output

Required:
  1st argument   Working directory / output root
  UNIPROT        UniProt ID
  FASTA          Input FASTA file
  ROSETTA_BIN    Path to per_residue_solvent_exposure

Optional:
  COLABFOLD_BIN          ColabFold executable (default: colabfold_batch)
  COLABFOLD_NUM_SEEDS    ColabFold seeds to sample (default: 300)
  COLABFOLD_MODELS_PER_SEED  AF2 models per seed, 1-5 (default: 5)
  COLABFOLD_NUM_MODELS   Legacy: target total models; converted to
                         num_seeds = ceil(NUM_MODELS / MODELS_PER_SEED)
  COLABFOLD_EXTRA_ARGS   Extra args for colabfold_batch (quoted string)
  NC_METHOD              Rosetta NC method: cone or sphere (default: cone)
  PAIRWISE_THRESHOLD     Minimum |ΔNC| for reporters (default: 5)
  RUN_LEGACY_DECISION    Set to 1 to also run step 04 absolute-gate decision
  LABELS_SOURCE          TSV mapping resname -> labels (legacy step 04 only)
  LABELS_DIR             Directory with *_top10_all_reps.tsv (legacy step 04 only)
  SKIP_COLABFOLD         Set to 1 if predictions already exist in colabfold/
  SKIP_CLUSTERING        Set to 1 if rep_info.tsv already exists in analysis/
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
COLABFOLD_MODELS_PER_SEED="${COLABFOLD_MODELS_PER_SEED:-5}"
if [[ -n "${COLABFOLD_NUM_MODELS:-}" ]]; then
  # Legacy knob: a total model count. Step 01 takes seeds x models-per-seed.
  COLABFOLD_NUM_SEEDS="${COLABFOLD_NUM_SEEDS:-$(( (COLABFOLD_NUM_MODELS + COLABFOLD_MODELS_PER_SEED - 1) / COLABFOLD_MODELS_PER_SEED ))}"
fi
COLABFOLD_NUM_SEEDS="${COLABFOLD_NUM_SEEDS:-300}"
ROSETTA_BIN="${ROSETTA_BIN:-per_residue_solvent_exposure.linuxgccrelease}"
NC_METHOD="${NC_METHOD:-cone}"
PAIRWISE_THRESHOLD="${PAIRWISE_THRESHOLD:-5}"
RUN_LEGACY_DECISION="${RUN_LEGACY_DECISION:-0}"
SKIP_COLABFOLD="${SKIP_COLABFOLD:-0}"
SKIP_CLUSTERING="${SKIP_CLUSTERING:-0}"
LABELS_SOURCE="${LABELS_SOURCE:-}"
LABELS_DIR="${LABELS_DIR:-}"

if [[ -z "$UNIPROT" ]]; then
  echo "ERROR: UNIPROT is required." >&2
  usage
  exit 1
fi
if [[ "$SKIP_COLABFOLD" != "1" && -z "$FASTA" ]]; then
  echo "ERROR: FASTA is required unless SKIP_COLABFOLD=1." >&2
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
PAIRWISE_OUT="${WORK_DIR}/pairwise"
DECISION_OUT="${WORK_DIR}/decision"

mkdir -p "$PRED_DIR" "$ANALYSIS_DIR" "$ROSETTA_OUT" "$PAIRWISE_OUT" "$DECISION_OUT"

if [[ "$SKIP_COLABFOLD" != "1" ]]; then
  echo "=== Step 1: ColabFold ==="
  echo "    seeds=${COLABFOLD_NUM_SEEDS} models_per_seed=${COLABFOLD_MODELS_PER_SEED}"
  if [[ ${#COLABFOLD_ARGS[@]} -gt 0 ]]; then
    python "$SCRIPTS_DIR/01_run_colabfold_batch.py" \
      --fasta "$FASTA" \
      --out-dir "$PRED_DIR" \
      --colabfold-bin "$COLABFOLD_BIN" \
      --num-seeds "$COLABFOLD_NUM_SEEDS" \
      --models-per-seed "$COLABFOLD_MODELS_PER_SEED" \
      --extra-args "${COLABFOLD_ARGS[@]}"
  else
    python "$SCRIPTS_DIR/01_run_colabfold_batch.py" \
      --fasta "$FASTA" \
      --out-dir "$PRED_DIR" \
      --colabfold-bin "$COLABFOLD_BIN" \
      --num-seeds "$COLABFOLD_NUM_SEEDS" \
      --models-per-seed "$COLABFOLD_MODELS_PER_SEED"
  fi
else
  echo "=== Step 1: ColabFold (skipped) ==="
fi

if [[ "$SKIP_CLUSTERING" != "1" ]]; then
  echo "=== Step 2: pLDDT/RMSD clustering ==="
  python "$SCRIPTS_DIR/02_plddt_rmsd_kmeans.py" \
    --uniprot "$UNIPROT" \
    --pred-dir "$PRED_DIR" \
    --out-dir "$ANALYSIS_DIR"
else
  echo "=== Step 2: Clustering (skipped) ==="
fi

echo "=== Step 3: Rosetta NC (method=${NC_METHOD}) ==="
python "$SCRIPTS_DIR/03_run_rosetta_nc.py" \
  --uniprot "$UNIPROT" \
  --rep-info "${ANALYSIS_DIR}/${UNIPROT}_rep_info.tsv" \
  --out-dir "$ROSETTA_OUT" \
  --rosetta-bin "$ROSETTA_BIN" \
  --method "$NC_METHOD"

echo "=== Step 5: Pairwise ΔNC + reagent mapping (T=${PAIRWISE_THRESHOLD}) ==="
python "$SCRIPTS_DIR/05_pairwise_delta_nc.py" \
  --uniprot "$UNIPROT" \
  --nc-tsv "${ROSETTA_OUT}/${UNIPROT}_rosetta_nc.tsv" \
  --out-dir "$PAIRWISE_OUT" \
  --threshold "$PAIRWISE_THRESHOLD"

if [[ "$RUN_LEGACY_DECISION" == "1" ]]; then
  echo "=== Step 4 (legacy): absolute-gate decision ==="
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
fi

echo "[DONE] Pipeline outputs in: ${WORK_DIR}"
echo "  rosetta/   cone NC tables"
echo "  pairwise/  ΔNC reporters + reagent targets"
