#!/usr/bin/env bash
set -euo pipefail

usage() {
cat <<'EOF'
Usage:
  entrypoint.sh --workdir <path> --input <path-to-job.json> --ncpus <N>

Required:
  --workdir   Working directory; all outputs are written here.
  --input     Path to a JSON file describing the job
              (job_id, sequence, num_seeds, models_per_seed,
              optional labels_source / labels_dir).
  --ncpus     Number of CPUs this container instance is allowed to use.

Optional env:
  THREADS_PER_WORKER   CPU threads per ColabFold worker process (default: 8).
  ROSETTA_BIN          Path to per_residue_solvent_exposure.linuxgccrelease
                       (default: baked into the image).
EOF
}

WORK_DIR="" INPUT_JSON="" NCPUS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) WORK_DIR="$2"; shift 2 ;;
    --input)   INPUT_JSON="$2"; shift 2 ;;
    --ncpus)   NCPUS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$WORK_DIR" || -z "$INPUT_JSON" || -z "$NCPUS" ]]; then
  echo "ERROR: --workdir, --input, and --ncpus are all required." >&2
  usage; exit 1
fi

ROSETTA_BIN="${ROSETTA_BIN:-/opt/conda/bin/per_residue_solvent_exposure.linuxgccrelease}"
SCRIPTS_DIR="/opt/pipeline/pipeline_steps"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-8}"

mkdir -p "$WORK_DIR"
FASTA="${WORK_DIR}/input.fasta"
META="${WORK_DIR}/job_meta.env"

python "$SCRIPTS_DIR/00_prepare_input.py" \
  --input-json "$INPUT_JSON" --out-fasta "$FASTA" --out-meta "$META"

# shellcheck disable=SC1090
source "$META"   # UNIPROT, NUM_SEEDS, MODELS_PER_SEED, LABELS_SOURCE, LABELS_DIR

PRED_DIR="${WORK_DIR}/colabfold"; ANALYSIS_DIR="${WORK_DIR}/analysis"
ROSETTA_OUT="${WORK_DIR}/rosetta"; DECISION_OUT="${WORK_DIR}/decision"
mkdir -p "$PRED_DIR" "$ANALYSIS_DIR" "$ROSETTA_OUT" "$DECISION_OUT"

# ---------------------------------------------------------------------------
# Step 1: ColabFold, fanned out across NUM_WORKERS concurrent processes so
# all NCPUS cores stay busy (single-process high-thread-count scales poorly).
# ---------------------------------------------------------------------------
NUM_WORKERS=$(( NCPUS / THREADS_PER_WORKER ))
if [[ "$NUM_WORKERS" -lt 1 ]]; then NUM_WORKERS=1; fi
SEEDS_PER_WORKER=$(( (NUM_SEEDS + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "HOST=$(hostname) NCPUS=${NCPUS} THREADS_PER_WORKER=${THREADS_PER_WORKER} NUM_WORKERS=${NUM_WORKERS}"
echo "UNIPROT=${UNIPROT} NUM_SEEDS=${NUM_SEEDS} MODELS_PER_SEED=${MODELS_PER_SEED} SEEDS_PER_WORKER=${SEEDS_PER_WORKER}"

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS_PER_WORKER} inter_op_parallelism_threads=${THREADS_PER_WORKER}"

pids=()
for ((w=0; w<NUM_WORKERS; w++)); do
  START_SEED=$(( w * SEEDS_PER_WORKER ))
  WORKER_OUT="${PRED_DIR}/worker_${w}"
  mkdir -p "$WORKER_OUT"

  CORE_START=$(( w * THREADS_PER_WORKER ))
  CORE_END=$(( CORE_START + THREADS_PER_WORKER - 1 ))

  taskset -c "${CORE_START}-${CORE_END}" \
    python "$SCRIPTS_DIR/01_run_colabfold_batch.py" \
      --fasta "$FASTA" --out-dir "$WORKER_OUT" \
      --num-seeds "$SEEDS_PER_WORKER" --models-per-seed "$MODELS_PER_SEED" \
      --random-seed "$START_SEED" \
      --extra-args --max-msa 16:32 --use-dropout \
      > "${WORKER_OUT}/worker_${w}.log" 2>&1 &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

for d in "${PRED_DIR}"/worker_*/; do
  cp -n "$d"*.pdb "$d"*.json "$d"*.a3m "$PRED_DIR"/ 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# Steps 2-4: clustering, Rosetta NC (only ~3 representative structures,
# no parallelization needed), and the decision step.
# ---------------------------------------------------------------------------
python "$SCRIPTS_DIR/02_plddt_rmsd_kmeans.py" \
  --uniprot "$UNIPROT" --pred-dir "$PRED_DIR" --out-dir "$ANALYSIS_DIR"

python "$SCRIPTS_DIR/03_run_rosetta_nc.py" \
  --uniprot "$UNIPROT" \
  --rep-info "${ANALYSIS_DIR}/${UNIPROT}_rep_info.tsv" \
  --out-dir "$ROSETTA_OUT" --rosetta-bin "$ROSETTA_BIN"

LABEL_ARGS=()
[[ -n "${LABELS_SOURCE:-}" ]] && LABEL_ARGS+=(--labels-source "$LABELS_SOURCE")
[[ -n "${LABELS_DIR:-}"    ]] && LABEL_ARGS+=(--labels-dir "$LABELS_DIR")

python "$SCRIPTS_DIR/04_labeling_decision_pipeline_all_in_one.py" \
  --uniprot "$UNIPROT" --nc-tsv "${ROSETTA_OUT}/${UNIPROT}_rosetta_nc.tsv" \
  --out-dir "$DECISION_OUT" "${LABEL_ARGS[@]}"

echo "[DONE] Pipeline outputs in: ${WORK_DIR}"
