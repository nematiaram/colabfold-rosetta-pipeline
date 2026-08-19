#!/usr/bin/env bash
set -euo pipefail

usage() {
cat <<'EOF'
Usage:
  entrypoint.sh --workdir <dir> --input <path-to-job.json> --ncpus <N>

Required:
  --workdir   Working directory; all outputs are written here.
  --input     JSON job file (job_id, sequence, num_seeds, models_per_seed,
              optional labels_source / labels_dir).
  --ncpus     CPUs available to this container instance.

Optional env:
  THREADS_PER_WORKER      ColabFold worker threads (default: 8).
                          For CPU tests, try 2 then 1. Seeds are split
                          exactly across workers (no ceil overflow).
  COLABFOLD_BIN           ColabFold executable (default: colabfold_batch).
  ROSETTA_BIN             per_residue_solvent_exposure binary.
  NC_METHOD               Rosetta NC method: cone or sphere (default: cone).
  PAIRWISE_THRESHOLD      Minimum |ΔNC| for reporters (default: 5).
  RUN_LEGACY_DECISION     Set to 1 to also run step 04 absolute-gate decision.
EOF
}

WORK_DIR="" INPUT_JSON="" NCPUS=""
while [[ $# -gt 0 ]]; do
 case "$1" in
 --workdir) WORK_DIR="$2"; shift 2 ;;
 --input) INPUT_JSON="$2"; shift 2 ;;
 --ncpus) NCPUS="$2"; shift 2 ;;
 -h|--help) usage; exit 0 ;;
 *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
 esac
done

if [[ -z "$WORK_DIR" || -z "$INPUT_JSON" || -z "$NCPUS" ]]; then
 echo "ERROR: --workdir, --input, and --ncpus are all required." >&2
 usage; exit 1
fi

COLABFOLD_BIN="${COLABFOLD_BIN:-colabfold_batch}"
ROSETTA_BIN="${ROSETTA_BIN:-/opt/conda/bin/per_residue_solvent_exposure.linuxgccrelease}"
SCRIPTS_DIR="/opt/pipeline/pipeline_steps"
# One thread per ColabFold process. AF2 on CPU does not use 8 OpenMP threads
# well, and fat workers cut the number of independent seed jobs in flight.
THREADS_PER_WORKER="${THREADS_PER_WORKER:-1}"
NC_METHOD="${NC_METHOD:-cone}"
PAIRWISE_THRESHOLD="${PAIRWISE_THRESHOLD:-5}"
RUN_LEGACY_DECISION="${RUN_LEGACY_DECISION:-0}"

mkdir -p "$WORK_DIR"
FASTA="${WORK_DIR}/input.fasta"
META="${WORK_DIR}/job_meta.env"

python "$SCRIPTS_DIR/00_prepare_input.py" \
 --input-json "$INPUT_JSON" --out-fasta "$FASTA" --out-meta "$META"

# shellcheck disable=SC1090
source "$META"

PRED_DIR="${WORK_DIR}/colabfold"
ANALYSIS_DIR="${WORK_DIR}/analysis"
ROSETTA_OUT="${WORK_DIR}/rosetta"
PAIRWISE_OUT="${WORK_DIR}/pairwise"
DECISION_OUT="${WORK_DIR}/decision"
mkdir -p "$PRED_DIR" "$ANALYSIS_DIR" "$ROSETTA_OUT" "$PAIRWISE_OUT" "$DECISION_OUT"

# Do not assume this job owns cores 0..NCPUS-1. Under SGE/SLURM, and in any
# cpuset-constrained container, the allocated CPU ids can be an arbitrary set
# such as 40-51; pinning to 0..N-1 then makes every taskset call fail with
# "Invalid argument" and kills the run. Ask the kernel what we may use.
ALLOWED_CPUS=()
while read -r cpu; do ALLOWED_CPUS+=("$cpu"); done < <(
  python -c 'import os; print("\n".join(str(c) for c in sorted(os.sched_getaffinity(0))))' 2>/dev/null
)
N_ALLOWED=${#ALLOWED_CPUS[@]}
if [[ "$N_ALLOWED" -eq 0 ]]; then
 echo "WARNING: could not read CPU affinity; assuming cores 0-$((NCPUS - 1))." >&2
 for ((c=0; c<NCPUS; c++)); do ALLOWED_CPUS+=("$c"); done
 N_ALLOWED=$NCPUS
fi
if [[ "$NCPUS" -gt "$N_ALLOWED" ]]; then
 echo "WARNING: --ncpus ${NCPUS} exceeds the ${N_ALLOWED} CPUs this process may use; clamping to ${N_ALLOWED}." >&2
 NCPUS=$N_ALLOWED
fi

if [[ "$THREADS_PER_WORKER" -gt "$NCPUS" ]]; then
 echo "WARNING: THREADS_PER_WORKER ${THREADS_PER_WORKER} exceeds the ${NCPUS} usable CPUs; clamping to ${NCPUS}." >&2
 THREADS_PER_WORKER=$NCPUS
fi

# Only pin when the cpuset actually reflects this job's allocation. Some
# schedulers (UGE on shared nodes, for one) hand out N slots but leave affinity
# open across the whole machine. Pinning to the first N cores there makes every
# concurrent job on the node pile onto the same low-numbered cores while the
# rest idle, which is worse than not pinning at all.
HAVE_TASKSET=1
command -v taskset >/dev/null 2>&1 || { HAVE_TASKSET=0; echo "WARNING: taskset not found; workers will not be pinned." >&2; }
if [[ "$HAVE_TASKSET" -eq 1 && "$N_ALLOWED" -gt "$NCPUS" ]]; then
 HAVE_TASKSET=0
 echo "NOTE: affinity is open across ${N_ALLOWED} CPUs but only ${NCPUS} were requested;" >&2
 echo "      the cpuset does not describe this allocation, so workers run unpinned" >&2
 echo "      to avoid colliding with other jobs on a shared node." >&2
fi

NUM_WORKERS=$(( NCPUS / THREADS_PER_WORKER ))
if [[ "$NUM_WORKERS" -lt 1 ]]; then NUM_WORKERS=1; fi
BASE_SEEDS=$(( NUM_SEEDS / NUM_WORKERS ))
REMAINDER=$(( NUM_SEEDS % NUM_WORKERS ))

echo "HOST=$(hostname) NCPUS=${NCPUS} THREADS_PER_WORKER=${THREADS_PER_WORKER} NUM_WORKERS=${NUM_WORKERS}"
echo "UNIPROT=${UNIPROT} NUM_SEEDS=${NUM_SEEDS} MODELS_PER_SEED=${MODELS_PER_SEED} NC_METHOD=${NC_METHOD}"
echo "SEED_SPLIT BASE=${BASE_SEEDS} REMAINDER=${REMAINDER} (first ${REMAINDER} workers get BASE+1)"
echo "ALLOWED_CPUS=${ALLOWED_CPUS[*]}"

# Ignore ~/.local site-packages. Container runtimes bind $HOME by default, and
# a newer Biopython there shadows the image's, breaking AlphaFold's
# `from Bio.Data import SCOPData` before colabfold_batch can start.
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS_PER_WORKER} inter_op_parallelism_threads=${THREADS_PER_WORKER}"

# Build the MSA exactly once. Otherwise every one of NUM_WORKERS processes
# queries the MMseqs2 API for the same sequence: N-1 redundant round-trips
# before any folding starts, and a reliable way to get rate-limited.
MSA_DIR="${WORK_DIR}/msa"
mkdir -p "$MSA_DIR"
WORKER_INPUT="$FASTA"
if compgen -G "${MSA_DIR}/*.a3m" > /dev/null 2>&1; then
 echo "[MSA] reusing existing a3m in ${MSA_DIR}"
else
 if "$COLABFOLD_BIN" --help 2>&1 | grep -q -- '--msa-only'; then
  echo "[MSA] building shared MSA (--msa-only)"
  "$COLABFOLD_BIN" --msa-only "$FASTA" "$MSA_DIR" > "${MSA_DIR}/msa.log" 2>&1 || true
 fi
 if ! compgen -G "${MSA_DIR}/*.a3m" > /dev/null 2>&1; then
  echo "[MSA] --msa-only unavailable; falling back to a 1-seed/1-model pass"
  "$COLABFOLD_BIN" --num-seeds 1 --num-models 1 "$FASTA" "$MSA_DIR" >> "${MSA_DIR}/msa.log" 2>&1 || true
 fi
fi
A3M=""
for candidate in "${MSA_DIR}"/*.a3m; do
 if [[ -f "$candidate" ]]; then A3M="$candidate"; break; fi
done
if [[ -n "$A3M" ]]; then
 WORKER_INPUT="$A3M"
 echo "[MSA] workers reuse ${A3M} (0 per-worker MSA queries)"
else
 echo "[MSA] WARNING: no a3m produced; each worker will build its own MSA." >&2
fi

pids=()
WORKER_IDS=()
START_SEED=0
for ((w=0; w<NUM_WORKERS; w++)); do
 WORKER_SEEDS=$BASE_SEEDS
 if (( w < REMAINDER )); then
  WORKER_SEEDS=$(( WORKER_SEEDS + 1 ))
 fi
 if (( WORKER_SEEDS == 0 )); then
  continue
 fi
 WORKER_OUT="${PRED_DIR}/worker_${w}"
 mkdir -p "$WORKER_OUT"
 CORE_LIST=""
 for ((t=0; t<THREADS_PER_WORKER; t++)); do
  idx=$(( w * THREADS_PER_WORKER + t ))
  if [[ "$idx" -ge "$N_ALLOWED" ]]; then break; fi
  CORE_LIST="${CORE_LIST:+${CORE_LIST},}${ALLOWED_CPUS[$idx]}"
 done
 PIN=(taskset -c "$CORE_LIST")
 if [[ "$HAVE_TASKSET" -eq 0 || -z "$CORE_LIST" ]]; then PIN=(); fi
 echo "WORKER ${w} seeds=${WORKER_SEEDS} random-seed=${START_SEED} cores=${CORE_LIST}"
 # ${PIN[@]+...} guard: bash < 4.4 treats an empty array as unbound under set -u.
 ${PIN[@]+"${PIN[@]}"} \
 python "$SCRIPTS_DIR/01_run_colabfold_batch.py" \
 --fasta "$WORKER_INPUT" --out-dir "$WORKER_OUT" \
 --colabfold-bin "$COLABFOLD_BIN" \
 --num-seeds "$WORKER_SEEDS" --models-per-seed "$MODELS_PER_SEED" \
 --random-seed "$START_SEED" \
 --extra-args --max-msa 16:32 --use-dropout \
 > "${WORKER_OUT}/worker_${w}.log" 2>&1 &
 pids+=($!)
 WORKER_IDS+=("$w")
 START_SEED=$(( START_SEED + WORKER_SEEDS ))
done
echo "LAUNCHED_SEEDS=${START_SEED} (expected ${NUM_SEEDS})"
if [[ "$START_SEED" -ne "$NUM_SEEDS" ]]; then
 echo "ERROR: launched seeds ${START_SEED} != NUM_SEEDS ${NUM_SEEDS}" >&2
 exit 1
fi
if [[ ${#pids[@]} -eq 0 ]]; then
 echo "ERROR: no ColabFold workers launched" >&2
 exit 1
fi
# Report which worker died and why. Bare `wait` under set -e aborted the run
# with nothing on stdout, leaving the reason buried in a per-worker log.
FAILED=0
for i in "${!pids[@]}"; do
 w="${WORKER_IDS[$i]}"
 if ! wait "${pids[$i]}"; then
  FAILED=$(( FAILED + 1 ))
  echo "ERROR: ColabFold worker ${w} failed; last lines of its log:" >&2
  tail -n 15 "${PRED_DIR}/worker_${w}/worker_${w}.log" >&2 || true
 fi
done
if [[ "$FAILED" -gt 0 ]]; then
 echo "ERROR: ${FAILED} of ${#pids[@]} ColabFold workers failed." >&2
 echo "Predictions from surviving workers are kept in ${PRED_DIR}/worker_*/." >&2
 echo "To resume from them: consolidate the PDBs and rerun with SKIP_COLABFOLD=1." >&2
fi

# Consolidate before bailing out, so a partial run stays resumable.
for d in "${PRED_DIR}"/worker_*/; do
 cp -n "$d"*.pdb "$d"*.json "$d"*.a3m "$PRED_DIR"/ 2>/dev/null || true
done
N_PDB=$(ls -1 "${PRED_DIR}"/*.pdb 2>/dev/null | wc -l)
echo "[COLABFOLD] consolidated ${N_PDB} PDBs into ${PRED_DIR}"
if [[ "$FAILED" -gt 0 ]]; then exit 1; fi
if [[ "$N_PDB" -eq 0 ]]; then
 echo "ERROR: no PDBs were produced." >&2
 exit 1
fi

python "$SCRIPTS_DIR/02_plddt_rmsd_kmeans.py" \
 --uniprot "$UNIPROT" --pred-dir "$PRED_DIR" --out-dir "$ANALYSIS_DIR"

python "$SCRIPTS_DIR/03_run_rosetta_nc.py" \
 --uniprot "$UNIPROT" \
 --rep-info "${ANALYSIS_DIR}/${UNIPROT}_rep_info.tsv" \
 --out-dir "$ROSETTA_OUT" --rosetta-bin "$ROSETTA_BIN" \
 --method "$NC_METHOD"

python "$SCRIPTS_DIR/05_pairwise_delta_nc.py" \
 --uniprot "$UNIPROT" \
 --nc-tsv "${ROSETTA_OUT}/${UNIPROT}_rosetta_nc.tsv" \
 --out-dir "$PAIRWISE_OUT" \
 --threshold "$PAIRWISE_THRESHOLD"

if [[ "$RUN_LEGACY_DECISION" == "1" ]]; then
 LABEL_ARGS=()
 [[ -n "${LABELS_SOURCE:-}" ]] && LABEL_ARGS+=(--labels-source "$LABELS_SOURCE")
 [[ -n "${LABELS_DIR:-}" ]] && LABEL_ARGS+=(--labels-dir "$LABELS_DIR")
 python "$SCRIPTS_DIR/04_labeling_decision_pipeline_all_in_one.py" \
 --uniprot "$UNIPROT" --nc-tsv "${ROSETTA_OUT}/${UNIPROT}_rosetta_nc.tsv" \
 --out-dir "$DECISION_OUT" "${LABEL_ARGS[@]}"
fi

echo "[DONE] Pipeline outputs in: ${WORK_DIR}"
echo "  rosetta/   NC (${NC_METHOD})"
echo "  pairwise/  ΔNC reporters + reagent targets (T=${PAIRWISE_THRESHOLD})"
