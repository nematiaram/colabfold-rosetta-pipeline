#!/usr/bin/env python3
"""Cluster/ROSIE entrypoint for the ColabFold + Rosetta cone-NC pipeline.

Replaces the former entrypoint.sh. Console output is deliberately ASCII-only:
cluster jobs frequently run under LANG=C, where printing a non-ASCII character
raises UnicodeEncodeError and kills the run. Files are written as UTF-8.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(os.environ.get("PIPELINE_SCRIPTS_DIR", "/opt/pipeline/pipeline_steps"))


def parse_args():
    p = argparse.ArgumentParser(description="Run the ColabFold/Rosetta NC pipeline.")
    p.add_argument("--workdir", required=True, type=Path,
                   help="Working directory; all outputs are written here.")
    p.add_argument("--input", required=True, type=Path, help="JSON job file.")
    p.add_argument("--ncpus", required=True, type=int,
                   help="CPUs available to this container instance.")
    return p.parse_args()


def run(cmd, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True, env=env)


def load_env_file(path):
    """Read the shell-style env file produced by 00_prepare_input.py."""
    values = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            values[key] = value
    return values


def usable_cpus(ncpus):
    """Return (allowed_cpu_ids, ncpus, pin).

    Never assume this job owns cores 0..ncpus-1. Two distinct situations:

      * A cpuset narrower than --ncpus (SLURM, cgroup-limited containers, and
        cpuset-constrained SGE). Pinning to 0..N-1 there fails outright with
        "Invalid argument". Clamp to what we may use and pin to those ids.

      * A cpuset wider than --ncpus (measured on UGE: a 4-slot job reports
        affinity over all 36 cores). The cpuset does not describe this
        allocation, so pinning to the first N cores makes every concurrent job
        on a shared node pile onto the same low cores. Do not pin at all.
    """
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = []

    if not allowed:
        print("WARNING: could not read CPU affinity; assuming cores 0-%d." % (ncpus - 1),
              file=sys.stderr)
        return list(range(ncpus)), ncpus, False

    if ncpus > len(allowed):
        print("WARNING: --ncpus %d exceeds the %d CPUs this process may use; clamping to %d."
              % (ncpus, len(allowed), len(allowed)), file=sys.stderr)
        ncpus = len(allowed)

    pin = True
    if shutil.which("taskset") is None:
        print("WARNING: taskset not found; workers will not be pinned.", file=sys.stderr)
        pin = False
    elif len(allowed) > ncpus:
        print("NOTE: affinity is open across %d CPUs but only %d were requested;"
              % (len(allowed), ncpus), file=sys.stderr)
        print("      the cpuset does not describe this allocation, so workers run", file=sys.stderr)
        print("      unpinned to avoid colliding with other jobs on a shared node.", file=sys.stderr)
        pin = False

    return allowed, ncpus, pin


def read_cgroup_memory_limit_bytes():
    """Best-effort read of this job's cgroup memory limit in bytes, or None.

    Each ColabFold worker holds its own full copy of the AlphaFold weights (not
    shared -- see "Shared model weights and memory sizing" in the README), so
    memory, not CPU, is what actually caps how many workers can run at once.
    Requesting many CPUs without matching --mem produces exactly the failure
    mode that looks like "the CPUs aren't being used": workers get OOM-killed
    mid-run and CPU usage collapses as a symptom.

    This is unreliable under rootless podman without a systemd user session
    (no dbus to delegate a per-container memory cgroup): the container's own
    memory.max then reads "unlimited" even though an ancestor cgroup (e.g.
    SLURM's, from --mem) is the thing actually enforcing and killing on OOM,
    invisible from inside the container's own cgroup namespace. Prefer the
    explicit MEM_LIMIT_GB env var over this when the deployment can set it.
    """
    v2 = Path("/sys/fs/cgroup/memory.max")
    try:
        if v2.is_file():
            raw = v2.read_text().strip()
            if raw != "max":
                return int(raw)
            return None
    except (OSError, ValueError):
        pass

    v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    try:
        if v1.is_file():
            val = int(v1.read_text().strip())
            # cgroup v1 reports a huge sentinel (near 2**63, arch-dependent)
            # instead of "unlimited"; treat anything absurd as "no real limit".
            if val < (1 << 62):
                return val
    except (OSError, ValueError):
        pass

    return None


def max_workers_for_memory(mem_limit_bytes, mem_per_worker_gb, reserve_gb=2.0):
    """Cap worker count so NUM_WORKERS copies of the weights fit in the cgroup limit."""
    usable_gb = mem_limit_bytes / 1e9 - reserve_gb
    if usable_gb <= 0:
        return 1
    return max(1, int(usable_gb // mem_per_worker_gb))


def build_shared_msa(colabfold_bin, fasta, msa_dir, env=None):
    """Build the MSA once and return the .a3m, or None.

    Without this every worker runs colabfold_batch on the FASTA and queries the
    MMseqs2 API for the same sequence: NUM_WORKERS-1 redundant lookups before
    any folding starts, and a reliable way to get rate-limited.
    """
    msa_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(msa_dir.glob("*.a3m"))
    if existing:
        print("[MSA] reusing existing a3m in %s" % msa_dir, flush=True)
        return existing[0]

    log = msa_dir / "msa.log"
    supports_msa_only = False
    try:
        help_text = subprocess.run([colabfold_bin, "--help"], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=300,
                                   env=env).stdout.decode("utf-8", "replace")
        supports_msa_only = "--msa-only" in help_text
    except (OSError, subprocess.SubprocessError) as e:
        print("[MSA] could not query %s --help (%s)" % (colabfold_bin, e), file=sys.stderr)

    with open(log, "wb") as lf:
        if supports_msa_only:
            print("[MSA] building shared MSA (--msa-only)", flush=True)
            subprocess.run([colabfold_bin, "--msa-only", str(fasta), str(msa_dir)],
                           stdout=lf, stderr=subprocess.STDOUT, env=env)
        if not sorted(msa_dir.glob("*.a3m")):
            print("[MSA] --msa-only unavailable; falling back to a 1-seed/1-model pass",
                  flush=True)
            subprocess.run([colabfold_bin, "--num-seeds", "1", "--num-models", "1",
                            str(fasta), str(msa_dir)], stdout=lf, stderr=subprocess.STDOUT,
                           env=env)

    found = sorted(msa_dir.glob("*.a3m"))
    if found:
        print("[MSA] workers reuse %s (0 per-worker MSA queries)" % found[0], flush=True)
        return found[0]

    print("[MSA] WARNING: no a3m produced; each worker will build its own MSA.",
          file=sys.stderr)
    return None


def split_seeds(num_seeds, num_workers):
    """Exact split: the first `remainder` workers take one extra seed.

    A ceil-per-worker split over-runs the request (1000 seeds over 12 workers
    becomes 1008; 8 seeds over 32 workers becomes 32).
    """
    base, remainder = divmod(num_seeds, num_workers)
    return [base + 1 if w < remainder else base for w in range(num_workers)]


def consolidate(pred_dir):
    """Collect worker outputs into pred_dir without overwriting (cp -n)."""
    for worker_dir in sorted(pred_dir.glob("worker_*")):
        if not worker_dir.is_dir():
            continue
        for pattern in ("*.pdb", "*.json", "*.a3m"):
            for src in worker_dir.glob(pattern):
                dst = pred_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
    return len(list(pred_dir.glob("*.pdb")))


def run_colabfold(work_dir, pred_dir, fasta, uniprot, num_seeds, models_per_seed,
                  ncpus, threads_per_worker, colabfold_bin, mem_per_worker_gb=4.0,
                  mem_limit_gb=None):
    allowed, ncpus, pin = usable_cpus(ncpus)

    if threads_per_worker > ncpus:
        print("WARNING: THREADS_PER_WORKER %d exceeds the %d usable CPUs; clamping to %d."
              % (threads_per_worker, ncpus, ncpus), file=sys.stderr)
        threads_per_worker = ncpus

    num_workers = max(1, ncpus // threads_per_worker)

    if mem_limit_gb is not None:
        mem_limit = mem_limit_gb * 1e9
        mem_source = "MEM_LIMIT_GB=%.0f (explicit)" % mem_limit_gb
    else:
        mem_limit = read_cgroup_memory_limit_bytes()
        mem_source = "cgroup memory.max (auto-detected)"

    if mem_limit is not None:
        mem_cap = max_workers_for_memory(mem_limit, mem_per_worker_gb)
        if mem_cap < num_workers:
            print("WARNING: %d CPU-based workers would need roughly %.0fGB "
                  "(%.0fGB memory limit via %s, ~%.1fGB/worker assumed for "
                  "the AlphaFold weights each worker loads); clamping to %d "
                  "workers to avoid an OOM kill mid-run. Raise MEM_PER_WORKER_GB "
                  "if your sequence needs less, or request more memory."
                  % (num_workers, num_workers * mem_per_worker_gb,
                     mem_limit / 1e9, mem_source, mem_per_worker_gb, mem_cap),
                  file=sys.stderr)
            num_workers = mem_cap
    else:
        print("NOTE: no memory limit available (cgroup auto-detection failed and "
              "MEM_LIMIT_GB was not set -- this is expected under rootless podman "
              "without a systemd user session, where the container's own cgroup "
              "reports 'unlimited' even though an ancestor cgroup enforces the "
              "real --mem and will OOM-kill workers). Skipping the memory-based "
              "worker cap. Each worker holds its own copy of the AlphaFold "
              "weights (~%.1fGB/worker assumed); set MEM_LIMIT_GB to the memory "
              "actually available to this container to enable the cap."
              % mem_per_worker_gb, file=sys.stderr)

    per_worker = split_seeds(num_seeds, num_workers)

    print("HOST=%s NCPUS=%d THREADS_PER_WORKER=%d NUM_WORKERS=%d"
          % (socket.gethostname(), ncpus, threads_per_worker, num_workers), flush=True)
    print("ALLOWED_CPUS=%s" % " ".join(map(str, allowed)), flush=True)
    print("SEED_SPLIT %s" % " ".join("w%d=%d" % (i, n) for i, n in enumerate(per_worker)),
          flush=True)

    env = os.environ.copy()
    env.update({
        # Container runtimes bind $HOME, so a user's ~/.local site-packages
        # shadows the image's. A Biopython >= 1.80 there breaks AlphaFold's
        # `from Bio.Data import SCOPData` before colabfold_batch can start.
        # Scoped to the ColabFold subprocesses: steps 02/03/05 legitimately
        # need numpy and pandas, which may themselves live in ~/.local.
        "PYTHONNOUSERSITE": "1",
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(threads_per_worker),
        "OPENBLAS_NUM_THREADS": str(threads_per_worker),
        "MKL_NUM_THREADS": str(threads_per_worker),
        "XLA_FLAGS": ("--xla_cpu_multi_thread_eigen=true "
                      "intra_op_parallelism_threads=%d "
                      "inter_op_parallelism_threads=%d"
                      % (threads_per_worker, threads_per_worker)),
    })

    msa_a3m = build_shared_msa(colabfold_bin, fasta, work_dir / "msa", env=env)
    worker_input = msa_a3m if msa_a3m is not None else fasta

    processes = []
    start_seed = 0
    try:
        for w, seeds in enumerate(per_worker):
            if seeds == 0:
                continue
            worker_out = pred_dir / ("worker_%d" % w)
            worker_out.mkdir(parents=True, exist_ok=True)

            cores = allowed[w * threads_per_worker:(w + 1) * threads_per_worker]
            prefix = ["taskset", "-c", ",".join(map(str, cores))] if (pin and cores) else []

            cmd = prefix + [
                sys.executable, str(SCRIPTS_DIR / "01_run_colabfold_batch.py"),
                "--fasta", str(worker_input),
                "--out-dir", str(worker_out),
                "--colabfold-bin", colabfold_bin,
                "--num-seeds", str(seeds),
                "--models-per-seed", str(models_per_seed),
                "--random-seed", str(start_seed),
                "--extra-args", "--max-msa", "16:32", "--use-dropout",
            ]
            print("WORKER %d seeds=%d random-seed=%d cores=%s"
                  % (w, seeds, start_seed, ",".join(map(str, cores)) if prefix else "unpinned"),
                  flush=True)

            log_path = worker_out / ("worker_%d.log" % w)
            log_file = open(log_path, "wb")
            processes.append((w, subprocess.Popen(cmd, env=env, stdout=log_file,
                                                  stderr=subprocess.STDOUT), log_file,
                              log_path))
            start_seed += seeds

        if start_seed != num_seeds:
            raise RuntimeError("launched seeds %d != NUM_SEEDS %d" % (start_seed, num_seeds))
        if not processes:
            raise RuntimeError("no ColabFold workers launched")

        failed = []
        for w, proc, log_file, log_path in processes:
            rc = proc.wait()
            log_file.close()
            if rc != 0:
                failed.append((w, rc, log_path))
    finally:
        for w, proc, log_file, log_path in processes:
            if proc.poll() is None:
                proc.terminate()
            if not log_file.closed:
                log_file.close()

    for w, rc, log_path in failed:
        print("ERROR: ColabFold worker %d failed (exit %d); last lines of its log:"
              % (w, rc), file=sys.stderr)
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            for line in tail:
                print("  " + line, file=sys.stderr)
        except OSError:
            pass

    # Consolidate before bailing out, so a partial run stays resumable.
    n_pdb = consolidate(pred_dir)
    print("[COLABFOLD] consolidated %d PDBs into %s" % (n_pdb, pred_dir), flush=True)

    if failed:
        print("ERROR: %d of %d ColabFold workers failed." % (len(failed), len(processes)),
              file=sys.stderr)
        print("Predictions from surviving workers are kept in %s/worker_*/." % pred_dir,
              file=sys.stderr)
        print("To resume from them, rerun with SKIP_COLABFOLD=1.", file=sys.stderr)
        raise SystemExit(1)
    if n_pdb == 0:
        raise SystemExit("ERROR: no PDBs were produced.")


def main():
    args = parse_args()
    work_dir = args.workdir.resolve()
    input_json = args.input.resolve()

    if args.ncpus < 1:
        raise SystemExit("ERROR: --ncpus must be >= 1")

    colabfold_bin = os.environ.get("COLABFOLD_BIN", "colabfold_batch")
    rosetta_bin = os.environ.get(
        "ROSETTA_BIN", "/opt/conda/bin/per_residue_solvent_exposure.linuxgccrelease")
    threads_per_worker = int(os.environ.get("THREADS_PER_WORKER", "1"))
    mem_per_worker_gb = float(os.environ.get("MEM_PER_WORKER_GB", "4"))
    mem_limit_gb_raw = os.environ.get("MEM_LIMIT_GB", "").strip()
    mem_limit_gb = float(mem_limit_gb_raw) if mem_limit_gb_raw else None
    nc_method = os.environ.get("NC_METHOD", "cone")
    pairwise_threshold = os.environ.get("PAIRWISE_THRESHOLD", "5")
    run_legacy_decision = os.environ.get("RUN_LEGACY_DECISION", "0") == "1"

    if threads_per_worker < 1:
        raise SystemExit("ERROR: THREADS_PER_WORKER must be >= 1")
    if mem_per_worker_gb <= 0:
        raise SystemExit("ERROR: MEM_PER_WORKER_GB must be > 0")
    if mem_limit_gb is not None and mem_limit_gb <= 0:
        raise SystemExit("ERROR: MEM_LIMIT_GB must be > 0")

    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / "input.fasta"
    meta = work_dir / "job_meta.env"

    run([sys.executable, SCRIPTS_DIR / "00_prepare_input.py",
         "--input-json", input_json, "--out-fasta", fasta, "--out-meta", meta])

    meta_vars = load_env_file(meta)
    try:
        uniprot = meta_vars["UNIPROT"]
        num_seeds = int(meta_vars["NUM_SEEDS"])
        models_per_seed = int(meta_vars["MODELS_PER_SEED"])
    except KeyError as e:
        raise RuntimeError("Required variable %r missing from %s" % (e.args[0], meta))

    pred_dir = work_dir / "colabfold"
    analysis_dir = work_dir / "analysis"
    rosetta_out = work_dir / "rosetta"
    pairwise_out = work_dir / "pairwise"
    decision_out = work_dir / "decision"
    for d in (pred_dir, analysis_dir, rosetta_out, pairwise_out, decision_out):
        d.mkdir(parents=True, exist_ok=True)

    print("UNIPROT=%s NUM_SEEDS=%d MODELS_PER_SEED=%d NC_METHOD=%s"
          % (uniprot, num_seeds, models_per_seed, nc_method), flush=True)

    run_colabfold(work_dir, pred_dir, fasta, uniprot, num_seeds, models_per_seed,
                  args.ncpus, threads_per_worker, colabfold_bin,
                  mem_per_worker_gb=mem_per_worker_gb, mem_limit_gb=mem_limit_gb)

    run([sys.executable, SCRIPTS_DIR / "02_plddt_rmsd_kmeans.py",
         "--uniprot", uniprot, "--pred-dir", pred_dir, "--out-dir", analysis_dir])

    run([sys.executable, SCRIPTS_DIR / "03_run_rosetta_nc.py",
         "--uniprot", uniprot,
         "--rep-info", analysis_dir / ("%s_rep_info.tsv" % uniprot),
         "--out-dir", rosetta_out, "--rosetta-bin", rosetta_bin, "--method", nc_method])

    run([sys.executable, SCRIPTS_DIR / "05_pairwise_delta_nc.py",
         "--uniprot", uniprot,
         "--nc-tsv", rosetta_out / ("%s_rosetta_nc.tsv" % uniprot),
         "--out-dir", pairwise_out, "--threshold", pairwise_threshold])

    if run_legacy_decision:
        cmd = [sys.executable, SCRIPTS_DIR / "04_labeling_decision_pipeline_all_in_one.py",
               "--uniprot", uniprot,
               "--nc-tsv", rosetta_out / ("%s_rosetta_nc.tsv" % uniprot),
               "--out-dir", decision_out]
        if meta_vars.get("LABELS_SOURCE"):
            cmd += ["--labels-source", meta_vars["LABELS_SOURCE"]]
        if meta_vars.get("LABELS_DIR"):
            cmd += ["--labels-dir", meta_vars["LABELS_DIR"]]
        run(cmd)

    print("[DONE] Pipeline outputs in: %s" % work_dir)
    print("  rosetta/   NC (%s)" % nc_method)
    print("  pairwise/  delta-NC reporters + reagent targets (T=%s)" % pairwise_threshold)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        # A pipeline step failed. Report which one plainly rather than dumping a
        # traceback through this driver, which says nothing about the cause.
        step = Path(str(e.cmd[1])).name if len(e.cmd) > 1 else str(e.cmd)
        print("ERROR: pipeline step %s failed with exit code %d." % (step, e.returncode),
              file=sys.stderr)
        raise SystemExit(e.returncode)
    except RuntimeError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
