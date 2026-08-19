#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Run ColabFold batch to generate structures.")
    ap.add_argument("--fasta", required=True,
                     help="Input FASTA, or a precomputed .a3m so workers skip the MSA step.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--colabfold-bin", default="colabfold_batch",
                     help="ColabFold batch executable (default: colabfold_batch).")
    ap.add_argument("--num-seeds", type=int, required=True,
                     help="Number of seeds this process should run.")
    ap.add_argument("--models-per-seed", type=int, required=True, choices=[1, 2, 3, 4, 5],
                     help="AlphaFold models run per seed; forwarded to colabfold_batch --num-models.")
    ap.add_argument("--random-seed", type=int, default=None,
                     help="Starting random seed (offset seeds across parallel workers).")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER,
                     help="Extra args to pass to colabfold_batch.")
    args = ap.parse_args()

    fasta = Path(args.fasta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.colabfold_bin,
        "--num-seeds", str(args.num_seeds),
        "--num-models", str(args.models_per_seed),
    ]
    if args.random_seed is not None:
        cmd += ["--random-seed", str(args.random_seed)]
    cmd += [str(fasta), str(out_dir)]
    if args.extra_args:
        cmd += args.extra_args

    total = args.num_seeds * args.models_per_seed
    print(f"[RUN] {' '.join(cmd)}")
    print(f"[INFO] this worker generates {args.num_seeds} seeds x {args.models_per_seed} models = {total} structures")
    subprocess.run(cmd, check=True)
    print("[DONE] ColabFold batch finished.")

if __name__ == "__main__":
    main()
