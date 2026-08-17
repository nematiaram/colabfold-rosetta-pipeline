#!/usr/bin/env python3
import argparse
import math
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Run ColabFold batch to generate many models."
    )
    ap.add_argument("--fasta", required=True, help="Input FASTA file.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--colabfold-bin", default="colabfold_batch",
                    help="ColabFold batch executable (default: colabfold_batch).")
    ap.add_argument("--num-models", type=int, default=1500,
                    help="Target number of models to generate (default: 1500).")
    ap.add_argument("--num-seeds", type=int, default=0,
                    help="Override number of seeds; if 0, computed from num-models.")
    ap.add_argument("--models-per-seed", type=int, default=5,
                    help="Models per seed (used to infer num-seeds).")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER,
                    help="Extra args to pass to colabfold_batch.")
    args = ap.parse_args()

    fasta = Path(args.fasta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.num_seeds and args.num_seeds > 0:
        num_seeds = args.num_seeds
    else:
        if args.models_per_seed <= 0:
            raise ValueError("--models-per-seed must be > 0")
        num_seeds = int(math.ceil(args.num_models / args.models_per_seed))

    cmd = [
        args.colabfold_bin,
        "--num-seeds", str(num_seeds),
        str(fasta),
        str(out_dir),
    ]
    if args.extra_args:
        cmd += args.extra_args

    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[DONE] ColabFold batch finished.")


if __name__ == "__main__":
    main()
