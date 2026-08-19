#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Prepare FASTA + job metadata from a job JSON.")
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--out-fasta", required=True)
    ap.add_argument("--out-meta", required=True)
    args = ap.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        job = json.load(f)

    for required in ("job_id", "sequence", "num_seeds", "models_per_seed"):
        if required not in job:
            raise ValueError(f"Job JSON is missing required field: {required}")

    uniprot = job["job_id"]
    sequence = job["sequence"].strip()
    num_seeds = int(job["num_seeds"])
    models_per_seed = int(job["models_per_seed"])
    labels_source = job.get("labels_source", "")
    labels_dir = job.get("labels_dir", "")

    Path(args.out_fasta).write_text(f">{uniprot}\n{sequence}\n", encoding="utf-8")

    with open(args.out_meta, "w", encoding="utf-8") as f:
        f.write(f"UNIPROT={uniprot}\n")
        f.write(f"NUM_SEEDS={num_seeds}\n")
        f.write(f"MODELS_PER_SEED={models_per_seed}\n")
        f.write(f"LABELS_SOURCE={labels_source}\n")
        f.write(f"LABELS_DIR={labels_dir}\n")

if __name__ == "__main__":
    main()
