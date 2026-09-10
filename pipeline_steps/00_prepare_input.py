#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _validate_threshold(value, field="pairwise_threshold"):
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a number, got %r" % (field, value))
    if t <= 0:
        raise ValueError("%s must be > 0, got %s" % (field, t))
    return t


def _validate_n_clusters(value, field="n_clusters"):
    try:
        k = int(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be an integer, got %r" % (field, value))
    if k < 2:
        raise ValueError("%s must be >= 2, got %s" % (field, k))
    return k


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

    pairwise_threshold = _validate_threshold(job.get("pairwise_threshold", 5.0))
    n_clusters = _validate_n_clusters(job.get("n_clusters", 3))

    # Current pairwise / Rosetta merge path assumes exactly 3 representatives.
    if n_clusters != 3:
        raise ValueError(
            "n_clusters=%s is not supported yet (pairwise NC currently requires "
            "exactly 3 cluster representatives). Use n_clusters=3."
            % n_clusters
        )

    custom_reagents = job.get("custom_reagents", [])
    if custom_reagents is None:
        custom_reagents = []
    if custom_reagents and not isinstance(custom_reagents, list):
        raise ValueError("custom_reagents must be a JSON list of {name, residues} objects")

    # Validate early (and normalize) using reagent_map helpers when available.
    if custom_reagents:
        from reagent_map import load_custom_reagents
        custom_reagents = load_custom_reagents(custom_reagents)

    out_fasta = Path(args.out_fasta)
    out_meta = Path(args.out_meta)
    work_dir = out_meta.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    out_fasta.write_text(f">{uniprot}\n{sequence}\n", encoding="utf-8")

    custom_path = ""
    if custom_reagents:
        custom_json = work_dir / "custom_reagents.json"
        custom_json.write_text(
            json.dumps(custom_reagents, indent=2) + "\n",
            encoding="utf-8",
        )
        custom_path = str(custom_json.resolve())

    with open(out_meta, "w", encoding="utf-8") as f:
        f.write(f"UNIPROT={uniprot}\n")
        f.write(f"NUM_SEEDS={num_seeds}\n")
        f.write(f"MODELS_PER_SEED={models_per_seed}\n")
        f.write(f"LABELS_SOURCE={labels_source}\n")
        f.write(f"LABELS_DIR={labels_dir}\n")
        f.write(f"PAIRWISE_THRESHOLD={pairwise_threshold}\n")
        f.write(f"N_CLUSTERS={n_clusters}\n")
        f.write(f"CUSTOM_REAGENTS_JSON={custom_path}\n")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        raise SystemExit("ERROR: %s" % e)
