# ColabFold + Rosetta cone NC + pairwise ΔNC pipeline

Docker-based workflow from FASTA to covalent-labeling reporter recommendations.

This repository implements the **pairwise neighbor-count (NC) approach** used in the
11-protein benchmark: Rosetta **cone** solvent exposure on three cluster
representatives, pairwise |ΔNC| ranking, and residue→reagent mapping (Table S2).

## Pipeline steps

| Step | Script | Output |
|------|--------|--------|
| 1 | `01_run_colabfold_batch.py` | `colabfold/*.pdb` |
| 2 | `02_plddt_rmsd_kmeans.py` | `analysis/{UID}_rep_info.tsv` (3 reps) |
| 3 | `03_run_rosetta_nc.py` | `rosetta/{UID}_rosetta_nc.tsv` (**cone** NC by default) |
| 5 | `05_pairwise_delta_nc.py` | `pairwise/` — ΔNC tables, reagent targets, best-per-pair picks |

Step 4 (`04_labeling_decision_pipeline_all_in_one.py`) is the **legacy**
absolute exposed/buried gate workflow. Set `RUN_LEGACY_DECISION=1` to run it
in addition to step 5.

## Key concepts

- **Cone NC**: Rosetta `per_residue_solvent_exposure` with
  `-solvent_exposure:method cone`. Directional neighbor count (side-chain oriented),
  distinct from the default soft **sphere** method.
- **Pairwise ΔNC**: For each residue, `d12 = |NC_rep1 − NC_rep2|`, etc. A residue
  is a **reporter** when max(d12, d13, d23) ≥ T (default **T = 5**).
- **Reagent mapping**: `reagent_map.py` assigns residue-selective reagents
  (DEPC, EDC/GEE, arginine dicarbonyls, …) and broadly reactive bins
  (OH-high/medium/low, diazirine, CF3). See Table S2 in the paper.

Lower NC ≈ more solvent-exposed ≈ more labeled in the more-exposed conformer.

## Build the Docker image

```bash
git clone https://github.com/nematiaram/colabfold-rosetta-pipeline.git
cd colabfold-rosetta-pipeline
docker build -t colabfold-rosetta-pipeline .
```

Rosetta is **not** bundled. Mount the binary at runtime via `ROSETTA_BIN`.

## Run (simple mode)

```bash
docker run --rm -it --gpus all \
  -v /path/to/data:/data \
  -v /path/to/rosetta/bin/per_residue_solvent_exposure.linuxgccrelease:/opt/rosetta/per_residue_solvent_exposure.linuxgccrelease:ro \
  -e UNIPROT=Q9X6R4 \
  -e FASTA=/data/input.fasta \
  -e ROSETTA_BIN=/opt/rosetta/per_residue_solvent_exposure.linuxgccrelease \
  -e NC_METHOD=cone \
  -e PAIRWISE_THRESHOLD=5 \
  colabfold-rosetta-pipeline \
  /data/output
```

## Run from existing predictions (no ColabFold)

If you already have PDBs under `colabfold/`:

```bash
UNIPROT=A0A075Q0W3 \
ROSETTA_BIN=/path/to/per_residue_solvent_exposure \
SKIP_COLABFOLD=1 \
./run_pipeline.sh /data/output
```

## Output structure

```
output/
  colabfold/          # ColabFold PDBs
  analysis/           # rep_info.tsv, clustering plots
  rosetta/            # *_rosetta_nc.tsv, raw *.out per rep
  pairwise/           # main results
    {UID}_all_residues_with_pair_diffs.tsv
    {UID}_residues_dNC_ge_5.tsv
    {UID}_reagent_residue_detail.tsv
    {UID}_reagent_target_counts.tsv
    {UID}_best_per_pair.tsv
  decision/           # only if RUN_LEGACY_DECISION=1
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NC_METHOD` | `cone` | Rosetta NC method (`cone` or `sphere`) |
| `PAIRWISE_THRESHOLD` | `5` | Minimum \|ΔNC\| for a reporter |
| `RUN_LEGACY_DECISION` | `0` | Run legacy step 04 absolute-gate decision |
| `SKIP_COLABFOLD` | `0` | Skip step 01 if PDBs exist |
| `SKIP_CLUSTERING` | `0` | Skip step 02 if rep_info exists |
| `COLABFOLD_NUM_MODELS` | `1500` | Target ColabFold model count |

## Cluster entrypoint

For multi-worker ColabFold fan-out on shared clusters, use `entrypoint.sh`
(JSON job input, `--ncpus`). It runs the same steps 2–5 after ColabFold completes.

## Reagent map

Edit `pipeline_steps/reagent_map.py` to change residue→reagent assignments.
Residue-selective sets follow published specificities; OH bins and all-20
diazirine/CF3 mappings are operational pipeline assumptions (see file docstring).

## Requirements

- Python 3.9+
- numpy, pandas, matplotlib (`requirements.txt`)
- Rosetta 3.x `per_residue_solvent_exposure` binary
- ColabFold (for step 1 only)
