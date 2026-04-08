# ColabFold + Rosetta NC Pipeline Container

This repository contains a Docker-based pipeline that runs:

1. ColabFold batch prediction
2. pLDDT/RMSD clustering
3. Rosetta neighbor count (NC)
4. Labeling/decision outputs

The pipeline starts from a **FASTA file** and runs the full workflow from structure prediction to final ranking outputs.

The Rosetta binary is **not** included in the container. Please mount it at runtime and pass its path through `ROSETTA_BIN`.

## Build the Docker image

First clone the repository and move into it:

```bash
git clone https://github.com/nematiaram/colabfold-rosetta-pipeline.git
cd colabfold-rosetta-pipeline
```

Then build the Docker image.

The base image tag must exist in GHCR. If a build fails, use a valid tag from:
https://github.com/sokrypton/ColabFold/pkgs/container/colabfold

```bash
docker build -t colabfold-rosetta-pipeline .
```

Override the base image tag if needed:

```bash
docker build \
  --build-arg COLABFOLD_IMAGE=ghcr.io/sokrypton/colabfold:1.5.3-cuda12.2.2 \
  -t colabfold-rosetta-pipeline .
```

## Run the pipeline

The container expects these environment variables:

- `UNIPROT`
- `FASTA`
- `ROSETTA_BIN`

The final argument to the container is the **working directory**, and all outputs will be written there.

Example:

```bash
docker run --rm -it --gpus all \
  -v /path/to/data:/data \
  -v /path/to/rosetta/bin/per_residue_solvent_exposure.linuxgccrelease:/opt/rosetta/per_residue_solvent_exposure.linuxgccrelease:ro \
  -e UNIPROT=Q9X6R4 \
  -e FASTA=/data/input.fasta \
  -e ROSETTA_BIN=/opt/rosetta/per_residue_solvent_exposure.linuxgccrelease \
  colabfold-rosetta-pipeline \
  /data/output
```

In this example, all pipeline outputs will be written under:

```bash
/data/output
```

If you do not have a GPU, remove `--gpus all`. ColabFold may run much slower depending on your setup.

The default ColabFold target is 1500 models. To override it, use for example:

```bash
docker run --rm -it --gpus all \
  -v /path/to/data:/data \
  -v /path/to/rosetta/bin/per_residue_solvent_exposure.linuxgccrelease:/opt/rosetta/per_residue_solvent_exposure.linuxgccrelease:ro \
  -e UNIPROT=Q9X6R4 \
  -e FASTA=/data/input.fasta \
  -e ROSETTA_BIN=/opt/rosetta/per_residue_solvent_exposure.linuxgccrelease \
  -e COLABFOLD_NUM_MODELS=2000 \
  colabfold-rosetta-pipeline \
  /data/output
```

## Output structure

The pipeline writes results inside the working directory you provide, including:

- `colabfold/`
- `analysis/`
- `rosetta/`
- `decision/`

## Label mapping inputs

The decision step can optionally use label mapping inputs:

- `LABELS_SOURCE`: TSV mapping `resname -> labels/label_non_specific`
- `LABELS_DIR`: directory to scan for `*_top10_all_reps.tsv`

## Using Apptainer

If your cluster lacks `mksquashfs`, you may not be able to build a `.sif` locally from a sandbox. Two practical options are:

1. Build the `.sif` on another machine with `mksquashfs`, then copy it over:

```bash
docker build -t colabfold-rosetta-pipeline .
apptainer build colabfold-rosetta-pipeline.sif docker-daemon://colabfold-rosetta-pipeline:latest
```

2. If your cluster supports pulling from Docker registries, build the `.sif` there once `mksquashfs` is available in `PATH`.
