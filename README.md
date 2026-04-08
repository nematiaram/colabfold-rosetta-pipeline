# ColabFold + Rosetta NC Pipeline Container

This repository includes a Docker image setup and a convenience script to run:

1) ColabFold batch prediction
2) pLDDT/RMSD clustering
3) Rosetta neighbor count (NC)
4) Labeling/decision outputs

The Rosetta binary is **not** bundled. Mount it at runtime and pass its path via
`ROSETTA_BIN`.

## Build the Docker image

The base image tag must exist in GHCR. If a build fails, set a valid tag from:
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

### Option A: run everything (including ColabFold)

```bash
Run the pipeline

docker run --rm -it all \
  -v /path/to/data:/data \
  -v /path/to/rosetta/bin/per_residue_solvent_exposure.linuxgccrelease:/opt/rosetta/per_residue_solvent_exposure.linuxgccrelease:ro \
  -e UNIPROT=Q9X6R4 \
  -e FASTA=/data/input.fasta \
  -e ROSETTA_BIN=/opt/rosetta/per_residue_solvent_exposure.linuxgccrelease \
  colabfold-rosetta-pipeline \
  /data/output
```

If you do not have a GPU, remove `--gpus all` and expect ColabFold to run much
slower or not at all depending on your setup.

Default ColabFold target is 1500 models. Override with:

```bash
COLABFOLD_NUM_MODELS=2000
```

```

## Label mapping inputs

The decision scripts can optionally use an existing label map:

- `LABELS_SOURCE`: TSV mapping `resname -> labels/label_non_specific`
- `LABELS_DIR`: directory to scan for `*_top10_all_reps.tsv`
- `COLABFOLD_RESULTS_DIR`: alternative default scan directory

## Using Apptainer

If your cluster lacks `mksquashfs`, you cannot build a `.sif` locally from a
sandbox. Two practical options:

1) Build the `.sif` on another machine with `mksquashfs`, then copy it over:

```bash
docker build -t colabfold-rosetta-pipeline .
apptainer build colabfold-rosetta-pipeline.sif docker-daemon://colabfold-rosetta-pipeline:latest
```

2) If your cluster supports pulling from Docker registries, you can also build
the `.sif` there once `mksquashfs` is available in PATH.

## GitHub Pages

This repo includes `docs/index.md` for a simple GitHub Pages site. After you
push to GitHub, enable Pages with **Source: `docs/`**.
Site URL: https://nematiaram.github.io/cl_conformation_pipline/

## Notes

- The Rosetta binary should be mounted read-only.
- `MPLBACKEND=Agg` is set in the image so matplotlib can render without a GUI.
