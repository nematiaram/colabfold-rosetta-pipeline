ARG COLABFOLD_IMAGE=ghcr.io/sokrypton/colabfold:1.5.3-cuda12.2.2
FROM ${COLABFOLD_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV MPLBACKEND=Agg

WORKDIR /opt/pipeline

COPY requirements.txt /opt/pipeline/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /opt/pipeline/requirements.txt


RUN if ! command -v conda >/dev/null 2>&1; then \
      curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      && bash /tmp/miniconda.sh -b -p /opt/conda-bootstrap \
      && rm /tmp/miniconda.sh; \
      CONDA_EXE=/opt/conda-bootstrap/bin/conda; \
    else \
      CONDA_EXE=conda; \
    fi; \
    $CONDA_EXE create -y -p /opt/rosetta -c https://conda.rosettacommons.org rosetta \
    && $CONDA_EXE clean -afy

RUN set -eux; \
    bin="$(find /opt/rosetta/bin -maxdepth 1 -name 'per_residue_solvent_exposure*' -type f -perm -u+x | sort | head -n1)"; \
    if [ -z "$bin" ]; then \
      echo "ERROR: per_residue_solvent_exposure not found. Contents of /opt/rosetta/bin:" >&2; \
      ls /opt/rosetta/bin >&2; \
      exit 1; \
    fi; \
    echo "Resolved Rosetta binary: $bin"; \
    ln -sf "$bin" /opt/rosetta-bin

ENV ROSETTA_BIN=/opt/rosetta-bin

COPY pipeline_steps /opt/pipeline/pipeline_steps
COPY entrypoint.sh /opt/pipeline/entrypoint.sh

ENV PYTHONPATH=/opt/pipeline/pipeline_steps

RUN chmod +x /opt/pipeline/entrypoint.sh

ENTRYPOINT ["/opt/pipeline/entrypoint.sh"]
