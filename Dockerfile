# ARG COLABFOLD_IMAGE=ghcr.io/sokrypton/colabfold:1.5.3-cuda12.2.2
# FROM ${COLABFOLD_IMAGE}

# ENV DEBIAN_FRONTEND=noninteractive
# ENV MPLBACKEND=Agg

# WORKDIR /opt/pipeline

# COPY requirements.txt /opt/pipeline/requirements.txt
# RUN python -m pip install --upgrade pip \
#     && python -m pip install --no-cache-dir -r /opt/pipeline/requirements.txt

# COPY pipeline_steps /opt/pipeline/pipeline_steps
# COPY run_pipeline.sh /opt/pipeline/run_pipeline.sh

# ENV PYTHONPATH=/opt/pipeline/pipeline_steps

# RUN chmod +x /opt/pipeline/run_pipeline.sh

# ENTRYPOINT ["/opt/pipeline/run_pipeline.sh"]


ARG COLABFOLD_IMAGE=ghcr.io/sokrypton/colabfold:1.5.3-cuda12.2.2
FROM ${COLABFOLD_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV MPLBACKEND=Agg

WORKDIR /opt/pipeline

COPY requirements.txt /opt/pipeline/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /opt/pipeline/requirements.txt

# --- Install Rosetta (per ROSIE's recommended conda channel) ---
RUN if ! command -v conda >/dev/null 2>&1; then \
        curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        && bash /tmp/miniconda.sh -b -p /opt/conda \
        && rm /tmp/miniconda.sh; \
    fi
ENV PATH=/opt/conda/bin:$PATH
RUN conda install -y -c https://conda.rosettacommons.org rosetta \
    && conda clean -afy

ENV ROSETTA_BIN=/opt/conda/bin/per_residue_solvent_exposure.linuxgccrelease

COPY pipeline_steps /opt/pipeline/pipeline_steps
COPY entrypoint.sh /opt/pipeline/entrypoint.sh

ENV PYTHONPATH=/opt/pipeline/pipeline_steps

RUN chmod +x /opt/pipeline/entrypoint.sh

ENTRYPOINT ["/opt/pipeline/entrypoint.sh"]
