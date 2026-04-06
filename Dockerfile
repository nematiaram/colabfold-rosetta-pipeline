ARG COLABFOLD_IMAGE=ghcr.io/sokrypton/colabfold:1.5.3-cuda12.2.2
FROM ${COLABFOLD_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV MPLBACKEND=Agg

WORKDIR /opt/pipeline

COPY requirements.txt /opt/pipeline/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /opt/pipeline/requirements.txt

COPY pipeline_steps /opt/pipeline/pipeline_steps
COPY run_pipeline.sh /opt/pipeline/run_pipeline.sh

ENV PYTHONPATH=/opt/pipeline/pipeline_steps

RUN chmod +x /opt/pipeline/run_pipeline.sh

CMD ["bash"]
