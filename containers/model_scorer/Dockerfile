# ============================================================================
# YARP conformer-pair model scorer
# ============================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniforge.sh

ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN mamba create -n model_scorer -c conda-forge -y \
    python=3.10 \
    numpy=1.26 \
    scipy \
    pandas \
    scikit-learn=1.3.0 \
    && mamba clean -afy

ENV CONDA_DEFAULT_ENV=model_scorer
ENV PATH="${CONDA_DIR}/envs/model_scorer/bin:${PATH}"

WORKDIR /opt/yarp_model_scorer
COPY containers/model_scorer/score_model.py /opt/yarp_model_scorer/score_model.py
COPY containers/model_artifacts/model_scorer/*.sav /opt/yarp_models/

RUN python -c "import importlib.metadata as md; assert md.version('scikit-learn') == '1.3.0', md.version('scikit-learn')" && \
    python - <<'PY'
import numpy
assert numpy.__version__.startswith("1.26."), numpy.__version__
PY
RUN test -s /opt/yarp_models/poor_model.sav && \
    test -s /opt/yarp_models/rich_model.sav

WORKDIR /work
CMD ["/bin/bash"]
