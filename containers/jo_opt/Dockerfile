# ============================================================================
# YARP xTB/OpenBabel Joint Optimization + Pysisyphus GSM Container
# ============================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    libopenblas-openmp-dev \
    liblapack-dev \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh

ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN mamba create -n jo_opt -c conda-forge -y \
    python=3.10 \
    xtb \
    openbabel \
    rdkit \
    numpy \
    scipy \
    scikit-learn=1.3.0 \
    ase \
    pyyaml \
    ipython \
    pip \
    && mamba clean -afy

ENV CONDA_DEFAULT_ENV=jo_opt
ENV PATH="${CONDA_DIR}/envs/jo_opt/bin:${PATH}"
ENV PYSISRC=/root/.pysisyphusrc

RUN pip install --no-cache-dir pysisyphus==0.7.6.post2
RUN python -c "import importlib.metadata as md; assert md.version('scikit-learn') == '1.3.0', md.version('scikit-learn'); assert md.version('pysisyphus') == '0.7.6.post2', md.version('pysisyphus')"

WORKDIR /opt/yarp
COPY . /opt/yarp
RUN pip install --no-cache-dir -e .

RUN echo "[xtb]" > /root/.pysisyphusrc && \
    echo "cmd=xtb" >> /root/.pysisyphusrc
RUN pysis --help > /tmp/pysis_help.txt

ENV OMP_STACKSIZE=1G
ENV OMP_MAX_ACTIVE_LEVELS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV LD_LIBRARY_PATH="/opt/conda/envs/jo_opt/lib:/usr/lib/x86_64-linux-gnu"

WORKDIR /work
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
