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

RUN mamba create -n yarp_jo_opt -c conda-forge -y \
    python=3.10 \
    xtb \
    openbabel \
    rdkit \
    numpy \
    scipy \
    scikit-learn \
    ase \
    pyyaml \
    pip \
    && mamba clean -afy

ENV CONDA_DEFAULT_ENV=yarp_jo_opt
ENV PATH="${CONDA_DIR}/envs/yarp_jo_opt/bin:${PATH}"

RUN pip install --no-cache-dir pysisyphus

WORKDIR /opt/yarp
COPY . /opt/yarp
RUN pip install --no-cache-dir -e .

RUN echo "[xtb]" > /root/.pysisyphusrc && \
    echo "cmd=xtb" >> /root/.pysisyphusrc

ENV OMP_STACKSIZE=1G
ENV OMP_MAX_ACTIVE_LEVELS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV LD_LIBRARY_PATH="/opt/conda/envs/yarp_jo_opt/lib:/usr/lib/x86_64-linux-gnu"

WORKDIR /work
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
