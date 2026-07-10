# ============================================================================
# YARP xTB/OpenBabel joint-optimization runner
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

RUN mamba create -n joint_opt -c conda-forge -y \
    python=3.10 \
    numpy \
    openbabel \
    xtb \
    && mamba clean -afy

ENV CONDA_DEFAULT_ENV=joint_opt
ENV PATH="${CONDA_DIR}/envs/joint_opt/bin:${PATH}"
ENV OMP_MAX_ACTIVE_LEVELS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

WORKDIR /opt/yarp_joint_opt
COPY containers/joint_opt/joint_opt.py /opt/yarp_joint_opt/joint_opt.py

WORKDIR /work
CMD ["/bin/bash"]
