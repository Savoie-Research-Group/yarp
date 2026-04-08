# ============================================================================
# YARP Stage 2.B: Pysisyphus (GSM) Container
# ============================================================================
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ============================================================================
# 1. System Dependencies 
# ============================================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    libopenblas-openmp-dev \
    liblapack-dev \
    libopenmpi-dev \
    openmpi-bin \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# 2. Install Miniforge
# ============================================================================
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh

ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN conda init bash && \
    echo "conda activate yarp_pysisyphus" >> ~/.bashrc

# ============================================================================
# 3. Create conda environment and install Pysisyphus / xTB
# ============================================================================
RUN mamba create -n yarp_pysisyphus -c conda-forge -y \
    python=3.10 \
    xtb \
    numpy \
    scipy \
    pip \
    && mamba clean -afy

ENV CONDA_DEFAULT_ENV=yarp_pysisyphus
ENV PATH="${CONDA_DIR}/envs/yarp_pysisyphus/bin:${PATH}"

RUN pip install --no-cache-dir pysisyphus

# ============================================================================
# 4. Configure OpenMP Environment Variables for Parallelization
# ============================================================================
ENV OMP_STACKSIZE=1G
ENV OMP_MAX_ACTIVE_LEVELS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# ============================================================================
# 5. Set up .pysisyphusrc Configuration
# ============================================================================
# Pysisyphus requires this file in the user's home directory to locate external 
# binaries. Because xTB is in our conda PATH, we can just use the 'xtb' command.
RUN echo "[xtb]" > /root/.pysisyphusrc && \
    echo "cmd=xtb" >> /root/.pysisyphusrc

# Note: If you ever add ORCA to this container, you would append:
# RUN echo "[orca]" >> /root/.pysisyphusrc && \
#     echo "cmd=/path/to/orca" >> /root/.pysisyphusrc

# ============================================================================
# 6. Set working directory and final PATH/LD_LIBRARY_PATH
# ============================================================================
WORKDIR /work

ENV LD_LIBRARY_PATH="/opt/conda/envs/yarp_pysisyphus/lib:/usr/lib/x86_64-linux-gnu"

SHELL ["/bin/bash", "-c"]

CMD ["/bin/bash"]