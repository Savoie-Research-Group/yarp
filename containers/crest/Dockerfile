# ============================================================================
# YARP Stage 2.A: CREST Conformer Generation Container
# ============================================================================
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ============================================================================
# 1. System Dependencies (OpenMPI is critical for xTB/CREST parallelization)
# ============================================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    libopenblas-openmp-dev \
    liblapack-dev \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# 2. Install Miniforge (avoids Anaconda ToS issues)
# ============================================================================
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh

ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Initialize conda
RUN conda init bash && \
    echo "conda activate yarp_crest" >> ~/.bashrc

# ============================================================================
# 3. Create conda environment and install CREST / xTB
# ============================================================================
# Using mamba directly to install the specific chemistry/physics dependencies
RUN mamba create -n yarp_crest -c conda-forge -y \
    python=3.10 \
    xtb \
    crest \
    numpy \
    scipy \
    && mamba clean -afy

# Set environment variables to use the new environment
ENV CONDA_DEFAULT_ENV=yarp_crest
ENV PATH="${CONDA_DIR}/envs/yarp_crest/bin:${PATH}"

# ============================================================================
# 4. Configure OpenMP Environment Variables for Parallelization
# ============================================================================
# These are recommended default stack sizes for xTB/CREST to prevent segfaults
ENV OMP_STACKSIZE=1G
ENV OMP_MAX_ACTIVE_LEVELS=1

# Prevent thread oversubscription between xTB/OpenMP and math libraries
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# ============================================================================
# 5. Set working directory and final PATH/LD_LIBRARY_PATH
# ============================================================================
WORKDIR /work

# Add conda lib and system lib to LD_LIBRARY_PATH for OpenMPI libraries
ENV LD_LIBRARY_PATH="/opt/conda/envs/yarp_crest/lib:/usr/lib/x86_64-linux-gnu"

# Shell initialization for interactive use
SHELL ["/bin/bash", "-c"]

CMD ["/bin/bash"]