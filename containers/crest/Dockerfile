# ============================================================================
# YARP Stage 2.A: CREST Conformer Generation Container
# ============================================================================
# Published as erm42/yarp:crest. YARP pins the image by digest in
# yarp/reaction/external/conf_gen.py (CREST_IMAGE); update that constant
# whenever the image is republished.
#
# Build for x86-64 (also on an Apple-silicon Mac), from the repo root:
#   docker build --platform linux/amd64 -t <user>/yarp:crest containers/crest
#
# Pinned so a rebuild gives the image YARP was validated with:
#   - Miniforge 26.1.0-0 (checksum-verified);
#   - the yarp_crest environment, installed from conda-linux-64.lock (an exact
#     export: every package's URL and hash, no solving);
#   - the CREST 3.0.2 sources (checksums in crest_3.0.2_sources.txt).
# To update xtb, CREST or any library: regenerate the lock file, rebuild, and
# revalidate (debug/261104_crest_crash/, local, not in the repo).
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
ARG MINIFORGE_VERSION=26.1.0-0
ARG MINIFORGE_SHA256=127b5e14cfe6c83b787f624487cdc2168645ed82fdca1b0c1937caa086aed6d5
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum -c - && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh

ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Initialize conda
RUN conda init bash && \
    echo "conda activate yarp_crest" >> ~/.bashrc

# ============================================================================
# 3. Create the conda environment (CREST / xTB) from the lock file
# ============================================================================
# conda-linux-64.lock is an exact --explicit export of the validated image's
# environment (crest 3.0.2, xtb 6.7.1, libgfortran 15.2.0, OpenBLAS 0.3.30, ...),
# so a rebuild installs those packages instead of the newest ones.
COPY conda-linux-64.lock /tmp/conda-linux-64.lock
RUN conda create -n yarp_crest -y --file /tmp/conda-linux-64.lock \
    && rm /tmp/conda-linux-64.lock \
    && conda clean -afy

# Set environment variables to use the new environment
ENV CONDA_DEFAULT_ENV=yarp_crest
ENV PATH="${CONDA_DIR}/envs/yarp_crest/bin:${PATH}"

# ============================================================================
# 4. Rebuild CREST 3.0.2 with the GFN-FF free-CO/NO fix
# ============================================================================
# Rebuilds crest from the CREST 3.0.2 sources with the build settings
# conda-forge uses (crest-feedstock: sources + checksums in
# crest_3.0.2_sources.txt, patches/0001, cmake RelWithDebInfo,
# -ffree-line-length-none), plus one fix:
#   patches/0002: GFN-FF no longer treats a free CO or NO as a carbonyl/nitro
#     H-bond acceptor, which segfaulted CREST's NCI flexibility step
#     (e.g. CCOO.[C-]#[O+]). Same guard as xtb >= 6.6. Drop it once a CREST
#     release bundles a GFN-FF with the guard.
# Compiler: conda-forge gcc/gfortran 14.3.0, the one that built conda-forge's
# crest 3.0.2 (gcc 15 fails: C23 makes `true` a keyword in symmetry_i.c).
# The build environment is temporary; the binary runs on the yarp_crest
# libraries above. conda-forge's binary stays at
# /opt/conda/envs/yarp_crest/bin/crest.conda-forge-3.0.2.
SHELL ["/bin/bash", "-c"]
ARG CREST_BUILD_JOBS=3

COPY crest_3.0.2_sources.txt /opt/crest_build/sources.txt
COPY patches/0001-double-allocation.patch \
     patches/0002-gfnff-free-co-no-hbond-guard.patch \
     /opt/crest_build/

RUN set -eux; \
    B=/opt/crest_build; \
    RT=/opt/conda/envs/yarp_crest; \
    \
    # Build environment: compilers and the same runtime library versions as yarp_crest.
    conda create -y -p $B/env --override-channels -c conda-forge \
        "gfortran=14.3.0" "gcc=14.3.0" "gxx=14.3.0" cmake ninja patch patchelf \
        "libgfortran5=15.2.0" "libblas=3.11.0=*openblas" "liblapack=3.11.0=*openblas" \
        "libopenblas=0.3.30" wget; \
    export PATH=$B/env/bin:$PATH; \
    \
    # Sources, checksum-verified, laid out as in the conda-forge recipe.
    mkdir -p $B/src && cd $B/src; \
    while read -r folder url sha; do \
        wget -q -O /tmp/src.tgz "$url"; \
        echo "$sha  /tmp/src.tgz" | sha256sum -c -; \
        mkdir -p "$folder"; \
        tar -xzf /tmp/src.tgz -C "$folder" --strip-components=1; \
        rm /tmp/src.tgz; \
    done < $B/sources.txt; \
    \
    patch -p1 < $B/0001-double-allocation.patch; \
    (cd subprojects/gfnff && patch -p1 < $B/0002-gfnff-free-co-no-hbond-guard.patch); \
    grep -q "nbnbk .gt. 1" subprojects/gfnff/src/gfnff_engrad.f90; \
    \
    # conda-forge build steps (recipe meta.yaml, build: script)
    cp -r subprojects/test-drive subprojects/toml-f/subprojects/test-drive; \
    export FC=$B/env/bin/gfortran CC=$B/env/bin/gcc; \
    # conda-forge's default linux-64 FFLAGS, plus the recipe's -ffree-line-length-none
    export CFLAGS="-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe"; \
    export FFLAGS="-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -ffree-line-length-none"; \
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_TESTS=OFF \
          -DCMAKE_PREFIX_PATH=$B/env -DCMAKE_INSTALL_PREFIX=$B/install \
          -B build_dir -GNinja; \
    cmake --build build_dir -j ${CREST_BUILD_JOBS}; \
    cmake --install build_dir; \
    \
    # Swap the binary into yarp_crest; resolve libraries from that environment.
    mv $RT/bin/crest $RT/bin/crest.conda-forge-3.0.2; \
    cp $B/install/bin/crest $RT/bin/crest; \
    patchelf --set-rpath '$ORIGIN/../lib' $RT/bin/crest; \
    mkdir -p $RT/share/crest-patches; \
    cp $B/000[12]-*.patch $RT/share/crest-patches/; \
    \
    cd / && rm -rf $B; \
    conda clean -afy

# ============================================================================
# 5. Configure OpenMP Environment Variables for Parallelization
# ============================================================================
# 256M, not the 1G often recommended for xTB/CREST: under SGE's h_vmem (a
# per-process virtual-memory limit) a 1G stack per thread makes CREST fail with
# "libgomp: Thread creation failed". YARP also passes 256M on every CREST job
# (conf_gen.py CREST_OMP_STACKSIZE).
ENV OMP_STACKSIZE=256M
ENV OMP_MAX_ACTIVE_LEVELS=1

# Prevent thread oversubscription between xTB/OpenMP and math libraries
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# ============================================================================
# 6. Set working directory and final PATH/LD_LIBRARY_PATH
# ============================================================================
WORKDIR /work

# Add conda lib and system lib to LD_LIBRARY_PATH for OpenMPI libraries
ENV LD_LIBRARY_PATH="/opt/conda/envs/yarp_crest/lib:/usr/lib/x86_64-linux-gnu"

LABEL org.opencontainers.image.title="yarp_crest" \
      org.opencontainers.image.description="CREST 3.0.2 rebuilt (conda-forge recipe) + GFN-FF free-CO/NO H-bond guard; xTB 6.7.1"

CMD ["/bin/bash"]
