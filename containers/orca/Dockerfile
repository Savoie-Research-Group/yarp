# Use a lightweight Debian base (pass --platform=linux/amd64 to your build command if on Apple Silicon)
FROM debian:bookworm-slim

# Set locale and environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root

# Install essential dependencies
# SSH is required for OpenMPI cross-node communication (even in single containers)
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        curl \
        bzip2 \
        ca-certificates \
        ssh \
        tar \
        xz-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Micromamba natively for environment management safely in /tmp
RUN cd /tmp && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba && \
    mv bin/micromamba /usr/local/bin/micromamba && \
    rm -rf /tmp/bin

# Create a non-root user since OpenMPI will refuse to execute as root
ARG USER=orcauser
ARG GROUP=orca
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${GROUP} && \
    useradd -l -u ${UID} -g ${GROUP} -s /bin/bash -m ${USER}

# Switch to the unprivileged user
USER ${UID}:${GID}
WORKDIR /home/${USER}

# Create a micromamba environment containing exactly OpenMPI 4.1.6, xTB, and CREST
RUN micromamba create -y -p /home/${USER}/env -c conda-forge openmpi=4.1.6 xtb crest && \
    micromamba clean --all --yes

# Copy the ORCA tarball into the image and extract it
# Assuming it is named 'orca_6_0_1_linux_x86-64_shared_openmpi416.tar.xz'
COPY --chown=${USER}:${GROUP} orca_6_0_1_linux_x86-64_shared_openmpi416.tar.xz /home/${USER}/
RUN mkdir /home/${USER}/orca && \
    tar -xf /home/${USER}/orca_6_0_1_linux_x86-64_shared_openmpi416.tar.xz -C /home/${USER}/orca --strip-components=1 && \
    rm /home/${USER}/orca_6_0_1_linux_x86-64_shared_openmpi416.tar.xz

# Expose the conda environment binaries/libraries and ORCA directly to the system paths
ENV PATH="/home/${USER}/env/bin:/home/${USER}/orca:${PATH}"
# Set LD_LIBRARY_PATH directly to avoid undefined variable warnings
ENV LD_LIBRARY_PATH="/home/${USER}/env/lib:/home/${USER}/orca"

# Prevent OpenMPI warnings regarding memory/network architectures when running inside Docker
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV OMPI_MCA_rmaps_base_oversubscribe=1
ENV OMPI_MCA_btl="^openib"

CMD ["/bin/bash"]