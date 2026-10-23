FROM ubuntu:22.04

RUN set -eux \
    && apt-get -y update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
        ca-certificates \
        bzip2 \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/micromamba/bin \
    && wget -qO- "https://micro.mamba.pm/api/micromamba/linux-64/latest" \
       | tar -xvj -C /opt/micromamba/bin --strip-components=1 bin/micromamba

# Auto3D 3.2.2 (the first release whose spin-aware AIMNet2 runs) on the minimum
# versions it allows: python 3.11, rdkit 2022.09.5, torch 2.9.1 CPU (torchani
# 2.8's conda-forge builds need libtorch >= 2.9.1), torchani 2.8.2. CPU only;
# no CUDA packages.
#
# conda-linux-64.lock is an exact --explicit export (full URLs + hashes for every
# conda package), so a rebuild reconstructs these packages instead of
# re-solving. The four pip-only packages are installed on top with --no-deps
# and pinned hashes: a resolving pip would replace conda's rdkit with PyPI's.
COPY conda-linux-64.lock requirements-pip.lock /opt/auto3d_conf/
RUN /opt/micromamba/bin/micromamba create -y -p /opt/auto3d-conf-env \
        --file /opt/auto3d_conf/conda-linux-64.lock \
    && /opt/auto3d-conf-env/bin/python -m pip install --no-cache-dir --no-deps --require-hashes \
        -r /opt/auto3d_conf/requirements-pip.lock \
    && /opt/micromamba/bin/micromamba clean -a -y

# AIMNet2 weights are not in the wheel; aimnet downloads them on first use.
# Fetch both models YARP offers now (aimnet checks each file's sha256 against
# its registry), so jobs never need the network, and make them world-readable:
# aimnet writes them 0600.
ENV AIMNET_CACHE_DIR=/opt/auto3d_conf/aimnet_cache
RUN /opt/auto3d-conf-env/bin/python -c "from aimnet.calculators.model_registry import get_model_path; [get_model_path(m) for m in ('aimnet2', 'aimnet2-nse')]" \
    && chmod -R a+rX /opt/auto3d_conf/aimnet_cache

COPY run_conf_gen.py /opt/auto3d_conf/run_conf_gen.py

ENV LC_ALL=C
ENV PATH="/opt/micromamba/bin:${PATH}"
ENV MAMBA_ROOT_PREFIX="/opt/micromamba"
ENV PYTHONNOUSERSITE=1
ENV TQDM_DISABLE=1

WORKDIR /work

ENTRYPOINT ["/opt/micromamba/bin/micromamba", "run", "-p", "/opt/auto3d-conf-env", \
            "python", "/opt/auto3d_conf/run_conf_gen.py"]
