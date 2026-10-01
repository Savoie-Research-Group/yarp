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

# Pinned to rdkit=2022.03.5 to match classy_yarp/environment_yamls/env.yaml
# (its only env file with an exact, reproducible rdkit pin). Initial
# benchmark version, not a final choice.
#
# Installed from conda-linux-64.lock (an exact --explicit export, full URLs +
# hashes for every transitive dependency) rather than a loose spec, so a
# rebuild reconstructs these exact packages instead of re-solving against
# conda-forge.
COPY conda-linux-64.lock /opt/rdkit_conf/conda-linux-64.lock
RUN /opt/micromamba/bin/micromamba create -y -p /opt/rdkit-conf-env \
        --file /opt/rdkit_conf/conda-linux-64.lock \
    && /opt/micromamba/bin/micromamba clean -a -y

COPY run_conf_gen.py /opt/rdkit_conf/run_conf_gen.py

ENV LC_ALL=C
ENV PATH="/opt/micromamba/bin:${PATH}"
ENV MAMBA_ROOT_PREFIX="/opt/micromamba"
ENV PYTHONNOUSERSITE=1

WORKDIR /work

ENTRYPOINT ["/opt/micromamba/bin/micromamba", "run", "-p", "/opt/rdkit-conf-env", \
            "python", "/opt/rdkit_conf/run_conf_gen.py"]
