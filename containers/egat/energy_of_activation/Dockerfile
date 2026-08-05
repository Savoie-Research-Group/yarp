FROM --platform=linux/amd64 ubuntu:22.04

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

RUN /opt/micromamba/bin/micromamba create -y -p /opt/egat-env \
        -c conda-forge -c pytorch -c dglteam/label/th24_cpu \
        python=3.11 pip numpy pandas scipy omegaconf rdkit openbabel ase \
        pytorch=2.4.* cpuonly dgl=2.4.* ipython xgboost pytest \
    && /opt/micromamba/bin/micromamba clean -a -y

COPY src            /opt/egat/src
COPY egat_predict_reaction_csv.py /opt/egat/egat_predict_reaction_csv.py
COPY models         /opt/egat/models
COPY tests          /opt/egat/tests
COPY environment.yml /opt/egat/environment.yml

RUN chmod +x /opt/egat/egat_predict_reaction_csv.py 2>/dev/null || true

ENV LC_ALL=C
ENV PATH="/opt/micromamba/bin:${PATH}"
ENV MAMBA_ROOT_PREFIX="/opt/micromamba"
ENV PYTHONNOUSERSITE=1
ENV PYTHONPATH=/opt/egat/src

ENTRYPOINT ["/opt/micromamba/bin/micromamba", "run", "-p", "/opt/egat-env", \
            "python", "/opt/egat/egat_predict_reaction_csv.py"]
