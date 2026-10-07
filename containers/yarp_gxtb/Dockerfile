# ============================================================================
# YARP g-xTB container  (published as erm42/yarp_gxtb:<version>)
#
# Used by YARP only for Pysisyphus tasks run with `lot: gxtb`. Ordinary
# `lot: xtb` tasks keep using erm42/yarp:pysis_xtb (stock pysisyphus + conda xtb).
#
# On top of erm42/yarp:pysis_xtb (Ubuntu 22.04, conda env yarp_pysisyphus) this adds
#   1. the g-xTB build of xtb (6.7.1-gxtb, statically linked) as /usr/local/bin/xtb-gxtb
#   2. pysisyphus at a pinned upstream commit + pysisyphus-gxtb.patch, which adds
#      XTB(gxtb=True) -> passes --gxtb instead of --gfn <n>
#   3. a .pysisyphusrc pointing pysisyphus' xtb calculator at xtb-gxtb
#
# Build (from this folder):
#   docker build --platform linux/amd64 -t erm42/yarp_gxtb:v1 .
# See README.md for tests and publishing.
# ============================================================================
# For reproducible builds, pin the base by digest, e.g.
#   FROM erm42/yarp:pysis_xtb@sha256:<digest>
# (get it with: docker inspect --format '{{index .RepoDigests 0}}' erm42/yarp:pysis_xtb)
# The target platform is set at build time (docker build --platform linux/amd64),
# as for the other YARP images; the g-xTB binary is x86-64 only.
FROM erm42/yarp:pysis_xtb

# --- g-xTB binary (GPL-3.0, https://github.com/grimme-lab/g-xtb) -------------
# Downloaded at build time and checked against the published SHA-256, so the
# 40 MB tarball never goes into git. Override GXTB_URL if the location changes;
# a wrong file fails the checksum and stops the build.
ARG GXTB_TARBALL=xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz
ARG GXTB_URL=https://github.com/grimme-lab/g-xtb/raw/main/binaries/${GXTB_TARBALL}
ARG GXTB_SHA256=cdfb81164f8d3300fc451dbbf2414528b533e71f7a30ec824b564cd30d6d9e5c

RUN set -eux; \
    wget -q -O /tmp/gxtb.tar.xz "${GXTB_URL}"; \
    echo "${GXTB_SHA256}  /tmp/gxtb.tar.xz" | sha256sum -c -; \
    mkdir -p /opt/gxtb; \
    tar -xJf /tmp/gxtb.tar.xz -C /opt/gxtb --strip-components=1; \
    ln -s /opt/gxtb/bin/xtb /usr/local/bin/xtb-gxtb; \
    rm /tmp/gxtb.tar.xz
COPY LICENSE.g-xtb /opt/gxtb/LICENSE
# No LD_LIBRARY_PATH change: the g-xTB binary is statically linked, and the
# base image's LD_LIBRARY_PATH (conda env + system libs) is inherited as is.

# --- pysisyphus with g-xTB support -------------------------------------------
ARG PYSIS_REPO=https://github.com/eljost/pysisyphus.git
ARG PYSIS_COMMIT=e531f04cd1a53ea9c0432070c543449f8007d6a2
COPY pysisyphus-gxtb.patch /tmp/pysisyphus-gxtb.patch
RUN set -eux; \
    git clone --quiet "${PYSIS_REPO}" /tmp/pysisyphus; \
    cd /tmp/pysisyphus; \
    git checkout --quiet "${PYSIS_COMMIT}"; \
    git apply /tmp/pysisyphus-gxtb.patch; \
    /opt/conda/envs/yarp_pysisyphus/bin/pip install --no-cache-dir --no-deps .; \
    cd /; \
    rm -rf /tmp/pysisyphus /tmp/pysisyphus-gxtb.patch

# --- point pysisyphus' xtb calculator at the g-xTB build ---------------------
# This sets the executable only. The method still comes from YARP's calc block
# (gxtb: True -> --gxtb); without it the patched calculator passes --gfn 2.
RUN printf '[xtb]\ncmd=/usr/local/bin/xtb-gxtb\n' > /root/.pysisyphusrc

LABEL org.opencontainers.image.title="yarp_gxtb" \
      org.opencontainers.image.description="Pysisyphus + g-xTB (xtb 6.7.1-gxtb) for YARP lot: gxtb tasks" \
      org.opencontainers.image.licenses="GPL-3.0"
