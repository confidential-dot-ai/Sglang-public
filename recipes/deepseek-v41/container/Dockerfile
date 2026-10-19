# DeepSeek-V4.1 PD image: the resolved requirements.lock environment plus this
# repository's sglang and V4.1 launcher. One image serves both containers in
# compose.yaml (the Engram holder and the PD pair). Build with build.sh.

# The host's CUDA toolkit (cuda-toolkit-13-0 13.0.3); FlashInfer and Triton JIT
# compile against it at runtime.
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04
FROM ghcr.io/astral-sh/uv:0.11.16 AS uv

# The PD router (sglang_router), built from this checkout's sgl-model-gateway
# instead of the PyPI wheel so router fixes ship with the image. Rust matches
# sgl-model-gateway/rust-toolchain.toml; the lockfile pins the dependency graph.
FROM rust:1.90-bookworm AS router
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build/sgl-model-gateway
COPY sgl-model-gateway .
COPY recipes/deepseek-v41/container/sglang-router.Cargo.lock Cargo.lock
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/sgl-model-gateway/target \
    cd bindings/python \
    && uv tool run --from 'maturin[patchelf]==1.9.6' maturin build --release --locked --out /wheels

FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
# Mooncake links ibverbs/mlx5, libnl, numa and curl even for intra-node NVLink.
# FFmpeg 6.1 matches the host; without it torchcodec logs load failures at import.
# OpenSSL headers: the HiCache storage hash extension JIT-builds against them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential ca-certificates curl git procps libssl-dev \
        ibverbs-providers libibverbs1 libnuma1 libcurl4t64 libnl-3-200 libnl-route-3-200 \
        libavcodec60 libavdevice60 libavfilter9 libavformat60 libavutil58 libswresample4 libswscale7 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/dsv41 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0
ENV PATH=/opt/dsv41/bin:${PATH}
RUN uv venv --python /usr/bin/python3.12 /opt/dsv41

COPY recipes/deepseek-v41/container/requirements.lock /tmp/requirements.lock
# Epoch mtimes: FlashInfer's ninja JIT cache compares source and object mtimes,
# so freshly installed sources would recompile every op (~20 min) per image rebuild.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -r /tmp/requirements.lock \
    && find /opt/dsv41 -xdev -exec touch -h -d @0 {} +

# Replaces the lock's PyPI sglang-router with the router built above; a later
# layer, so a router change doesn't reinstall the environment.
COPY --from=router /wheels /tmp/router-wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps --reinstall /tmp/router-wheels/sglang_router-*.whl \
    && rm -rf /tmp/router-wheels

# Mooncake's transfer engine links the CUDA 12 runtime from this wheel.
ENV LD_LIBRARY_PATH=/opt/dsv41/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}

WORKDIR /opt/sglang
COPY python python
# Editable, so a bind mount of a checkout over /opt/sglang/python replaces the code.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install setuptools-scm==10.3.4 wheel==0.48.0 \
    && SGLANG_BUILD_RUST_EXTS=none uv pip install --no-deps --no-build-isolation -e python

COPY recipes/deepseek-v41 recipes/deepseek-v41

ARG SOURCE_RECORD
RUN test -n "${SOURCE_RECORD}" && printf '%s\n' "${SOURCE_RECORD}" > SOURCE_RECORD.json \
    && python -c "import json; json.load(open('SOURCE_RECORD.json'))"

ENTRYPOINT ["recipes/deepseek-v41/container/entrypoint.sh"]
CMD ["pd"]
