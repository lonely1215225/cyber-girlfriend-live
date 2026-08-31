FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:/root/.pixi/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      ffmpeg \
      git \
      iproute2 \
      libsndfile1 \
      nginx \
      openssl \
      pkg-config \
      procps \
      python3 \
      python3-pip \
      python3-venv \
      unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /opt/cyber-girlfriend
COPY . /opt/cyber-girlfriend

RUN chmod +x /opt/cyber-girlfriend/install.sh \
      /opt/cyber-girlfriend/scripts/*.sh \
      /opt/cyber-girlfriend/deploy/docker/entrypoint.sh \
    && cp /opt/cyber-girlfriend/deploy/docker/entrypoint.sh \
      /usr/local/bin/cyber-girlfriend-entry \
    && chmod +x /usr/local/bin/cyber-girlfriend-entry

EXPOSE 19800 8189/udp 8190

ENTRYPOINT ["/usr/local/bin/cyber-girlfriend-entry"]
