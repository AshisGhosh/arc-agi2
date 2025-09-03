FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# install uv and add to PATH in the same RUN command
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv --version

# set working directory
WORKDIR /workspace

# copy requirements and install dependencies
COPY requirements.txt .
# base requirements
RUN export PATH="/root/.local/bin:$PATH" && \
    uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install torch>=2.0.0 numpy pyyaml && \
    uv pip install flash-attn --no-build-isolation && \
    uv pip install adam-atan2 einops

# additional requirements
RUN export PATH="/root/.local/bin:$PATH" && \
    uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install tqdm

# set python path and activate venv
ENV PYTHONPATH="/workspace:$PYTHONPATH"
ENV PATH="/workspace/.venv/bin:$PATH"

# default command
CMD ["/bin/bash"]
