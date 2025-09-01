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
RUN export PATH="/root/.local/bin:$PATH" && \
    uv venv && \
    uv pip install -r requirements.txt

# set python path
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# default command
CMD ["/bin/bash"] 