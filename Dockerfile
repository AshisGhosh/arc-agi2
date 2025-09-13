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

# install uv and add to PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# set working directory
WORKDIR /workspace

# copy requirements and install dependencies
COPY requirements.txt .
RUN uv venv .venv && \
    uv pip install -r requirements.txt --python .venv/bin/python

# activate the virtual environment and set python path
ENV VIRTUAL_ENV="/workspace/.venv"
ENV PATH="/workspace/.venv/bin:$PATH"
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# default command
CMD ["/bin/bash"] 