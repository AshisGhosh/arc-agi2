FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

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
    uv pip install adam-atan2-pytorch einops



# clone original HRM repo and apply RTX 4090 patches
# (avoid ssh authentication)
RUN git clone https://github.com/sapientinc/HRM.git HRM && \
    cd HRM && \
    git config --global url."https://github.com/".insteadOf "git@github.com:" && \
    git submodule update --init --recursive && \
    sed -i 's/adam-atan2/adam-atan2-pytorch/g' requirements.txt && \
    sed -i 's/adam_atan2/adam_atan2_pytorch/g' pretrain.py && \
    sed -i 's/AdamATan2/AdamAtan2/g' pretrain.py && \
    sed -i 's/lr=0,/lr=0.0001,/g' pretrain.py

# additional requirements for original HRM
RUN export PATH="/root/.local/bin:$PATH" && \
    uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install -r HRM/requirements.txt

# additional requirements for original HRM
RUN export PATH="/root/.local/bin:$PATH" && \
    uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install tqdm

# install streamlit for testing interface
RUN export PATH="/root/.local/bin:$PATH" && \
    uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install streamlit>=1.28.0 matplotlib

# set python path and activate venv
ENV PYTHONPATH="/workspace"
ENV PATH="/workspace/.venv/bin:$PATH"

# create a startup script that activates venv and runs streamlit
RUN echo '#!/bin/bash\n\
source /workspace/.venv/bin/activate\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

# default command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
