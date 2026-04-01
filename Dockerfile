FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

ENV HF_HOME=/app/models

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3 python3-pip python3-dev python3-venv \
    git git-lfs wget curl htop nano \
    openssh-server nginx \
    build-essential cmake \
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Python/pip symlinks
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy unsloth custom library to image
COPY ./unsloth /opt/unsloth/

# Copy execution files to services
COPY run.sh /services/
COPY start.sh /services/
COPY main.py /services/

# Set execution rights
RUN chmod +x /services/run.sh /services/start.sh

# Install jupyter notebook
RUN uv pip install --system notebook ipywidgets ipykernel

# Setup DeepLearning Trainer libraries
ENTRYPOINT ["/bin/bash", "/opt/unsloth/setup.sh"]

# Change directory to workspace    
WORKDIR /workspace

# Execute script & Make the container active
CMD ["/bin/bash", "/services/run.sh"]