ARG PYTORCH="1.10.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0+PTX;7.5;8.0;8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y vim git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root `appuser`
# Obtain your USER_ID from the bash command `id -u`
ARG USER_ID
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"
WORKDIR /home/appuser

# Set ENV to download pretrained PyTorch models
ENV TORCH_HOME="/home/appuser/sesemi/.cache/torch/"

# Add and install dependencies
COPY --chown=appuser:appuser . sesemi/
WORKDIR /home/appuser/sesemi

RUN conda clean -y --all
RUN pip install --no-cache-dir --user --upgrade pip
RUN pip install --no-cache-dir --user -e .
