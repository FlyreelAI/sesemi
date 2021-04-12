ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
# Set ENV to download pretrained PyTorch models
ARG PYTORCH_HOME
ENV TORCH_HOME=${PYTORCH_HOME}

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

# Clone SESEMI and install dependencies
RUN git clone https://github.com/FlyreelAI/sesemi.git
WORKDIR /home/appuser/sesemi
RUN conda clean -y --all
RUN pip install --no-cache-dir --user --upgrade pip
RUN pip install --no-cache-dir --user -r docker/requirements.txt

