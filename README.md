<p><p align="center"><img height="350px" src="assets/sesemi-banner.png" /></p></p>

# Image Classification with Self-Supervised Regularization
<span><img src="https://img.shields.io/badge/license-Apache-blue" /> <img src="https://img.shields.io/badge/python->=3.6-green" /> <img src="https://img.shields.io/badge/pytorch->=1.6.0-light" /> <img src="https://img.shields.io/badge/%20-contributions--welcome-5429E6" /></span>

## Why SESEMI?
SESEMI is an open source image classification library built on PyTorch. SESEMI enables various modern supervised classifiers to be robust semi-supervised learners based on the principles of self-supervised regularization.

### Highlights and Features

* Integration with the popular [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) (timm) library for access to contemporary, high-performance supervised architectures with optional pretrained ImageNet weights. See the list of [supported backbones](https://github.com/FlyreelAI/sesemi/blob/master/models/sesemi.py)
* Demonstrated utility on large realistic image datasets and is currently competitive on the [FastAI Imagenette benchmarks](https://github.com/fastai/imagenette)
* Easy to use out-of-the-box requiring little hyper-parameter tuning across many tasks related to supervised learning, semi-supervised learning, and learning with noisy labels. In most use cases, one only needs to tune the learning rate, batch size, and backbone architecture
* Simply add unlabeled data for improved image classification without any tricks

Our goal is to expand the utility of SESEMI for the ML/CV practitioner by incorporating the latest advances in self-supervised, semi-supervised, and few-shot learning to boost the accuracy performance of conventional supervised classifiers in the limited labeled data setting. Contributions are welcome!

## Installation
Our preferred installation method is Docker, however, you can use any virtual environment tool to install the necessary Python dependencies. Below are instructions for both these methods.

First, clone this repository to your machine and enter the root directory.

```bash
$ git clone https://github.com/FlyreelAI/sesemi.git
$ cd sesemi
```

### Pip

To use pip, configure a virtual environment of choice with at least Python 3.6 (e.g. [miniconda](https://docs.conda.io/en/latest/miniconda.html)). Then install the requirements as follows:

```bash
$ pip install -r requirements.txt
```

### Docker

If you would like to use docker, then ensure you have it installed by following the instructions [here](https://docs.docker.com/get-docker/). The Dockerfile at the root can be used to build an image with the 
code in this repository. To build the image, run the following `bash` command from the project root:

```bash
$ USER_ID=$(id -u) SESEMI_IMAGE=sesemi
$ docker build \
    --build-arg USER_ID=${USER_ID} \
    -t ${SESEMI_IMAGE}:latest .
```

Note that your OS user ID is obtained through the bash command `id -u`. This command will create an image named
`sesemi:latest`.

## Getting Started

This section will go through the process of using SESEMI to train a model on [FastAI's imagewoof2 dataset](https://github.com/fastai/imagenette#imagewoof). If you don't have access to a GPU machine, 
training will work but will take a very long time.

1. Enter the project root directory.

    ```bash
    $ cd sesemi
    ```

2. Download and extract the imagewoof2 dataset to the data directory.

    ```bash
    $ mkdir data
    $ curl https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz | tar -xzv -C ./data
    ```

3. Run training using SESEMI for 80 epochs.

    If you're not using docker this can be done as follows:

    ```bash
    $ python -u open_sesemi.py \
        --data-dir ./data/imagewoof2 \
        --lr 0.1 --optimizer sgd --batch-size 32 \
        --epochs 80 --warmup-lr 0.001 --warmup-epochs 10 \
        --backbone resnet50 --run-id imagewoof_run01
    ```

    If you use docker and have [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker) installed you can instead use:

    ```bash
    $ USER_ID=$(id -u) SESEMI_IMAGE=sesemi GPUS=all
    $ docker run \
        --gpus ${GPUS} \
        -u ${USER_ID} \
        --rm --ipc=host \
        --mount type=bind,src=$(pwd),dst=/home/appuser/sesemi \
        ${SESEMI_IMAGE}:latest \
        python -u open_sesemi.py \
        --data-dir /home/appuser/sesemi/data/imagewoof2 \
        --lr 0.1 --optimizer sgd --batch-size 32 \
        --epochs 80 --warmup-lr 0.001 --warmup-epochs 10 \
        --backbone resnet50 --run-id imagewoof_run01
    ```

3. Run evaluation on the trained checkpoint.

    Without docker:

    ```bash
    $ python -u open_sesemi.py evaluate-only \
        --data-dir ./data/imagewoof2 \
        --checkpoint-path ./checkpoints/imagewoof_run01/best_val.pth
    ```

    With docker:

    ```bash
    $ USER_ID=$(id -u) SESEMI_IMAGE=sesemi GPUS=all
    $ docker run \
        --gpus ${GPUS} \
        -u ${USER_ID} \
        --rm --ipc=host \
        --mount type=bind,src=$(pwd),dst=/home/appuser/sesemi \
        ${SESEMI_IMAGE}:latest \
        python -u open_sesemi.py evaluate-only \
        --data-dir /home/appuser/sesemi/data/imagewoof2 \
        --checkpoint-path /home/appuser/sesemi/checkpoints/imagewoof_run01/best_val.pth
    ```

## Citation
If you find this work useful, consider citing the related paper:

```
@inproceedings{tran-sesemi,
  title="{Exploring Self-Supervised Regularization for Supervised and Semi-Supervised Learning}",
  author={Phi Vu Tran},
  booktitle={NeurIPS Workshop on Learning with Rich Experience: Integration of Learning Paradigms},
  year={2019}
}
```

