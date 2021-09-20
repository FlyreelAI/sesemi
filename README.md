<p><p align="center"><img height="350px" src="https://github.com/FlyreelAI/sesemi/raw/master/assets/sesemi-banner.png" /></p></p>

# Image Classification with Self-Supervised Regularization
<span><img src="https://img.shields.io/badge/license-Apache-blue" /> <img src="https://img.shields.io/badge/python->=3.6-green" /> <img src="https://img.shields.io/badge/pytorch->=1.6.0-light" /> <img src="https://img.shields.io/badge/%20-contributions--welcome-5429E6" /></span>

## Why SESEMI?
SESEMI is an open source image classification library built on PyTorch and PyTorch Lightning. SESEMI enables various modern supervised classifiers to be robust semi-supervised learners based on the principles of self-supervised regularization.

### Highlights and Features

* Integration with the popular [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) (timm) library for access to contemporary, high-performance supervised architectures with optional pretrained ImageNet weights. See the list of [supported backbones](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/models/backbones/timm.py)
* Demonstrated utility on large realistic image datasets and is currently competitive on the [FastAI Imagenette benchmarks](https://github.com/fastai/imagenette)
* Easy to use out-of-the-box requiring little hyper-parameter tuning across many tasks related to supervised learning, semi-supervised learning, and learning with noisy labels. In most use cases, one only needs to tune the learning rate, batch size, and backbone architecture
* Simply add unlabeled data for improved image classification without any tricks

Our goal is to expand the utility of SESEMI for the ML/CV practitioner by incorporating the latest advances in self-supervised, semi-supervised, and few-shot learning to boost the accuracy performance of conventional supervised classifiers in the limited labeled data setting. If you find this work useful please star this repo to let us know. Contributions are also welcome!

## Installation
Our preferred installation method is Docker, however, you can use any virtual environment tool to install the necessary Python dependencies. Below are instructions for both these methods.

### Pip

To use pip, configure a virtual environment of choice with at least Python 3.6 (e.g. [miniconda](https://docs.conda.io/en/latest/miniconda.html)). Then install the requirements as follows:

```bash
$ pip install git+https://github.com/FlyreelAI/sesemi.git
```

While the above installs the latest version from the main branch, a version from PyPI can be installed instead as follows:

```bash
$ pip install sesemi
```

### Docker

If you would like to use docker, then ensure you have it installed by following the instructions [here](https://docs.docker.com/get-docker/). The Dockerfile at the root can be used to build an image with the 
code in this repository. To build the image, run the following `bash` command :

```bash
$ USER_ID=$(id -u) SESEMI_IMAGE=sesemi
$ DOCKER_BUILDKIT=1 docker build \
    --build-arg USER_ID=${USER_ID} \
    -t ${SESEMI_IMAGE}:latest https://github.com/FlyreelAI/sesemi.git
```

Note that your OS user ID is obtained through the bash command `id -u`. This command will create an image named
`sesemi:latest`.

## Getting Started

You can find more detailed documentation which is hosted [here](https://flyreelai.github.io/sesemi/), however, this section will guide you through the process of using SESEMI to train a model on [FastAI's imagewoof2 dataset](https://github.com/fastai/imagenette#imagewoof). If you don't have access to a GPU machine, 
training will work but will take a very long time.

1. Create a directory for the experiment and enter it.

    ```bash
    $ mkdir sesemi-experiments
    $ cd sesemi-experiments
    $ mkdir data runs .cache
    ```

2. Download and extract the imagewoof2 dataset to the data directory.

    ```bash
    $ curl https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz | tar -xzv -C ./data
    ```

3. Run training using SESEMI for 80 epochs. You should get 90-91% accuracy on the imagewoof2 dataset, which is competitive on the [FastAI leaderboard](https://github.com/fastai/imagenette#imagewoof-leaderboard), using a standard training protocol + unlabeled data, without fancy tricks.

    If you're not using docker this can be done as follows:

    ```bash
    $ open_sesemi -cn imagewoof_rotpred
    ```

    If you use docker and have [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed you can instead use:

    ```bash
    $ USER_ID=$(id -u) SESEMI_IMAGE=sesemi GPUS=all
    $ docker run \
        --gpus ${GPUS} \
        -u ${USER_ID} \
        --rm --ipc=host \
        --mount type=bind,src=$(pwd),dst=/home/appuser/sesemi-experiments/ \
        -w /home/appuser/sesemi-experiments \
        ${SESEMI_IMAGE}:latest \
        open_sesemi -cn imagewoof_rotpred
    ```

    The training logs with all relevant training statistics (accuracy, losses, learning rate, etc.) are written to the `./runs` directory. You can use [TensorBoard](https://www.tensorflow.org/tensorboard) to view and monitor them in your browser during training.
    
    ```bash
    $ tensorboard --logdir ./runs
    ```
    
3. Run evaluation on the trained checkpoint.

    Without docker:

    ```bash
    $ CHECKPOINT_PATH=$(echo ./runs/imagewoof_rotpred/*/lightning_logs/version_0/checkpoints/last.ckpt)
    $ open_sesemi -cn imagewoof_rotpred \
        run.mode=VALIDATE \
        run.pretrained_checkpoint_path=$CHECKPOINT_PATH
    ```

    With docker:

    ```bash
    $ USER_ID=$(id -u) SESEMI_IMAGE=sesemi GPUS=all
    $ CHECKPOINT_PATH=$(echo ./runs/imagewoof_rotpred/*/lightning_logs/version_0/checkpoints/last.ckpt)
    $ docker run \
        --gpus ${GPUS} \
        -u ${USER_ID} \
        --rm --ipc=host \
        --mount type=bind,src=$(pwd),dst=/home/appuser/sesemi-experiments/ \
        -w /home/appuser/sesemi-experiments \
        ${SESEMI_IMAGE}:latest \
        open_sesemi -cn imagewoof_rotpred \
            run.mode=VALIDATE \
            run.pretrained_checkpoint_path=$CHECKPOINT_PATH
    ```

## Citation
If you find this work useful, consider citing the related paper:

```
@inproceedings{TranSESEMI,
  title="{Exploring Self-Supervised Regularization for Supervised and Semi-Supervised Learning}",
  author={Phi Vu Tran},
  booktitle={NeurIPS Workshop on Learning with Rich Experience: Integration of Learning Paradigms},
  year={2019}
}
```

