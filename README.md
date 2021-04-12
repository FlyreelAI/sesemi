# Image Classification with Self-Supervised Regularization

## Installation
Our preferred installation method is Docker. Please refer to the provided `docker/Dockerfile` and related instructions. Otherwise, `pytorch>=1.6.0` and `torchvision>=0.7.0` along with `docker/requirements.txt` should satisfy the dependencies to run this repository outside of Docker, e.g., in an Anaconda environment.

## Usage

The following instructions are intended for usage with the Docker installation.

* Issue the following `docker run` command to train and evaluate a SESEMI model with ResNet-50 backbone on the [`imagewoof2`](https://github.com/fastai/imagenette) dataset. The command assumes the `imagewoof2` dataset is mounted onto the image and is accessible via path `/data/imagewoof2` inside the container. The script takes about 8 hours to complete using 2 TITAN X GPUs each with 12GB of video memory, and achieves ~90% classification accuracy.

```bash
docker run \
  -w /home/appuser/sesemi \
  --gpus '"device=0,1"' \
  -u $(id -u) \
  --rm --ipc=host \
  -v /path/to/host-home-dir/sesemi:/home/appuser/sesemi \
  -v /data:/data \
  <my-docker-image-id> \
  python -u open_sesemi.py \
  --data-dir /data/imagewoof2 \
  --lr 0.1 --optimizer sgd --batch-size 32 \
  --epochs 80 --warmup-lr 0.001 --warmup-epochs 10 \
  --backbone resnet50 --run-id imagewoof_run01
```

* To evaluate a trained SESEMI model on a validation set, do:

```bash
docker run \
  -w /home/appuser/sesemi \
  --gpus '"device=0,1"' \
  -u $(id -u) \
  --rm --ipc=host \
  -v /path/to/host-home-dir/sesemi:/home/appuser/sesemi \
  -v /data:/data \
  <my-docker-image-id> \
  python -u open_sesemi.py evaluate-only \
  --data-dir /data/imagewoof2 \
  --checkpoint-path </path/to/trained/sesemi.pth>
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

