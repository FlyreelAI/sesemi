<p><p align="center"><img height="350px" src="https://github.com/FlyreelAI/sesemi/raw/master/assets/sesemi-banner.png" /></p></p>

# Image Classification with Self-Supervised Regularization
<span><img src="https://img.shields.io/badge/license-Apache-blue" /> <img src="https://img.shields.io/badge/python->=3.7-green" /> <img src="https://img.shields.io/badge/pytorch->=1.11.0-light" /> <img src="https://img.shields.io/badge/coverage-91%25-green" /> <img src="https://img.shields.io/badge/%20-contributions%20welcome-5429E6" /></span>

## Why SESEMI?

SESEMI is an open source image classification library built on PyTorch Lightning. SESEMI enables various modern supervised classifiers to be robust semi-supervised learners based on the principles of self-supervised regularization.

### Highlights and Features

* Integration with the popular [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) (timm) library for access to contemporary, high-performance supervised architectures with optional pretrained ImageNet weights. See the list of [recommended backbones](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/models/backbones/timm.py)
* Demonstrated utility on large realistic image datasets and is currently competitive on the [FastAI Imagenette benchmarks](https://github.com/fastai/imagenette)
* Easy to use out-of-the-box requiring little hyper-parameter tuning across many tasks related to supervised learning, semi-supervised learning, and learning with noisy labels. In most use cases, one only needs to tune the learning rate, batch size, and backbone architecture
* Simply add unlabeled data for improved image classification without any tricks

Our goal is to expand the utility of SESEMI for the ML/CV practitioner by incorporating the latest advances in self-supervised, semi-supervised, and few-shot learning to boost the accuracy performance of conventional supervised classifiers in the limited labeled data setting. If you find this work useful please star this repo to let us know. Contributions are also welcome!

### Documentation

Click [here](https://sesemi.readthedocs.io/) to view the full documentation hosted on readthedocs. 
For convenience, we have provided links to some of the content below:

* [Overview](https://sesemi.readthedocs.io/en/latest/overview.html)
* [Installation](https://sesemi.readthedocs.io/en/latest/installation.html)
* [Quickstart](https://sesemi.readthedocs.io/en/latest/quickstart.html)
* [Tutorials](https://sesemi.readthedocs.io/en/latest/tutorials/project_setup.html)
* [Methods](https://sesemi.readthedocs.io/en/latest/methods/rotation_prediction.html)
* [Ops](https://sesemi.readthedocs.io/en/latest/ops/inference.html)
* [API Reference](https://sesemi.readthedocs.io/en/latest/api/sesemi.html)

### Supported Methods

We currently support variants of the following methods:

* [Rotation Prediction](https://sesemi.readthedocs.io/en/latest/methods/rotation_prediction.html)
* [Entropy Minimization](https://sesemi.readthedocs.io/en/latest/methods/entropy_minimization.html)
* [Jigsaw Prediction](https://sesemi.readthedocs.io/en/latest/methods/jigsaw_prediction.html)
* [Pi Model](https://sesemi.readthedocs.io/en/latest/methods/pi_model.html)
* [Mean Teacher](https://sesemi.readthedocs.io/en/latest/methods/mean_teacher.html)
* [FixMatch](https://sesemi.readthedocs.io/en/latest/methods/fixmatch.html)
* [Noisy Student](https://sesemi.readthedocs.io/en/latest/methods/noisy_student.html)

### Built-in Configurations

| Config Name                      | Dataset    | Methods    | Training Logs   |
|----------------------------------|------------|-----------------|
| [cifar10_wrn_28_10](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/cifar10_wrn_28_10.yaml)                | CIFAR-10   | Supervised | N/A             |
| [cifar10](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/cifar10.yaml)                          | CIFAR-10   | Supervised | N/A             |
| [cifar100](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/cifar100.yaml)                         | CIFAR-100   | Supervised | N/A             |
| [imagewang_consistency](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewang_consistency.yaml)            | Imagewang   | Mean Teacher | N/A             |
| [imagewang_entmin](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewang_entmin.yaml)                 | Imagewang   | Entropy Minimization | N/A     |
| [imagewang_fixmatch_randaugment](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewang_fixmatch_randaugment.yaml)   | Imagewang   | FixMatch | N/A     |
| [imagewang_fixmatch](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewang_fixmatch.yaml)   | Imagewang   | FixMatch | N/A     |
| [imagewang_jigsaw_entmin](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewang_jigsaw_entmin.yaml)   | Imagewang   | Jigsaw Prediction + Entropy Minimization | N/A     |
| [imagewang_noisy_student_stage_1](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewang_noisy_student_stage_1.yaml)   | Imagewang   | Noisy Student | N/A     |
| [imagewang_noisy_student_stage_n](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewang_noisy_student_stage_n.yaml)   | Imagewang   | Noisy Student | N/A     |
| [imagewang_rotation_entmin](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewang_rotation_entmin.yaml)   | Imagewang   | Rotation Prediction + Entropy Minimization | N/A     |
| [imagewang_rotation](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewang_rotation.yaml)   | Imagewang   | Rotation Prediction | N/A     |
| [imagewang](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewang.yaml)   | Imagewang   | Supervised | N/A     |
| [imagewoof_entmin](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewoof_entmin.yaml)   | Imagewoof   | Entropy Minimization | N/A     |
| [imagewoof_rotation](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewoof_rotation.yaml)   | Imagewoof   | Rotation Prediction | N/A     |
| [imagewoof](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/imagewoof.yaml)   | Imagewoof   | Supervised | N/A     |
| [stl10](https://github.com/FlyreelAI/sesemi/blob/master/sesemi/trainer/conf/stl10.yaml)   | STL-10   | Supervised | N/A     |

## Copyright

We have released SESEMI under the permissive Apache 2.0 license.
Any contributions made will also be subject to the same licensing.

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

