FixMatch
--------

This method showed that combining consistency regularization and pseudo-labeling
techniques can result in a simple and effective semi-supervised learning approach.
Diagramed below, it involves applying consistency regularization between
a weakly and strongly augmented view of the same unlabeled image and only
on predictions where the weakly augmented sample meets a specific confidence
threshold.

.. raw:: html
    
    <img src="../_static/images/fixmatch-diagram.png" class="align-center" width="100%" />

Usage
^^^^^

To use this method, you must incorporate the `FixMatchLossHead`
into the model. Additionally, you'll need to provide an unlabeled
data branch that provides two augmented views of the same image.
The `TwoViewsTranform` and `MultiViewTransform` are built-in
for this purpose.

We have provided a sample configuration for the imagewang dataset
that can be easily adapated to run on other datasets:

.. code-block:: bash

  $ open_sesemi -cn imagewang_fixmatch --cfg job

  run:
    seed: 42
    num_epochs: 100
    num_iterations: null
    gpus: 2
    num_nodes: 1
    accelerator: dp
    batch_size_per_gpu: 16
    data_root: ./data/imagewang
    id: imagewang_fixmatch
    dir: ./runs
    mode: FIT
    resume_from_checkpoint: null
    pretrained_checkpoint_path: null
  data:
    train:
      supervised:
        dataset:
          root: null
          subset: train
          image_transform:
            _target_: sesemi.transforms.train_transforms
            random_resized_crop: true
            resize: 256
            crop_dim: 224
            scale:
            - 0.3
            - 1.0
          _target_: sesemi.dataset
          name: image_folder
          classes:
          - n02086240
          - n02087394
          - n02088364
          - n02089973
          - n02093754
          - n02096294
          - n02099601
          - n02105641
          - n02111889
          - n02115641
        batch_size: null
        batch_size_per_gpu: null
        shuffle: true
        sampler: null
        batch_sampler: null
        num_workers: 4
        collate_fn: null
        pin_memory: true
        drop_last: true
        timeout: 0.0
        worker_init_fn: null
        _target_: sesemi.DataLoader
      fixmatch:
        dataset:
          root: null
          subset:
          - train
          - val
          - unsup
          image_transform:
            _target_: sesemi.transforms.MultiViewTransform
            num_views: 2
            image_augmentations:
            - _target_: sesemi.T.Compose
              transforms:
              - _target_: sesemi.T.Resize
                size: 256
              - _target_: sesemi.T.CenterCrop
                size: 224
              - _target_: sesemi.T.RandomHorizontalFlip
                p: 0.5
              - _target_: sesemi.T.RandomAffine
                degrees: 0
                translate:
                - 0.125
                - 0.125
              - _target_: sesemi.T.ToTensor
              - _target_: sesemi.T.Normalize
                mean:
                - 0.485
                - 0.456
                - 0.406
                std:
                - 0.229
                - 0.224
                - 0.225
            - ${data.train.supervised.dataset.image_transform}
          _target_: sesemi.dataset
          name: image_file
        batch_size: null
        batch_size_per_gpu: null
        shuffle: true
        sampler: null
        batch_sampler: null
        num_workers: 4
        collate_fn: null
        pin_memory: true
        drop_last: true
        timeout: 0.0
        worker_init_fn: null
        _target_: sesemi.DataLoader
    val:
      dataset:
        root: null
        subset: val
        image_transform:
          _target_: sesemi.transforms.center_crop_transforms
          resize: 256
          crop_dim: 224
        _target_: sesemi.dataset
        name: image_folder
        classes:
        - n02086240
        - n02087394
        - n02088364
        - n02089973
        - n02093754
        - n02096294
        - n02099601
        - n02105641
        - n02111889
        - n02115641
      batch_size: null
      batch_size_per_gpu: null
      shuffle: false
      sampler: null
      batch_sampler: null
      num_workers: 4
      collate_fn: null
      pin_memory: true
      drop_last: false
      timeout: 0.0
      worker_init_fn: null
      _target_: sesemi.DataLoader
    test: null
  learner:
    _target_: sesemi.Classifier
    hparams:
      num_classes: 10
      model:
        backbone:
          _target_: sesemi.PyTorchImageModels
          name: resnet50d
          freeze: false
          pretrained: false
          global_pool: avg
          drop_rate: 0.5
        head:
          _target_: sesemi.models.heads.base.LinearHead
        loss:
          callable:
            _target_: torch.nn.CrossEntropyLoss
          scheduler: null
          reduction: mean
          scale_factor: 1.0
        regularization_loss_heads:
          fixmatch:
            head:
              _target_: sesemi.models.heads.loss.FixMatchLossHead
              data: fixmatch
              student_backbone: supervised_backbone
              student_head: supervised_head
              teacher_backbone: supervised_backbone_ema
              teacher_head: supervised_head_ema
            scheduler: null
            reduction: mean
            scale_factor: 1.0
        ema:
          decay: 0.999
      optimizer:
        _target_: torch.optim.SGD
        lr: 0.1
        momentum: 0.9
        nesterov: true
        weight_decay: 0.0005
      lr_scheduler:
        scheduler:
          _target_: sesemi.PolynomialLR
          warmup_epochs: 10
          iters_per_epoch: ${sesemi:iterations_per_epoch}
          warmup_lr: 0.001
          lr_pow: 0.5
          max_iters: ${sesemi:max_iterations}
        frequency: 1
        interval: step
        monitor: null
        strict: true
        name: null
  trainer:
    callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: val/top1
      mode: max
      save_top_k: 1
      save_last: true

References
^^^^^^^^^^

.. code-block:: bibtex

  @article{Sohn2020FixMatchSS,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin Dogus Cubuk and Alexey Kurakin and Han Zhang and Colin Raffel},
    journal={Neural Information Processing System},
    year={2020},
  }