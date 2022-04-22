Noisy Student
-------------

This method demonstrated that pseudo-labels can be effectively used for self-training
by introducing specific kinds of noise during training. In particular,
it was shown that using strong data augmentation, dropout, and stochastic depth
were effective in providing a learning signal for a model to bootstrap off its
own pseudo-labels. Illustrated below, it involves iterating between training
and pseudo-label generation on unlabeled data.

.. raw:: html
    
    <img src="../_static/images/noisy-student-diagram.png" class="align-center" width="100%" />

Usage
^^^^^

To use this method, you must alternate between supervised training
and pseudo-label generation on unlabeled data.

We have provided sample configurations that illustrates how to
run the different stages of noisy student training including:

1. Supervised training on a labeled dataset.
2. Pseudo-label generation on an unlabeled dataset.
3. Supervised training on the combined labeled and pseudo-labeled data.
4. Repeat steps 2 and 3 as necessary.

Step 1
++++++

This step can be performed on the imagewang dataset using the following configuration that can be adpated:

.. code-block:: bash

  $ open_sesemi -cn imagewang_noisy_student_stage_1 --cfg job

  run:
    seed: 42
    num_epochs: 800
    num_iterations: null
    gpus: 2
    num_nodes: 1
    accelerator: dp
    batch_size_per_gpu: 32
    data_root: ./data/imagewang
    id: imagewang_noisy_student_stage_1
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
          drop_path_rate: 0.8
        head:
          _target_: sesemi.models.heads.base.LinearHead
        loss:
          callable:
            _target_: torch.nn.CrossEntropyLoss
          scheduler: null
          reduction: mean
          scale_factor: 1.0
        regularization_loss_heads: null
        ema: null
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

Step 2
++++++

This step can be performed by running a built-in pseudo-labeling operation as follows:

.. code-block:: bash

  $ python -m sesemi.ops.pseudo_dataset -cn imagewang_noisy_student_stage_n \
      output_dir=${OUTPUT_DIR} \
      checkpoint_path=${CHECKPOINT_PATH} \
      dataset.root=${DATASET_ROOT} \
      --cfg job

  checkpoint_path: ${CHECKPOINT_PATH}
  seed: 42
  output_dir: ${OUTPUT_DIR}
  dataset:
    root: ${DATASET_ROOT}
    subset:
    - val
    - unsup
    image_transform: null
    _target_: sesemi.dataset
    name: image_folder
  preprocessing_transform:
    _target_: sesemi.transforms.center_crop_transforms
    resize: 256
    crop_dim: 224
  test_time_augmentation: null
  postaugmentation_transform: null
  image_getter: null
  gpus: -1
  batch_size: 16
  num_workers: 6
  symlink_images: true

Step 3
++++++

This and subsequent retraining steps can use the following separate configuration:

.. code-block:: bash

  $ open_sesemi -cn imagewang_noisy_student_stage_n \
      run.data.train.supervised.dataset.datasets[1].root=${PSEUDO_DATASET_ROOT} \
      --cfg job

  run:
    seed: 42
    num_epochs: 100
    num_iterations: null
    gpus: 2
    num_nodes: 1
    accelerator: dp
    batch_size_per_gpu: 32
    data_root: ./data/imagewang
    id: imagewang_noisy_student_stage_n
    dir: ./runs
    mode: FIT
    resume_from_checkpoint: null
    pretrained_checkpoint_path: null
  data:
    train:
      supervised:
        dataset:
          root: null
          subset: null
          image_transform:
            _target_: sesemi.transforms.train_transforms
            random_resized_crop: true
            resize: 256
            crop_dim: 224
            scale:
            - 0.3
            - 1.0
          _target_: sesemi.dataset
          name: concat
          datasets:
          - _target_: sesemi.dataset
            name: image_folder
            root: ./data/imagewang
            subset: train
            image_transform: ${...image_transform}
          - _target_: sesemi.dataset
            name: pseudo
            root: ${PSEUDO_DATASET_ROOT}
            image_transform: ${...image_transform}
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
          drop_path_rate: 0.8
        head: ???
        loss:
          callable:
            _target_: torch.nn.CrossEntropyLoss
          scheduler: null
          reduction: mean
          scale_factor: 1.0
        regularization_loss_heads: null
        ema: null
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

  @article{Xie2020SelfTrainingWN,
    title={Self-Training With Noisy Student Improves ImageNet Classification},
    author={Qizhe Xie and Eduard H. Hovy and Minh-Thang Luong and Quoc V. Le},
    journal={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2020},
    pages={10684-10695}
  }