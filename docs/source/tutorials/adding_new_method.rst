Custom SSL Methods
------------------

..
  Goal

  * Users know how add custom loss heads.

  Outline

  * Setup
  * Method
  * Loss Head
  * Configuration
  * Training

In this tutorial, we will describe how to train models using custom
SSL methods. While we support a number of SSL methods out of the box,
you may want to experiment with other new or existing ones without
having to make significant modifications. To demonstrate how this
can be done, we will go through the different steps involved
with implementing the FixMatch algorithm.

For the rest of this tutorial, we will be assuming that you have set up
a project following the approach described in the "Setting Up a Standard Project"
tutorial page. If not, you should do so before moving forward.

------
Method
------

FixMatch is a semi-supervised algorithm that uses consistency regularization and pseudo-labeling.
It is relatively simple while still providing strong baseline performance.
The key idea is summarized well in the accompanied diagram from the original
paper. 

.. raw:: html
    
    <img src="../_static/images/fixmatch-diagram.png" class="align-center" width="100%" />

Like many other methods, it computes a loss over unlabeled data
after the following steps:

1. Get an unlabeled example.
2. Pass the unlabeled example through weak and strong augmentations.
3. Pass the augmented example through a classifier to get predictive probabilities.
4. If the maximum probability of the weakly augmented example is greater than a threshold continue with the loss computation, otherwise ignore this example.
5. Compute a one-hot pseudo-label from the weakly augmented example.
6. Use the pseudo-label to optimize the predictions over the strongly augmented example using cross-entropy minimization.

We support mixing in different losses using the concept of a loss head which
was described in our overview document. To provide a recap, note that
loss heads are passed lookup tables for data batches, backbones, heads, and
computed features and are expected to return a scalar or vector loss which
is then reduced and jointly optimized with other losses through a summation.

In addition to loss heads, we make mixing in a variety of different.

---------
Loss Head
---------

Now that you are familiar with the main concepts, we will dig into the implementation
of a loss head. The base interface for one is provided here as context:

.. code-block:: python

  @dataclass
  class LossOutputs:
      """The outputs of a loss head.

      Attributes:
          losses: The per-sample losses.
          weights: Optional per-sample weights.
      """

      losses: Tensor
      weights: Optional[Tensor] = None

      def asdict(self) -> Dict[str, Any]:
          """Converts this into a dictionary."""
          return dict(
              losses=self.losses,
              weights=self.weights,
          )


  class LossHead(nn.Module):
      """The interface for loss heads."""

      def build(
          self,
          backbones: Dict[str, Backbone],
          heads: Dict[str, Head],
          **kwargs,
      ):
          """Builds the loss head.

          Args:
              backbones: A dictionary of shared backbones. During the build, new backbones can be
                  added. As this is actually an `nn.ModuleDict` which tracks the parameters,
                  these new backbones should not be saved to the loss head object to avoid
                  double-tracking.
              heads: A dictionary of shared heads similar to the backbones
          """

      def forward(
          self,
          data: Dict[str, Any],
          backbones: Dict[str, Backbone],
          heads: Dict[str, Head],
          features: Dict[str, Any],
          step: int,
          logger_wrapper: Optional[LoggerWrapper] = None,
          **kwargs,
      ) -> LossOutputs:
          """Computes the loss.

          Args:
              data: A dictionary of data batches.
              backbones: A dictionary of shared backbones. This should not be altered.
              heads: A dictionary of shared heads. This should not be altered.
              features: A dictionary of shared features. Additional tensors can be added to this.
              logger_wrapper: An optional wrapper around the lightning logger.
              step: The training step number.
              **kwargs: Placeholder for other arguments that may be added.

          Returns:
              The losses and optional per-sample weights.
          """
          raise NotImplementedError()

As you can see, loss heads primarily consist of a `build` and `forward` function.
We will describe each of these now.

^^^^^
Build
^^^^^

The build function can be used to construct backbones and heads while also being
able to lookup previously built modules. An important detail to note is that weights
should be stored either on the loss head or in one of the lookup tables, however,
not both.

^^^^^^^
Forward
^^^^^^^

This function is generally where most of the logic for your custom SSL method will be located.
You will have access to data batches, backbones, heads, and features from the current iteration.
You may also mutate the features lookup table as necessary, however, the other lookup tables should
not be modified here.

^^^^^^^^
FixMatch
^^^^^^^^

Putting these concepts together, we can now implement FixMatch. We will do so in a `loss.py` file
placed under the `src` of our project as follows::

    /experiments
        /configs
        /data
        /runs
        /src
            /sesemi_tutorials
            __init__.py
            loss.py

Then assuming that our current working directory is `/experiments`, we will 
add the `src` directory of our project to our PYTHONPATH:

.. code-block:: bash

  $ PYTHONPATH=$PYTHONPATH:$PWD/src/

Now, we can fill in the contents of the `loss.py` file with the following code:

.. code-block:: python

  import torch
  import torch.nn.functional as F

  from torch import Tensor
  from typing import Dict, Any, Optional

  from sesemi.logger import LoggerWrapper

  from ..backbones.base import Backbone
  from ..heads.base import Head
  from .base import LossHead, LossOutputs


  class FixMatchLossHead(LossHead):
      """The FixMatch loss head.

      @article{Sohn2020FixMatchSS,
          title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
          author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin Dogus Cubuk and Alexey Kurakin and Han Zhang and Colin Raffel},
          journal={ArXiv},
          year={2020},
          volume={abs/2001.07685}
      }
      """

      def __init__(
          self,
          data: str,
          student_backbone: str = "supervised_backbone",
          teacher_backbone: Optional[str] = None,
          student_head: str = "supervised_head",
          teacher_head: Optional[str] = None,
          threshold: float = 0.5,
      ):
          """Initializes the loss head.

          Args:
              data: The data key.
              student_backbone: The student's backbone key.
              teacher_backbone: The teacher's backbone key. Defaults to the student's.
              student_head: The student's head key.
              teacher_head: The teacher's head key. Defaults to the student's.
              threshold: The threshold used to filter low confidence predictions
                  made by the teacher.
          """
          super().__init__()
          self.data = data
          self.student_backbone = student_backbone
          self.teacher_backbone = teacher_backbone or student_backbone
          self.student_head = student_head
          self.teacher_head = teacher_head or student_head
          self.threshold = threshold

      def forward(
          self,
          data: Dict[str, Any],
          backbones: Dict[str, Backbone],
          heads: Dict[str, Head],
          features: Dict[str, Any],
          step: int,
          logger_wrapper: Optional[LoggerWrapper] = None,
          **kwargs,
      ) -> Tensor:
          weakly_augmented, strongly_augmented = data[self.data]
          student_backbone = backbones[self.student_backbone]
          student_head = heads[self.student_head]
          teacher_backbone = backbones[self.teacher_backbone]
          teacher_head = heads[self.teacher_head]

          weakly_augmented_features = teacher_backbone(weakly_augmented)
          strongly_augmented_features = student_backbone(strongly_augmented)

          weakly_augmented_logits = teacher_head(weakly_augmented_features).detach()
          strongly_augmented_logits = student_head(strongly_augmented_features)

          weakly_augmented_probs = torch.softmax(weakly_augmented_logits, dim=-1)
          weakly_augmented_labels = torch.argmax(weakly_augmented_probs, dim=-1).to(
              torch.long
          )

          loss_weights = (weakly_augmented_probs.max(dim=-1)[0] >= self.threshold).to(
              weakly_augmented.dtype
          )

          losses = F.cross_entropy(
              strongly_augmented_logits,
              weakly_augmented_labels,
              reduction="none",
          )

          if logger_wrapper:
              logger_wrapper.log_images(
                  "fixmatch/images/weak", weakly_augmented, step=step
              )
              logger_wrapper.log_images(
                  "fixmatch/images/strong", strongly_augmented, step=step
              )

          return LossOutputs(losses=losses, weights=loss_weights)


Note that here we don't use need to use the `build` function as we are referencing
the existing classifier's named backbone and head modules instead.

--------------
Configurations
--------------

The starting point of our FixMatch configuration will be a built-in supervised
baseline for the imagewang dataset which we will then augment with our FixMatch loss.
To start it will look like this:

.. code-block:: yaml

  # @package _global_
  defaults:
    - /base/supervised/model/baseline
    - /base/supervised/data/imagewang

Place this in the file `configs/custom_imagewang_fixmatch.yaml`.
You can inspect what the full configuration actually looks like
after importing the defaults by running the following command:

.. code-block:: bash

  $ open_sesemi -cd configs -cn custom_imagewang_fixmatch --cfg job

Using the defaults instead of manually specifying everything can make it easier to
focus on the components that you want to override while keeping everything else fixed.

In particular, the key changes we will need to make are:

1. Add a training data loader that generates a weakly and strongly augmented pair of images.
2. Add a FixMatch loss head to the model.

^^^^
Data
^^^^

As described in the overview document, training data batches are stored in lookup tables that
are passed around each iteration. The lookup table is defined by the training data configuration.
For reference, here are the relevant config data structures:

.. code-block:: python

  @dataclass
  class IgnoredDataConfig:
      """A configuration to specify data loaders that should be ignored.

      Hydra currently has the limitation that `Dict[str, Optional[DataLoaderConfig]]`
      is not considered a valid type to use with a structured config due to the optional value.
      This makes it impossible to override a configuration and set one of the data loaders
      as null. To enable ignoring data loaders in these kind of dictionaries, this supplemental
      configuration supports marking which data loader not to build. A benefit of this approach
      is that the configuration will still be accessible elsewhere.

      Attributes:
          train: An optional dictionary marking which data loaders to ignore.
          extra: An optional dictionary marking which data loaders to ignore.
      """

      train: Optional[Dict[str, bool]] = None
      extra: Optional[Dict[str, bool]] = None


  @dataclass
  class DataConfig:
      """The data group configuration.

      Attributes:
          train: An optional dictionary of data loader configurations. This configuration is directly
              mapped into dictionaries of data batches.
          val: An optional data loader configuration to use during validation.
          test: An optional data loader configuration to use for testing.
          extra: An optional dictionary of data loader configurations. This configuration is directly
              mapped into dictionaries of data batches.
      """

      train: Optional[Dict[str, DataLoaderConfig]] = None
      val: Optional[DataLoaderConfig] = None
      test: Optional[DataLoaderConfig] = None
      extra: Optional[Dict[str, DataLoaderConfig]] = None

      ignored: IgnoredDataConfig = IgnoredDataConfig()
  
  
  @dataclass
  class DataLoaderConfig:
      """The data loader configuration.

      Most of the attributes are taken directly from PyTorch's DataLoader object.

      Attributes:
          dataset: The dataset configuration.
          batch_size: An optional batch size to use for a PyTorch data loader. Cannot be set with
              `batch_size_per_device`.
          batch_size_per_device: An optional batch size per device to use. Cannot be set with `batch_size`.
          shuffle: Whether to shuffle the dataset at each epoch.
          sampler: An optional sampler configuration.
          batch_sampler: An optional batch sampler configuration.
          num_workers: The number of workers to use for data loading (0 means use main process).
          collate_fn: An optional collation callable configuration.
          pin_memory: Whether to pin tensors into CUDA.
          drop_last: Whether to drop the last unevenly sized batch.
          timeout: The timeout to use get data batches from workers.
          worker_init_fn: An optional callable that is invoked for each worker on initialization.
          repeat: The number of times to repeat the dataset on iteration.
          prefetch_factor: The number of samples to prefetch per worker.
          persistent_workers: Whether or not to persist workers after iterating through a dataset.

      References:
          * https://pytorch.org/docs/1.6.0/data.html?highlight=dataloader#torch.utils.data.DataLoader
      """

      dataset: DatasetConfig = field(default_factory=DatasetConfig)
      batch_size: Optional[int] = None
      batch_size_per_device: Optional[int] = None
      shuffle: bool = False
      sampler: Optional[Any] = None
      batch_sampler: Optional[Any] = None
      num_workers: Optional[int] = 0
      collate_fn: Optional[Any] = None
      pin_memory: bool = False
      drop_last: Optional[bool] = False
      timeout: float = 0
      worker_init_fn: Optional[Any] = None
      repeat: Optional[int] = None
      prefetch_factor: Optional[int] = 2
      persistent_workers: Optional[bool] = False
      _target_: str = "sesemi.RepeatableDataLoader"

Note that the named training data loaders map directly to the named
data batches in the training lookup table. We will just need to add
a `fixmatch` data loader alongside the existing `supervised` data loader.

In this case we will be leveraging a specialized data augmentation
provided within SESEMI for applying different augmentations on
the same input example to produce multiple views.

.. code-block:: yaml

  # @package _global_
  defaults:
    - /base/supervised/model/baseline
    - /base/supervised/data/imagewang
  data:
    train:
      fixmatch:
        dataset:
          name: image_file
          subset: [train, unsup]
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
                  translate: [0.125, 0.125]
                - _target_: sesemi.T.ToTensor
                - _target_: sesemi.T.Normalize
                  mean: [0.485, 0.456, 0.406]
                  std: [0.229, 0.224, 0.225]
              - ${data.train.supervised.dataset.image_transform}
        shuffle: True
        pin_memory: True
        num_workers: 4
        drop_last: True

Note that the first data augmentation is weak and consists of just
a flip and translate while the second data augmentation is stronger
and corresponds to the standard supervised data augmentations
provided within SESEMI.

^^^^^
Model
^^^^^

Separate from the data configuration, the model also needs to be updated.
Important data structures needed to make these kinds of changes are
shown below.

.. code-block:: python

  @dataclass
  class ClassifierModelConfig:
      """The classifier learner's model configuration.

      Attributes:
          backbone: A backbone config that can be instantiated.
          supervised_loss: A callable loss config.
          regularization_loss_heads: An optional dictionary of loss head configs.
          ema: An optional config for the ema decay coefficient.
      """

      backbone: Any = MISSING
      supervised_loss: LossCallableConfig = LossCallableConfig()
      regularization_loss_heads: Optional[Dict[str, LossHeadConfig]] = None
      ema: Optional[EMAConfig] = None
  

  @dataclass
  class LossHeadConfig:
      """The loss head configuration.

      Attributes:
          head: A loss head configuration that can be instantiated.
          scheduler: An optional learning rate scheduler that can be instantiated.
          reduction: The loss reduction method to use (e.g. "mean" or "sum").
          scale_factor: The loss scale factor.
      """

      head: Any
      scheduler: Any = None
      reduction: str = "mean"
      scale_factor: float = 1.0

Specifically we will need to add a regularization loss head that makes use of the custom code:

.. code-block:: yaml

  # @package _global_
  defaults:
    - /base/supervised/model/baseline
    - /base/supervised/data/imagewang
  learner:
    hparams:
      model:
        regularization_loss_heads:
          fixmatch:
            head:
              _target_: sesemi_tutorials.loss.FixMatchLossHead
              data: fixmatch
  data:
    train:
      fixmatch:
        dataset:
          name: image_file
          subset: [train, unsup]
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
                  translate: [0.125, 0.125]
                - _target_: sesemi.T.ToTensor
                - _target_: sesemi.T.Normalize
                  mean: [0.485, 0.456, 0.406]
                  std: [0.229, 0.224, 0.225]
              - ${data.train.supervised.dataset.image_transform}
        shuffle: True
        pin_memory: True
        num_workers: 4
        drop_last: True

The last piece we will need to define is the run configuration that includes
things like the number of epochs, number of training GPUs, batch size per GPU,
and more. Altogether the final configuration is given below, though feel free
to make adjustments according to the hardware you have available. For example, 
if you strapped for GPUs you can approximate the global batch size by utilizing the
`accumulate_grad_batches` parameter of the PyTorch Lightning trainer.

.. code-block:: yaml

  # @package _global_
  defaults:
    - /base/supervised/model/baseline
    - /base/supervised/data/imagewang
  run:
    seed: 42
    devices: 2
    batch_size_per_device: 16
    num_epochs: 100
    id: custom_imagewang_fixmatch
  learner:
    hparams:
      model:
        regularization_loss_heads:
          fixmatch:
            head:
              _target_: sesemi_tutorials.loss.FixMatchLossHead
              data: fixmatch
  data:
    train:
      fixmatch:
        dataset:
          name: image_file
          subset: [train, unsup]
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
                  translate: [0.125, 0.125]
                - _target_: sesemi.T.ToTensor
                - _target_: sesemi.T.Normalize
                  mean: [0.485, 0.456, 0.406]
                  std: [0.229, 0.224, 0.225]
              - ${data.train.supervised.dataset.image_transform}
        shuffle: True
        pin_memory: True
        num_workers: 4
        drop_last: True

--------
Training
--------

Now that we have defined the loss head in code and configuration, we are ready
to begin training using SESEMI.

First, make sure that you have the imagewang dataset downloaded to the local
`data` directory. If not you can do so using:

.. code-block:: bash

  $ curl https://s3.amazonaws.com/fast-ai-imageclas/imagewang.tgz | tar -xzv -C ./data

Then simply run the following:

.. code-block:: bash

  $ open_sesemi -cd configs -cn custom_imagewang_fixmatch


