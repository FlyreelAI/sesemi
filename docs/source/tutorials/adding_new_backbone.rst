Custom Backbones and Heads
--------------------------

..
  Goal

  * Users know how add custom backbone and head.

  Outline

  * Setup
  * Backbone
  * Head
  * Configuration
  * Training

In this tutorial, we will demonstrate how to add custom backbones
and heads. Though we support standard TorchVision and timm backbones,
there are cases where a specialized backbone or head can be useful.
As long as the modules conform to our APIs they can be used.

Similar to other tutorials, we will be making use of the following
project structure with some slight adjustments::

    /experiments
        /configs
        /data
        /runs
        /src
          /sesemi_tutorials
          __init__.py
          models.py

--------
Backbone
--------

The backbone API is defined as follows:

.. code-block:: python

  class Backbone(nn.Module):
      """The interface for image classification backbones.

      Attributes:
          out_features: The number of output features generated by the backbone.
      """

      out_features: int

The only requirement here is that the `out_features` attribute
is exposed.

We can for example implement LeNet as follows:

.. code-block:: python

  import torch.nn as nn

  from torch import Tensor
  from sesemi.models.backbones.base import Backbone


  class LeNet5(Backbone):
      def __init__(self, out_features: int = 84):
          super().__init__()
          self.layers = nn.Sequential(
              nn.Conv2d(3, 6, 5, 2),
              nn.ReLU(inplace=True),
              nn.Conv2d(6, 16, 5, 2),
              nn.ReLU(inplace=True),
              nn.Flatten(),
              nn.Linear(400, 120),
              nn.ReLU(inplace=True),
              nn.Linear(120, out_features),
              nn.ReLU(inplace=True),
          )
          self.out_features = out_features

      def forward(self, inputs: Tensor) -> Tensor:
          return self.layers(inputs)

----
Head
----

Much like the backbone, the API for heads requires exposing a few
dimensions such as `in_features` and `out_features`.

.. code-block:: python

  class Head(nn.Module):
      """The interface for image classification heads.

      Attributes:
          in_features: The number of inputs features to the head.
          out_features: The number of output features from the head.
      """

      in_features: int
      out_features: int
  
We can for example implement a cosine similarity head as follows:

.. code-block:: python

  import torch
  import torch.nn as nn
  
  from torch import Tensor
  from sesemi.models.heads.base import Head


  class CosineSimilarityHead(Head):
      def __init__(self, in_features: int, out_features: int):
          super().__init__()
          self.in_features = in_features
          self.out_features = out_features
          
          self.weight = nn.Parameter(
            torch.zeros(
              (out_features, in_features), dtype=torch.float32))
          torch.nn.init.kaiming_normal_(self.weight)
      
      def forward(self, features: Tensor) -> Tensor:
        return torch.cosine_similarity(self.weight[None], features[:, None], dim=-1)

Placing these implementations in the `models.py` file, we can then
define a configuration, 'mnist_lenet5_cosine.yaml`, that uses them:

.. code-block:: yaml

  # @package _global_
  defaults:
    - /base/supervised/model
    - /base/supervised/data/mnist
    - /base/supervised/optimizer/sgd
    - /base/supervised/lr_scheduler/polynomial
  run:
    seed: 42
    devices: 1
    batch_size_per_device: 128
    num_epochs: 100
    id: mnist_lenet5_cosine
  learner:
    hparams:
      model:
        head:
          _target_: sesemi_tutorials.models.CosineSimilarityHead
        backbone:
          _target_: sesemi_tutorials.models.LeNet5