Custom Datasets and Augmentations
---------------------------------

..
  Goal

  * Users know how add custom datasets.

  Outline

  * Setup
  * Dataset
  * Configuration
  * Training

Here we will demonstrate the steps needed to use a custom dataset
with SESEMI. We will reuse a few built-ins
that we have provided as illustrative examples.

To start, make sure that you have a project setup similar to what
was described in the "Setting Up a Standard Project" tutorial page.
We will add a couple new modules to it.

-------
Dataset
-------

The custom dataset we will be implementing here is an unlabeled
dataset that collects all images that exist within a given root
subdirectory including those that are arbitrarily nested.

Recall that the project structure we are working with is similar to::

    /experiments
        /configs
        /data
        /runs
        /src

We will be adding a package to the `src` directory that will contain
a new dataset module.

Specifically, create the following source code structure::

    /src
      /sesemi_tutorials
      __init__.py
      datasets.py

Then assuming that our current working directory is `/experiments`, we will 
add the `src` directory of our project to our PYTHONPATH:

.. code-block:: bash

  $ PYTHONPATH=$PYTHONPATH:$PWD/src/

Prior to implementing the custom dataset, it is important
to know the API of datasets within SESEMI. In particular,
we require that datasets conform to the following
structured configuration:

.. code-block:: python

  class DatasetConfig(DictConfig):
      """The SESEMI dataset configuration.

      Attributes:
          name: The name of the dataset in the registry.
          root: The absolute or relative (with respect to the run's data root) path to the dataset.
              If this is None, then the run's `data_root` attribute will be used instead.
          subset: Can either be a the string name of the subset, a list of subsets, or None (null)
              to indicate the full set.
          image_transform: An optional callable transform configuration that is applied to images.
      """

      name: str
      root: Optional[str]
      subset: Any
      image_transform: Any
      _target_: str

      def __init__(self, defaults: Optional[Mapping[str, Any]] = None):
          super().__init__(
              {
                  "root": None,
                  "subset": None,
                  "image_transform": None,
                  "_target_": "sesemi.dataset",
              }
          )
          if defaults is not None:
              self.update(defaults)

As is shown, datasets can be any instantiable objects (such as classes or functions)
though they should at a minimum support the given parameters above. You may also
supplement these parameters with arbitrary dataset-specific keyword arguments
during configuration as is common for complex datasets.

Moving to the custom dataset implementation, we will be reusing functionality
from the standard TorchVision `ImageFolder` dataset. Altogether, the full
dataset is given below:

.. code-block:: python

  import os

  from torch.utils.data import ConcatDataset, Dataset
  from torchvision.datasets.folder import (
      default_loader,
      has_file_allowed_extension,
      IMG_EXTENSIONS,
  )

  from typing import Any, Callable, List, Optional, Union


  def get_image_files(directory: str, is_valid_file: Callable[[str], bool]) -> List[str]:
      """Finds the full list of image files recursively under a directory path.

      Args:
          directory: The root directory to search for image files.
          is_valid_file: A callable to determine if a file is a valid image.

      Returns:
          The list of paths to the image files.
      """
      directory = os.path.expanduser(directory)

      files: List[str] = []
      for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
          for fname in sorted(fnames):
              if is_valid_file(fname):
                  path = os.path.join(root, fname)
                  files.append(path)

      return files


  def default_is_vaild_file(path: str) -> bool:
      """The default callable to determine if a file is a valid image."""
      return has_file_allowed_extension(path, IMG_EXTENSIONS)


  class ImageFile(Dataset):
      def __init__(
          self,
          root: str,
          transform: Optional[Callable] = None,
          loader: Callable[[str], Any] = default_loader,
          is_valid_file: Callable[[str], bool] = default_is_vaild_file,
      ):
          """An image-file dataset.

          Args:
              root: The path to the dataset root.
              transform: An optional image transform.
              loader: The image loading callable.
              is_valid_file: A callable to determine if a file is a valid image.
          """
          super().__init__()
          self.transform = transform
          self.loader = loader
          self.is_valid_file = is_valid_file
          self.image_files = get_image_files(root, self.is_valid_file)

      def __len__(self) -> int:
          return len(self.image_files)

      def __getitem__(self, index: int):
          image = self.loader(self.image_files[index])
          if self.transform is not None:
              image = self.transform(image)
          return image


  def image_file(
      name: str,
      root: str,
      subset: Optional[Union[str, List[str]]] = None,
      image_transform: Optional[Callable] = None,
      **kwargs,
  ) -> Dataset:
      """An image file dataset builder.

      Args:
          name: Expected to simply be "image_file".
          root: The path to the image file dataset.
          subset: The subset(s) to use.
          image_transform: The image transformations to apply.

      Returns:
          An `ImageFile` dataset.
      """
      # The name argument makes it possible to support multiple different
      # kinds of datasets using a single object or function. Though we
      # expect this function to only be used with the `image_file` dataset.
      assert name == "image_file", "only support image_file datasets"

      if isinstance(subset, str):
          # If the subset is a single string then only a subdirectory with
          # that name is used to collect image filenames.
          return ImageFile(os.path.join(root, subset), transform=image_transform)
      else:
          if subset is None:
              # If None is used as the subset, then all available subsets are implied.
              subsets = [
                  x
                  for x in os.listdir(root)
                  if not x.startswith(".") and os.path.isdir(os.path.join(root, x))
              ]
          else:
              subsets = subset

          dsts = [
              ImageFile(os.path.join(root, s), transform=image_transform) for s in subsets
          ]

          return ConcatDataset(dsts)

This dataset is primarily intended for use with label-free SSL methods. As
this is already a built-in, we will show the key difference between this custom
implementation and the built-in by overriding an existing configuration.
Specifically, the built-in FixMatch configs make use of the `image_file` dataset.
We can use the following configuration to override the built-in dataset with this
custom one below that you can save to `configs/custom_dataset_imagewang_fixmatch.yaml.
Again, feel free to make adjustments according to the hardware you have available.

.. code-block:: yaml

  # @package _global_
  defaults:
    - /base/supervised/model/baseline
    - /base/supervised/data/imagewang
    - /base/fixmatch/model
    - /base/fixmatch/data/imagewang
  run:
    seed: 42
    devices: 2
    batch_size_per_device: 16
    num_epochs: 100
    id: custom_dataset_imagewang_fixmatch
  data:
    train:
      fixmatch:
        dataset:
          _target_: sesemi_tutorials.datasets.image_file

As you can see, we only needed to change the Hydra `_target_`
parameter to point to our custom implementation.

You can verify this update by inspecting the full configuration using:

.. code-block:: bash

  $ open_sesemi -cd configs -cn custom_dataset_imagewang_fixmatch --cfg job

If you have the imagewang dataset downloaded to the `data`
directory you can run this configuration directly.

To download imagewang locally:

.. code-block:: bash

  $ curl https://s3.amazonaws.com/fast-ai-imageclas/imagewang.tgz | tar -xzv -C ./data

Then simply run the following:

.. code-block:: bash

  $ open_sesemi -cd configs -cn custom_dataset_imagewang_fixmatch