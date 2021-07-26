Quickstart
----------

To install the sesemi package you can run::

    $ pip install sesemi

This will download and configure an *open_sesemi* CLI which you can inspect as follows::

    $ open_sesemi -h
    cli is powered by Hydra.

    == Configuration groups ==
    Compose your configuration from those groups (group=option)

    learner: classifier


    == Config ==
    Override anything in the config (foo.bar=value)

    run:
        seed: null
        num_epochs: null
        num_iterations: null
        gpus: -1
        num_nodes: 1
        accelerator: null
        batch_size_per_gpu: null
        data_root: ./data
        id: default
        dir: ./runs
        mode: FIT
        resume_from_checkpoint: null
        pretrained_checkpoint_path: null
    data:
        train: null
        val: null
        test: null
    learner:
        _target_: sesemi.Classifier
        hparams:
            num_classes: ???
            model:
                backbone: ???
                supervised_loss:
                    callable: ???
                    scheduler: null
                    reduction: mean
                    scale_factor: 1.0
                regularization_loss_heads: null
            optimizer: ???
            lr_scheduler: null
    trainer: null


    Powered by Hydra (https://hydra.cc)
    Use --hydra-help to view Hydra specific help

The CLI is configured using `Hydra <https://hydra.cc/>`_. This is useful for the following reasons:

* Enables defining YAML configuration files that fully specify the run configuration.
* Cleanly maps configurations to internal and external objects through the use of the `instantiation <https://hydra.cc/docs/advanced/instantiate_objects/overview>`_ features.
* Supports composition of configurations through different override mechanisms.
* Makes it easy to define a collection of built-in configuration files accessible by the user.

Although Hydra is a flexible system for configuration management, it is somewhat complex which is why we have
developed a brief primer on the subject. Please see that page for additional background.

-------------------------
Structured Configurations
-------------------------

The config shown by the help text above is actually defined using a set of Python data structures. In particular,
data classes with type annotations are used. For example, some of the root configuration data structures are shown below::

    @dataclass
    class SESEMIConfig:
        """The full SESEMI configuration.

        Attributes:
            run: The run config.
            data: The data config.
            learner: The learner config.
            trainer: Optional additional parameters that can be passed to a PyTorch Lightning Trainer
                object.

        References:
            * https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api
        """

        run: RunConfig = RunConfig()
        data: DataConfig = DataConfig()
        learner: LearnerConfig = LearnerConfig()
        trainer: Any = None
    

    @dataclass
    class RunConfig:
        """The run configuration.

        Attributes:
            seed: An optional random seed used on initialization.
            num_epochs: The number of training epochs to train. Cannot be set with `num_iterations`.
            num_iterations: The number of training iterations to run. Cannot be set with `num_epochs`.
            gpus: Either an integer specifying the number of GPUs to use, a list of GPU
                integer IDs, a comma-separated list of GPU IDs, or None to train on the CPU. Setting
                this to -1 uses all GPUs and setting it to 0 also uses the CPU.
            num_nodes: The number of nodes to use during training (defaults to 1).
            accelerator: Supports either "dp" or "ddp" (the default).
            batch_size_per_gpu: An optional default batch size per GPU to use with all data loaders.
            data_root: The directory to use as the parent of relative dataset root directories
                (see `DatasetConfig`).
            id: The identifier to use for the run.
            dir: The directory to store run outputs (e.g. logs, configurations, etc.).
            mode: The run's mode.
            resume_from_checkpoint: An optional checkpoint path to restore trainer state.
            pretrained_checkpoint_path: An optional checkpoint path to load pretrained model weights.
        """

        seed: Optional[int] = None
        num_epochs: Optional[int] = None
        num_iterations: Optional[int] = None
        gpus: Any = -1
        num_nodes: int = 1
        accelerator: Optional[str] = None
        batch_size_per_gpu: Optional[int] = None
        data_root: Optional[str] = "./data"
        id: str = "default"
        dir: str = "./runs"
        mode: RunMode = RunMode.FIT
        resume_from_checkpoint: Optional[str] = None
        pretrained_checkpoint_path: Optional[str] = None


    @dataclass
    class DataConfig:
        """The data group configuration.

        Attributes:
            train: An optional dictionary of data loader configurations. This configuration is directly
                mapped into dictionaries of data batches.
            val: An optional data loader configuration to use during validation.
            test: An optional data loader configuration to use for testing.
        """

        train: Optional[Dict[str, DataLoaderConfig]] = None
        val: Optional[DataLoaderConfig] = None
        test: Optional[DataLoaderConfig] = None
    

    @dataclass
    class LearnerConfig:
        """A base learner configuration."""

        _target_: str = MISSING


    @dataclass
    class ClassifierConfig(LearnerConfig):
        """The classifier configuration.

        Attributes:
            hparams: The classifier's hyperparameters.
        """

        hparams: ClassifierHParams = ClassifierHParams()
        _target_: str = "sesemi.Classifier"

Each of the config attributes in turn have their own structure which may also be defined using a similar data structure.
All of these are specified in the *sesemi.config.structs* module.

These structured configurations can map directly to YAML and also enable type-checking when parsing user inputs.
Any of the nested attributes can be set both through config files as well as through the CLI.

As an example, to run the CLI and set the number of epochs to 100::

    $ open_sesemi run.num_epochs=100

-----------------------
Built-in Configurations
-----------------------

We have a couple built-in configurations which are packaged with the library. For instance, to use the imagewoof
configuration you can run::

    $ open_sesemi -cn imagewoof

This assumes you have downloaded the imagewoof dataset to the *./data/imagewoof2* directory, but otherwise it should work out of the box.

--------
Datasets
--------

Currently, the torchvision image folder dataset is the main one that is supported, however, datasets that follow a certain
format can be easily registered. The interface used to construct datasets is shown below::

    def dataset(
        name: str,
        root: str,
        subset: Optional[Union[str, List[str]]] = None,
        image_transform: Optional[Callable] = None,
        **kwargs,
    ) -> Union[Dataset, IterableDataset]:
        """Builds a dataset.

        Args:
            name: The name of the dataset to build.
            root: The path to the image folder dataset.
            subset: The subset(s) to use.
            image_transform: The image transformations to apply.
            **kwargs: Any other arguments to forward to the underlying dataset builder.

        Returns:
            The dataset.
        """

Note that the name used for image folder datasets is *image_folder*. Additionally, registering datasets is done using a
*register_dataset* decorator function.

-------
Example
-------

The following is a look at part of the imagewoof configuration::

    defaults:
      - sesemi_config
      - learner: classifier
    run:
      seed: 42
      num_epochs: 80
      gpus: 2
      accelerator: dp
      batch_size_per_gpu: 16
    data:
      train:
        supervised:
          dataset:
            _target_: sesemi.dataset
            name: image_folder
            root: imagewoof2
            subset: train
            image_transform:
              _target_: sesemi.transforms.train_transforms
          shuffle: True
          num_workers: 4
        rotation_prediction:
          dataset:
            _target_: sesemi.dataset
            name: image_folder
            root: imagewoof2
            image_transform:
              _target_: sesemi.transforms.train_transforms
          shuffle: True
          pin_memory: True
          num_workers: 4
          collate_fn:
            _target_: sesemi.collation.RotationTransformer
          drop_last: True
      val:
        dataset:
          _target_: sesemi.dataset
          name: image_folder
          root: imagewoof2
          subset: val
          image_transform:
            _target_: sesemi.transforms.center_crop_transforms
        shuffle: False
        pin_memory: True
        num_workers: 4
        drop_last: False
    learner:
      hparams:
        num_classes: 10
        model:
          backbone:
            _target_: sesemi.PyTorchImageModels
            name: resnet50d
            freeze: False
            pretrained: False
            global_pool: avg
            drop_rate: 0.5
          supervised_loss:
            callable:
              _target_: torch.nn.CrossEntropyLoss
          regularization_loss_heads:
            rotation_prediction:
              head:
                _target_: sesemi.models.heads.loss.RotationPredictionLossHead
                input_data: rotation_prediction
                input_backbone: backbone
        optimizer:
          _target_: torch.optim.SGD
          lr: 0.1
          momentum: 0.9
          nesterov: True
          weight_decay: 0.
        lr_scheduler:
          scheduler:
            _target_: sesemi.PolynomialLR
            warmup_epochs: 10
            iters_per_epoch: ${sesemi:iterations_per_epoch}
            warmup_lr: 0.001
            lr_pow: 0.5
            max_iters: ${sesemi:max_iterations}
    trainer:
      callbacks:
        - _target_: pytorch_lightning.callbacks.ModelCheckpoint
          monitor: val/top1
          mode: max
          save_top_k: 1
          save_last: True

In it, you will find that there are sections defining data loaders for supervised and unsupervised (rotation prediction)
datasets. Additionally, their is a cross entropy loss function defined for the supervised branch as well as a
rotation prediction loss head defined as a regularization branch.

Also note how there are variable interpolations of the form ${sesemi:name}. These variables are filled in at runtime
and enable referencing specific kinds of information from the configuration that may not be known ahead of time. The
set of these variables that are available are defined by the *sesemi.config.resolvers.SESEMIConfigAttributes* object
which is shown below::

  class SESEMIConfigAttributes(AttributeResolver):
      """The attributes exposed to SESEMI configuration files.

      These attributes can be referenced in the config files by following the omegaconf syntax for
      custom resolvers. For example, ${sesemi:iterations_per_epoch} will reference the
      `iterations_per_epoch` attribute.

      Attributes:
          iterations_per_epoch: The number of training iterations per epoch if training data is
              available.
          max_iterations: The maximum number of training iterations if training data is available.
          num_gpus: The number of GPUs that will be used.
          num_nodes: The number of compute nodes that will be used.
      """

      iterations_per_epoch: Optional[int]
      max_iterations: Optional[int]
      num_gpus: int
      num_nodes: int

Going back to the first section of the config, there is a *defaults* section which is used to essentially import
configurations from other sources. In this case, the *sesemi_config* default specifies that the *SESEMIConfig* data 
class should be used for type checking and specifying default attributes. Additionally, a classifier learner is
set to be used.