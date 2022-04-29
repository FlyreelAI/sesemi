A Primer on Hydra Configurations
--------------------------------

..
  Goal

  * Users familiar with the problem Hydra was designed to solve.
  * Users familiar with the approach that Hydra takes in addressing th e problem.
  * Users familiar with why, how, and where SESEMI makes use of Hydra.
  * Users familiar with the basics of defining Hydra configurations (structured and unstructed configurations).
  * Users familiar with using Hydra as a CLI.
  * Users familiar with Hydra YAML support and defaults lists.
  * Users familiar with configuration groups.
  * Users familiar with layout of CLIs and config files within SESEMI.

  Outline

  * Introduction to problem Hydra designed to solve
  * Hydra Overview (approach to solve problem)
  * Hydra with SESEMI
  * Basics
  * Structured Configuration
  * CLI Functionality
  * YAML Functionality
  * Defaults lists
  * Configuration Groups
  * Built-in Configs and CLIs

If you are not familiar with Hydra, we have provided a brief primer on the topic here, however,
you are encouraged to go through the associated
`documentation <https://hydra.cc/docs/intro/>`_. To quote their introduction:

  Hydra is an open-source Python framework that simplifies the development of research and other
  complex applications. The key feature is the ability to dynamically create a hierarchical configuration
  by composition and override it through config files and the command line. The name Hydra comes from its
  ability to run multiple similar jobs - much like a Hydra with multiple heads.

  Key features:
  
  * Hierarchical configuration composable from multiple sources
  * Configuration can be specified or overridden from the command line
  * Dynamic command line tab completion
  * Run your application locally or launch it to run remotely
  * Run multiple jobs with different arguments with a single command

========
Overview
========

Hydra builds on the sister-project `OmegaConf <https://omegaconf.readthedocs.io/en/2.1_branch/>`_.
In fact, it is primarily a way of constructing OmegaConf objects from multiple sources,
with the main ones being YAML files and CLI arguments.

The YAML files which are used as sources are interpreted by OmegaConf and Hydra
and have unique capabilities such as variable interpolation, setting of defaults
from other files, and more.

The CLI also operates differently than other existing frameworks. Rather than working with
single-level flags, you work with entire config objects whose parameters are set via
the command-line. For an introduction on these semantics please see the
`extensive documentation <https://hydra.cc/docs/intro/>`_.

..
  * Builds on Omegaconf.
  * In fact, it is a way of constructing OmegaConf objects from multiple sources.
  * Main sources are YAML files and command-line arguments.
  * The YAML files are interpreted by OmegaConf and Hydra and have unique capabilities
    such as variable interpolation, setting defaults defined in other files, and more.
  * CLI also works differently than other existing frameworks. Rather than working with
    flags, you work with config objects whose parameters are set via the command-line.

======
Basics
======

As mentioned, the main data structure within Hydra is the config object (i.e. OmegaConf).
Such objects can be either structured or unstructured, with the main difference being
that structured configs generally have well-defined typed parameters while unstructured
configs are more akin to dictionaries that support arbitrary parameters.

In most cases, it is beneficial to add structure to the configurations as this allows
Hydra to perform type-checking and more. However, there are instances where it is preferable
to allow arbitrary objects.

Config objects are the main entrypoints to your applications with Hydra. They create a
single interface to your app regardless of whether a file or command-line arguments are
used to parameterize it. This ultimately is what simplifies the process of supporting
various complex usecases.

..
  * Main data structure in Hydra is the config object (OmegaConf).
  * These objects can be structured or unstructured.
  * They are typically the main entrypoint for your application.
  * Benefits of structured configurations are that they enable type checking and
    command-completion abilities.
  * Illustrate a complete Hydra application including CLI, config file, and structured config.
    (take from Hydra examples).

=================
Usage with SESEMI
=================

The key areas that Hydra is used within SESEMI are:

* Training CLI (open_sesemi)
* ML Operation CLIs

As described earlier, these CLIs are fundamentally defined using objects.
For instance, the `open_sesemi` command is defined as follows:

.. code-block:: python

  @hydra.main(config_path="conf", config_name="/base/config")
  def open_sesemi(config: SESEMIBaseConfig):
      """The trainer's main function.

      Args:
          config: The trainer config.
      """

Note that the function `open_sesemi` is the entrypoint to the application
and its only argument is a structured configuration object. We define
these structures as dataclasses in the `sesemi.config.structs` module.
The definition for `SESEMIBaseConfig` is shown here as a reference,
however, you are encouraged to inspect the API reference for more detail:

.. code-block:: python

  class SESEMIBaseConfig:
    """The base SESEMI configuration.

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

At present, these objects serve as documentation of the CLIs and configuration files
that we support. Since they map to the same data structure, there is not much of a
difference in what's supported via YAML files compared to command-line arguments.
There is an `open issue <https://github.com/facebookresearch/hydra/issues/633>`_ for Hydra
to allow adding help text to the CLI and it is intended to be addressed in the next release milestone,
however, the documented structured configurations provide detailed information as well.

***********
Open SESEMI
***********

The `open_sesemi` CLI exposes a number of configurations from the config path
`sesemi/trainer/conf/`. Included are both supervised and semi-supervised baselines.
Below we provide a partial list of their names:

* cifar10
* imagewang
* imagewoof
* imagewang_consistency
* imagewang_entmin
* imagewang_fixmatch
* imagewang_rotation
* imagewoof_entmin
* imagewoof_rotation

In addition to these baselines, we have organized the configurations to be reusable.
This is done by factoring out logical parts of the configurations following a
directory hierarchy.

The current layout is as follows::

  /base                           # Stores the factorized configs.
    /supervised                   # Supervised configs.
      /backbone                   # Contains supervised backbone configs.
        {BACKBONE_TYPE}.yaml
      /head                       # Contains supervised head configs.
        {HEAD_TYPE}.yaml
      /data                       # Configs for labeled data.
        {DATASET_TYPE}.yaml
      /lr_scheduler               # Configs for LR schedulers.
        {LR_SCHEDULER_TYPE}.yaml
      /baselines                  # Supervised baseline configs without including data.
        resnet50.yaml
      /optimizer                  # Configs for optimizers.
        {OPTIMIZER_TYPE}.yaml
      model.yaml                  # The base supervised model config used by the baselines.
    /{SSL_METHOD_1}               # Configs for self-supervised and semi-supervised methods.
      /data                       # Unlabeled data configs for the SSL method.
        {DATASET_TYPE}.yaml
      model.yaml                  # Loss head and other model-specific configs.
    /{SSL_METHOD_N}               # An additional SSL method config.
      model.yaml
    config.yaml                   # The base config that serves as a default for other configs.

These configs can be combined and customized in a variety of different ways. As an example,
this is what the imagewang entropy minimization config looks like:

.. code-block:: yaml

  # @package _global_
  defaults:
    - /base/supervised/baselines/resnet50
    - /base/supervised/data/imagewang
    - /base/entropy_minimization/model
    - /base/entropy_minimization/data/imagewang
  run:
    seed: 42
    gpus: 2
    batch_size_per_gpu: 16
    num_epochs: 100
    id: imagewang_entmin

You can reuse these components in much the same way and override sections as needed.
Generally though, you'll want to include the following in your own config files:

* A supervised baseline model config.
* A labeled dataset config.
* A self-supervised or semi-supervised method's model config.
* A self-supervised or semi-supervised method's data config.