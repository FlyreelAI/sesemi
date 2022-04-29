Rotation Prediction
-------------------

This method employs a self-supervised pretext task
that involves predicting which of a set of possible rotations
an image has been transformed into. Despite its simplicity
it has been demonstrated to yield strong baseline performance
when integrated into a semi-supervised training pipeline.

.. raw:: html
    
    <img src="../_static/images/sesemi-diagram.png" class="align-center" width="100%" />

Usage
^^^^^

To use this method, you must apply the `RotationCollator`
that is built-in to the supervised data branch.
You'll need to use the `RotationPredictionLossHead`
in the model as well.

We have provided a baseline training configuration that
can be run on the imagewang dataset or adapated to others:

.. command-output:: open_sesemi -cn imagewang_rotation --cfg job

References
^^^^^^^^^^

.. code-block:: bibtex

  @inproceedings{TranSESEMI,
    title="{Exploring Self-Supervised Regularization for Supervised and Semi-Supervised Learning}",
    author={Phi Vu Tran},
    booktitle={NeurIPS Workshop on Learning with Rich Experience: Integration of Learning Paradigms},
    year={2019}
  }

  @article{Zhai2019S4LSS,
    title={S4L: Self-Supervised Semi-Supervised Learning},
    author={Xiaohua Zhai and Avital Oliver and Alexander Kolesnikov and Lucas Beyer},
    journal={2019 IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2019},
    pages={1476-1485}
  }

  @inproceedings{
    gidaris2018unsupervised,
    title={Unsupervised Representation Learning by Predicting Image Rotations},
    author={Spyros Gidaris and Praveer Singh and Nikos Komodakis},
    journal={International Conference on Learning Representations},
    year={2018},
    url={https://openreview.net/forum?id=S1v4N2l0-},
  }