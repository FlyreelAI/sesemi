Pi Model
--------

This method showed that applying consistency regularization
between two stochastically augmented views of the same unlabeled
image can yield improved semi-supervised performance.

.. raw:: html
    
    <img src="../_static/images/pi-model-diagram.png" class="align-center" width="100%" />

Usage
^^^^^

To use this method, you must incorporate the `ConsistencyLossHead`
into the model. Additionally, you'll need to provide an unlabeled
data branch that provides two augmented views of the same image.
The `TwoViewsTranform` and `MultiViewTransform` are built-in
for this purpose.

The mean teacher algorithm is an extension of this work so we
have only provided a configuration for mean teacher that can
easily be adapted for the Pi model. This can be done by turning off EMA
and using the the aforementioned `ConsistencyLossHead` instead of
the `EMAConsistencyLossHead`. The consistency configuration is
shown below for reference:

.. command-output:: open_sesemi -cn imagewang_consistency --cfg job

References
^^^^^^^^^^

.. code-block:: bibtex

  @article{Laine2017TemporalEF,
    title={Temporal Ensembling for Semi-Supervised Learning},
    author={Samuli Laine and Timo Aila},
    booktitle={ICLR},
    year={2017},
  }