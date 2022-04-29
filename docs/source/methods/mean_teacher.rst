Mean Teacher
------------

This method showed that applying consistency regularization
between a student model and it's exponentially averaged
teacher model can result in significant performance improvements.

.. raw:: html
    
    <img src="../_static/images/mean-teacher-diagram.png" class="align-center" width="100%" />

Usage
^^^^^

To use this method, you must incorporate the `EMAConsistencyLossHead`
into the model. Additionally, you'll need to provide an unlabeled
data branch that provides two augmented views of the same image.
The `TwoViewsTranform` and `MultiViewTransform` are built-in
for this purpose.

We have provided a baseline training configuration that
can be run on the imagewang dataset or adapated to others:

.. command-output:: open_sesemi -cn imagewang_consistency --cfg job


References
^^^^^^^^^^

.. code-block:: bibtex

  @inproceedings{Tarvainen2017MeanTA,
    title={Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results},
    author={Antti Tarvainen and Harri Valpola},
    booktitle={NIPS},
    year={2017}
  }