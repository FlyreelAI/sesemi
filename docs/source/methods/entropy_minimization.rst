Entropy Minimization
--------------------

This method minimizes the model's entropy on unlabeled data as a form of regularization.
It is a classical semi-supervised learning technique that is highly effective.

Usage
^^^^^

Apart from providing an unlabeled image branch, you'll need to
use the `EntropyMinimizationLossHead` in the model as well.

We have provided a baseline training configuration that
can be run on the imagewang dataset or adapated to others:

.. command-output:: open_sesemi -cn imagewang_entmin --cfg job
  
References
^^^^^^^^^^

.. code-block:: bibtex

  @inproceedings{Grandvalet2004SemisupervisedLB,
    title={Semi-supervised Learning by Entropy Minimization},
    author={Yves Grandvalet and Yoshua Bengio},
    booktitle={CAP},
    year={2004}
  }
