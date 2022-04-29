Jigsaw Prediction
-----------------

This method regularizes a model by learning to predict both the
shuffle order and class label of images from the supervised data branch.
Images are shuffled using a jigsaw pattern based on a set of predefined
permutations.

.. raw:: html
    
    <img src="../_static/images/jigsaw-diagram.png" class="align-center" width="100%" />

Usage
^^^^^

To use this method, you must apply the `JigsawCollator`
that is built-in to the supervised data branch.
You'll need to use the `JigsawPredictionLossHead`
in the model as well.

We have provided a baseline training configuration that
can be run on the imagewang dataset or adapated to others:

.. command-output:: open_sesemi -cn imagewang_jigsaw --cfg job

References
^^^^^^^^^^

.. code-block:: bibtex

  @article{Carlucci2019DomainGB,
    title={Domain Generalization by Solving Jigsaw Puzzles},
    author={Fabio Maria Carlucci and Antonio D'Innocente and Silvia Bucci and Barbara Caputo and Tatiana Tommasi},
    journal={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019},
    pages={2224-2233}
  }