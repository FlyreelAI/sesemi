FixMatch
--------

This method showed that combining consistency regularization and pseudo-labeling
techniques can result in a simple and effective semi-supervised learning approach.
Diagramed below, it involves applying consistency regularization between
a weakly and strongly augmented view of the same unlabeled image and only
on predictions where the weakly augmented sample meets a specific confidence
threshold.

.. raw:: html
    
    <img src="../_static/images/fixmatch-diagram.png" class="align-center" width="100%" />

Usage
^^^^^

To use this method, you must incorporate the `FixMatchLossHead`
into the model. Additionally, you'll need to provide an unlabeled
data branch that provides two augmented views of the same image.
The `TwoViewsTranform` and `MultiViewTransform` are built-in
for this purpose.

We have provided a sample configuration for the imagewang dataset
that can be easily adapated to run on other datasets:

.. command-output:: open_sesemi -cn imagewang_fixmatch --cfg job

References
^^^^^^^^^^

.. code-block:: bibtex

  @article{Sohn2020FixMatchSS,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin Dogus Cubuk and Alexey Kurakin and Han Zhang and Colin Raffel},
    journal={Neural Information Processing System},
    year={2020},
  }