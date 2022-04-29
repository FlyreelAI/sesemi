Noisy Student
-------------

This method demonstrated that pseudo-labels can be effectively used for self-training
by introducing specific kinds of noise during training. In particular,
it was shown that using strong data augmentation, dropout, and stochastic depth
were effective in providing a learning signal for a model to bootstrap off its
own pseudo-labels. Illustrated below, it involves iterating between training
and pseudo-label generation on unlabeled data.

.. raw:: html
    
    <img src="../_static/images/noisy-student-diagram.png" class="align-center" width="100%" />

Usage
^^^^^

To use this method, you must alternate between supervised training
and pseudo-label generation on unlabeled data.

We have provided sample configurations that illustrate how to
run the different stages of noisy student training including:

1. Supervised training on a labeled dataset.
2. Pseudo-label generation on an unlabeled dataset.
3. Supervised training on the combined labeled and pseudo-labeled data.
4. Repeat steps 2 and 3 as necessary.

Step 1
++++++

This step can be performed on the imagewang dataset using the following configuration that can be adpated:

.. command-output:: open_sesemi -cn imagewang_noisy_student_stage_1 --cfg job

Step 2
++++++

This step can be performed by running a built-in pseudo-labeling operation as follows:

.. command-output:: python -m sesemi.ops.pseudo_dataset -cn imagewang_noisy_student_pseudo_dataset output_dir='${OUTPUT_DIR}' checkpoint_path='${CHECKPOINT_PATH}' dataset.root='${DATASET_ROOT}' --cfg job

Step 3
++++++

This and subsequent retraining steps can use the following separate configuration:

.. command-output:: open_sesemi -cn imagewang_noisy_student_stage_n data.train.supervised.dataset.datasets.1.root='${PSEUDO_DATASET_ROOT}' --cfg job

References
^^^^^^^^^^

.. code-block:: bibtex

  @article{Xie2020SelfTrainingWN,
    title={Self-Training With Noisy Student Improves ImageNet Classification},
    author={Qizhe Xie and Eduard H. Hovy and Minh-Thang Luong and Quoc V. Le},
    journal={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2020},
    pages={10684-10695}
  }