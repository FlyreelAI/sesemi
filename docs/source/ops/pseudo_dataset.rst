Pseudo Dataset
--------------

To enable training with pseudo-labels we have exposed an operation
that is able to take in a pretrained model and an arbitrary dataset
of images to generate a pseudo-labeled dataset in a format
standardized internally.

The structure of the standardized dataset is illustrated below::

  /pseudo_dataset
    /images
      ${EXAMPLE_1}.jpg
      ...
      ${EXAMPLE_N}.jpg
    /predictions
      ${EXAMPLE_1}.h5
      ...
      ${EXAMPLE_N}.h5
    metadata.yaml

Where the metadata file has the following schema::

  ids: [str]
  details: [{id: str, filename: str}]

Additionally, the format of the prediction H5 files is shown below::

  probabilities: [float32]
  logits: [float32]

Usage
^^^^^

A sample built-in configuration that can be used to generate pseudo-datasets from the
imagewang dataset is expanded below. It can be adapated for other datasets as well:

.. command-output:: python -m sesemi.ops.pseudo_dataset -cn imagewang_noisy_student_pseudo_dataset output_dir='${OUTPUT_DIR}' checkpoint_path='${CHECKPOINT_PATH}' dataset.root='${DATASET_ROOT}' --cfg job
