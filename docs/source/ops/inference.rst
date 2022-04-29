Inference
---------

This op can be used to generate predictions over a set of images for a more detailed inspection.
It expects the images to be located under some directory and outputs a CSV file named `labels.csv`
with the following columns:

* id - a unique image identifier.
* filename - the original image filename used to generate the prediction.
* label - the label of the prediction.
* score - the probability of the prediction.

Usage
^^^^^

A sample built-in configuration that can be used to generate predictions using a multi-crop
test-time augmentation is provided and shown below:

.. code-block:: yaml

    defaults:
    - /ops/inference
    test_time_augmentation:
        _target_: sesemi.tta.MultiCropTTA
        resize: 256
        crop_dim: 224

It can be used as follows:

.. command-output:: python -m sesemi.ops.inference -cn multi_crop_inference checkpoint_path='${CHECKPOINT_PATH}' data_dir='${DATA_DIR}' output_dir='${OUTPUT_DIR}' --cfg job
