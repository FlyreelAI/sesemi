Evaluating Trained Models
-------------------------

Once you have trained a model, you can evaluate it one of the following non-exhaustive ways:

1. Regenerate metrics on the validation set by using the VALIDATE or TEST run modes.

For example, you can generate the validation metrics using the same training configuration as follows:

.. code-block:: bash

    $ open_sesemi -cd configs -cn baseline run.mode=VALIDATE

This assumes the training configuration is located at `configs/baseline.yaml`.

2. Use the inference op to generate detailed model predictions for further inspection.

.. code-block:: bash

    $ python -m sesemi.ops.inference \
        checkpoint_path=${CHECKPOINT_PATH} \
        data_dir=${DATA_DIR} \
        output_dir=${OUTPUT_DIR}

This produces a directory at `${OUTPUT_DIR}` that contains a CSV file with the model predictions
over the images located under `${DATA_DIR}`.

3. Inspect the detailed tensorboard training logs for metrics and other details.