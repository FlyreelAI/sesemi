import pytest
import h5py
import numpy as np

import os
import os.path as osp

from PIL import Image
import yaml
from sesemi.datasets import dataset


@pytest.mark.parametrize("num_predictions", [0, 1, 100])
@pytest.mark.parametrize("use_probability_target", [False, True])
def test_pseudo(
    tmp_path,
    num_predictions,
    use_probability_target,
):
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()

    images_dir = tmp_path / "images"
    images_dir.mkdir()

    metadata = dict(ids=[], details={})

    image = Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8))
    for i in range(num_predictions):
        data_id = str(i)

        image_filename = str(images_dir / f"{data_id}.jpg")
        image.save(image_filename)

        metadata["ids"].append(data_id)
        metadata["details"][data_id] = dict(id=data_id, filename=image_filename)
        with h5py.File(osp.join(predictions_dir, f"{data_id}.h5"), "w") as predictions:
            predictions.create_dataset("logits", data=np.ones(10, dtype=np.float32))
            predictions.create_dataset(
                "probabilities", data=0.1 * np.ones(10, dtype=np.float32)
            )

    with open(tmp_path / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)

    dst = dataset(
        name="pseudo", root=str(tmp_path), use_probability_target=use_probability_target
    )

    assert len(dst) == num_predictions
