#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import os
import numpy as np
from tqdm import trange
import logging

import torch
import hydra
from torchvision import datasets
from hydra.core.config_store import ConfigStore

from ..config.structs import SESEMIInferenceConfig
from ..learners import Classifier
from ..utils import validate_paths
from ..transforms import center_crop_transforms, multi_crop_transforms


logger = logging.getLogger(__name__)

config_store = ConfigStore.instance()
config_store.store(
    name="inference",
    node=SESEMIInferenceConfig,
    group="ops",
    package="_global_",
)


class Predictor:
    def __init__(self, model_path, classes, config):
        self.config = config
        self.model_path = model_path
        self.classes = np.array(classes)
        self.device = torch.device(
            "cpu" if config.no_cuda or not torch.cuda.is_available() else "cuda"
        )
        self._init_model()

    def _init_model(self):
        self.model = Classifier.load_from_checkpoint(
            self.model_path, map_location=self.device
        )
        assert len(self.classes) == self.model.hparams.num_classes
        logger.info(f"=> Model checkpoint loaded from {self.model_path}")
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.eval()

    def predict(self, x, ncrops, topk=1):
        with torch.no_grad():
            x = x.to(self.device)
            batch_size = x.size(0)
            w, h, c = x.shape[-1:-4:-1]
            outputs = self.model(x.view(-1, c, h, w))  # fuse batch size and ncrops
            outputs = torch.softmax(outputs, dim=-1)
            outputs = outputs.view(batch_size, ncrops, -1).mean(1)  # avg over crops
            scores, indices = torch.topk(outputs, k=topk, largest=True, sorted=True)
            scores = scores.cpu().numpy()
            indices = indices.cpu().numpy()
            labels = self.classes[indices]
            return (labels, scores)


@hydra.main(config_path="./conf", config_name="/ops/inference")
def predict(config: SESEMIInferenceConfig):
    # Data loading
    validate_paths([config.data_dir])
    if config.oversample:
        ncrops = config.ncrops
        test_transformations = multi_crop_transforms(
            config.resize, config.crop_dim, ncrops, interpolation=3
        )
    else:
        ncrops = 1
        test_transformations = center_crop_transforms(
            config.resize, config.crop_dim, interpolation=3
        )
    dataset = datasets.ImageFolder(config.data_dir, test_transformations)
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=False,
    )

    classifier = Predictor(config.checkpoint_path, dataset.classes, config)

    # Write prediction results to file
    if os.path.exists(config.outfile):
        os.remove(config.outfile)
    with open(config.outfile, "a") as f:
        header = ",".join(["Id", "Category", "Score"])
        f.write(header + "\n")
    index = 0

    dataset_iterator = iter(dataset_loader)
    for _ in trange(
        len(dataset_loader),
        desc=f"Inferencing on {len(dataset.imgs)} files",
        position=1,
    ):
        inputs, _ = next(dataset_iterator)
        labels, scores = classifier.predict(inputs, ncrops, config.topk)
        # Write prediction results to file
        with open(config.outfile, "a") as f:
            for label, score in zip(labels, scores):
                img_path = dataset.imgs[index][0]
                img_id = os.path.splitext(os.path.basename(img_path))[0]
                label = " ".join(label)
                score = [f"{s:.6f}" for s in score]
                score = " ".join(score)
                f.write(",".join([img_id, label, score]) + "\n")
                index += 1


if __name__ == "__main__":
    predict()
