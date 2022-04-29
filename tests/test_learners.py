from glob import glob
from typing import Dict, List

import pytest
import numpy as np

import torch
import torch.nn.functional as F

import os.path as osp
import pytorch_lightning as pl

import yaml

from omegaconf import OmegaConf
from hydra.utils import instantiate
from natsort import natsorted
from sesemi.learners import Classifier

from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MeanMetric

from torch.utils.data import DataLoader

from tqdm import tqdm

from sesemi.utils import copy_config

from .utils import (
    run_trainer,
    check_state_dicts_equal,
    check_state_dicts_not_equal,
    load_tensorboard_scalar_events,
)


def compute_metrics(
    backbone, head, dataset, device: str = "cuda", batch_size: int = 128
) -> Dict[str, float]:
    features_arr: List[np.ndarray] = []
    logits_arr: List[np.ndarray] = []

    images_arr: List[np.ndarray] = []
    targets_arr: List[np.ndarray] = []

    accuracy = Accuracy(top_k=1).to(device)
    mean_metric = MeanMetric().to(device)

    with torch.no_grad():
        backbone = backbone.to(device).eval()
        head = head.to(device).eval()

        dataloader = DataLoader(dataset, batch_size=batch_size)
        for image, target in tqdm(dataloader):
            image = image.to(device)
            target = target.to(device)

            feats = backbone(image)
            logits = head(feats)

            features_arr.append(feats.detach().cpu().numpy())
            logits_arr.append(logits.detach().cpu().numpy())
            images_arr.append(image.detach().cpu().numpy())
            targets_arr.append(target.detach().cpu().numpy())

            probabilities = torch.softmax(logits, dim=-1)
            loss = F.cross_entropy(
                probabilities,
                target,
                reduction="none",
            )

            accuracy.update(probabilities, target)
            mean_metric.update(loss)

        top1 = accuracy.compute().item()
        mean_loss = mean_metric.compute().item()

        features_np = np.concatenate(features_arr, axis=0)
        logits_np = np.concatenate(logits_arr, axis=0)
        images_np = np.concatenate(images_arr, axis=0)
        targets_np = np.concatenate(targets_arr, axis=0)

        return dict(
            top1=top1,
            loss=mean_loss,
            features=features_np,
            logits=logits_np,
            images=images_np,
            targets=targets_np,
            accuracy_metric=accuracy,
            mean_metric=mean_metric,
        )


def get_experiment_filenames(run_dir: str, run_id: str) -> Dict[str, str]:
    event_files = glob(
        osp.join(run_dir, f"{run_id}/*/lightning_logs/version_0/events.*")
    )
    assert len(event_files) == 1

    hparams_files = glob(
        osp.join(run_dir, f"{run_id}/*/lightning_logs/version_0/hparams.yaml")
    )
    assert len(hparams_files) == 1

    return dict(events=event_files[0], hparams=hparams_files[0])


@pytest.mark.parametrize("strategy", ["dp", "ddp"])
@pytest.mark.parametrize(
    "num_iterations,warmup_iters,val_check_interval,check_val_every_n_epoch,checkpoint_on_train_step,min_top_1",
    [
        (10, 5, 1, 1, True, None),
        (1000, 100, 100, 1, False, None),
    ],
)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="insufficient GPUs")
def test_learner_strategy(
    tmp_path,
    strategy,
    num_iterations,
    warmup_iters,
    val_check_interval,
    check_val_every_n_epoch,
    checkpoint_on_train_step,
    min_top_1,
):
    seed = 42
    batch_size = 32

    pl.seed_everything(seed)

    data_root = tmp_path / "data"
    data_root.mkdir()

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()

    config = dict(
        defaults=["/cifar10_wrn_28_10"],
        run=dict(
            seed=seed,
            data_root=str(data_root),
            batch_size_per_device=batch_size,
            dir=str(run_dir),
            strategy=strategy,
            devices=2,
            accelerator="gpu",
            num_iterations=num_iterations,
            num_epochs=None,
        ),
        learner=dict(
            hparams=dict(
                lr_scheduler=dict(
                    scheduler=dict(
                        warmup_iters=warmup_iters,
                        warmup_epochs=None,
                    )
                )
            )
        ),
        trainer=dict(
            sync_batchnorm=True,
            precision=32,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            callbacks=[
                dict(
                    _target_="sesemi.callbacks.DebuggingCallback",
                    debug_dir=str(debug_dir),
                    checkpoint_on_train_step=checkpoint_on_train_step,
                )
            ],
        ),
    )

    std, err, retcode = run_trainer(tmp_path, config)
    if retcode != 0:
        print(std)
        raise Exception(err)

    experiment_fns = get_experiment_filenames(run_dir, "cifar10_wrn_28_10")
    events = load_tensorboard_scalar_events(experiment_fns["events"])

    with open(experiment_fns["hparams"], "r") as f:
        sesemi_config = OmegaConf.create(yaml.safe_load(f)).sesemi_config

    val_dataset_config = copy_config(sesemi_config.data.val.dataset)
    val_dataset_config.pop("root")

    val_dataset = instantiate(val_dataset_config, root=str(data_root))

    val_top1 = events[events.tag == "val/top1"]
    val_top1_by_step = {int(x.step): float(x.value) for _, x in val_top1.iterrows()}

    val_loss = events[events.tag == "val/loss"]
    val_loss_by_step = {int(x.step): float(x.value) for _, x in val_loss.iterrows()}

    checkpoints = natsorted(
        glob(osp.join(debug_dir, "checkpoint-worker-0-validation-*.ckpt"))
    )
    checkpoint_i = [int(x.split(".")[-2].split("-")[-1]) for x in checkpoints]

    for i, ckpt_path in zip(checkpoint_i, checkpoints):
        if i == 0:
            continue
        classifier = Classifier.load_from_checkpoint(ckpt_path)
        classifier.eval()
        classifier.to("cuda")

        val_metrics = compute_metrics(
            classifier.backbone, classifier.head, val_dataset, batch_size=batch_size
        )

        assert np.isclose(val_metrics["top1"], val_top1_by_step[i - 1], atol=1e-2)

    if strategy == "ddp":
        for x in ["start", "end"]:
            ckpt0_path = osp.join(debug_dir, f"checkpoint-worker-0-{x}.ckpt")
            ckpt1_path = osp.join(debug_dir, f"checkpoint-worker-1-{x}.ckpt")
            with open(ckpt0_path, "rb") as ckpt0_file, open(
                ckpt1_path, "rb"
            ) as ckpt1_file:
                ckpt0 = torch.load(ckpt0_file)
                ckpt1 = torch.load(ckpt1_file)
                check_state_dicts_equal(ckpt0["state_dict"], ckpt1["state_dict"])

    ckpt0_path = osp.join(debug_dir, f"checkpoint-worker-0-start.ckpt")
    ckpt1_path = osp.join(debug_dir, f"checkpoint-worker-0-end.ckpt")
    with open(ckpt0_path, "rb") as ckpt0_file, open(ckpt1_path, "rb") as ckpt1_file:
        ckpt0 = torch.load(ckpt0_file)
        ckpt1 = torch.load(ckpt1_file)
        check_state_dicts_not_equal(ckpt0["state_dict"], ckpt1["state_dict"])

    if strategy == "ddp":
        if checkpoint_on_train_step:
            for i in range(1, num_iterations + 1):
                ckpt0_path = osp.join(debug_dir, f"checkpoint-worker-0-train-{i}.ckpt")
                ckpt1_path = osp.join(debug_dir, f"checkpoint-worker-1-train-{i}.ckpt")
                with open(ckpt0_path, "rb") as ckpt0_file, open(
                    ckpt1_path, "rb"
                ) as ckpt1_file:
                    ckpt0 = torch.load(ckpt0_file)
                    ckpt1 = torch.load(ckpt1_file)
                    check_state_dicts_equal(ckpt0["state_dict"], ckpt1["state_dict"])

    if min_top_1 is not None:
        best_top_1 = max(val_top1.values())
        assert best_top_1 >= min_top_1
