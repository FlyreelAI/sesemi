#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import os.path as osp

import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback


class DebuggingCallback(Callback):
    def __init__(self, debug_dir: str, checkpoint_on_train_step: bool = False):
        super().__init__()
        self.debug_dir = debug_dir
        self.checkpoint_on_train_step = checkpoint_on_train_step

    def save_checkpoint(
        self,
        trainer: "pl.Trainer",
        filepath: str,
    ) -> None:
        trainer.strategy.checkpoint_io.save_checkpoint(
            trainer._checkpoint_connector.dump_checkpoint(), filepath
        )

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        checkpoint_path = osp.join(
            self.debug_dir, f"checkpoint-worker-{trainer.global_rank}-start.ckpt"
        )
        self.save_checkpoint(trainer, checkpoint_path)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        checkpoint_path = osp.join(
            self.debug_dir,
            f"checkpoint-worker-{trainer.global_rank}-validation-{trainer.global_step}.ckpt",
        )
        self.save_checkpoint(trainer, checkpoint_path)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        *args,
        **kwargs,
    ) -> None:
        if self.checkpoint_on_train_step:
            checkpoint_path = osp.join(
                self.debug_dir,
                f"checkpoint-worker-{trainer.global_rank}-train-{trainer.global_step}.ckpt",
            )
            self.save_checkpoint(trainer, checkpoint_path)

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        checkpoint_path = osp.join(
            self.debug_dir, f"checkpoint-worker-{trainer.global_rank}-end.ckpt"
        )
        self.save_checkpoint(trainer, checkpoint_path)
