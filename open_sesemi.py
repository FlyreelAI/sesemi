# Copyright 2021, Flyreel. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import os
import argparse
import numpy as np
from pytorch_lightning import callbacks

import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from torchvision import datasets
from pprint import pprint

from models import SESEMI
from utils import load_checkpoint, validate_paths, assert_same_classes
from dataset import (
    UnlabeledDataset, RotationTransformer,
    train_transforms, center_crop_transforms
)

from pytorch_lightning.callbacks import ModelCheckpoint

import logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='Supervised and Semi-Supervised Image Classification',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Optional subparser to evaluate trained model on validation dataset and exit
subparsers = parser.add_subparsers(dest='mode', help='module modes')
evaluate_parser = subparsers.add_parser('evaluate-only', help='model evaluation')
# Run arguments
parser.add_argument('--run-id', default='run01',
                    help='experiment ID to name checkpoints and logs')
parser.add_argument('--log-dir', default='./logs',
                    help='directory to output checkpoints and metrics')
parser.add_argument('--resume-from-checkpoint', default='',
                    help='path to saved checkpoint')
parser.add_argument('--pretrained-checkpoint-path', default='',
                    help='path to pretrained model weights')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable cuda')
parser.add_argument('--resume', default='',
                    help='path to latest checkpoint')
# Data loading arguments
parser.add_argument('--data-dir', nargs='+', default=[],
                    help='path(s) to dataset containing "train" and "val" subdirs')
parser.add_argument('--unlabeled-dir', nargs='+', default=[],
                    help='path(s) to unlabeled dataset with one or more subdirs containing images')
parser.add_argument('--batch-size', default=16, type=int,
                    help='mini-batch size')
parser.add_argument('--workers', default=6, type=int,
                    help='number of data loading workers')
parser.add_argument('--resize', default=256, type=int,
                    help='resize smaller edge to this resolution while maintaining aspect ratio')
parser.add_argument('--crop-dim', default=224, type=int,
                    help='dimension for center or multi cropping')
# Training arguments
parser.add_argument('--backbone', default='resnet50d',
                    help='choice of backbone architecture')
parser.add_argument('--freeze-backbone', action='store_true',
                    help='freeze backbone weights from updating')
parser.add_argument('--pretrained', action='store_true',
                    help='use backbone architecture with pretrained ImageNet weights')
parser.add_argument('--optimizer', default='SGD',
                    choices=['SGD'.lower(), 'Adam'.lower()],
                    help='optimizer to use')
parser.add_argument('--lr', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--epochs', default=30, type=int,
                    help='number of total epochs to run')
parser.add_argument('--lr-pow', default=0.5, type=float,
                    help='power to drop LR in polynomial scheduler')
parser.add_argument('--warmup-lr', default=1e-6, type=float,
                    help='initial learning rate for warmup')
parser.add_argument('--warmup-epochs', default=0, type=int,
                    help='number of warmup epochs')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum parameter in SGD or beta1 parameter in Adam')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    help='weight decay')
parser.add_argument('--num-gpus', default=1, type=int,
                    help='the number of GPUs to use')
parser.add_argument('--fully-supervised', action='store_true')


def open_sesemi():
    args = parser.parse_args()
    
    args.device = torch.device(
        'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda'
    )
    run_dir = os.path.join(args.log_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Data loading
    traindir, valdir = [], []
    for datadir in args.data_dir:
        for d in os.scandir(datadir):
            if d.is_dir():
                if d.name == 'train':
                    traindir.append(os.path.join(datadir, d))
                elif d.name == 'val':
                    valdir.append(os.path.join(datadir, d))
                else:
                    continue
    data_dirs = traindir + valdir
    if args.unlabeled_dir:
        data_dirs.extend(args.unlabeled_dir)
    validate_paths(data_dirs)
    
    train_transformations = train_transforms(
        random_resized_crop=True, resize=args.resize, crop_dim=args.crop_dim, scale=(0.2, 1.0), p_erase=0.0, interpolation=3
    )
    test_transformations = center_crop_transforms(resize=args.resize, crop_dim=args.crop_dim, interpolation=3)
    
    train_dataset = torch.utils.data.ConcatDataset([
        datasets.ImageFolder(datadir, train_transformations) for datadir in traindir
    ])
    val_dataset = torch.utils.data.ConcatDataset([
        datasets.ImageFolder(datadir, test_transformations) for datadir in valdir
    ])
    unlabeled_dataset = torch.utils.data.ConcatDataset([
        UnlabeledDataset(datadir, train_transformations) for datadir in data_dirs
    ])
    
    for dataset in [train_dataset, val_dataset]:
        assert_same_classes(dataset.datasets)

    rotate = RotationTransformer()
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=lambda x: np.random.seed(),
        collate_fn=rotate, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True,
        worker_init_fn=lambda x: np.random.seed(), drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    num_unlabeled_classes = rotate.num_rotation_labels
    args.classes = train_dataset.datasets[0].classes

    # Initialize variables
    args.epoch_over = max(len(train_loader), len(unlabeled_loader))
    args.warmup_iters = args.warmup_epochs * args.epoch_over
    args.max_iters = args.epochs * args.epoch_over
    args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr
    args.stop_rampup = int(0.0 * args.max_iters) # try 0.1-0.5
    args.loss_weight = 1.0
    args.curr_iter = 0
    args.best_val_score = 0
    
    # Initialize model with optional pretrained backbone
    hparams = OmegaConf.create(dict(
        backbone=args.backbone,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        num_labeled_classes=len(args.classes),
        num_unlabeled_classes=num_unlabeled_classes if not args.fully_supervised else 0,
        classes=args.classes,
        dropout_rate=0.5,
        global_pool='avg',
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        initial_loss_weight=args.loss_weight,
        stop_rampup=args.stop_rampup,
        warmup_iters=args.warmup_iters,
        warmup_lr=args.warmup_lr,
        lr=args.lr,
        lr_pow=args.lr_pow,
        max_iters=args.max_iters,
    ))

    model = SESEMI(hparams)

    model_checkpoint_callback = ModelCheckpoint(
        monitor='val/top1',
        mode='max',
        save_top_k=1, 
        save_last=True)

    trainer = pl.Trainer(
        gpus=args.num_gpus, 
        accelerator='dp', 
        max_steps=args.max_iters,
        default_root_dir=run_dir,
        progress_bar_refresh_rate=0,
        resume_from_checkpoint=args.resume_from_checkpoint or None,
        callbacks=[model_checkpoint_callback])

    if not args.resume_from_checkpoint and args.pretrained_checkpoint_path:
        load_checkpoint(model, args.pretrained_checkpoint_path)

    print(f'CLI hyperparameters:')
    pprint(dict(hparams))
    print()
    print(f'Fully loaded hyperparameters:')
    pprint(dict(model.hparams))
    print()
    print(f'Steps per epoch: {args.epoch_over}')
    print()
    
    if args.mode == 'evaluate-only':
        # Evaluate model on validation set and exit
        trainer.validate(model, val_loader)
        return

    if args.fully_supervised:
        loaders = dict(supervised=train_loader)
    else:
        loaders = dict(supervised=train_loader, unsupervised_rotation=unlabeled_loader)

    trainer.fit(model, loaders, val_loader)
    

if __name__ == '__main__':
    open_sesemi()

