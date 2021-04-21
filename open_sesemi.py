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
import os, errno
import argparse
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter

import torch
from torchvision import datasets

from models import SESEMI
from utils import train, evaluate, save_model, load_model
from dataset import (
    UnlabeledDataset, RotationTransformer,
    train_transforms, center_crop_transforms
)

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
evaluate_parser.add_argument('--data-dir', required=True,
                             help='path to dataset containing "train" and "val" subdirs')
evaluate_parser.add_argument('--checkpoint-path', required=True,
                             help='path to saved checkpoint')
# Run arguments
parser.add_argument('--run-id', default='run01',
                    help='experiment ID to name checkpoints and logs')
parser.add_argument('--checkpoint-dir', default='./checkpoints',
                    help='directory to output checkpoints')
parser.add_argument('--checkpoint-path', default='',
                    help='path to saved checkpoint')
parser.add_argument('--logs', default='./logs',
                    help='directory to logging')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable cuda')
parser.add_argument('--resume', default='',
                    help='path to latest checkpoint')
# Data loading arguments
parser.add_argument('--data-dir', default='',
                    help='path to dataset containing "train" and "val" subdirs')
parser.add_argument('--unlabeled-dir', default='',
                    help='path to unlabeled dataset with one or more subdirs containing images')
parser.add_argument('--batch-size', default=16, type=int,
                    help='mini-batch size')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers')
# Training arguments
parser.add_argument('--backbone', default='resnet50',
                    help='choice of backbone architecture')
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
parser.add_argument('--pretrained', action='store_true',
                    help='use backbone architecture with pretrained ImageNet weights')
parser.add_argument('--freeze-backbone', action='store_true',
                    help='freeze backbone weights from updating')


def open_sesemi():
    args = parser.parse_args()
    
    args.device = torch.device(
        'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda'
    )
    os.makedirs(os.path.join(args.checkpoint_dir, args.run_id), exist_ok=True)

    # Data loading
    if not args.data_dir:
        logging.info('Exiting. --data-dir argument is required.')
        exit()
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    data_dirs = [traindir, valdir]
    if args.unlabeled_dir:
        data_dirs.append(args.unlabeled_dir)
    for path in data_dirs:
        if not os.path.exists(path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path
            )
    
    train_transformations = train_transforms(
            random_resized_crop=True, resize=256, crop_dim=224, scale=(0.2, 1.0), p_erase=0.0
    )
    test_transformations = center_crop_transforms(resize=256, crop_dim=224)
    
    train_dataset = datasets.ImageFolder(traindir, train_transformations)
    val_dataset = datasets.ImageFolder(valdir, test_transformations)
    unlabeled_dataset = torch.utils.data.ConcatDataset([
        UnlabeledDataset(datadir, train_transformations) for datadir in data_dirs
    ])
    rotate = RotationTransformer()
    unlabeled_classes = rotate.rotation_labels
    
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=lambda x: np.random.seed(),
        collate_fn=rotate, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size * unlabeled_classes,
        shuffle=True, num_workers=args.workers, pin_memory=True,
        worker_init_fn=lambda x: np.random.seed(), drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    # Model loading
    if args.checkpoint_path:
        # Load saved model for finetuning or evaluation
        model = load_model(SESEMI, args.checkpoint_path, args.device)
        logging.info(f'=> Model checkpoint loaded from {args.checkpoint_path}')
        if args.mode == 'evaluate-only':
            # Evaluate model on validation set and exit
            args.curr_epoch = '###'
            with torch.no_grad():
                evaluate(model, val_loader, args)
            return
    else:
        # Initialize model with optional pretrained backbone
        model = SESEMI(
            args.backbone,
            pretrained=args.pretrained,
            labeled_classes=len(train_dataset.classes),
            unlabeled_classes=unlabeled_classes
        )
    
    if args.freeze_backbone:
        logging.info(f'Freezing {model.backbone} backbone')
        for m in model.feature_extractor.children():
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    model = torch.nn.DataParallel(model).to(args.device)

    # Optimizer options
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, momentum=args.momentum, nesterov=True,
            weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, betas=(args.momentum, 0.999), weight_decay=0.0)
    else:
        raise NotImplementedError()    
    
    # TODO: Tensorboard for monitoring training statistics
    tb_path = os.path.join(args.logs, args.run_id)
    os.makedirs(tb_path, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_path)

    # Initialize variables
    args.warmup_iters = args.warmup_epochs * len(unlabeled_loader)
    args.max_iters = args.epochs * len(unlabeled_loader)
    args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr
    args.stop_rampup = int(0.0 * args.max_iters) # try 0.1-0.5
    args.loss_weight = 1.0
    args.curr_iter = 0
    args.best_val_score = 0
    args.classes = train_dataset.classes
        
    # Start training and evaluation
    for epoch in trange(1, args.epochs + 1, desc='Epoch', unit='epoch', position=0):
        args.curr_epoch = epoch
        
        # Train for one epoch
        train(model, train_loader, unlabeled_loader, optimizer, args)
        
        # Evaluate after each epoch
        with torch.no_grad():
            history = evaluate(model, val_loader, args)
            
        # Save best validation model
        curr_val_score = history['validation top1 accuracy'].avg
        logging.info('Epoch {:03d} =====> {:.4f} vs. Best {:.4f}'.format(
                epoch, curr_val_score, args.best_val_score))
        if curr_val_score > args.best_val_score:
            args.best_val_score = curr_val_score
            save_model(model, args, path=os.path.join(args.checkpoint_dir, args.run_id, 'best_val.pth'))
    

if __name__ == '__main__':
    open_sesemi()

