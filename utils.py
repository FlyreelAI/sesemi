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
import torch
import numpy as np
from tqdm import trange
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from collections import OrderedDict
from itertools import combinations
import logging
import os, errno


def to_device(objects, device):
    objs = []
    for obj in objects:
        objs.append(obj.to(device))
    return objs


def forward_pass(model, x, y, device):
    x = x.to(device)
    y = y.to(device)
    outputs = model(x)
    loss = F.cross_entropy(outputs, y, reduction='mean')
    return outputs, loss


def sigmoid_rampup(curr_iter, rampup_iters):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_iters == 0:
        return 1.0
    else:
        current = np.clip(curr_iter, 0.0, rampup_iters)
        phase = 1.0 - current / rampup_iters
        return float(np.exp(-5.0 * phase * phase))


def train(model, train_loader, unlabeled_loader, optimizer, args):
    # Initialize history object to record and compute statistics
    history = HistoryDict()
    
    # Switch to train mode
    model.train()

    train_iterator = iter(train_loader)
    unlabeled_iterator = iter(unlabeled_loader)
    for _ in trange(args.epoch_over, desc=f'Train Epoch {args.curr_epoch}', position=2):
        adjust_polynomial_lr(optimizer, args)
        args.curr_iter += 1
        history.update('lr', optimizer.param_groups[0]['lr'], n=1)
        
        try:
            inputs_t, targets_t = next(train_iterator)
            inputs_u, targets_u = next(unlabeled_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            unlabeled_iterator = iter(unlabeled_loader)
            inputs_t, targets_t = next(train_iterator)
            inputs_u, targets_u = next(unlabeled_iterator)
        
        # Forward pass
        inputs_t, targets_t, inputs_u, targets_u = to_device(
            (inputs_t, targets_t, inputs_u, targets_u), device=args.device
        )
        outputs_t, outputs_u = model(inputs_t, inputs_u)
        
        loss_weight = args.loss_weight * sigmoid_rampup(args.curr_iter, args.stop_rampup)
        
        loss_t = F.cross_entropy(outputs_t, targets_t, reduction='mean')
        loss_u = F.cross_entropy(outputs_u, targets_u, reduction='mean')
        loss = loss_t + loss_u * loss_weight

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure accuracy and record statistics
        acc, batch_size = accuracy(outputs_t.data, targets_t, topk=(1,))
        history.update('training loss', loss.item(), n=batch_size)
        history.update('training top1 accuracy', acc[0], n=batch_size)
        history.update('self-supervised weight', loss_weight, n=1)
    
    logging.info(
        'Train Loss: {history[training loss].avg:.4f}  '
        'Train Acc: {history[training top1 accuracy].avg:.4f}  '
        'LR: {history[lr].value:.6f}  '
        'loss_w: {history[self-supervised weight].value:.2f}'.format(history=history)
    )


def evaluate(model, val_loader, args):
    # Initialize history object to record and compute statistics
    history = HistoryDict()

    # Switch to evaluate mode
    model.eval()

    val_iterator = iter(val_loader)
    for _ in trange(len(val_loader), desc=f'Val Epoch {args.curr_epoch}', position=2):
        inputs, targets = next(val_iterator)
        
        # Forward pass
        outputs, loss = forward_pass(model, inputs, targets, args.device)

        # measure accuracy and record loss
        acc, batch_size = accuracy(outputs.data, targets, topk=(1,))
        history.update('validation loss', loss.item(), n=batch_size)
        history.update('validation top1 accuracy', acc[0], n=batch_size)
    
    logging.info(
        'Valid Loss: {history[validation loss].avg:.4f}  '
        'Valid Acc: {history[validation top1 accuracy].avg:.4f}  '
        'Valid Counts: {history[validation loss].count:d}'.format(history=history)
    )
    return history


class History():
    """Compute and store history statistics"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class HistoryDict():
    def __init__(self):
        self.history = {}

    def __getitem__(self, key):
        return self.history[key]

    def update(self, name, value, n=1):
        if not name in self.history:
            self.history[name] = History()
        self.history[name].update(value, n)

    def reset(self):
        for value in self.history.values():
            value.reset()

    def values(self, postfix=''):
        return {name + postfix: value.value for name, value in self.history.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: value.avg for name, value in self.history.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: value.sum for name, value in self.history.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: value.count for name, value in self.history.items()}


class GammaCorrection():
    def __init__(self, r=(0.5, 2.0)):
        self.gamma_range = r
        
    def __call__(self, x):
        gamma = np.random.uniform(*self.gamma_range)
        return TF.adjust_gamma(x, gamma, gain=1)

    def __repr__(self):
        return self.__class__.__name__ + '(r={})'.format(self.gamma_range)

    
def adjust_polynomial_lr(optimizer, curr_iter, *, warmup_iters, warmup_lr, lr, lr_pow, max_iters):
    """Decay learning rate according to polynomial schedule with warmup"""
    if curr_iter < warmup_iters:
        frac = curr_iter / warmup_iters
        step = lr - warmup_lr
        running_lr = warmup_lr + step * frac
    else:
        frac = (float(curr_iter) - warmup_iters) / (max_iters - warmup_iters)
        scale_running_lr = max((1.0 - frac), 0.) ** lr_pow
        running_lr = lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr
    
    return running_lr
        
        
def accuracy(output, target, topk=(1,)):
    """Compute topk accuracy"""
    output = output.cpu()
    target = target.cpu()
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return (res, batch_size)


def save_model(model, args, path):
    state_dict = OrderedDict({
        'meta': args.__dict__,
        'model_params': {
            'backbone': model.module.backbone,
            'pretrained': model.module.pretrained,
            'num_labeled_classes': model.module.num_labeled_classes,
            'num_unlabeled_classes': model.module.num_unlabeled_classes,
            'dropout_rate': model.module.dropout_rate,
            'global_pool': model.module.global_pool,
        },
        'state_dict': model.module.state_dict(),
    })
    torch.save(state_dict, path)
    logging.info('=> Model checkpoint saved to {:s}'.format(path))


def load_model(Model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint['model_params']['pretrained'] = False
    model = Model(**checkpoint['model_params'])
    model.load_state_dict(checkpoint['state_dict'])
    model.CLASSES = checkpoint['meta']['classes']
    return model.to(device)


def assert_same_classes(datasets):
    if len(datasets) == 1:
        return True
    same_classes = [x.class_to_idx == y.class_to_idx for x, y in combinations(datasets, r=2)]
    assert all(same_classes), \
    f'The following have mismatched subdirectory names. Check the `Root location`.\n{datasets}'


def validate_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path
            )

