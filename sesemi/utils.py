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
import numpy as np
import os, errno
import torch
import torchvision.transforms.functional as TF

from itertools import combinations


def sigmoid_rampup(curr_iter, rampup_iters):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_iters == 0:
        return 1.0
    else:
        current = np.clip(curr_iter, 0.0, rampup_iters)
        phase = 1.0 - current / rampup_iters
        return float(np.exp(-5.0 * phase * phase))


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


def load_checkpoint(model, checkpoint_path):
    print(f'Loading {checkpoint_path}')
    print()
    with open(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f)

    pretrained_state_dict = checkpoint['state_dict']
    pretrained_state_dict.pop('current_learning_rate', None)
    pretrained_state_dict.pop('best_validation_top1_accuracy', None)
    
    current_state_dict = model.state_dict()
    if 'fc_unlabeled.weight' in pretrained_state_dict:
        if 'fc_unlabeled.weight' not in current_state_dict or (
                pretrained_state_dict['fc_unlabeled.weight'].shape != current_state_dict['fc_unlabeled.weight'].shape):
            pretrained_state_dict.pop('fc_unlabeled.weight')
            pretrained_state_dict.pop('fc_unlabeled.bias')
    
    if 'fc_labeled.weight' in pretrained_state_dict:
        if 'fc_labeled.weight' not in current_state_dict or (
                pretrained_state_dict['fc_labeled.weight'].shape != current_state_dict['fc_labeled.weight'].shape):
            pretrained_state_dict.pop('fc_labeled.weight')
            pretrained_state_dict.pop('fc_labeled.bias')

    incompatible_keys = model.load_state_dict(pretrained_state_dict, strict=False)
    if incompatible_keys.missing_keys:
        print('missing keys:')
        print('---')
        print('\n'.join(incompatible_keys.missing_keys))
        print()
    
    if incompatible_keys.unexpected_keys:
        print('unexpected keys:')
        print('---')
        print('\n'.join(incompatible_keys.unexpected_keys))
        print()