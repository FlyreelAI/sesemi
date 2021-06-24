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
import torch.nn as nn
from .timm import PyTorchImageModels
from utils import sigmoid_rampup, adjust_polynomial_lr
import logging

import torch.nn.functional as F

import pytorch_lightning as pl

from torchmetrics.classification.accuracy import Accuracy


SUPPORTED_BACKBONES = (
    # The following backbones strike a balance between accuracy and model size, with optional
    # pretrained ImageNet weights. For a summary of their ImageNet performance, see
    # <https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv>.

    # Compared with the defaults, the "d" variants (e.g., resnet50d, resnest50d)
    # replace the 7x7 conv in the input stem with three 3x3 convs.
    # And in the downsampling block, a 2x2 avg_pool with stride 2 is added before conv,
    # whose stride is changed to 1. Described in `Bag of Tricks <https://arxiv.org/abs/1812.01187>`.
    
    # ResNet models.
    'resnet18', 'resnet18d', 'resnet34', 'resnet34d', 'resnet50', 'resnet50d', 'resnet101d', 'resnet152d',
    
    # ResNeXt models.
    'resnext50_32x4d', 'resnext50d_32x4d', 'resnext101_32x8d',
    
    # Squeeze and Excite models.
    'seresnet50', 'seresnet152d', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext50_32x4d',
    
    # ResNeSt models.
    'resnest14d', 'resnest26d', 'resnest50d', 'resnest101e', 'resnest200e', 'resnest269e',
    'resnest50d_1s4x24d', 'resnest50d_4s2x40d',
    
    # ResNet-RS models.
    'resnetrs50', 'resnetrs101', 'resnetrs152', 'resnetrs200',
    
    # DenseNet models.
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    
    # Inception models.
    'inception_v3', 'inception_v4', 'inception_resnet_v2',
    
    # Xception models.
    'xception', 'xception41', 'xception65', 'xception71',
    
    # EfficientNet models.
    'tf_efficientnet_b0', 'tf_efficientnet_b1', 'tf_efficientnet_b2',
    'tf_efficientnet_b3', 'tf_efficientnet_b4', 'tf_efficientnet_b5',
    'tf_efficientnet_b6', 'tf_efficientnet_b7',

    # EfficientNet models trained with noisy student.
    'tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2_ns',
    'tf_efficientnet_b3_ns', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns',
    'tf_efficientnet_b6_ns', 'tf_efficientnet_b7_ns',
)


class SESEMI(pl.LightningModule):
    def __init__(self, hparams):
        super(SESEMI, self).__init__()
        self.save_hyperparameters(hparams)

        assert self.hparams.backbone in SUPPORTED_BACKBONES, f'--backbone must be one of {SUPPORTED_BACKBONES}'
        
        if not self.hparams.global_pool:
            # If no global pooling method specified, fall back to "avg"
            self.hparams.global_pool = 'avg'

        self.feature_extractor = PyTorchImageModels(self.hparams.backbone, self.hparams.pretrained, self.hparams.global_pool)

        if self.hparams.freeze_backbone:
            logging.info(f'Freezing {self.hparams.backbone} backbone')
            for m in self.feature_extractor.modules():
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        if self.hparams.pretrained:
            logging.info(f'Initialized with pretrained {self.hparams.backbone} backbone')

        self.in_features = self.feature_extractor.in_features
        self.dropout = nn.Dropout(self.hparams.dropout_rate)
        self.fc_labeled = nn.Linear(self.in_features, self.hparams.num_labeled_classes)
        self.fc_unlabeled = nn.Linear(self.in_features, self.hparams.num_unlabeled_classes)
        self.register_buffer(
            'current_learning_rate',
            torch.tensor(self.hparams.warmup_lr, dtype=torch.float32, device=self.device))
        self.register_buffer(
            'best_validation_top1_accuracy',
            torch.tensor(0., dtype=torch.float32, device=self.device))

        self.validation_top1_accuracy = Accuracy(top_k=1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.fc_labeled(features)
        return torch.softmax(logits, dim=-1)

    def forward_train(self, x_labeled, x_unlabeled=None):
        # Compute output for labeled input
        x_labeled = self.feature_extractor(x_labeled)
        if self.hparams.dropout_rate > 0.0:
            x_labeled = self.dropout(x_labeled)
        output_labeled = self.fc_labeled(x_labeled)
        
        if x_unlabeled is not None:
            # Compute output for unlabeled input and return both outputs
            x_unlabeled = self.feature_extractor(x_unlabeled)
            output_unlabeled = self.fc_unlabeled(x_unlabeled)
            return output_labeled, output_unlabeled

        return output_labeled, None

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        **kwargs,
    ):
        optimizer.step(closure=optimizer_closure)
        self.current_learning_rate = torch.tensor(adjust_polynomial_lr(
                optimizer.optimizer, self.global_step,
                warmup_iters=self.hparams.warmup_iters,
                warmup_lr=self.hparams.warmup_lr,
                lr=self.hparams.lr,
                lr_pow=self.hparams.lr_pow,
                max_iters=self.hparams.max_iters),
                dtype=self.current_learning_rate.dtype,
                device=self.current_learning_rate.device)

    def configure_optimizers(self):
        if self.hparams.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.hparams.lr, momentum=self.hparams.momentum, nesterov=True,
                weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.hparams.lr, betas=(self.hparams.momentum, 0.999), weight_decay=0.0)
        else:
            raise NotImplementedError()    
        
        return optimizer

    def training_step(self, batch, batch_index):
        inputs_t, targets_t = batch['supervised']
        inputs_u, targets_u = batch.get('unsupervised_rotation', (None, None))

        message = {}
        
        # Forward pass
        outputs_t, outputs_u = self.forward_train(inputs_t, inputs_u)
        
        loss_weight = self.hparams.initial_loss_weight * sigmoid_rampup(
            self.global_step, self.hparams.stop_rampup)
        
        loss_t = F.cross_entropy(outputs_t, targets_t, reduction='mean')
        
        if outputs_u is not None:
            loss_u = F.cross_entropy(outputs_u, targets_u, reduction='mean')
        else:
            loss_u = 0.

        loss = loss_t + loss_u * loss_weight

        self.log('train/loss_labeled', loss_t)
        self.log('train/loss_unlabeled', loss_u)
        self.log('train/loss_unlabeled_weight', loss_weight)
        self.log('train/loss', loss)
        self.log('train/learning_rate', self.current_learning_rate)

        message['rank'] = self.global_rank
        message['epoch'] = self.trainer.current_epoch
        message['step'] = self.global_step
        message['loss'] = float(loss)
        message['loss_labeled'] = float(loss_t)
        message['loss_unlabeled'] = float(loss_u)
        message['loss_weight'] = float(loss_weight)

        if self.global_rank == 0:
            logging.info(str(message))

        return loss

    def validation_step(self, batch, batch_index):
        inputs_t, targets_t = batch
        outputs_t = self.forward(inputs_t)
        return outputs_t, targets_t
        
    def validation_step_end(self, outputs):
        outputs_t, targets_t = outputs
        self.validation_top1_accuracy.update(outputs_t, targets_t)

    def validation_epoch_end(self, outputs):
        top1 = self.validation_top1_accuracy.compute()
        self.validation_top1_accuracy.reset()

        if top1 > self.best_validation_top1_accuracy:
            self.best_validation_top1_accuracy =  torch.tensor(
                float(top1),
                dtype=self.best_validation_top1_accuracy.dtype,
                device=self.best_validation_top1_accuracy.device)

        self.log('val/top1', top1)
        logging.info('Epoch {:03d} =====> Validation Accuracy {:.4f} [Best {:.4f}]'.format(
            self.trainer.current_epoch,
            top1,
            self.best_validation_top1_accuracy))
