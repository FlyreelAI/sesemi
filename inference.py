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
import logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)

import torch
import torch.nn.functional as F
from torchvision import datasets

from models import SESEMI
from utils import load_model
from dataset import center_crop_transforms, multi_crop_transforms


parser = argparse.ArgumentParser(description='Perform inference on test data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Run arguments
parser.add_argument('--checkpoint-path', default='',
                    help='path to saved checkpoint')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable cuda')
# Data loading arguments
parser.add_argument('--data-dir', default='',
                    help='path to test dataset with one or more subdirs containing images')
parser.add_argument('--batch-size', default=16, type=int,
                    help='mini-batch size')
parser.add_argument('--workers', default=6, type=int,
                    help='number of data loading workers')
# Inference arguments
parser.add_argument('--oversample', action='store_true',
                    help='enable test-time augmentation')
parser.add_argument('--ncrops', default=5, type=int,
                    help='number of crops to oversample')
parser.add_argument('--topk', default=1, type=int,
                    help='return topk predictions')
parser.add_argument('--resize', default=256, type=int,
                    help='resize smaller edge to this resolution while maintaining aspect ratio')
parser.add_argument('--crop-dim', default=224, type=int,
                    help='dimension for center or multi cropping')
parser.add_argument('--outfile', default='inference_results.csv',
                    help='write prediction results to file')


class Classifier():
    def __init__(self, model_path, args):
        self.args = args
        self.model_path = model_path
        self.device = torch.device(
            'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda'
        )
        self._init_model()

    def _init_model(self):
        self.model = load_model(SESEMI, self.model_path, self.device)
        logging.info(f'=> Model checkpoint loaded from {self.model_path}')
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.classes = np.array(self.model.module.CLASSES)
        self.model.eval()

    def predict(self, x, ncrops, topk=1):
        with torch.no_grad():
            x = x.to(self.device)
            batch_size = x.size(0)
            w, h, c = x.shape[-1:-4:-1]
            outputs = self.model(x.view(-1, c, h, w)) # fuse batch size and ncrops
            outputs = outputs.view(batch_size, ncrops, -1).mean(1) # avg over crops
            outputs = F.softmax(outputs, dim=1)
            scores, indices = torch.topk(outputs, k=topk, largest=True, sorted=True)
            scores = scores.cpu().numpy()
            indices = indices.cpu().numpy()
            labels = self.classes[indices]
            return (labels, scores)


def predict():
    args = parser.parse_args()
    classifier = Classifier(args.checkpoint_path, args)
    # Data loading
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), args.data_dir
        )
    if args.oversample:
        ncrops = args.ncrops
        test_transformations = multi_crop_transforms(
            args.resize, args.crop_dim, ncrops
        )
    else:
        ncrops = 1
        test_transformations = center_crop_transforms(
            args.resize, args.crop_dim
        )
    dataset = datasets.ImageFolder(args.data_dir, test_transformations)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False)
    
    # Write prediction results to file
    if os.path.exists(args.outfile):
        os.remove(args.outfile)
    with open(args.outfile, 'a') as f:
        header = ','.join(['Id', 'Category', 'Score'])
        f.write(header + '\n')
    index = 0

    dataset_iterator = iter(dataset_loader)
    for _ in trange(len(dataset_loader), desc=f'Inferencing on {len(dataset.imgs)} files', position=1):
        inputs, _ = next(dataset_iterator)
        labels, scores = classifier.predict(inputs, ncrops, args.topk)
        # Write prediction results to file
        with open(args.outfile, 'a') as f:
            for label, score in zip(labels, scores):
                img_path = dataset.imgs[index][0]
                img_id = os.path.splitext(os.path.basename(img_path))[0]
                label = ' '.join(label)
                score = [f'{s:.6f}' for s in score]
                score = ' '.join(score)
                f.write(','.join([img_id, label, score]) + '\n')
                index += 1


if __name__ == '__main__':
    predict()

