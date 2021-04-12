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
import torch
from torchvision import datasets, transforms

from utils import GammaCorrection


channel_mean = [0.485, 0.456, 0.406] 
channel_std  = [0.229, 0.224, 0.225]

def train_transforms(random_resized_crop=True, resize=256, crop_dim=224, scale=(0.2, 1.0),
                     gamma_range=(0.5, 1.5), p_hflip=0.5, norms=(channel_mean, channel_std), p_erase=0.5):
    default_transforms = [
        GammaCorrection(gamma_range),
        transforms.RandomHorizontalFlip(p_hflip),
        transforms.ToTensor(),
        transforms.Normalize(*norms),
        transforms.RandomErasing(p=p_erase, value='random')
    ]
    if random_resized_crop:
        return transforms.Compose(
            [transforms.RandomResizedCrop(crop_dim, scale=scale)] + default_transforms
        )
    else:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(crop_dim)
        ] + default_transforms)


def center_crop_transforms(resize=256, crop_dim=224, norms=(channel_mean, channel_std)):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop_dim),
        transforms.ToTensor(),
        transforms.Normalize(*norms)
    ])


def multi_crop_transforms(resize=256, crop_dim=224, num_crop=5,
                          norms=(channel_mean, channel_std)):
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(*norms)
    Lambda = transforms.Lambda
    if num_crop == 5:
        multi_crop = transforms.FiveCrop
    elif num_crop == 10:
        multi_crop = transforms.TenCrop
    else:
        raise NotImplementedError('Number of crops should be integer of 5 or 10')
    return transforms.Compose([
        transforms.Resize(resize),
        multi_crop(crop_dim), # this is a list of PIL Images
        Lambda(lambda crops: torch.stack([to_tensor(crop) for crop in crops])),
        Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
    ])


class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transformations):
        # `img_dir` must have one or more subdirs containing images
        self.dataset = datasets.ImageFolder(img_dir, transformations)

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        x, subdir_index = self.dataset[index]
        return (x, subdir_index)


class RotationTransformer():
    def __init__(self):
        self.rotation_labels = 4

    def __call__(self, batch):
        tensors, labels = [], []
        for tensor, _ in batch:
            for k in range(self.rotation_labels):
                if k == 0:
                    t = tensor
                else:
                    t = torch.rot90(tensor, k, dims=[1, 2])
                tensors.append(t)
                labels.append(torch.LongTensor([k]))
        x = torch.stack(tensors, dim=0)
        y = torch.cat(labels, dim=0)
        return (x, y)


if __name__ == '__main__':
    import argparse
    from tqdm import trange

    parser = argparse.ArgumentParser(description='Dataset Visualization',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img-dir', required=True,
                        help='`img_dir` must have one or more subdirs containing images')
    parser.add_argument('--head', default=10, type=int,
                        help='visualize first k images')
    parser.add_argument('--hflip', action='store_true',
                        help='apply random horizontal flip')
    parser.add_argument('--erase', action='store_true',
                        help='apply random erase')
    parser.add_argument('--gamma', action='store_true',
                        help='apply random luminance and gamma correction')
    parser.add_argument('--normalize', action='store_true',
                        help='apply channel-wise mean-std normalization')
    parser.add_argument('--visualize-rotations', action='store_true',
                        help='visualize rotation transformations')
    parser.add_argument('--out-dir', default='./sample_dataset_vis',
                        help='directory to save images for visualization')
    args = parser.parse_args()

    if not os.path.exists(args.img_dir):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), args.img_dir
        )
    os.makedirs(args.out_dir, exist_ok=True)

    print('args:')
    for key, val in args.__dict__.items():
        print('    {:20} {}'.format(key, val))
    
    p_hflip = 0.5 if args.hflip else 0.0
    p_erase = 0.5 if args.erase else 0.0
    gamma_range = (0.5, 1.5) if args.gamma else (1.0, 1.0)
    (mean, std) = (channel_mean, channel_std) \
        if args.normalize else ([0., 0., 0.], [1., 1., 1.])
    
    transformations = train_transforms(
        gamma_range=gamma_range, p_hflip=p_hflip, norms=(mean, std), p_erase=p_erase
    )
    dataset = datasets.ImageFolder(args.img_dir, transformations)
    print('transforms:\n', transformations)
    print('len(dataset): {}'.format(len(dataset)))
    to_pil_image = transforms.ToPILImage()
    rotate = RotationTransformer()
    for i in trange(args.head):
        fpath = dataset.imgs[i][0]
        fname = fpath.split('/')[-1]
        x, dummy = dataset[i]
        if args.visualize_rotations:
            tensors, indices = rotate([(x, dummy)])
            for x, ind in zip(*(tensors, indices)):
                image = to_pil_image(x)
                image.save(os.path.join(args.out_dir, f'rotated_{ind}_' + fname))
        else:
            image = to_pil_image(x)
            image.save(os.path.join(args.out_dir, fname))

