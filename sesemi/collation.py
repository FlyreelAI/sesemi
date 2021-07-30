#
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
# ========================================================================#
"""PyTorch collation callables."""
import torch

from torch import Tensor
from typing import List, Tuple


class RotationTransformer:
    """A collation callable to transform a data batch for use with a rotation prediction task.

    Input images are rotated 0, 90, 180, and 270 degrees.
    """

    def __call__(self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Generates a transformed data batch of rotation prediction data.

        Arguments:
            batch: A tuple containing a batch of input images and associated labels.

        Returns:
            A tuple of rotated images and their associated rotation prediction labels where:

                0  corresponds to 0 degrees.
                1  corresponds to 90 degrees.
                2  corresponds to 180 degrees.
                3  corresponds to 270 degrees.
        """
        tensors, labels = [], []
        for tensor, _ in batch:
            for k in range(4):
                if k == 0:
                    t = tensor
                else:
                    t = torch.rot90(tensor, k, dims=[1, 2])
                tensors.append(t)
                labels.append(torch.LongTensor([k]))
        x = torch.stack(tensors, dim=0)
        y = torch.cat(labels, dim=0)  # type: ignore
        return (x, y)
