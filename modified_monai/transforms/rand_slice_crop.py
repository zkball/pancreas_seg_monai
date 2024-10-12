# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A collection of "vanilla" transforms for crop and pad operations.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from itertools import chain
from math import ceil
from typing import Any

import numpy as np
import torch

from monai.config import IndexSelection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms.croppad.functional import crop_func, pad_func
from monai.transforms.inverse import InvertibleTransform, TraceableTransform
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import LazyTransform, Randomizable, Transform
from monai.transforms.utils import (
    compute_divisible_spatial_size,
    generate_label_classes_crop_centers,
    generate_pos_neg_label_crop_centers,
    generate_spatial_bounding_box,
    is_positive,
    map_binary_to_indices,
    map_classes_to_indices,
    weighted_patch_samples,
)
from monai.utils import ImageMetaKey as Key
from monai.utils import (
    LazyAttr,
    Method,
    PytorchPadMode,
    TraceKeys,
    TransformBackends,
    convert_data_type,
    convert_to_tensor,
    deprecated_arg_default,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    pytorch_after,
)

from monai.transforms.croppad.array import (
    Crop, CenterSpatialCrop
)

import os

class RandSpatialCrop(Randomizable, Crop):
    """
    Crop image with random size or specific size ROI. It can crop at a random position as center
    or at the image center. And allows to set the minimum and maximum size to limit the randomly generated ROI.

    Note: even `random_size=False`, if a dimension of the expected ROI size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected ROI, and the cropped results
    of several images may not have exactly the same shape.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        max_roi_size: if `random_size` is True and `roi_size` specifies the min crop region size, `max_roi_size`
            can specify the max crop region size. if None, defaults to the input image size.
            if its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            if True, the actual size is sampled from `randint(roi_size, max_roi_size + 1)`.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    def __init__(
        self,
        roi_size: Sequence[int] | int,
        slice_dict: dict,
        # max_roi_size: Sequence[int] | int | None = None,
        # random_center: bool = True,
        # random_size: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self.roi_size = roi_size
        self.slice_dict = slice_dict

    # def randomize(self, img_size: Sequence[int]) -> None:
    #     self._size = fall_back_tuple(self.roi_size, img_size)
    #     if self.random_size:
    #         max_size = img_size if self.max_roi_size is None else fall_back_tuple(self.max_roi_size, img_size)
    #         if any(i > j for i, j in zip(self._size, max_size)):
    #             raise ValueError(f"min ROI size: {self._size} is larger than max ROI size: {max_size}.")
    #         self._size = tuple(self.R.randint(low=self._size[i], high=max_size[i] + 1) for i in range(len(img_size)))
    #     if self.random_center:
    #         valid_size = get_valid_patch_size(img_size, self._size)
    #         self._slices = get_random_patch(img_size, valid_size, self.R)

    def __call__(self, img: torch.Tensor, randomize: bool = True, lazy: bool | None = None) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:] ## in [B, H, W, S]

        ## [ADD] added in image_dataset
        num_slices = self.roi_size[-1]
        idx_info = self.slice_dict[os.path.basename(img.meta["filename_or_obj"])]
        idx_start, idx_end, idx_center, interval, idx_last = idx_info["start"], idx_info["end"], (idx_info["end"]+idx_info["start"])//2, idx_info["interval"], img_size[-1]

        idx_lack = num_slices - interval
        if idx_lack>0: ## we lack valid slices
            rnd_begin = max(0, idx_start-np.random.randint(0,idx_lack))
        elif idx_lack<0: ## there are abundant frames
            rnd_begin = idx_start+np.random.randint(0,-idx_lack)
        else:
            rnd_begin = idx_start
            
        img = img[..., rnd_begin:rnd_begin+num_slices]

        return img

        # if randomize:
        #     self.randomize(img_size)
        # if self._size is None:
        #     raise RuntimeError("self._size not specified.")
        # lazy_ = self.lazy if lazy is None else lazy
        # if self.random_center:
        #     return super().__call__(img=img, slices=self._slices, lazy=lazy_)
        # cropper = CenterSpatialCrop(self._size, lazy=lazy_)
        # return super().__call__(img=img, slices=cropper.compute_slices(img_size), lazy=lazy_)
