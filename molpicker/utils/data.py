# coding=utf-8
from typing import List, Any, Mapping, Tuple, Optional, Union

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor


def load_coco_detection_dataset(
        root: str,
        ann_file: str,
) -> CocoDetection:
    ds = CocoDetection(
        root,
        ann_file,
        transforms=TransformCocoDetectionToGeneralizedRCNNInput()
    )
    return ds


class TransformCocoDetectionToGeneralizedRCNNInput:
    """Class for transformation from CocoDetection dataset to Generalized R-CNN
        module inputs
    """

    def __init__(self):
        self.to_tensor = ToTensor()

    def __call__(self,
                 image: Image,
                 annotations: Optional[List[Mapping[str, Any]]] = None
                 ) -> Tuple[Tensor, Optional[Mapping[str, Tensor]]]:
        image_tensor = self.to_tensor(image)
        if annotations is None or len(annotations) == 0:
            return image_tensor, None

        target = {
            'image_id': annotations[0]['image_id'],
            'boxes': [],
            'area': [],
            'labels': [],
            'masks': [],
        }
        for ann in annotations:
            assert ann['image_id'] == target['image_id']

            x0, y0, width, height = ann['bbox']
            target['boxes'].append([x0, y0, x0 + width, y0 + height])

            area = ann.get('area', None)
            if area is None:
                target['area'].append(width * height)
            else:
                target['area'].append(area)
            target['labels'].append(ann['category_id'])

            seg = ann.get('segmentation', None)
            if seg is not None:
                mask = self.get_binary_mask(seg, image.height, image.width)
                target['masks'].append(mask)

        assert len(target['masks']) == len(annotations)

        if len(target['masks']) == 0:
            del target['masks']

        target['image_id'] = torch.tensor(target['image_id'], dtype=torch.int64)
        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float)
        target['area'] = torch.tensor(target['area'], dtype=torch.float)
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        target['iscrowd'] = torch.zeros(len(annotations), dtype=torch.uint8)

        if 'masks' in target:
            target['masks'] = torch.tensor(target['masks'],
                                           dtype=torch.uint8).squeeze(-1)

        return image_tensor, target

    @staticmethod
    def get_binary_mask(seg: Union[List, Mapping],
                        height: int,
                        width: int) -> np.ndarray:
        if type(seg) == list:
            # polygon
            encoded_rles = mask_utils.frPyObjects(
                seg, height, width
            )
            # a single object might consist of multiple parts
            encoded_rle = mask_utils.merge(encoded_rles)
        else:
            # mask
            if type(seg['counts']) == list:
                # Change uncompressed rle to compressed rle
                encoded_rle = mask_utils.frPyObjects([seg], height, width)
            else:
                encoded_rle = [seg]
        mask = mask_utils.decode(encoded_rle)
        return mask


class CollateCocoDetectionOutput:

    def __init__(self):
        ...

    def __call__(
            self,
            batch_data: List[Tuple[Tensor, Optional[Mapping[str, Tensor]]]]
    ) -> Tuple[List[Tensor], Optional[Mapping[str, Tensor]]]:
        image_list = []
        target_list = []

        for image, target in batch_data:
            image_list.append(image)
            if target is not None:
                target_list.append(target)

        assert len(target_list) == 0 or len(image_list) == len(target_list)
        if len(target_list) == 0:
            target_list = None
        return image_list, target_list
