# coding=utf-8
from abc import ABCMeta
from collections import OrderedDict
from typing import List, Mapping, Tuple, Any, Optional

import torch.optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch import nn
from torchvision.models.detection.image_list import ImageList


class GeneralizedRCNN(LightningModule, metaclass=ABCMeta):
    """Modified Generalized R-CNN from torchvision to use with pytorch lightning"""

    def __init__(self,
                 backbone: nn.Module,
                 rpn: nn.Module,
                 roi_heads: nn.Module,
                 transform: nn.Module,
                 lr: float):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        self.lr = lr

    def forward(self, images: List[Tensor], **kwargs) -> Mapping[str, Any]:
        """
        Arguments:
             images (List[Tensor]): image list (image shape should be [N, C, H, W])

        Returns:
            result (Mapping[str, Any]): dictionary that contains the output of each modules.
        """
        original_image_sizes: List[Tuple[int, int]] = []
        for image in images:
            image_resolution = image.shape[-2:]
            assert len(image_resolution) == 2
            original_image_sizes.append(
                (image_resolution[0], image_resolution[1]))

        # ImageList contains batched images as one tensor
        images: ImageList = self.transform(images)

        features = self.backbone(images.tensors)
        if isinstance(features, Tensor):
            features = OrderedDict([('0', features)])

        proposals, proposal_logits = self.rpn(images, features)

        output_dict = {
            'original_image_sizes': original_image_sizes
        }

        # TODO: Return logits
        return output_dict

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        ...

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        ...

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        ...
