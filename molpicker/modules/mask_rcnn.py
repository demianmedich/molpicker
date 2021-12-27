# coding=utf-8
import copy
from typing import Any, Optional, Tuple

from dataclasses import dataclass
from dataclasses import field
import torch
from pytorch_lightning import LightningModule
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


@dataclass
class MaskRcnnModelConfig:
    """Mask R-CNN Model configuration

    Attributes:
        backbone_name (str): the network used to compute the features for the model.
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
    """
    backbone_name: str
    num_classes: int
    min_size: int = 800
    max_size: int = 1333
    image_mean: Tuple[float, float, float] = None
    image_std: Tuple[float, float, float] = None
    rpn_pre_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_test: int = 1000
    rpn_nms_thresh: float = 0.7
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5
    box_score_thresh = 0.05
    box_nms_thresh = 0.5
    box_detections_per_img = 100
    box_fg_iou_thresh = 0.5
    box_bg_iou_thresh = 0.5
    box_batch_size_per_image = 512
    box_positive_fraction = 0.25
    bbox_reg_weights: Tuple[float, float, float, float] = None

    @property
    def optional_arguments(self):
        return {k: v for k, v in self.__dict__.items() if
                k not in ('backbone_name', 'num_classes')}


@dataclass
class MaskRcnnOptimizerConfig:
    lr: float = 0.005
    # momentum:


@dataclass
class MaskRcnnLightningModuleConfig:
    model_config: MaskRcnnModelConfig


class MaskRcnnLightningModule(LightningModule):

    def __init__(
            self,
            cfg: MaskRcnnLightningModuleConfig,
            **kwargs: Any) -> None:
        super().__init__()
        self.cfg = copy.deepcopy(cfg)

        model_config = self.cfg.model_config

        self.mask_rcnn = maskrcnn_resnet_fpn(
            model_config.backbone_name,
            model_config.num_classes,
            kwargs=model_config.optional_arguments
        )


def maskrcnn_resnet_fpn(
        backbone_name: str,
        num_classes: int,
        pretrained_ckpt: Optional[str] = None,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        **kwargs
):
    """

    Arguments:
        backbone_name (str): The name of backbone. One of ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        num_classes (int): number of output classes of the model (including the background)
        pretrained_ckpt (str or None): If exists, returns a model pre-trained on COCO train2017
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.

    Returns:
        Mask R-CNN model
    """
    assert 0 <= trainable_backbone_layers <= 5
    assert backbone_name in (
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')

    if not (pretrained_ckpt or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained_ckpt:
        # No need to download backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone(backbone_name,
                                   pretrained_backbone,
                                   trainable_layers=trainable_backbone_layers)
    model = MaskRCNN(backbone, num_classes, **kwargs)
    if pretrained_ckpt:
        sd = torch.load(pretrained_ckpt)
        model.load_state_dict(sd)
    return model
