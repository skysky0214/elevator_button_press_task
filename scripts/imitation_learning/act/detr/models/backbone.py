# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
DINOv2 backbone added alongside original ResNet for richer spatial features.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed


class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class DINOv2Backbone(nn.Module):
    """DINOv2 vision transformer backbone.

    Produces a dense (B, C, H/14, W/14) feature map from the final transformer
    layer, dropping the [CLS] token.  Interface mirrors ResNet Backbone:
    forward(tensor) -> dict {"0": feature_map}
    """

    # Common IMAGENET normalization constants (DINOv2 was pretrained with these)
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    NAME_TO_HF = {
        "dinov2_small": "facebook/dinov2-small",
        "dinov2_base": "facebook/dinov2-base",
        "dinov2_large": "facebook/dinov2-large",
    }

    def __init__(self, name: str = "dinov2_small", train_backbone: bool = False):
        super().__init__()
        from transformers import Dinov2Model
        hf_name = self.NAME_TO_HF.get(name, name)
        self.model = Dinov2Model.from_pretrained(hf_name)
        self.patch_size = self.model.config.patch_size  # 14
        self.num_channels = self.model.config.hidden_size  # 384 (small), 768 (base), 1024 (large)
        self.train_backbone = train_backbone
        if not train_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = self.MEAN.to(tensor.device).to(tensor.dtype)
        std = self.STD.to(tensor.device).to(tensor.dtype)
        return (tensor - mean) / std

    def forward(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        # tensor: (B, 3, H, W) already in [0,1] range
        B, C, H, W = tensor.shape
        # Resize to multiples of patch_size (14)
        H_new = (H // self.patch_size) * self.patch_size
        W_new = (W // self.patch_size) * self.patch_size
        if H_new != H or W_new != W:
            tensor = F.interpolate(tensor, size=(H_new, W_new), mode="bilinear", align_corners=False)
        x = self._normalize(tensor)
        ctx = torch.enable_grad() if self.train_backbone else torch.no_grad()
        with ctx:
            out = self.model(pixel_values=x)
        hidden = out.last_hidden_state          # (B, 1 + n_patches, D)
        patches = hidden[:, 1:, :]               # drop CLS
        h_p = H_new // self.patch_size
        w_p = W_new // self.patch_size
        features = patches.transpose(1, 2).reshape(B, self.num_channels, h_p, w_p)
        return {"0": features}


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    name = args.backbone
    if name.startswith("dinov2"):
        backbone = DINOv2Backbone(name=name, train_backbone=train_backbone)
    else:
        backbone = Backbone(name, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
