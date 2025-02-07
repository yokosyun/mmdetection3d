# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig


@MODELS.register_module()
class SECOND(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels: int = 128,
                 out_channels: Sequence[int] = [128, 128, 256],
                 layer_nums: Sequence[int] = [3, 5, 5],
                 layer_strides: Sequence[int] = [2, 2, 2],
                 norm_cfg: ConfigType = dict(
                     type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg: ConfigType = dict(type='Conv2d', bias=False),
                 init_cfg: OptMultiConfig = None,
                 pretrained: Optional[str] = None) -> None:
        super(SECOND, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []

        # print(torch.max(x))
        # x[:,28,:,:] /= 10.0
        # x[:,62,:,:] /= 10.0
        # x[:,12,:,:] /= 10.0
        # x[:,19,:,:] /= 10.0
        # x[:,6,:,:] /= 10.0

        # if True:
        #     vis(x)

        # for i in range(len(self.blocks)):
        #     x = self.blocks[i](x)
        #     outs.append(x)
        # return tuple(outs)
        # high_vals = [
        #     [28, 62, 12, 19,  6], # [28, 12, 19, 62, 21]
        #     [44, 24, 51, 19, 4], # [35,  5, 19, 24, 43] [ 5, 19, 24, 48, 44]
        #     [12, 50, 11, 26, 41],# [12, 50, 11, 26, 15] [12, 50, 15, 11,  9]
        #     [3, 10, 28, 15, 33], # [ 3,  4, 28,  0, 10] [ 3, 28,  0, 17,  4]
        # ]

        for i in range(len(self.blocks)):
            for layer_idx, block in enumerate(self.blocks[i]):
                if type(block).__name__ == 'QuantConv2d' and i == 0:
                    vis(x, str(i) + '.' + str(layer_idx))
                x = block(x)
            outs.append(x)
        return tuple(outs)


def vis(x, layer_name):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    N, C, H, W = x.shape
    x_c_hw = x.reshape(C, H * W)
    mask = x_c_hw.abs().sum(0) != 0.0
    x_c_hw = x_c_hw[:, mask]

    if True:
        max_vals = torch.max(x_c_hw, dim=1).values
        topk = torch.topk(max_vals, 32)
        # print(topk)
        # indices = [28, 62, 12, 19,  6]
        indices = topk.indices
        # print(indices)
        # indices = [35, 58,  1, 20, 25, 50, 38, 14, 41, 27, 11,  0] # xy
        # indices = [32, 60,  2, 57, 63, 17, 21, 36, 16, 29, 61,  5] # center_x, center_y
        # indices = [60, 36, 16,  5, 29,  9, 31, 15, 28,  7, 58, 2] # time
        indices = [60, 56, 26, 24, 19, 5, 32, 15, 12, 61, 59, 57]
        x_c_hw = x_c_hw[indices]
    if True:
        max_vals = torch.max(x_c_hw, dim=0).values
        sorted_vals, sorted_idx = torch.sort(max_vals, descending=True)
        x_c_hw = x_c_hw[:, sorted_idx]
        x_c_hw = x_c_hw[:, ::20]

    C, HW = x_c_hw.shape

    X, Y = np.meshgrid(np.arange(HW), np.arange(C))
    z = np.abs(x_c_hw.cpu().numpy()).flatten()
    cmap = plt.cm.get_cmap('coolwarm')

    colors = cmap(z / np.max(z))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(z), 1, 1, z, color=colors)
    ax.set_xlabel('BEV Grids')
    ax.set_ylabel('Input Channel')
    ax.set_zlabel('Absolute Input Activation Value')
    ax.set_title(layer_name)
    plt.show()
