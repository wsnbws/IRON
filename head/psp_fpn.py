# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet."""

    def __init__(self, pool_scales, in_channels, channels, align_corners=False,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), 
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        in_channels, 
                        channels, 
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.
    
    Args:
        in_channels_list (list[int]): 输入特征的通道数列表，如[512, 1024, 2048]（3层）或[256, 512, 1024, 2048]（4层）
        channels (int): 输出通道数，默认512
        pool_scales (tuple[int]): PSP模块的池化尺度，默认(1, 2, 3, 6)
        align_corners (bool): 插值时是否对齐角点，默认False
        conv_cfg (dict|None): 卷积层配置
        norm_cfg (dict): 归一化层配置
        act_cfg (dict): 激活层配置
    
    输入:
        inputs (list[Tensor]): 多尺度特征列表 [p1, p2, p3] 或 [p1, p2, p3, p4]
            - 3层示例:
                - p1: [B, C1, H/8, W/8]
                - p2: [B, C2, H/16, W/16]
                - p3: [B, C3, H/32, W/32]
            - 4层示例:
                - p1: [B, C1, H/4, W/4]
                - p2: [B, C2, H/8, W/8]
                - p3: [B, C3, H/16, W/16]
                - p4: [B, C4, H/32, W/32]
    
    输出:
        list[Tensor]: PSP+FPN处理后的多尺度特征列表
    """

    def __init__(self, 
                 in_channels_list, 
                 channels=512, 
                 pool_scales=(1, 2, 3, 6), 
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.channels = channels
        self.align_corners = align_corners
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        
        # PSP Module (应用在最后一层特征上)
        self.psp_modules = PPM(
            pool_scales,
            in_channels_list[-1],
            channels,
            align_corners=align_corners,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        self.bottleneck = ConvModule(
            in_channels_list[-1] + len(pool_scales) * channels,
            channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        # FPN Module - Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list[:-1]:  # 跳过最后一层
            l_conv = ConvModule(
                in_channels, 
                channels, 
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                channels, 
                channels, 
                3, 
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def psp_forward(self, x):
        """PSP模块前向传播"""
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): 多尺度特征列表 [p1, p2, p3, p4]
        
        Returns:
            list[Tensor]: PSP+FPN处理后的特征列表，每个特征通道数都是self.channels
        """
        assert len(inputs) == len(self.in_channels_list), \
            f"输入特征数量({len(inputs)})与配置不匹配({len(self.in_channels_list)})"
        
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        laterals.append(self.psp_forward(inputs[-1]))
        
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        fpn_outs.append(laterals[-1])
        
        return fpn_outs
