import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
from typing import Optional, Tuple

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from .psp_fpn import UPerHead
from typing import List, Optional, Tuple, Type
import torch
from torch import nn
import torch.nn.functional as F
from head.untils import LayerNorm2d, MLP
from .loss import otdr_loss
from .sam.transformer import TwoWayTransformer

class CoarseGuidance(nn.Module):

    def __init__(self, in_channels: int, embed_channels: int):
        super().__init__()
        self.mask_mlp = nn.Sequential(
            nn.Conv2d(1, embed_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.fuser = nn.Sequential(
            nn.Conv2d(embed_channels, embed_channels, kernel_size=1),
            nn.SyncBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=1),
            nn.SyncBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, fine_feat: torch.Tensor, coarse_mask: torch.Tensor):

        mask_up = F.interpolate(coarse_mask, scale_factor=2, mode='nearest')
        mask_embed = self.mask_mlp(mask_up)
        fused_feat = self.fuser(fine_feat + mask_embed)  
        return fused_feat

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        use_high_res_features: bool = False,
    ) -> None:
        """
        Simplified mask decoder for point-prompt based segmentation.
        Only outputs a single mask.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          use_high_res_features (bool): whether to use high-res features
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        # Single mask output only
        self.coarse_fine_token = nn.ModuleList([nn.Embedding(1, transformer_dim) for _ in range(1)])
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        
        self.output_hypernetwork_mlps = nn.ModuleList([MLP(
            transformer_dim, transformer_dim, transformer_dim, 3) for _ in range(1)])
        
        # self.coarse_up_mid = CoarseGuidance(transformer_dim, transformer_dim)
        # self.mid_up_fine = CoarseGuidance(transformer_dim, transformer_dim)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:

        B = image_embeddings.size(0)
        src, pos_src= image_embeddings, torch.zeros_like(image_embeddings).to(image_embeddings.device, dtype=image_embeddings.dtype)

        tokens = torch.cat([token.weight.unsqueeze(0).expand(B, -1, -1) for token in self.coarse_fine_token], dim=1)
        
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        mask_token_coarse  = hs[:, 0, :]
        # mask_token_mid = hs[:, 1, :]
        # mask_token_fine = hs[:, 2, :]
        src = src.transpose(1, 2).view(b, c, h, w) # (B, C, H, W)

        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        # up_feat0 = F.interpolate(src, scale_factor=(1/8), mode='bilinear', align_corners=False)
        up_feat1  = act1(ln1(dc1(src) + feat_s1))
        up_feat2  = act2(dc2(up_feat1) + feat_s0)

        # Generate mask using hypernetwork
        hyper_in_coarse = self.output_hypernetwork_mlps[0](mask_token_coarse)  # (B, C)
        masks_coarse = (hyper_in_coarse.unsqueeze(1) @ up_feat2.view(b, c, -1)).view(b, 1, *up_feat2.shape[-2:])
        # hyper_in_mid = self.output_hypernetwork_mlps[1](mask_token_mid)  # (B, C)
        # hyper_in_fine = self.output_hypernetwork_mlps[2](mask_token_fine)  # (B, C)

        # masks_coarse = (hyper_in_coarse.unsqueeze(1) @ up_feat0.view(b, c, -1)).view(b, 1, *up_feat0.shape[-2:])
        # masks_coarse = F.interpolate(masks_coarse, scale_factor=8, mode='bilinear', align_corners=False)
        # masks_mid = (hyper_in_mid.unsqueeze(1) @ self.coarse_up_mid(up_feat1, torch.sigmoid(masks_coarse)).view(b, c, -1)).view(b, 1, h*2, w*2)
        # masks_fine = (hyper_in_coarse.unsqueeze(1) @ self.mid_up_fine(up_feat2, torch.sigmoid(masks_mid)).view(b, c, -1)).view(b, 1, h*4, w*4)

        return masks_coarse, None, None

@HEADS.register_module()
class StaticUPerHead(nn.Module):
    
    def __init__(self, **kwargs):

        super(StaticUPerHead, self).__init__()

        self.in_channels = kwargs.get('in_channels', [256, 512, 1024, 2048])  # Input channel dimensions
        self.channels = kwargs.get('channels', 512)  # Decoder feature channels
        self.align_corners = kwargs.get('align_corners', False)  # Interpolation alignment
        self.conv_cfg = kwargs.get('conv_cfg', None)  # Convolution config
        self.norm_cfg = kwargs.get('norm_cfg', dict(type='SyncBN', requires_grad=True))  # Normalization config
        self.act_cfg = kwargs.get('act_cfg', dict(type='ReLU'))  # Activation config
        self.use_high_res_features = kwargs.get('use_high_res_features', False)  # Whether to use high-res features
        self.num_classes = kwargs.get('num_classes', 2)  # Number of segmentation classes
        
        self.psp_fpn = UPerHead(
            in_channels_list=self.in_channels,
            channels=self.channels,
            pool_scales=(1, 2, 3, 6, 12),   
            align_corners=self.align_corners,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        self.mask_decoder = MaskDecoder(
            transformer_dim=self.channels,  # Transformer embedding dimension
            transformer=TwoWayTransformer(
                depth=2,  # Number of transformer layers
                embedding_dim=self.channels,  # Embedding dimension
                mlp_dim=2048,  # MLP hidden dimension
                num_heads=8,  # Number of attention heads
            ),
            use_high_res_features=True  # Enable high-resolution feature fusion
        )

        self.unified_loss = otdr_loss(
            cls_weight=1.0,
            reg_weight=1.0,
            seg_weight=1.0,
            normalize_by_image_size=False,
            min_area_ratio=0.0,
            ignore_index=255
        )
    
    def init_weights(self):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        pass

    def _forward(self, inputs):

        fpn_outs = self.psp_fpn(inputs) # List(tensor(B, C, H, W))
        cur_features = fpn_outs[-1] 
        B, C, H, W = cur_features.shape 

        masks_fine, masks_mid, masks_coarse = self.mask_decoder.forward(
            image_embeddings=cur_features,
            high_res_features=fpn_outs[:-1], 
        )
        
        return masks_fine, masks_mid, masks_coarse
    
    def forward_train(self, inputs, gt_semantic_seg):

        masks_list = self._forward(
            inputs
        )

        losses = self.unified_loss(
            pred_masks=masks_list,
            gt_semantic_seg=gt_semantic_seg, 
            target_class=1
        )
        
        return losses

    def forward_test(self, inputs, img_metas):

        final_output, _, _ = self._forward(inputs)
        return final_output

    def forward(self, inputs):
        return None
    