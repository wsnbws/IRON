# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type
import torch
from torch import nn
import torch.nn.functional as F
from head.untils import LayerNorm2d, MLP
from head.flag import get_task_state, get_test_task_state


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
        self.coarse_fine_token = nn.ModuleList([nn.Embedding(1, transformer_dim) for _ in range(3)])
        
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
            transformer_dim, transformer_dim, transformer_dim, 3) for _ in range(3)])
        
        self.coarse_up_mid = CoarseGuidance(transformer_dim, transformer_dim)
        self.mid_up_fine = CoarseGuidance(transformer_dim, transformer_dim)

    def forward(
        self,
        ori_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor = None,
        hist_cont_prompt_embeddings: torch.Tensor = None,
        high_res_features: Optional[List[torch.Tensor]] = None,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Predict a single mask given image and point prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder (B, C, H, W)
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings (1, C, H, W)
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points (B, N, C)
          high_res_features (Optional[List[torch.Tensor]]): high-res features for upsampling

        Returns:
          torch.Tensor: predicted mask (B, 1, H, W)
        """
        B = image_embeddings.size(0)
        src, pos_src= image_embeddings, image_pe

        tokens = torch.cat([token.weight.unsqueeze(0).expand(B, -1, -1) for token in self.coarse_fine_token], dim=1)
        
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        mask_token_coarse  = hs[:, 0, :]
        mask_token_mid = hs[:, 1, :]
        mask_token_fine = hs[:, 2, :]
        src = src.transpose(1, 2).view(b, c, h, w) # (B, C, H, W)

        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        up_feat0 = src
        up_feat1  = act1(ln1(dc1(up_feat0) + feat_s1))
        up_feat2  = act2(dc2(up_feat1) + feat_s0)

        # Generate mask using hypernetwork
        hyper_in_coarse = self.output_hypernetwork_mlps[0](mask_token_coarse)  # (B, C)
        hyper_in_mid = self.output_hypernetwork_mlps[1](mask_token_mid)  # (B, C)
        hyper_in_fine = self.output_hypernetwork_mlps[2](mask_token_fine)  # (B, C)

        masks_coarse = (hyper_in_coarse.unsqueeze(1) @ up_feat0.view(b, c, -1)).view(b, 1, h, w)
        masks_mid = (hyper_in_mid.unsqueeze(1) @ self.coarse_up_mid(up_feat1, torch.sigmoid(masks_coarse)).view(b, c, -1)).view(b, 1, h*2, w*2)
        masks_fine = (hyper_in_fine.unsqueeze(1) @ self.mid_up_fine(up_feat2, torch.sigmoid(masks_mid)).view(b, c, -1)).view(b, 1, h*4, w*4)

        return masks_fine, masks_mid, masks_coarse