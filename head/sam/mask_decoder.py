# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type
import torch
from torch import nn
from head.untils import LayerNorm2d, MLP
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
        self.mask_token = nn.Embedding(1, transformer_dim)

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
        
        self.output_hypernetwork_mlp = MLP(
            transformer_dim, transformer_dim, transformer_dim, 3
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        hist_cont_prompt_embeddings: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
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
        B = sparse_prompt_embeddings.size(0)
        src, pos_src= image_embeddings, image_pe

        output_tokens = self.mask_token.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 1, C)
        hist_cont_prompt_embeddings = hist_cont_prompt_embeddings.unsqueeze(1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings, hist_cont_prompt_embeddings), dim=1)  # (B, 1+N+M, C)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        mask_token_out = hs[:, 0, :]               # (B, C)
        src = src.transpose(1, 2).view(b, c, h, w) # (B, C, H, W)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # Generate mask using hypernetwork
        hyper_in = self.output_hypernetwork_mlp(mask_token_out)  # (B, C)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in.unsqueeze(1) @ upscaled_embedding.view(b, c, h * w)).view(b, 1, h, w)

        return masks