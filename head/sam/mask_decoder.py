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
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        # Single mask hypernetwork
        self.output_hypernetwork_mlp = MLP(
            transformer_dim, transformer_dim, transformer_dim // 8, 3
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Predict a single mask given image and point prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder (B, C, H, W)
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings (1, C, H, W)
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points (B, N, C)
          dense_prompt_embeddings (torch.Tensor): the dense embeddings (B, C, H, W)
          high_res_features (Optional[List[torch.Tensor]]): high-res features for upsampling

        Returns:
          torch.Tensor: predicted mask (B, 1, H, W)
        """
        B = sparse_prompt_embeddings.size(0)
        
        # Use only mask token
        output_tokens = self.mask_token.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 1, C)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  # (B, 1+N, C)

        # Prepare image embeddings
        assert image_embeddings.shape[0] == B, f"Batch size mismatch: image_embeddings={image_embeddings.shape[0]}, prompts={B}"
        src = image_embeddings + dense_prompt_embeddings
        
        assert image_pe.size(0) == 1, "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = image_pe.expand(B, -1, -1, -1)  # (1, C, H, W) -> (B, C, H, W)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_token_out = hs[:, 0, :]  # (B, C) - only mask token

        # Upscale mask embeddings and predict masks using the mask token
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # Generate mask using hypernetwork
        hyper_in = self.output_hypernetwork_mlp(mask_token_out)  # (B, C//8)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in.unsqueeze(1) @ upscaled_embedding.view(b, c, h * w)).view(b, 1, h, w)

        return masks