# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Type

import torch
from torch import nn

from head.position_embed import PositionEmbeddingRandom
 


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes point prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): Unused placeholder kept for API compatibility.
          activation (nn.Module): Unused for points-only encoding; kept for API compatibility.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Only positive points are supported
        self.point_embedding = nn.Embedding(1, embed_dim)
        # "No-point" token to represent empty/absent prompt
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
        confidence_is_logit: bool = False,
    ) -> torch.Tensor:
        """Embeds point prompts with optional confidence gating.

        Args:
          points: (B, N, 2) pixel coordinates (x, y)
          confidences: optional (B, N) or (B, N, 1) values in [0, 1] or logits
          confidence_is_logit: apply sigmoid on confidences if True
        """
        points = points + 0.5  # Shift to center of pixel
        pos_embed = self.pe_layer.forward_with_coords(points, self.input_image_size)

        if confidences is None:
            gate = torch.ones(points.shape[:2], device=points.device, dtype=pos_embed.dtype)
        else:
            if confidences.dim() == 3 and confidences.size(-1) == 1:
                confidences = confidences.squeeze(-1)
            gate = confidences.to(pos_embed.dtype)
            if confidence_is_logit:
                gate = torch.sigmoid(gate)
            # Clamp to [0, 1] for safety
            gate = torch.clamp(gate, 0.0, 1.0)

        # Broadcast gate to embedding dim
        gate = gate.unsqueeze(-1)  # (B, N, 1)
        token = gate * self.point_embedding.weight + (1.0 - gate) * self.not_a_point_embed.weight
        # token: (B, N, C) via broadcasting
        point_embedding = pos_embed + token
        return point_embedding

    def _get_batch_size(
        self,
        points: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embedding.weight.device

    def forward(
        self,
        points: Optional[torch.Tensor],
        confidences: Optional[torch.Tensor] = None,
        confidence_is_logit: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds point prompts, returning both sparse and dense embeddings.

        Arguments:
          points (torch.Tensor or none): point coordinates with shape (B, N, 2),
            where B is batch size, N is number of points, and each point is (x, y).
            All points are treated as positive points.

        Returns:
          torch.Tensor: sparse embeddings for the points, with shape
            BxNx(embed_dim), where N is determined by the number of input points.
          torch.Tensor: dense embeddings (no-mask embedding), in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            point_embeddings = self._embed_points(points, confidences, confidence_is_logit)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
        )

        return sparse_embeddings, dense_embeddings
