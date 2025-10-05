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
        image_embedding_size_test: Tuple[int, int],
        input_image_size_test: Tuple[int, int],
    ) -> None:
        """
        Encodes point prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.point_embedding = nn.Embedding(1, embed_dim)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

    def _embed_points(
        self,
        points: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds point prompts with optional confidence gating.

        Args:
          points: (B, N, 2) pixel coordinates (x, y)
          confidences: optional (B, N) in [0, 1]
        """
        points = points + 0.5  # Shift to center of pixel
        if self.training:
          input_image_size = self.input_image_size
        else:
          input_image_size = self.input_image_size_test
        pos_embed = self.pe_layer.forward_with_coords(points, input_image_size)
        gate = confidences.to(pos_embed.dtype).unsqueeze(-1)  # (B, N, 1)
        token = gate * self.point_embedding.weight + (1.0 - gate) * self.not_a_point_embed.weight
        point_embedding = pos_embed + token # token: (B, N, C) via broadcasting
        return point_embedding
    
    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        if self.training:  
          return self.pe_layer(self.image_embedding_size).unsqueeze(0)
        else:
          return self.pe_layer(self.image_embedding_size_test).unsqueeze(0)

    def _get_device(self) -> torch.device:
        return self.point_embedding.weight.device

    def forward(
        self,
        points: Optional[torch.Tensor],
        confidences: Optional[torch.Tensor] = None,
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
        """
        sparse_embeddings = torch.empty(
            (points.shape[0], 0, self.embed_dim), device=self._get_device()
        )
        assert points is not None, "points is None"
        point_embeddings = self._embed_points(points, confidences)
        sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        return sparse_embeddings
