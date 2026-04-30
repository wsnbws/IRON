import torch
import torch.nn as nn

from mmseg.models.builder import BACKBONES
from .convmae import ConvMAE


@BACKBONES.register_module()
class VideoConvMAE(ConvMAE):
    """Video-aware ConvMAE wrapper that supports 5D inputs.

    Accepts inputs of shape (B, C, H, W) or (B, T, C, H, W).
    For 5D inputs it flattens the temporal dimension into the batch dimension
    so downstream heads can fuse temporally (e.g., TemporalUPerHead expects
    features with batch = B*T and will reshape using num_frames).
    """

    def forward(self, x):
        if x.dim() == 5:
            batch_size, num_frames, num_channels, height, width = x.shape
            x = x.view(batch_size * num_frames, num_channels, height, width)
            return super().forward(x)
        elif x.dim() == 4:
            return super().forward(x)
        else:
            raise ValueError(f'Unsupported input dim: {x.dim()}, expected 4D or 5D tensor.')


