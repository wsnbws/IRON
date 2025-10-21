import torch
import torch.nn as nn

from mmseg.models.builder import BACKBONES
from .static_dinov3 import StaticDINOv3


@BACKBONES.register_module()
class VideoDINOv3(StaticDINOv3):
    """Video-aware DINOv3 wrapper that supports 5D inputs.

    Accepts inputs of shape (B, C, H, W) or (B, T, C, H, W).
    For 5D inputs it flattens the temporal dimension into the batch dimension
    so downstream heads can fuse temporally (e.g., TemporalUPerHead expects
    features with batch = B*T and will reshape using num_frames).
    
    Args:
        model_name (str): Model size. Options: 'small', 'base', 'large', 'giant2'
        img_size (int): Input image size. Default: 518
        patch_size (int): Patch size. Default: 14
        out_indices (list): Indices of output feature layers. Default: [3, 5, 7, 11]
        pretrained (str, optional): Path to pre-trained weights
        freeze_backbone (bool): Whether to freeze the backbone. Default: False
    """

    def forward(self, x):
        """Forward pass supporting both 4D and 5D inputs.
        
        Args:
            x (Tensor): Input images of shape (B, C, H, W) or (B, T, C, H, W)
            
        Returns:
            tuple: Multi-scale feature maps
        """
        if x.dim() == 5:
            # Video input: (B, T, C, H, W)
            batch_size, num_frames, num_channels, height, width = x.shape
            # Flatten temporal dimension into batch dimension
            x = x.view(batch_size * num_frames, num_channels, height, width)
            return super().forward(x)
        elif x.dim() == 4:
            # Static image input: (B, C, H, W)
            return super().forward(x)
        else:
            raise ValueError(f'Unsupported input dim: {x.dim()}, expected 4D or 5D tensor.')


# Convenience functions for specific video model sizes
@BACKBONES.register_module()
class VideoDINOv3Small(VideoDINOv3):
    def __init__(self, **kwargs):
        super().__init__(model_name='small', **kwargs)


@BACKBONES.register_module()
class VideoDINOv3Base(VideoDINOv3):
    def __init__(self, **kwargs):
        super().__init__(model_name='base', **kwargs)


@BACKBONES.register_module()
class VideoDINOv3Large(VideoDINOv3):
    def __init__(self, **kwargs):
        super().__init__(model_name='large', **kwargs)


@BACKBONES.register_module()
class VideoDINOv3Giant(VideoDINOv3):
    def __init__(self, **kwargs):
        super().__init__(model_name='giant2', **kwargs)

