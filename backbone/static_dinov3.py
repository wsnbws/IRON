import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv_custom import load_checkpoint
from .dino_3.dinov3 import (
    DinoVisionTransformer,
    vit_small,
    vit_base,
    vit_large,
    vit_giant2,
)


@BACKBONES.register_module()
class StaticDINOv3(nn.Module):
    """Static DINOv3 backbone for semantic segmentation.
    
    Args:
        model_name (str): Model size. Options: 'small', 'base', 'large', 'giant2'
        img_size (int): Input image size. Default: 518
        patch_size (int): Patch size. Default: 14
        out_indices (list): Indices of output feature layers. Default: [3, 5, 7, 11]
        pretrained (str, optional): Path to pre-trained weights
        embed_dim (int, optional): Embedding dimension (auto-set based on model_name if None)
        depth (int, optional): Depth (auto-set based on model_name if None)
        num_heads (int, optional): Number of attention heads (auto-set based on model_name if None)
        freeze_backbone (bool): Whether to freeze the backbone. Default: False
    """
    
    MODEL_CONFIGS = {
        'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
        'giant2': {'embed_dim': 1536, 'depth': 40, 'num_heads': 24},
    }
    
    def __init__(
        self,
        model_name='base',
        img_size=518,
        patch_size=14,
        out_indices=[3, 5, 7, 11],
        pretrained=None,
        embed_dim=None,
        depth=None,
        num_heads=None,
        freeze_backbone=False,
        **kwargs
    ):
        super().__init__()
        
        # Get config for the model
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"model_name must be one of {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[model_name]
        embed_dim = embed_dim or config['embed_dim']
        depth = depth or config['depth']
        num_heads = num_heads or config['num_heads']
        
        self.model_name = model_name
        self.out_indices = out_indices
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.freeze_backbone = freeze_backbone
        
        # Create DINOv3 model
        self.dino_model = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            **kwargs
        )
        
        # Note: init_weights will be called by segmentor, not here
    
    def _freeze_backbone(self):
        """Freeze all parameters in the backbone."""
        for param in self.dino_model.parameters():
            param.requires_grad = False
        self.dino_model.eval()
    
    def init_weights(self, pretrained=None):
        """Initialize weights from pretrained checkpoint.
        
        This method is called by the segmentor's init_weights.
        
        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        # Initialize DINOv3 model weights
        self.dino_model.init_weights(pretrained=pretrained)
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            self._freeze_backbone()
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            tuple: Multi-scale feature maps
        """
        B, C, H, W = x.shape
        
        # Get intermediate layers from DINOv3
        # Returns tuple of feature tensors
        intermediate_output = self.dino_model.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,  # Reshape to spatial format (B, C, H', W')
            return_class_token=False,
            norm=True,
        )
        
        # intermediate_output is a tuple of (B, C, H', W') tensors
        # We need to upsample them to common sizes for FPN-like processing
        features = list(intermediate_output)
        
        # Compute output spatial size based on patch size
        h_out = H // self.patch_size
        w_out = W // self.patch_size
        
        # Define target sizes for multi-scale features (similar to ConvMAE)
        # Typically we want features at different scales: 1/8, 1/16, 1/32 of input
        target_sizes = [
            (h_out * 2, w_out * 2),  # ~1/8 scale
            (h_out, w_out),           # ~1/16 scale  
            (h_out // 2, w_out // 2), # ~1/32 scale
        ]
        
        # Ensure we have at least 3 feature levels
        if len(features) < 3:
            # If we have fewer features, duplicate the last one
            while len(features) < 3:
                features.append(features[-1])
        
        # Take the last 3 features and resize to target sizes
        output_features = []
        selected_features = features[-3:]  # Take last 3 feature maps
        
        for feat, target_size in zip(selected_features, target_sizes):
            # Resize to target size
            if feat.shape[2:] != target_size:
                feat_resized = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                feat_resized = feat
            output_features.append(feat_resized)
        
        return tuple(output_features)
    
    def train(self, mode=True):
        """Override train mode to handle frozen backbone."""
        super().train(mode)
        if self.freeze_backbone:
            self.dino_model.eval()
        return self


# Convenience functions to create specific model sizes
@BACKBONES.register_module()
class DINOv3Small(StaticDINOv3):
    def __init__(self, **kwargs):
        super().__init__(model_name='small', **kwargs)


@BACKBONES.register_module()
class DINOv3Base(StaticDINOv3):
    def __init__(self, **kwargs):
        super().__init__(model_name='base', **kwargs)


@BACKBONES.register_module()
class DINOv3Large(StaticDINOv3):
    def __init__(self, **kwargs):
        super().__init__(model_name='large', **kwargs)


@BACKBONES.register_module()
class DINOv3Giant(StaticDINOv3):
    def __init__(self, **kwargs):
        super().__init__(model_name='giant2', **kwargs)

