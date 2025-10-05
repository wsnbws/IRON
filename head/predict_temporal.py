import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
from typing import Optional, Tuple

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmcv.cnn import ConvModule
from .memory_encoder import MemoryEncoder
from .memory_attention import MemoryAttention
from .position_embed import PositionEmbeddingSine
from .history_queue import TemporalQueue
from .untils import LayerNorm2d
from .point_predictor import PointPredictor
from .loss import otdr_loss
from .sam.prompt_encoder import PromptEncoder
from .sam.mask_decoder import MaskDecoder
from .sam.transformer import TwoWayTransformer


class PSPModule(nn.Module):
    """Pyramid Scene Parsing Module
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels  
        pool_scales (tuple): Pooling scales for pyramid pooling
    """
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Pyramid pooling branches
        self.psp_modules = nn.ModuleList()
        for scale in pool_scales:
            self.psp_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, in_channels // len(pool_scales), kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_channels // len(pool_scales)),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final conv to adjust channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_scales) * (in_channels // len(pool_scales)), 
                out_channels, 
                kernel_size=3, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """Forward pass of PSP module
        
        Args:
            x (Tensor): Input feature tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Enhanced feature tensor of shape (B, out_channels, H, W)
        """
        input_size = x.size()
        psp_outs = [x]
        
        for psp_module in self.psp_modules:
            psp_out = psp_module(x)
            psp_out = F.interpolate(
                psp_out, 
                size=input_size[2:], 
                mode='bilinear', 
                align_corners=False
            )
            psp_outs.append(psp_out)
        
        # Concatenate all pyramid features
        psp_outs = torch.cat(psp_outs, dim=1)
        
        # Final convolution to get desired output channels
        output = self.final_conv(psp_outs)
        
        return output

@HEADS.register_module()
class PredictiveTemporalUPerHead(nn.Module):
    """
    Predictive Temporal UPerHead for video segmentation with memory-based attention.
    
    This head processes temporal sequences by maintaining a history queue of previous frames
    and their segmentation masks, then uses memory attention to enhance current frame features
    with historical context. It incorporates point prediction for prompt-based segmentation
    and uses a SAM-style decoder for final mask generation.
    
    Key Components:
        - Temporal queue for maintaining frame history
        - Memory encoder for processing historical frames
        - Memory attention for temporal feature fusion
        - Point predictor for generating segmentation prompts
        - SAM decoder for final mask prediction
    """
    
    def __init__(self, **kwargs):
        """
        Initialize PredictiveTemporalUPerHead.
        
        Args:
            in_channels (list[int]): Input channels for each backbone level. Default: [256, 512, 1024, 2048]
            in_index (list[int]): Feature level indices to use from backbone. Default: [0, 1, 2, 3]
            channels (int): Number of channels in decoder features. Default: 512
            dropout_ratio (float): Dropout ratio for regularization. Default: 0.1
            num_classes (int): Number of segmentation classes (including background). Default: 2
            align_corners (bool): Whether to align corners in interpolation. Default: False
            ignore_index (int): Index to ignore in loss computation. Default: 255
            conv_cfg (dict): Configuration for convolution layers. Default: None
            norm_cfg (dict): Configuration for normalization layers. Default: SyncBN
            act_cfg (dict): Configuration for activation layers. Default: ReLU
            input_transform (str): Input transformation method. Default: 'multiple_select'
            streaming (bool): Whether to use streaming mode for online inference. Default: False
            detach_every (int): Detach gradients every N steps (0=disabled). Default: 0
            history_length (int): Number of historical frames to maintain. Default: 2
            mask_ratio (int): Downsampling ratio for mask resolution. Default: 8
            use_topk_memory (bool): Whether to use top-K foreground memory selection. Default: False
            topk_memory_size (int): Number of top-K memory tokens to select. Default: 256
        """
        super(PredictiveTemporalUPerHead, self).__init__()
    
        # Base decoder configuration
        self.in_channels = kwargs.get('in_channels', [256, 512, 1024, 2048])  # Input channel dimensions
        self.in_index = kwargs.get('in_index', [0, 1, 2, 3])  # Backbone feature indices
        self.channels = kwargs.get('channels', 512)  # Decoder feature channels
        self.dropout_ratio = kwargs.get('dropout_ratio', 0.1)  # Dropout probability
        self.num_classes = kwargs.get('num_classes', 2)  # Number of segmentation classes
        self.align_corners = kwargs.get('align_corners', False)  # Interpolation alignment
        self.ignore_index = kwargs.get('ignore_index', 255)  # Loss ignore index
        
        # Layer configuration
        self.conv_cfg = kwargs.get('conv_cfg', None)  # Convolution config
        self.norm_cfg = kwargs.get('norm_cfg', dict(type='SyncBN', requires_grad=True))  # Normalization config
        self.act_cfg = kwargs.get('act_cfg', dict(type='ReLU'))  # Activation config
        self.input_transform = kwargs.get('input_transform', 'multiple_select')  # Input transform method
        
        # Temporal processing configuration
        self.streaming = bool(kwargs.get('streaming', False))  # Enable streaming inference
        self.detach_every = int(kwargs.get('detach_every', 0))  # Gradient detachment frequency
        self.history_length = int(kwargs.get('history_length', 2))  # Historical frame count
        self.mask_ratio = int(kwargs.get('mask_ratio', 8))  # Mask downsampling ratio
        
        # Memory selection configuration
        self.use_topk_memory = bool(kwargs.get('use_topk_memory', False))  # Enable top-K memory selection
        self.topk_memory_size = int(kwargs.get('topk_memory_size', 256))  # Top-K memory token count

        # ===== Core Components Initialization =====
        
        # Temporal processing components
        self.temporal_queue = TemporalQueue(
            history_length=self.history_length,  # Number of historical frames to store
            streaming=self.streaming  # Enable streaming mode for online inference
        )
        self.temporal_time_test_queue = None  # Test-time timestamp queue
        self.last_m1_logits = None  # Legacy placeholder for M1 outputs
        
        # Feature processing modules
        self._build_fpn_module()  # Build FPN for multi-scale feature fusion
        self.memory_attention = MemoryAttention()  # Cross-attention with historical features
        self.memory_encoder = MemoryEncoder(total_stride=self.mask_ratio)  # Encode historical frames
        
        # Position encoding for spatial awareness
        self.pos_embed = PositionEmbeddingSine(
            num_pos_feats=256,  # Position encoding dimension
            normalize=True,  # Normalize position values
            scale=None,  # Auto-scale based on feature size
            temperature=10000  # Temperature for sinusoidal encoding
        )
        
        # Pyramid Scene Parsing for enhanced semantic understanding
        self.psp_module = PSPModule(
            in_channels=self.channels,  # Input feature channels
            out_channels=self.channels,  # Output feature channels
            pool_scales=(1, 2, 3, 6)  # Multi-scale pooling sizes
        )
        
        # Temporal embedding: maps scalar timestamps to feature embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),  # Input: scalar timestamp
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),  # Hidden layer
            nn.ReLU(inplace=True),
            nn.Linear(128, self.channels//4)  # Output: temporal embedding
        )

        # Point prediction for generating segmentation prompts
        self.point_predictor = PointPredictor(
            current_dim=self.channels,  # Current frame feature dimension
            memory_dim=64,  # Memory feature dimension
            num_points=1,  # Number of points to predict per object
            hidden_dim=512,  # Hidden layer dimension
            num_heads=4,  # Number of attention heads
            use_topk_features=True,  # Use top-K feature selection
            topk_size=self.topk_memory_size,  # Top-K selection size
        )

        # Build SAM-style decoder components
        self._build_sam_decoder()

        # Unified loss function for point prediction and mask segmentation
        self.unified_loss = otdr_loss(
            cls_weight=1.0,  # Weight for point classification loss
            reg_weight=1.0,  # Weight for point regression loss
            seg_weight=1.0,  # Weight for mask segmentation loss
            normalize_by_image_size=True,  # Normalize coordinates by image size
            min_area_ratio=0.0,  # Minimum area ratio for valid targets
        )
        self.point_prompt_threshold = 0.5  # Confidence threshold for point prompts

    def _build_sam_decoder(self):
        """
        Build SAM-style prompt encoder and mask decoder components.
        
        The prompt encoder processes point prompts and confidence scores into embeddings,
        while the mask decoder uses a two-way transformer to generate final segmentation masks.
        """
        # SAM decoder configuration
        self.sam_prompt_embed_dim = self.channels  # Embedding dimension for prompts
        self.sam_prompt_image_embedding_size = (32, 32)  # Training image embedding size
        self.sam_prompt_test_image_embedding_size = (32, 40)  # Test image embedding size
        self.sam_prompt_input_image_size = (512, 512)  # Training input image size
        self.sam_prompt_test_image_size = (512, 640)  # Test input image size

        # Prompt encoder: converts points and confidence to embeddings
        self.prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,  # Embedding dimension
            image_embedding_size=self.sam_prompt_image_embedding_size,  # Feature map size
            input_image_size=self.sam_prompt_input_image_size,  # Input image resolution
            image_embedding_size_test=self.sam_prompt_test_image_embedding_size,  # Test feature size
            input_image_size_test=self.sam_prompt_test_image_size  # Test image resolution
        )
        
        # Mask decoder: generates segmentation masks from features and prompts
        self.mask_decoder = MaskDecoder(
            transformer_dim=self.sam_prompt_embed_dim,  # Transformer embedding dimension
            transformer=TwoWayTransformer(
                depth=2,  # Number of transformer layers
                embedding_dim=self.sam_prompt_embed_dim,  # Embedding dimension
                mlp_dim=2048,  # MLP hidden dimension
                num_heads=8,  # Number of attention heads
            ),
            use_high_res_features=True  # Enable high-resolution feature fusion
        )
        
    def _build_fpn_module(self):
        """
        Build Feature Pyramid Network (FPN) module for multi-scale feature fusion.
        
        Creates lateral and FPN convolutions to process multi-level backbone features.
        Lateral convs reduce channel dimensions, FPN convs refine fused features.
        """
        # FPN lateral and output convolutions
        self.lateral_convs = nn.ModuleList()  # Channel reduction convolutions
        self.fpn_convs = nn.ModuleList()  # Feature refinement convolutions
        
        for in_channels in self.in_channels:
            # Lateral conv: reduces channels from backbone to decoder dimension
            l_conv = ConvModule(
                in_channels,  # Input channels from backbone level
                self.channels,  # Output channels (decoder dimension)
                1,  # 1x1 kernel for channel reduction
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            
            # FPN conv: refines fused features with 3x3 convolution
            fpn_conv = ConvModule(
                self.channels,  # Input channels (decoder dimension)
                self.channels,  # Output channels (decoder dimension)
                3,  # 3x3 kernel for feature refinement
                padding=1,  # Maintain spatial dimensions
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
    def _transform_inputs(self, inputs):
        """
        Transform backbone features according to input_transform configuration.
        
        Args:
            inputs (list[Tensor]): Multi-level backbone features, each with shape (B, C_i, H_i, W_i)
                where C_i varies by level and H_i, W_i are spatial dimensions
            
        Returns:
            list[Tensor] | Tensor: Transformed features
                - If 'multiple_select': list of selected features [(B, C_i, H_i, W_i), ...]
                - If 'resize_concat': concatenated upsampled features (B, sum(C_i), H_0, W_0)
                - Otherwise: single selected feature (B, C_idx, H_idx, W_idx)
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs
    
    # ===== Helpers for streaming mode =====
    def _fpn_forward_single(self, inputs):
        """
        Forward pass through FPN module for single frame processing.
        
        Args:
            inputs (list[Tensor]): Multi-level backbone features
                [(B, C_0, H_0, W_0), (B, C_1, H_1, W_1), (B, C_2, H_2, W_2), (B, C_3, H_3, W_3)]
        
        Returns:
            list[Tensor]: FPN output features at multiple scales
                [(B, channels, H_0, W_0), (B, channels, H_1, W_1), ...]
                where channels is the unified decoder channel dimension
        """
        inputs = self._transform_inputs(inputs)  # Transform according to input_transform config

        # Step 1: Apply lateral convolutions to reduce channel dimensions
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))  # (B, channels, H_i, W_i)

        # Step 2: Top-down pathway with lateral connections
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]  # Spatial size of higher resolution level
            # Upsample lower resolution features and add to higher resolution
            laterals[i - 1] += resize(
                laterals[i],  # Lower resolution features
                size=prev_shape,  # Target spatial size
                mode='bilinear',  # Interpolation method
                align_corners=self.align_corners)  # Corner alignment setting

        # Step 3: Apply FPN convolutions for feature refinement
        fpn_outs = [
            self.fpn_convs[i](laterals[i])  # Refine fused features
            for i in range(used_backbone_levels)
        ]
        return fpn_outs  # [(B, channels, H_0, W_0), (B, channels, H_1, W_1), ...]

    def _get_memory(self, cur_features, t=0, timestamps: torch.Tensor = None):
        """
        Retrieve and process historical memory features from temporal queue.
        
        Args:
            cur_features (Tensor): Current frame features (B, C, H, W)
            t (int): Current time step, reset queue when t=0. Default: 0
            timestamps (Tensor): Timestamp information (B, history_length+1) for temporal encoding
        
        Returns:
            tuple[Tensor, Tensor]: Processed memory features
                - memory_enc (Tensor): Memory position encodings
                    Shape: (T*H*W, B, C) if use_topk_memory=False, else (T*K, B, C)
                - mem_features (Tensor): Memory visual features  
                    Shape: (T*H*W, B, C) if use_topk_memory=False, else (T*K, B, C)
                where T=history_length, K=topk_memory_size
        """
        B, C, H, W = cur_features.shape  # Extract feature dimensions
        
        # Initialize temporal queue with appropriate dimensions
        mask_shape = (self.num_classes - 1, H * self.mask_ratio, W * self.mask_ratio)  # Mask dimensions
        self.temporal_queue.ensure_allocation(cur_features, mask_shape)
        
        # Reset temporal state at sequence boundaries
        if int(t) == 0:
            self.temporal_queue.reset_state(all_batch=True)  # Clear all historical data

        # Retrieve historical data from temporal queue
        # Clone to prevent inplace operation conflicts with gradient computation
        historical_frames, historical_masks = self.temporal_queue.get_history_frames()  # (T, B, C, H, W), (T, B, num_classes-1, H_mask, W_mask)
        
        # Reshape for efficient batch processing
        batch_frames = historical_frames.view(-1, *historical_frames.shape[2:])  # (T*B, C, H, W)
        batch_masks = historical_masks.view(-1, *historical_masks.shape[2:])  # (T*B, num_classes-1, H_mask, W_mask)

        # Process temporal information for time-aware encoding
        timestamps = timestamps.to(cur_features.device, dtype=cur_features.dtype)  # Move to correct device
        
        # Normalize timestamps to [0, 1] range per sequence
        tmin = timestamps.min(dim=1, keepdim=True)[0]  # Minimum timestamp per batch
        tmax = timestamps.max(dim=1, keepdim=True)[0]  # Maximum timestamp per batch
        denom = (tmax - tmin).clamp_min(1e-6)  # Avoid division by zero
        timestamps = (timestamps - tmin) / denom  # Normalize to [0, 1]
        
        # Extract historical timestamps (exclude current frame)
        historical_timestamps = timestamps[:, :-1].transpose(0, 1)  # (history_length, B)
        batch_timestamps = historical_timestamps.reshape(-1, 1)  # (history_length * B, 1)
        
        # Generate temporal embeddings and broadcast to spatial dimensions
        batch_time_enc = self.time_mlp(batch_timestamps)  # (history_length * B, channels//4)
        batch_time_enc = batch_time_enc.unsqueeze(2).unsqueeze(3)  # Add spatial dimensions
        batch_time_enc = batch_time_enc.expand(-1, -1, *historical_frames.shape[3:])  # (history_length * B, channels//4, H, W)

        # Encode historical frames with their segmentation masks
        batch_memory_output = self.memory_encoder(batch_frames, batch_masks)  # Process all historical frames
        mem_features = batch_memory_output["vision_features"]  # Encoded visual features (T*B, C, H, W)
        batch_pos_enc = batch_memory_output["vision_pos_enc"]  # Spatial position encodings (T*B, C, H, W)
        
        # Combine spatial and temporal encodings
        batch_pos_enc = batch_pos_enc + batch_time_enc  # Add temporal information to position encoding
        
        # Select memory tokens based on configuration
        if self.use_topk_memory:
            # Use top-K foreground selection for efficient memory usage
            k_size = self.topk_memory_size if self.training else 320  # Different K for train/test
            selected_memory, selected_pos_enc = self._select_topk_foreground_memory(
                mem_features, batch_pos_enc, batch_masks, k=k_size
            )
            
            # Reshape for attention mechanism: (sequence_length, batch_size, channels)
            k_tokens = selected_memory.shape[0] // (len(historical_frames) * B)  # Tokens per sample
            memory_enc = selected_pos_enc.view(len(historical_frames), B, k_tokens, -1)  # (T, B, k, C)
            memory_enc = memory_enc.permute(0, 2, 1, 3).flatten(0, 1)  # (T*k, B, C)
            
            mem_features = selected_memory.view(len(historical_frames), B, k_tokens, -1)  # (T, B, k, C)
            mem_features = mem_features.permute(0, 2, 1, 3).flatten(0, 1)  # (T*k, B, C)
        else:
            # Use all spatial locations as memory tokens (backward compatibility)
            batch_pos_enc = batch_pos_enc.view(-1, B, *batch_pos_enc.shape[1:])  # (T, B, C, H, W)
            memory_enc = batch_pos_enc.permute(0, 3, 4, 1, 2).flatten(0, 2)  # (T*H*W, B, C)
            
            mem_features = mem_features.view(-1, B, *mem_features.shape[1:])  # (T, B, C, H, W)
            mem_features = mem_features.permute(0, 3, 4, 1, 2).flatten(0, 2)  # (T*H*W, B, C)
            
        return memory_enc, mem_features

    def _select_topk_foreground_memory(self, mem_features, batch_pos_enc, batch_masks, k=256):
        """
        Select top-K foreground memory tokens based on segmentation masks.
        
        This method identifies the most relevant spatial locations by ranking pixels
        according to their foreground probability from historical segmentation masks.
        
        Args:
            mem_features (Tensor): Historical frame features (T*B, C, H, W)
                where T=history_length, B=batch_size, C=channels, H,W=spatial dims
            batch_pos_enc (Tensor): Position encodings (T*B, C, H, W)
                Positional embeddings for spatial locations
            batch_masks (Tensor): Historical segmentation masks (T*B, num_classes-1, H_mask, W_mask)
                Binary/probability masks for foreground classes
            k (int): Number of top-K tokens to select per sample. Default: 256
            
        Returns:
            tuple[Tensor, Tensor]: Selected memory tokens
                - selected_memory (Tensor): Top-K memory features (T*B*k, C)
                - selected_pos_enc (Tensor): Corresponding position encodings (T*B*k, C)
        """
        TB, C, H, W = mem_features.shape  # Extract tensor dimensions
        
        # Compute foreground probability maps from segmentation masks
        masks_resized = F.interpolate(
            batch_masks.sum(dim=1, keepdim=True),  # Sum across classes: (T*B, 1, H_mask, W_mask)
            size=(H, W),  # Resize to feature map resolution
            mode='bilinear',  # Bilinear interpolation
            align_corners=False  # Don't align corners
        ).squeeze(1)  # Remove channel dimension: (T*B, H, W)
        
        # Flatten spatial dimensions for efficient processing
        masks_flat = masks_resized.view(TB, H*W)  # Flattened masks: (T*B, H*W)
        mem_flat = mem_features.view(TB, C, H*W)  # Flattened features: (T*B, C, H*W)
        pos_flat = batch_pos_enc.view(TB, C, H*W)  # Flattened position encodings: (T*B, C, H*W)
        
        # Select top-K spatial locations with highest foreground probability
        _, topk_indices = torch.topk(masks_flat, k=min(k, H*W), dim=1)  # (T*B, k)
        
        # Gather features and position encodings at selected locations
        topk_indices_expanded = topk_indices.unsqueeze(1).expand(-1, C, -1)  # Expand for feature dimension: (T*B, C, k)
        
        selected_memory = torch.gather(mem_flat, 2, topk_indices_expanded)  # Selected features: (T*B, C, k)
        selected_pos_enc = torch.gather(pos_flat, 2, topk_indices_expanded)  # Selected positions: (T*B, C, k)
        
        # Reshape to final output format
        selected_memory = selected_memory.transpose(1, 2).reshape(TB*k, C)  # (T*B*k, C)
        selected_pos_enc = selected_pos_enc.transpose(1, 2).reshape(TB*k, C)  # (T*B*k, C)
        
        return selected_memory, selected_pos_enc

    def _forward_stream_batch(self, inputs, t: int, timestamps: torch.Tensor = None, **kargs):
        """
        Streaming forward pass for temporal video segmentation.
        
        Processes current frame with historical context from memory queue to generate
        segmentation masks. Integrates FPN features, memory attention, point prediction,
        and SAM-style decoding for final mask generation.

        Args:
            inputs (list[Tensor]): Multi-level backbone features for current frame
                [(B, C_0, H_0, W_0), (B, C_1, H_1, W_1), (B, C_2, H_2, W_2), (B, C_3, H_3, W_3)]
            t (int): Current time step index, resets temporal state when t=0
            timestamps (Tensor, optional): Temporal information (B, history_length+1)
                Contains timestamp values for temporal encoding
            **kargs: Additional keyword arguments passed to memory attention

        Returns:
            tuple[Tensor, Tensor, Tensor]: Segmentation results and predictions
                - masks (Tensor): Generated segmentation masks (B, 1, H_out, W_out)
                    where H_out, W_out depend on decoder upsampling
                - confidence (Tensor): Point prediction confidence scores (B, 1)
                    Probability that predicted points are valid
                - points (Tensor): Predicted point coordinates (B, 1, 2)
                    Normalized (x, y) coordinates in [0, 1] range
        """
        # Step 1: Extract and process current frame features
        fpn_outs = self._fpn_forward_single(inputs)  # Multi-scale FPN features
        if len(fpn_outs) > 1:
            cur_features = fpn_outs[-1]  # Use highest resolution features: (B, channels, H, W)
        
        # Step 2: Enhance features with Pyramid Scene Parsing
        cur_features = self.psp_module(cur_features)  # Enhanced semantic features: (B, channels, H, W)
        
        B, C, H, W = cur_features.shape  # Extract tensor dimensions
        
        # Step 3: Generate position encodings for spatial awareness
        cur_pos_enc = self.pos_embed(cur_features)  # Spatial position encodings: (B, C, H, W)
        
        # Step 4: Reshape for attention mechanism (sequence-first format)
        cur_features_seq = cur_features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        cur_pos_enc_seq = cur_pos_enc.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
    
        # Step 5: Retrieve and process historical memory
        memory_enc, mem_features = self._get_memory(cur_features, t, timestamps)

        # Step 6: Fuse current features with historical context via attention
        fus_feat, _ = self.memory_attention(
            cur_features_seq,  # Current frame queries: (H*W, B, C)
            mem_features,  # Historical memory keys/values: (T*K, B, C) or (T*H*W, B, C)
            cur_pos_enc_seq,  # Current position encodings: (H*W, B, C)
            memory_enc,  # Memory position encodings: (T*K, B, C) or (T*H*W, B, C)
            spatial_shape=(H, W),  # Spatial dimensions for RoPE
            **kargs
        )
        fus_feat = fus_feat.permute(1, 2, 0).reshape(B, C, H, W)  # Reshape back: (B, C, H, W)

        # Step 7: Predict segmentation points and confidence
        final_global_token, confidence, points = self.point_predictor.forward(
            cur_features=cur_features,  # Current frame features: (B, C, H, W)
            mem_features=mem_features,  # Memory features: (T*K, B, C) or (T*H*W, B, C)
            memory_pos_enc=memory_enc  # Memory position encodings: (T*K, B, C) or (T*H*W, B, C)
        )
        
        # Step 8: Encode predicted points as prompts
        sparse_prompt_embeddings = self.prompt_encoder.forward(points, confidence)  # Point embeddings: (B, N, C)
        sparse_prompt_embeddings = sparse_prompt_embeddings.to(fus_feat.device, fus_feat.dtype)

        # Step 9: Generate final segmentation masks
        masks = self.mask_decoder.forward(
            image_embeddings=fus_feat,  # Fused features: (B, C, H, W)
            image_pe=self.prompt_encoder.get_dense_pe(),  # Dense position encodings: (1, C, H, W)
            sparse_prompt_embeddings=sparse_prompt_embeddings,  # Point prompts: (B, N, C)
            hist_cont_prompt_embeddings=final_global_token,  # Historical context: (B, 1, C)
            high_res_features=[]  # High-resolution features (empty for now)
        )

        # Step 10: Update temporal queue with current results
        self.temporal_queue.push(cur_features, masks)  # Store features and masks

        return masks, confidence, points
    
    def forward(self, inputs):
        return None
    
    def forward_test(self, inputs, img_metas, test_cfg):
        """
        Test-time forward pass with streaming support for online inference.
        
        Manages temporal state across video sequences, automatically detecting
        sequence boundaries and maintaining appropriate history for each frame.

        Args:
            inputs (list[Tensor]): Multi-level backbone features for current frame
                [(B, C_0, H_0, W_0), (B, C_1, H_1, W_1), (B, C_2, H_2, W_2), (B, C_3, H_3, W_3)]
            img_metas (list[dict]): Image metadata containing:
                - 'filename': Image filename for timestamp extraction
                - 'group_index': Position within video sequence
                - 'is_seq_start': Boolean indicating sequence start
            test_cfg (dict): Test configuration (unused but required for interface)

        Returns:
            Tensor: Segmentation masks (B, 1, H_out, W_out)
                Final predicted segmentation masks for current frame
        """
        # Extract metadata and determine sequence state
        metas = img_metas
        do_reset = any(bool((mb.get('is_seq_start', False))) for mb in metas if isinstance(mb, dict))
        
        # Extract timestamp from filename for temporal encoding
        basename = os.path.basename(metas[0]['filename'])
        cur_timestamp = float(os.path.splitext(basename)[0].split("_")[-1])  # Extract timestamp
        group_index = int(metas[0]['group_index'])  # Position in sequence

        # Determine temporal state
        t_val = 0 if do_reset else 1  # Reset temporal queue if sequence starts
        
        # Manage timestamp queue for temporal encoding
        if group_index == 0:
            # Initialize timestamp queue for new sequence
            self.temporal_time_test_queue = [cur_timestamp] * (self.history_length + 1)
        else:
            # Update timestamp queue with current timestamp
            self.temporal_time_test_queue.pop(0)  # Remove oldest timestamp
            self.temporal_time_test_queue.append(cur_timestamp)  # Add current timestamp
            
            # Fill missing history for early frames in sequence
            if group_index + 1 < self.history_length + 1:
                for i in range(self.history_length - group_index):
                    self.temporal_time_test_queue[i] = cur_timestamp  # Duplicate current timestamp

        # Convert timestamps to tensor and perform forward pass
        ts_tensor = torch.tensor([self.temporal_time_test_queue], dtype=torch.float32)  # (1, history_length+1)
        final_output, _, _ = self._forward_stream_batch(
            inputs, t_val, timestamps=ts_tensor, basename=basename
        )
        return final_output  # (B, 1, H_out, W_out)
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, t=0, timestamps: torch.Tensor = None):
        """
        Training forward pass with loss computation.
        
        Processes current frame with temporal context and computes training losses
        including point prediction losses. Segmentation losses need custom implementation.

        Args:
            inputs (list[Tensor]): Multi-level backbone features
                [(B, C_0, H_0, W_0), (B, C_1, H_1, W_1), (B, C_2, H_2, W_2), (B, C_3, H_3, W_3)]
            img_metas (list[dict]): Image metadata for each sample in batch
            gt_semantic_seg (Tensor): Ground truth segmentation masks (B, 1, H_gt, W_gt)
                Integer class labels for each pixel
            t (int): Current time step, resets temporal state when t=0. Default: 0
            timestamps (Tensor, optional): Temporal information (B, history_length+1)
                Timestamp values for temporal encoding

        Returns:
            dict[str, Tensor]: Training losses
                - 'loss_point': Total point prediction loss
                - 'loss_point_cls': Point classification loss (existence prediction)
                - 'loss_point_reg': Point regression loss (coordinate prediction)
                - 'point_targets': Mean number of valid point targets per sample
                Note: Segmentation losses are commented out and need custom implementation
        """
        # Forward pass through streaming pipeline
        masks, confidence, points = self._forward_stream_batch(
            inputs, t, timestamps=timestamps
        )
        
        # Compute unified losses (point prediction + mask segmentation)
        total_loss, loss_components = self.unified_loss(
            pred_has_point=confidence,  # Point existence confidence: (B, 1)
            pred_points=points,  # Predicted point coordinates: (B, 1, 2)
            pred_masks=masks,  # Predicted mask logits: (B, 1, H, W)
            gt_semantic_seg=gt_semantic_seg,  # Ground truth masks: (B, 1, H_gt, W_gt)
            target_class=1,  # Target foreground class index
        )
        
        # Prepare training losses with unified loss components
        losses = {
            'loss_total': total_loss,  # Combined weighted loss
            **loss_components  # Individual loss components for monitoring
        }
        
        return losses
