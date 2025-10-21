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
from .untils import (LayerNorm2d, init_conv_module_weights, init_psp_weights, 
                     init_mlp_weights, init_attention_weights, init_encoder_weights,
                     init_predictor_weights, init_sam_weights)
from .point_predictor import PointPredictor
from .loss import otdr_loss
from .sam.prompt_encoder import PromptEncoder
from .sam.mask_decoder import MaskDecoder
from .sam.transformer import TwoWayTransformer
from .position_embed import PositionEmbeddingRandom
from .flag import get_task_state
from .psp_fpn import UPerHead

@HEADS.register_module()
class PredictiveTemporalUPerHead(nn.Module):
    
    def __init__(self, **kwargs):

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
        self.use_topk_memory = bool(kwargs.get('use_topk_memory', True))  # Enable top-K memory selection
        self.topk_memory_size = int(kwargs.get('topk_memory_size', 256))  # Top-K memory token count
        self.test_topk_memory_size = int(kwargs.get('test_topk_memory_size', 320))  # Top-K memory token count for test

        # ===== Core Components Initialization =====
        
        # Temporal processing components
        self.temporal_queue = TemporalQueue(
            history_length=self.history_length,  # Number of historical frames to store
            streaming=self.streaming  # Enable streaming mode for online inference
        )
        self.temporal_time_test_queue = None  # Test-time timestamp queue
        self.last_m1_logits = None  # Legacy placeholder for M1 outputs
        
        # # Feature processing modules
        # self._build_fpn_module()  # Build FPN for multi-scale feature fusion
        self.memory_attention = MemoryAttention()  # Cross-attention with historical features
        self.memory_encoder = MemoryEncoder(total_stride=self.mask_ratio)  # Encode historical frames
        
        # Position encoding for spatial awareness
        self.pos_embed = PositionEmbeddingSine(
            num_pos_feats=256,  # Position encoding dimension
            normalize=True,  # Normalize position values
            scale=None,  # Auto-scale based on feature size
            temperature=10000  # Temperature for sinusoidal encoding
        )
        
        self.psp_fpn = UPerHead(
            in_channels_list=self.in_channels,
            channels=self.channels,
            pool_scales=(1, 2, 3, 6, 12),   
            align_corners=self.align_corners,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        
        # Temporal embedding: maps scalar timestamps to feature embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),  # Input: scalar timestamp
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),  # Hidden layer
            nn.ReLU(inplace=True),
            nn.Linear(128, self.channels//4)  # Output: temporal embedding
        )
        
        self.pe_layer = PositionEmbeddingRandom(self.channels // 2)
        self._build_sam_decoder()

        # Unified loss function for point prediction and mask segmentation
        self.unified_loss = otdr_loss(
            cls_weight=1.0,  # Weight for point classification loss
            reg_weight=1.0,  # Weight for point regression loss
            seg_weight=1.0,  # Weight for mask segmentation loss
            normalize_by_image_size=False,  # Normalize coordinates by image size
            min_area_ratio=0.0,  # Minimum area ratio for valid targets
        )
        
        self.hist_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2
            ),
            nn.SyncBatchNorm(self.channels),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2
            ),
            nn.SyncBatchNorm(self.channels),
            nn.GELU(),
        )
        
        # Project hist_feat from C channels to 1 channel for loss computation
        self.hist_mask_proj = nn.Conv2d(self.channels, 1, kernel_size=3, padding=1)



    def init_weights(self):
        """Initialize weights for all components using utils functions."""
        # Initialize temporal MLP
        init_mlp_weights(self.time_mlp)
        
        # Initialize memory attention
        if hasattr(self.memory_attention, 'init_weights'):
            self.memory_attention.init_weights()
        else:
            init_attention_weights(self.memory_attention)
            
        # Initialize memory encoder
        if hasattr(self.memory_encoder, 'init_weights'):
            self.memory_encoder.init_weights()
        else:
            init_encoder_weights(self.memory_encoder)
            
        init_sam_weights(self.mask_decoder)
        
        # Initialize hist mask projection
        nn.init.xavier_uniform_(self.hist_mask_proj.weight)
        if self.hist_mask_proj.bias is not None:
            nn.init.constant_(self.hist_mask_proj.bias, 0)

    def _build_sam_decoder(self):

        # SAM decoder configuration
        self.sam_prompt_embed_dim = self.channels  # Embedding dimension for prompts
        self.sam_prompt_image_embedding_size = (32, 32)  # Training image embedding size
        self.sam_prompt_test_image_embedding_size = (32, 40)  # Test image embedding size
        self.sam_prompt_input_image_size = (512, 512)  # Training input image size
        self.sam_prompt_test_image_size = (512, 640)  # Test input image size

        # Prompt encoder: converts points and confidence to embeddings
        # self.prompt_encoder = PromptEncoder(
        #     embed_dim=self.sam_prompt_embed_dim,  # Embedding dimension
        #     image_embedding_size=self.sam_prompt_image_embedding_size,  # Feature map size
        #     input_image_size=self.sam_prompt_input_image_size,  # Input image resolution
        #     test_image_embedding_size=self.sam_prompt_test_image_embedding_size,  # Test feature size
        #     test_input_image_size=self.sam_prompt_test_image_size  # Test image resolution
        # )
        
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

    def _get_memory_base(self, cur_features, t=0, timestamps: torch.Tensor = None):
        """Get base memory features before formatting (shared computation)."""
        B, C, H, W = cur_features.shape
        mask_shape = (self.num_classes - 1, H * self.mask_ratio, W * self.mask_ratio)
        self.temporal_queue.ensure_allocation(cur_features, mask_shape)
        
        if int(t) == 0:
            self.temporal_queue.reset_state(all_batch=True)

        historical_frames, historical_masks = self.temporal_queue.get_history_frames()
        batch_frames = historical_frames.view(-1, *historical_frames.shape[2:])
        batch_masks = historical_masks.view(-1, *historical_masks.shape[2:])

        # Process temporal encoding
        timestamps = timestamps.to(cur_features.device, dtype=cur_features.dtype)
        tmin = timestamps.min(dim=1, keepdim=True)[0]
        tmax = timestamps.max(dim=1, keepdim=True)[0]
        denom = (tmax - tmin).clamp_min(1e-6)
        timestamps = (timestamps - tmin) / denom
        
        historical_timestamps = timestamps[:, :-1].transpose(0, 1)
        batch_timestamps = historical_timestamps.reshape(-1, 1)
        batch_time_enc = self.time_mlp(batch_timestamps)
        batch_time_enc = batch_time_enc.unsqueeze(2).unsqueeze(3)
        batch_time_enc = batch_time_enc.expand(-1, -1, *historical_frames.shape[3:])

        # Encode memory features
        batch_memory_output = self.memory_encoder(batch_frames, batch_masks)
        mem_features = batch_memory_output["vision_features"]
        batch_pos_enc = batch_memory_output["vision_pos_enc"]
        batch_pos_enc = batch_pos_enc + batch_time_enc
        
        return mem_features, batch_pos_enc, batch_masks

    def _format_memory_full(self, mem_features, batch_pos_enc, B):
        """Format base memory as full spatial memory for attention."""
        # Use all spatial locations (T*H*W, B, C)
        batch_pos_enc = batch_pos_enc.view(-1, B, *batch_pos_enc.shape[1:])
        memory_enc = batch_pos_enc.permute(0, 3, 4, 1, 2).flatten(0, 2)
        
        mem_features = mem_features.view(-1, B, *mem_features.shape[1:])
        mem_features = mem_features.permute(0, 3, 4, 1, 2).flatten(0, 2)
        
        return memory_enc, mem_features

    def _forward_stream_batch(self, inputs, t: int, timestamps: torch.Tensor = None, **kargs):

        fpn_outs = self.psp_fpn(inputs) # List(tensor(B, C, H, W))
        cur_features = fpn_outs[-1] 
        B, C, H, W = cur_features.shape 

        cur_pos_enc = self.pos_embed(cur_features)
        cur_features_seq = cur_features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        cur_pos_enc_seq = cur_pos_enc.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
    
        base_mem_features, base_pos_enc, batch_masks = self._get_memory_base(cur_features, t, timestamps)
        memory_enc_full, mem_features_full = self._format_memory_full(base_mem_features, base_pos_enc, B)
        
        fus_feat, hist_feat = self.memory_attention(
            cur_features_seq,  # Current frame queries: (H*W, B, C)
            mem_features_full,  # Full historical memory: (T*H*W, B, C)
            cur_pos_enc_seq,  # Current position encodings: (H*W, B, C)
            memory_enc_full,  # Full memory position encodings: (T*H*W, B, C)
            spatial_shape=(H, W),  # Spatial dimensions for RoPE
            **kargs
        )
        fus_feat = fus_feat.permute(1, 2, 0).reshape(B, C, H, W)  # Reshape back: (B, C, H, W)
        hist_feat = hist_feat.permute(1, 2, 0).reshape(B, C, H, W)  # Reshape back: (B, C, H, W)
        hist_feat = self.hist_upscaling(hist_feat)
        hist_feat = self.hist_mask_proj(hist_feat)  # Project to 1 channel: (B, 1, H, W)

        masks_fine, masks_mid, masks_coarse = self.mask_decoder.forward(
            ori_embeddings = cur_features,
            image_embeddings=fus_feat,  # Fused features: (B, C, H, W)
            image_pe=self.pe_layer(self.sam_prompt_image_embedding_size if self.training else self.sam_prompt_test_image_embedding_size).unsqueeze(0),  # Dense position encodings: (1, C, H, W)
            # sparse_prompt_embeddings=sparse_prompt_embeddings,  # Point prompts: (B, N, C)
            # hist_cont_prompt_embeddings=final_global_token,  # Historical context: (B, 1, C)
            high_res_features=fpn_outs[:-1],  # High-resolution features (empty for now),
            step=t
        )

        self.temporal_queue.push(cur_features, masks_fine)  # Store features and masks

        return masks_fine, masks_mid, masks_coarse, hist_feat
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, t=0, timestamps: torch.Tensor = None):
        
        masks_list = self._forward_stream_batch(
            inputs, t, timestamps=timestamps
        )
        losses = self.unified_loss(
            pred_masks=masks_list,  # Predicted mask logits: (B, 1, H, W)
            gt_semantic_seg=gt_semantic_seg,  # Ground truth masks: (B, 1, H_gt, W_gt)
            target_class=1
        )
        
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):

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
        final_output, _, _, hist_feat = self._forward_stream_batch(
            inputs, t_val, timestamps=ts_tensor, basename=basename
        )
        return final_output, None, None# (B, 1, H_out, W_out)

    def forward(self, inputs):
        return None
    