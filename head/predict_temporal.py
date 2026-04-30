import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import re
from typing import Optional, Tuple
import math
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
from .loss import otdr_loss
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
        self.logic_history_length = int(kwargs.get('history_length', 2))  # Logical history count
        self.frame_stride = int(kwargs.get('frame_stride', 1)) # Frame stride
        self.history_length = self.logic_history_length * self.frame_stride # Physical history length for buffer
        self.mask_ratio = int(kwargs.get('mask_ratio', 8))  # Mask downsampling ratio
        
        # Memory selection configuration
        self.use_topk_memory = bool(kwargs.get('use_topk_memory', True))  # Enable top-K memory selection
        self.topk_memory_size = int(kwargs.get('topk_memory_size', 256))  # Top-K memory token count
        self.test_topk_memory_size = int(kwargs.get('test_topk_memory_size', 320))  # Top-K memory token count for test

        self.use_queue_hint = bool(kwargs.get('use_queue_hint', True))
        self.sam_prompt_test_image_embedding_size = kwargs.get('sam_prompt_test_image_embedding_size', (45, 80))
        self.sam_prompt_test_image_size = kwargs.get('sam_prompt_test_image_size', (720, 1280))

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
        
        # self.hist_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         self.channels, self.channels, kernel_size=2, stride=2
        #     ),
        #     nn.SyncBatchNorm(self.channels),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(
        #         self.channels, self.channels, kernel_size=2, stride=2
        #     ),
        #     nn.SyncBatchNorm(self.channels),
        #     nn.GELU(),
        # )
        
        # # Project hist_feat from C channels to 1 channel for loss computation
        # self.hist_mask_proj = nn.Conv2d(self.channels, 1, kernel_size=3, padding=1)



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
        
        # # Initialize hist mask projection
        # nn.init.xavier_uniform_(self.hist_mask_proj.weight)
        # if self.hist_mask_proj.bias is not None:
        #     nn.init.constant_(self.hist_mask_proj.bias, 0)

    def _build_sam_decoder(self):

        # SAM decoder configuration
        self.sam_prompt_embed_dim = self.channels  # Embedding dimension for prompts
        self.sam_prompt_image_embedding_size = (32, 32)  # Training image embedding size
        self.sam_prompt_input_image_size = (512, 512)  # Training input image size

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

        # Handle history sampling based on mode
        if self.training:
            # In training, inputs are already strided (physically adjacent pushes are logically adjacent)
            # So we just take the last 'logic_history_length' frames from the queue
            if historical_frames is not None:
                historical_frames = historical_frames[-self.logic_history_length:]
                historical_masks = historical_masks[-self.logic_history_length:]
        elif self.streaming and self.frame_stride > 1:
            # In inference, inputs are continuous, so we sample with stride from the physical buffer
            # We want: [t - stride*(L-1), ..., t - stride, t] (current frame is separate)
            # The queue stores [t - history_length, ..., t-1] (oldest to newest)
            # Slicing logic: take every Kth element starting from the appropriate offset
            
            # Example: logic=2, stride=4. history_len = 8.
            # Queue: [t-8, t-7, t-6, t-5, t-4, t-3, t-2, t-1]
            # We want: t-4 (index 4).
            # If logic=3: Queue len 12. We want t-8, t-4.
            
            # Start index: length - 1 - (logic-1)*stride
            start_idx = self.history_length - 1 - (self.logic_history_length - 1) * self.frame_stride
            # Ensure start_idx is non-negative (should be 0 if math is right)
            start_idx = max(0, start_idx)
            
            historical_frames = historical_frames[start_idx::self.frame_stride]
            historical_masks = historical_masks[start_idx::self.frame_stride]
            
            if timestamps is not None:
                # Slice timestamps to match logic history
                # timestamps passed here has shape (B, Physical_History + 1)
                # We need to slice it to (B, Logic_History + 1)
                
                # Split history and current
                T_phys = timestamps.shape[1] - 1
                hist_ts = timestamps[:, :T_phys]
                cur_ts = timestamps[:, T_phys:]
                
                # Sample history timestamps using the same logic
                hist_ts = hist_ts[:, start_idx::self.frame_stride]
                
                # Recombine
                timestamps = torch.cat([hist_ts, cur_ts], dim=1)

        queue_has_hint = self._check_queue_hints(historical_masks)
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
        batch_memory_output = self.memory_encoder(batch_frames, batch_masks, skip_mask_sigmoid=True)
        mem_features = batch_memory_output["vision_features"]
        batch_pos_enc = batch_memory_output["vision_pos_enc"]
        batch_pos_enc = batch_pos_enc + batch_time_enc
        
        return mem_features, batch_pos_enc, batch_masks, queue_has_hint

    def _check_queue_hints(self, historical_masks, ratio=0.1):
        """ hist T,B,1,H,W """
        T, B, C, mask_H, mask_W = historical_masks.shape

        high_confidence_pixels = (historical_masks > 0.5).sum(dim=(-2, -1))  # (T, B, 1)

        max_t_batch_pixels = high_confidence_pixels.max(dim=0).values  # (B, 1)

        total_pixels = mask_H * mask_W
        threshold_pixels = total_pixels * ratio

        queue_has_hint = (max_t_batch_pixels > threshold_pixels).float()  # (B, 1)

        return queue_has_hint


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
    
        base_mem_features, base_pos_enc, batch_masks, queue_has_hint = self._get_memory_base(cur_features, t, timestamps)
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
        # hist_feat = hist_feat.permute(1, 2, 0).reshape(B, C, H, W)  # Reshape back: (B, C, H, W)
        # hist_feat = self.hist_upscaling(hist_feat)
        # hist_feat = self.hist_mask_proj(hist_feat)  # Project to 1 channel: (B, 1, H, W)

        masks_fine, masks_mid, masks_coarse, masks_empty = self.mask_decoder.forward(
            ori_embeddings = cur_features,
            image_embeddings=fus_feat,  # Fused features: (B, C, H, W)
            image_pe=self.pe_layer(self.sam_prompt_image_embedding_size if self.training else self.sam_prompt_test_image_embedding_size).unsqueeze(0),  # Dense position encodings: (1, C, H, W)
            # sparse_prompt_embeddings=sparse_prompt_embeddings,  # Point prompts: (B, N, C)
            # hist_cont_prompt_embeddings=final_global_token,  # Historical context: (B, 1, C)
            high_res_features=fpn_outs[:-1],  # High-resolution features (empty for now),
            step=t,
            queue_has_hint=queue_has_hint 
        )
        
        if self.training:
            use_pred_prob = self.sigmoid_schedule(kargs["current_iter"]["now"], kargs["current_iter"]["total"])
            use_pred = np.random.rand(1) < use_pred_prob
            gt_seg = kargs["gt_seg"].to(dtype=torch.float32)
            gt_mask = F.interpolate(gt_seg if get_task_state() == 1 else (1- gt_seg), scale_factor=0.25, mode='bilinear', align_corners=False)
            mask2push = F.sigmoid(masks_fine) if use_pred else gt_mask
            self.temporal_queue.push(cur_features, mask2push)  # Store features and masks
        else:
            self.temporal_queue.push(cur_features, F.sigmoid(masks_fine))  # Store features and masks

        return masks_fine, masks_mid, masks_coarse, None, masks_empty
    
    def sigmoid_schedule(self, step, total_steps, k=10):
        # x = (step / total_steps) * 2 - 1  # [-1, 1] 范围
        # return 1 / (1 + math.exp(-k * x))  # 输出 [0,1]
        return step / total_steps
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, t=0, timestamps: torch.Tensor = None, current_iter=None):
        
        masks_list = self._forward_stream_batch(
            inputs, t, timestamps=timestamps, gt_seg = gt_semantic_seg, current_iter=current_iter
        )
        losses = self.unified_loss(
            pred_masks=masks_list,  # Predicted mask logits: (B, 1, H, W)
            gt_semantic_seg=gt_semantic_seg,  # Ground truth masks: (B, 1, H_gt, W_gt)
            target_class=get_task_state()
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
        # ts_tensor = torch.tensor([self.temporal_time_test_queue], dtype=torch.float32)  # (1, history_length+1)
        ts_tensor = torch.arange(len(self.temporal_time_test_queue), dtype=torch.float32).unsqueeze(0)
        final_output, _, _, hist_feat,_ = self._forward_stream_batch(
            inputs, t_val, timestamps=ts_tensor, basename=basename
        )
        return final_output, None, None# (B, 1, H_out, W_out)

    def forward(self, inputs):
        return None
    