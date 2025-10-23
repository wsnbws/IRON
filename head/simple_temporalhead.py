import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from typing import Optional, Tuple
import math
from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmcv.cnn import ConvModule
from .memory_attention import MemoryAttention
from .position_embed import PositionEmbeddingSine
from .history_queue import TemporalQueue
from .untils import (init_mlp_weights, init_attention_weights)
from .psp_fpn import UPerHead
from .flag import get_task_state
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class DeconvFusionHead(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, num_classes=2, use_concat=False):
        """
        Args:
            in_channels (int): 各尺度特征通道（已对齐）
            out_channels (int): 融合后的通道数
            num_classes (int): 输出类别数
            use_concat (bool): True->concat融合, False->相加融合
        """
        super().__init__()
        self.use_concat = use_concat
        fusion_in_ch = in_channels * 2 if use_concat else in_channels

        norm_cfg = dict(type='SyncBN', requires_grad=True)

        # ---- 32x32 -> 64x64 ----
        self.deconv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.fuse1 = ConvModule(
            fusion_in_ch, out_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')
        )

        # ---- 64x64 -> 128x128 ----
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.fuse2 = ConvModule(
            fusion_in_ch, out_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')
        )

        # ---- 128x128 -> 256x256 ----
        self.deconv3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.fuse3 = ConvModule(
            out_channels, out_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')
        )

        # ---- 256x256 -> 512x512 ----
        self.deconv4 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.refine = ConvModule(
            out_channels, out_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')
        )

        # ---- 最终分类层 ----
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, feats):
        """
        Args:
            feats: [f1(128x128), f2(64x64), f3(32x32)]
        Returns:
            seg_out: (B, num_classes, 512, 512)
        """
        f1, f2, f3 = feats

        # ---- 32x32 -> 64x64 ----
        f3_up = self.deconv1(f3)
        f2_fuse = torch.cat([f2, f3_up], dim=1) if self.use_concat else f2 + f3_up
        f2_out = self.fuse1(f2_fuse)

        # ---- 64x64 -> 128x128 ----
        f2_up = self.deconv2(f2_out)
        f1_fuse = torch.cat([f1, f2_up], dim=1) if self.use_concat else f1 + f2_up
        f1_out = self.fuse2(f1_fuse)

        # ---- 128x128 -> 256x256 ----
        x = self.deconv3(f1_out)
        x = self.fuse3(x)

        # ---- 256x256 -> 512x512 ----
        x = self.deconv4(x)
        x = self.refine(x)

        # ---- 分类输出 ----
        seg_out = self.classifier(x)
        return seg_out

@HEADS.register_module()
class SimplifiedTemporalUPerHead(nn.Module):
    """
    Simplified Temporal UPerHead with DeconvFusionHead integration.
    
    This head combines:
    1. Temporal memory attention for historical frame processing
    2. Multi-scale feature fusion using DeconvFusionHead
    3. Progressive upsampling with deconvolution layers
    
    Key features:
    - Memory attention applied to multiple scales
    - DeconvFusionHead for progressive feature fusion
    - Configurable segmentation head (DeconvFusionHead or simple head)
    """
    
    def __init__(self, **kwargs):
        super(SimplifiedTemporalUPerHead, self).__init__()
    
        # Base decoder configuration
        self.in_channels = kwargs.get('in_channels', [256, 512, 1024, 2048])
        self.in_index = kwargs.get('in_index', [0, 1, 2, 3])
        self.channels = kwargs.get('channels', 512)
        self.dropout_ratio = kwargs.get('dropout_ratio', 0.1)
        self.num_classes = kwargs.get('num_classes', 2)
        self.align_corners = kwargs.get('align_corners', False)
        self.ignore_index = kwargs.get('ignore_index', 255)
        
        # Layer configuration
        self.conv_cfg = kwargs.get('conv_cfg', None)
        self.norm_cfg = kwargs.get('norm_cfg', dict(type='SyncBN', requires_grad=True))
        self.act_cfg = kwargs.get('act_cfg', dict(type='ReLU'))
        self.input_transform = kwargs.get('input_transform', 'multiple_select')
        
        # Temporal processing configuration
        self.streaming = bool(kwargs.get('streaming', False))
        self.history_length = int(kwargs.get('history_length', 2))
        
        # Segmentation head configuration
        self.use_deconv_fusion = bool(kwargs.get('use_deconv_fusion', True))  # Default to DeconvFusionHead
        
        # ===== Core Components Initialization =====
        
        # Temporal queue to store historical features (without masks)
        self.temporal_queue = TemporalQueue(
            history_length=self.history_length,
            streaming=self.streaming
        )
        self.temporal_time_test_queue = None
        
        # Memory attention for cross-attention with historical features
        self.memory_attention = MemoryAttention()
        
        # Position encoding for spatial awareness
        self.pos_embed = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000
        )
        self.hist_pos_embed = PositionEmbeddingSine( 
                                num_pos_feats=64,
                                normalize=True,
                                scale=None,
                                temperature=10000
                                )
        
        # FPN for multi-scale feature fusion
        self.psp_fpn = UPerHead(
            in_channels_list=self.in_channels,
            channels=self.channels,
            pool_scales=(1, 2, 3, 6),   
            align_corners=self.align_corners,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        
        # Temporal embedding: maps scalar timestamps to feature embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )
        
        # DeconvFusionHead: multi-scale feature fusion and segmentation
        self.deconv_fusion_head = DeconvFusionHead(
            in_channels=self.channels,
            out_channels=self.channels,
            num_classes=self.num_classes,
            use_concat=kwargs.get('use_concat_fusion', False)  # Default to additive fusion
        )
        
        self.hist_linear = ConvModule(
                            in_channels=256,        # 原通道数
                            out_channels=64,       # 目标通道数
                            kernel_size=1,          # 不改变空间尺寸
                            stride=1,
                            padding=0,
                            norm_cfg=dict(type='SyncBN', requires_grad=True),  # 分布式BN
                            act_cfg=dict(type='ReLU'),                         # 可改成 None
                            inplace=True
                        )
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def init_weights(self):
        """Initialize weights for all components."""
        # Initialize temporal MLP
        init_mlp_weights(self.time_mlp)
        
        # Initialize memory attention
        if hasattr(self.memory_attention, 'init_weights'):
            self.memory_attention.init_weights()
        else:
            init_attention_weights(self.memory_attention)
        
        # Initialize DeconvFusionHead
        for m in self.deconv_fusion_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def _get_memory_features(self, cur_features, t=0, timestamps: torch.Tensor = None):
        """Get historical features and prepare for attention."""
        B, C, H, W = cur_features.shape
        dummy_mask_shape = (1, H, W)
        self.temporal_queue.ensure_allocation(cur_features, dummy_mask_shape)
        if int(t) == 0:
            self.temporal_queue.reset_state(all_batch=True)
        historical_frames, _ = self.temporal_queue.get_history_frames()  # (T, B, C, H, W) 
        batch_frames = historical_frames.permute(1, 0, 2, 3, 4).contiguous().view(-1, C, H, W)
        batch_frames = self.hist_linear(batch_frames)
        timestamps = timestamps.to(cur_features.device, dtype=cur_features.dtype)
        normalized_timestamps = self._normalize_timestamps(timestamps)
        historical_timestamps = normalized_timestamps[:, :-1]  # (B, T)
        batch_timestamps = historical_timestamps.reshape(-1, 1)
        batch_time_enc = self.time_mlp(batch_timestamps)  # (B*T, C)
        batch_time_enc = batch_time_enc.unsqueeze(2).unsqueeze(3)  # (B*T, C, 1, 1)
        batch_time_enc = batch_time_enc.expand(-1, -1, H, W)  # (B*T, C, H, W)
        batch_frames = batch_frames + batch_time_enc
        batch_pos_enc = self.hist_pos_embed(batch_frames)  # (B*T, C, H, W)
        
        return batch_frames, batch_pos_enc, B

    def _format_memory_for_attention(self, mem_features, batch_pos_enc, B):
        """Format memory features for attention mechanism with optimized memory usage."""
        T = self.history_length
        mem_features = mem_features.view(T, B, *mem_features.shape[1:])  # (T, B, C, H, W)
        batch_pos_enc = batch_pos_enc.view(T, B, *batch_pos_enc.shape[1:])  # (T, B, C, H, W)
        mem_features_seq = mem_features.permute(0, 3, 4, 1, 2).flatten(0, 2)  # (T*H*W, B, C)
        memory_enc_seq = batch_pos_enc.permute(0, 3, 4, 1, 2).flatten(0, 2)  # (T*H*W, B, C)
        
        return memory_enc_seq, mem_features_seq
    
    def _normalize_timestamps(self, timestamps):
        """Normalize timestamps to [0, 1] range per batch."""
        B = timestamps.shape[0]
        tmin = timestamps.min(dim=1, keepdim=True)[0]
        tmax = timestamps.max(dim=1, keepdim=True)[0]
        denom = (tmax - tmin).clamp_min(1e-6)
        return (timestamps - tmin) / denom

    def _forward_stream_batch(self, inputs, t: int, timestamps: torch.Tensor = None, **kwargs):
        """Forward pass for streaming batch processing."""
        # Extract multi-scale features
        fpn_outs = self.psp_fpn(inputs)  # List[Tensor(B, C, H, W)]
        cur_features = fpn_outs[-1]  # Use highest level features
        B, C, H, W = cur_features.shape

        # Compute position encoding for current frame
        cur_pos_enc = self.pos_embed(cur_features)
        cur_features_seq = cur_features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        cur_pos_enc_seq = cur_pos_enc.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
    
        # Get historical features (no masks needed)
        mem_features, mem_pos_enc, B = self._get_memory_features(cur_features, t, timestamps)
        
        # Format memory for attention
        memory_enc_full, mem_features_full = self._format_memory_for_attention(
            mem_features, mem_pos_enc, B
        )
        
        # Apply memory attention
        fus_feat, _ = self.memory_attention(
            cur_features_seq,  # Current frame queries: (H*W, B, C)
            mem_features_full,  # Historical memory: (T*H*W, B, C)
            cur_pos_enc_seq,  # Current position encodings: (H*W, B, C)
            memory_enc_full,  # Memory position encodings: (T*H*W, B, C)
            spatial_shape=(H, W),  # Spatial dimensions
            **kwargs
        )
        
        # Reshape back to spatial dimensions
        fus_feat = fus_feat.permute(1, 2, 0).reshape(B, C, H, W)  # (B, C, H, W)
        multi_scale_feats = fpn_outs[:-1] + [fus_feat]

        masks_fine = self.deconv_fusion_head(multi_scale_feats)

        dummy_mask = torch.zeros(
            (fus_feat.shape[0], 1, fus_feat.shape[2], fus_feat.shape[3]),
            device=fus_feat.device,
            dtype=fus_feat.dtype
        )
        self.temporal_queue.push(fus_feat, dummy_mask)
        
        return masks_fine
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, t=0, timestamps: torch.Tensor = None, current_iter=None,**ignore_kargs):
        """Training forward pass."""
        seg_logits = self._forward_stream_batch(
            inputs, t, timestamps=timestamps, current_iter=current_iter
        )
        
        # Compute loss
        losses = {}
        losses['loss_seg'] = self.criterion(seg_logits, gt_semantic_seg.squeeze(1).long())
        
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, **ignore_kargs):
        """Testing forward pass."""
        # Extract metadata and determine sequence state
        metas = img_metas
        do_reset = any(bool((mb.get('is_seq_start', False))) for mb in metas if isinstance(mb, dict))
        
        # Extract timestamp from filename
        basename = os.path.basename(metas[0]['filename'])
        cur_timestamp = float(os.path.splitext(basename)[0].split("_")[-1])
        group_index = int(metas[0]['group_index'])

        # Determine temporal state
        t_val = 0 if do_reset else 1
        
        # Manage timestamp queue
        if group_index == 0:
            self.temporal_time_test_queue = [cur_timestamp] * (self.history_length + 1)
        else:
            self.temporal_time_test_queue.pop(0)
            self.temporal_time_test_queue.append(cur_timestamp)
            
            if group_index + 1 < self.history_length + 1:
                for i in range(self.history_length - group_index):
                    self.temporal_time_test_queue[i] = cur_timestamp

        # Convert timestamps to tensor and perform forward pass
        ts_tensor = torch.tensor([self.temporal_time_test_queue], dtype=torch.float32)
        seg_logits = self._forward_stream_batch(inputs, t_val, timestamps=ts_tensor)
        
        return seg_logits

    def forward(self, inputs):
        """Placeholder for base forward."""
        return None