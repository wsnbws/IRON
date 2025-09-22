import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmcv.cnn import ConvModule
from .memory_encoder import MemoryEncoder
from .memory_attention import MemoryAttention
from .position_embed import PositionEmbeddingSine
from .history_queue import TemporalQueue
from .untils import LayerNorm2d

@HEADS.register_module()
class PredictiveTemporalUPerHead(BaseDecodeHead):
    """
    Predictive Temporal UPerHead that uses T-2 and T-1 frames to predict T frame,
    then fuses with current frame T for final prediction.
    """
    
    def __init__(self, **kwargs):
        # Pop custom streaming args before passing to parent
        self.streaming = bool(kwargs.pop('streaming', False))
        self.detach_every = int(kwargs.pop('detach_every', 0))
        self.history_length = int(kwargs.pop('history_length', 2))  # Configurable history length
        self.mask_ratio = int(kwargs.pop('mask_ratio', 8))
        super(PredictiveTemporalUPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        # ===== Streaming (true online) inference/training state (no new learnable params) =====
        self.temporal_queue = TemporalQueue(
            history_length=self.history_length, 
            streaming=self.streaming
        )
        self.temporal_time_test_queue = None

        self.last_m1_logits = None
        self._build_fpn_module()
        self.memory_attention = MemoryAttention()
        self.memory_encoder = MemoryEncoder(total_stride=self.mask_ratio)
        self.pos_embed = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000
        )
        # time embedding: simple MLP mapping scalar timestamp -> channel embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.channels//4)
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2
            ),
            LayerNorm2d(self.channels),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2
            ),
            nn.GELU(),
        )

   
    def _build_fpn_module(self):
     # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels:
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
    # ===== Helpers for streaming mode =====
    def _fpn_forward_single(self, inputs):
        """UPer forward for a single frame batch. Returns (B, C, H, W) features after fpn_bottleneck."""
        inputs = self._transform_inputs(inputs)

        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]
        return fpn_outs

    def _get_memory(self, cur_features, t = 0, timestamps: torch.Tensor = None):
        B, C, H, W = cur_features.shape 
        # Ensure queue allocation
        mask_shape = (self.num_classes - 1, H * self.mask_ratio, W * self.mask_ratio)
        self.temporal_queue.ensure_allocation(cur_features, mask_shape)
        # Reset at sequence start
        if int(t) == 0:
            self.temporal_queue.reset_state(all_batch=True)

        # Read historical frames from queue
        # Clone to decouple from underlying queue storage to avoid inplace version conflicts
        historical_frames, historical_masks = self.temporal_queue.get_history_frames() # (T, B, C, H, W)
        
        # Reshape for batch processing: (T*B, C, H, W)
        batch_frames = historical_frames.view(-1, *historical_frames.shape[2:])
        batch_masks = historical_masks.view(-1, *historical_masks.shape[2:])

        # Vectorized time embedding processing
        timestamps = timestamps.to(cur_features.device, dtype=cur_features.dtype)
        # normalize UNIX timestamps per-sample to [0, 1]
        tmin = timestamps.min(dim=1, keepdim=True)[0]
        tmax = timestamps.max(dim=1, keepdim=True)[0]
        denom = (tmax - tmin).clamp_min(1e-6)
        timestamps = (timestamps - tmin) / denom
        
        historical_timestamps = timestamps[:, :-1].transpose(0, 1)                                    # (num_historical, B)
        batch_timestamps = historical_timestamps.reshape(-1, 1)                                       # (num_historical * B, 1)
        batch_time_enc = self.time_mlp(batch_timestamps).unsqueeze(2).unsqueeze(3).expand(-1, -1, *historical_frames.shape[3:]) # (num_historical * B, C_time_embed, H, W)

        # Single forward pass for all historical frames
        batch_memory_output = self.memory_encoder(batch_frames, batch_masks)
        mem_features = batch_memory_output["vision_features"]  # (T*B, C, H, W)
        batch_pos_enc = batch_memory_output["vision_pos_enc"]    # (T*B, C, H, W)
        batch_pos_enc = batch_pos_enc + batch_time_enc
        batch_pos_enc = batch_pos_enc.view(-1, B, *batch_pos_enc.shape[1:]) #(T, B, C, H, W)
        memory_enc = batch_pos_enc.permute(0, 3, 4, 1, 2).flatten(0, 2) #(T*H*W, B,C)
        
        mem_features = mem_features.view(-1, B, *mem_features.shape[1:])
        mem_features = mem_features.permute(0, 3, 4, 1, 2).flatten(0, 2)

        return memory_enc, mem_features

    def _forward_stream_batch(self, inputs, t: int, timestamps: torch.Tensor = None, **kargs):
        """Streaming forward for a batch of single-frame features using per-slot memory.

        Args:
            inputs: list of tensors for a single time step, each (B, C, H, W)
            t: current time index; reset state when t == 0
            timestamps: (B, 3) float tensor for [t-2, t-1, t] (optional)

        Returns:
            M1: (B, num_classes, H, W)
            final_output: (B, num_classes, H, W)
        """
        # Compute current frame features (B, C, H, W)
        fpn_outs = self._fpn_forward_single(inputs)
        if len(fpn_outs) > 1:
            cur_features = fpn_outs[-1]  # (B, 768, 32, 32)
        B, C, H, W = cur_features.shape 
        
        cur_features = self._corrupt_current_features(cur_features)
        cur_pos_enc = self.pos_embed(cur_features)  # (B, C, H, W)
        cur_features_seq = cur_features.flatten(2).permute(2, 0, 1)  # (S, B, C)
        cur_pos_enc_seq = cur_pos_enc.flatten(2).permute(2, 0, 1)
    
        memory_enc, mem_features = self._get_memory(cur_features, t, timestamps)
        
        fus_feat = self.memory_attention(cur_features_seq, mem_features, cur_pos_enc_seq, memory_enc, spatial_shape=(H, W), **kargs)
        fus_feat = fus_feat.permute(1, 2, 0).reshape(B, C, H, W)  # (bs, c, h, w)

        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = fpn_outs[0], fpn_outs[1]
        upscaled_embedding = act1(ln1(dc1(fus_feat) + 0*feat_s1))
        upscaled_embedding = act2(dc2(upscaled_embedding) + 0*feat_s0)
        final_output = self.cls_seg(upscaled_embedding)

        # Update queues: keep pure current features and final outputs
        self.temporal_queue.push(cur_features, final_output[:,-1:,:,:])

        return final_output

    def _corrupt_current_features(self, features: torch.Tensor) -> torch.Tensor:

        if not self.training:
            return features
            
        corruption_prob = 0.8
        if torch.rand(1).item() > corruption_prob:
            return features
            
        corrupted = features.clone()
        B, C, H, W = corrupted.shape
        
        dropout_ratio = 0.5
        mask = torch.rand(B, 1, H, W, device=features.device) > dropout_ratio
        corrupted = corrupted * mask.float()

        return corrupted

    def forward(self, inputs):
        return None
    
    def forward_test(self, inputs, img_metas, test_cfg):
        """Test-time forward. Supports streaming with single-frame inputs.

        If ``self.streaming`` is True, expects per-step single-frame features
        and uses ``img_metas`` to determine sequence start for reset.
        """
        metas = img_metas
        do_reset = any(bool((mb.get('is_seq_start', False))) for mb in metas if isinstance(mb, dict))
        basename = os.path.basename(metas[0]['filename'])
        cur_timestamp = float(os.path.splitext(basename)[0].split("_")[-1])
        group_index = int(metas[0]['group_index'])

        t_val = 0 if do_reset else 1
        if group_index == 0:
            self.temporal_time_test_queue = [cur_timestamp] * (self.history_length + 1)
        else:
            self.temporal_time_test_queue.pop(0)
            self.temporal_time_test_queue.append(cur_timestamp)
            if group_index + 1 < self.history_length + 1:
                for i in range(self.history_length - group_index):
                    self.temporal_time_test_queue[i] = cur_timestamp

        ts_tensor = torch.tensor([self.temporal_time_test_queue], dtype=torch.float32)
        final_output = self._forward_stream_batch(inputs, t_val, timestamps=ts_tensor, basename = basename)
        return final_output
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, t=0, timestamps: torch.Tensor = None,):
        """Forward function for training with two heads (M1 and final).

        Computes losses for both M1 and final outputs using the base ``losses``.
        """
        # Streaming path: inputs contain only current frame features (B, ...)
        final_output = self._forward_stream_batch(inputs, t, timestamps=timestamps)
        loss_final = self.losses(final_output, gt_semantic_seg)
        losses = {}
        for key, value in loss_final.items():
            losses[f'{key}_final'] = value
        return losses
